#!/usr/bin/env python3
"""Profile Tenstorrent device power and host CPU energy for one benchmark run.

Host energy (RAPL)
------------------
Linux exposes one top-level RAPL zone per *die* on first-generation AMD EPYC
(node117: package-0-die-0 ... package-1-die-7), but every die zone of a socket
reports the SAME socket-wide counter.  Summing all top-level zones therefore
counts each socket once per die (4x on node117).  This profiler reads exactly
one zone per physical package (the lowest-numbered zone whose name starts with
``package-<N>``), which is correct both there and on Intel hosts that expose a
single ``package-<N>`` zone per socket.  Sub-zones (core, uncore, dram) are
never added.  Counters are unwrapped with ``max_energy_range_uj``.

Host and device are sampled by two independent threads: RAPL every
``--rapl-interval`` seconds (0.1 s by default) and tt-smi as fast as it
returns (about 0.7 s on node117), each sample stamped with the time at which
it was actually read.

Outputs (unchanged names, so the existing analysis script keeps working):
  power_log_<nlist>_<nprobe>_tt.csv   Timestamp,Power_Watts   (device)
  power_log_<nlist>_<nprobe>_cpu.csv  Timestamp,Power_Watts   (host, derived)
  energy_summary_<nlist>_<nprobe>_<target>.csv
  events.csv, command.txt, benchmark.log
plus, for auditing:
  rapl_zones.csv      which zones were read
  rapl_counters.csv   raw cumulative counters of the selected zones
"""

import argparse
import json
import math
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TT_EXECUTABLE_PATH = "./build/programming_examples/ann_ivf_energy"
FAISS_SCRIPT_PATH = "run_faiss_benchmark.py"
DATASET = "glove-100-angular"
K = 10
RAPL_ROOT = Path("/sys/class/powercap")
TOP_LEVEL_ZONE = re.compile(r"^intel-rapl:(\d+)$")
PACKAGE_NAME = re.compile(r"^package-(\d+)")


# --------------------------------------------------------------------------
# tt-smi
# --------------------------------------------------------------------------
def _numeric_power(value, key_hint=""):
    if isinstance(value, dict):
        for nested_key in ("value", "current", "reading"):
            if nested_key in value:
                return _numeric_power(value[nested_key], key_hint)
        return None
    if isinstance(value, (int, float)):
        power = float(value)
        unit_text = key_hint.lower()
    elif isinstance(value, str):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        if match is None:
            return None
        power = float(match.group(0))
        unit_text = f"{key_hint} {value}".lower()
    else:
        return None
    if not math.isfinite(power) or power < 0:
        return None
    if re.search(r"(?:^|[^a-z])uw(?:$|[^a-z])|microwatt", unit_text):
        power /= 1_000_000.0
    elif re.search(r"(?:^|[^a-z])mw(?:$|[^a-z])|milliwatt", unit_text):
        power /= 1_000.0
    return power


def extract_tt_power_watts(output):
    """Extract the telemetry power reading from tt-smi JSON output."""
    text = output.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        first, last = text.find("{"), text.rfind("}")
        if first < 0 or last <= first:
            raise ValueError("tt-smi output does not contain JSON")
        payload = json.loads(text[first : last + 1])

    candidates = []

    def visit(value, path):
        if isinstance(value, dict):
            for raw_key, nested in value.items():
                normalized = re.sub(r"[^a-z0-9]+", "_", str(raw_key).lower()).strip("_")
                nested_path = (*path, normalized)
                if "power" in normalized and not any(
                    word in normalized for word in ("limit", "maximum", "max", "cap")
                ):
                    power = _numeric_power(nested, normalized)
                    if power is not None:
                        score = 0
                        if normalized in {"power", "power_w", "board_power",
                                          "board_power_w", "power_draw"}:
                            score += 100
                        if "telemetry" in path:
                            score += 50
                        if "board" in normalized:
                            score += 10
                        candidates.append((score, nested_path, power))
                visit(nested, nested_path)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, (*path, str(index)))

    visit(payload, ())
    if not candidates:
        raise ValueError("no power field was found in tt-smi JSON")
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return candidates[0][2]


def read_tt_power_once():
    before = time.time()
    result = subprocess.run(["tt-smi", "-s", "--snapshot_no_tty"],
                            capture_output=True, text=True, timeout=3)
    after = time.time()
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"tt-smi exited with {result.returncode}: {detail}")
    return (before + after) * 0.5, extract_tt_power_watts(result.stdout), result.stdout


# --------------------------------------------------------------------------
# RAPL: one zone per physical package, unwrapped
# --------------------------------------------------------------------------
def discover_package_zones(root=RAPL_ROOT):
    """Return [(name, energy_path, max_range_uj)], one zone per package."""
    by_package = {}
    for directory in root.glob("intel-rapl:*"):
        match = TOP_LEVEL_ZONE.match(directory.name)
        if match is None:
            continue  # sub-zones such as intel-rapl:0:0 (core) are skipped
        try:
            name = (directory / "name").read_text().strip()
            int((directory / "energy_uj").read_text().strip())
        except (OSError, ValueError):
            continue
        package = PACKAGE_NAME.match(name)
        if package is None:
            continue  # e.g. psys
        max_range = None
        try:
            max_range = int((directory / "max_energy_range_uj").read_text().strip())
        except (OSError, ValueError):
            pass
        zone = (int(match.group(1)), name, directory / "energy_uj", max_range)
        package_id = int(package.group(1))
        # All die zones of a package report the same counter: keep the first.
        if package_id not in by_package or zone[0] < by_package[package_id][0]:
            by_package[package_id] = zone
    zones = [by_package[p][1:] for p in sorted(by_package)]
    if not zones:
        raise RuntimeError(f"no readable RAPL package zones below {root}")
    return zones


def read_counters(zones):
    return tuple(int(path.read_text().strip()) for _, path, _ in zones)


class RaplUnwrapper:
    """Turn raw cumulative counters into a monotonic total in joules."""

    def __init__(self, zones, first_raw):
        self.zones = zones
        self.last_raw = list(first_raw)
        self.accumulated_uj = [0.0] * len(zones)

    def update(self, raw):
        for i, value in enumerate(raw):
            delta = value - self.last_raw[i]
            if delta < 0:
                max_range = self.zones[i][2]
                if not max_range:
                    raise RuntimeError(f"RAPL {self.zones[i][0]} wrapped without max range")
                delta += max_range
            self.accumulated_uj[i] += delta
            self.last_raw[i] = value
        return sum(self.accumulated_uj) / 1_000_000.0


# --------------------------------------------------------------------------
# Profiler with independent RAPL and tt-smi threads
# --------------------------------------------------------------------------
class PowerProfiler:
    def __init__(self, target, zones, rapl_interval_s):
        self.target = target
        self.zones = zones
        self.rapl_interval_s = rapl_interval_s
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.tt_samples = []          # (timestamp, watts)
        self.rapl_energy = []         # (timestamp, cumulative joules)
        self.rapl_raw = []            # (timestamp, raw counters)
        self.last_tt_error = None
        self.last_tt_output = ""
        self.threads = [threading.Thread(target=self._poll_rapl, daemon=True)]
        if target == "tt":
            self.threads.append(threading.Thread(target=self._poll_tt, daemon=True))

    def start(self):
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()

    def _poll_rapl(self):
        raw = read_counters(self.zones)
        stamp = time.time()
        unwrapper = RaplUnwrapper(self.zones, raw)
        with self.lock:
            self.rapl_energy.append((stamp, 0.0))
            self.rapl_raw.append((stamp, raw))
        while not self.stop_event.wait(self.rapl_interval_s):
            raw = read_counters(self.zones)
            stamp = time.time()
            energy = unwrapper.update(raw)
            with self.lock:
                self.rapl_energy.append((stamp, energy))
                self.rapl_raw.append((stamp, raw))

    def _poll_tt(self):
        while not self.stop_event.is_set():
            try:
                stamp, power, output = read_tt_power_once()
                with self.lock:
                    self.tt_samples.append((stamp, power))
                self.last_tt_output = output
            except Exception as error:  # keep sampling; report at the end
                self.last_tt_error = str(error)
                time.sleep(0.1)

    def host_power_samples(self):
        """Power between consecutive RAPL readings, stamped at the midpoint."""
        samples = []
        for (t0, e0), (t1, e1) in zip(self.rapl_energy, self.rapl_energy[1:]):
            if t1 > t0:
                samples.append(((t0 + t1) / 2.0, (e1 - e0) / (t1 - t0)))
        return samples


def interpolate(points, timestamp):
    if timestamp <= points[0][0]:
        return points[0][1]
    if timestamp >= points[-1][0]:
        return points[-1][1]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= timestamp <= t1:
            return v0 + (v1 - v0) * (timestamp - t0) / (t1 - t0)
    return points[-1][1]


def integrate_power(samples, start, end):
    """Trapezoidal energy of a power trace over [start, end]."""
    ordered = sorted(samples)
    inside = [s for s in ordered if start <= s[0] <= end]
    if not ordered or end <= start:
        return 0.0, 0.0, len(inside)
    points = [(start, interpolate(ordered, start))]
    points += [s for s in ordered if start < s[0] < end]
    points.append((end, interpolate(ordered, end)))
    energy = sum((p0 + p1) * 0.5 * (t1 - t0)
                 for (t0, p0), (t1, p1) in zip(points, points[1:]))
    return energy / (end - start), energy, len(inside)


def counter_energy(rapl_energy, start, end):
    """Exact host energy from the cumulative counter, interpolated at markers."""
    inside = [s for s in rapl_energy if start <= s[0] <= end]
    energy = interpolate(rapl_energy, end) - interpolate(rapl_energy, start)
    return energy / (end - start), energy, len(inside)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def save_energy_summary(path, args, duration, samples, avg_power, energy, queries):
    with open(path, "w") as output:
        output.write("target,dataset,nlist,nprobe,k,runs,queries,duration_s,samples,"
                     "average_power_w,energy_j,energy_per_query_j\n")
        per_query = energy / queries if queries else 0.0
        output.write(f"{args.target},{args.dataset},{args.nlist},{args.nprobe},{args.k},"
                     f"{args.runs},{queries},{duration:.9f},{samples},"
                     f"{avg_power:.9f},{energy:.9f},{per_query:.12f}\n")
    print(f"-> Saved energy summary to: {path}")


def save_power_log(samples, name, nlist, nprobe, start, avg_power, output_dir):
    if not samples:
        return
    csv_file = output_dir / f"power_log_{nlist}_{nprobe}_{name}.csv"
    with open(csv_file, "w") as handle:
        handle.write("Timestamp,Power_Watts\n")
        for t, p in samples:
            handle.write(f"{t:.9f},{p:.6f}\n")
    print(f"-> Saved {name} power data to: {csv_file}")
    if len(samples) < 2:
        return
    t0 = samples[0][0]
    plt.figure(figsize=(10, 5))
    plt.plot([t - t0 for t, _ in samples], [p for _, p in samples],
             color="#ff7f0e", label=f"{name.upper()} power")
    if start is not None:
        plt.axvline(start - t0, color="gray", linestyle="--", label="Measurement start")
        if avg_power > 0:
            plt.axhline(avg_power, color="green", label=f"Average: {avg_power:.1f} W")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title(f"{name.upper()} power (NList={nlist}, NProbe={nprobe})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"power_log_{nlist}_{nprobe}_{name}_timeline.png", dpi=200)
    plt.close()


def save_rapl_audit(output_dir, zones, raw_samples):
    with open(output_dir / "rapl_zones.csv", "w") as handle:
        handle.write("Column,ZoneName,EnergyPath,MaxEnergyRange_uJ\n")
        for i, (name, path, max_range) in enumerate(zones):
            handle.write(f"zone{i}_energy_uJ,{name},{path},{max_range or ''}\n")
    with open(output_dir / "rapl_counters.csv", "w") as handle:
        handle.write("Timestamp," + ",".join(f"zone{i}_energy_uJ" for i in range(len(zones))) + "\n")
        for stamp, raw in raw_samples:
            handle.write(f"{stamp:.9f}," + ",".join(str(v) for v in raw) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["tt", "cpu"], default="tt")
    parser.add_argument("--nlist", type=int, default=2048)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--max-num-queries", type=int, default=10000)
    parser.add_argument("--result-staging", choices=["dram", "core0-l1"], default="dram")
    parser.add_argument("--cluster-chunk-blocks", type=int, default=128)
    parser.add_argument("--executable", default=TT_EXECUTABLE_PATH)
    parser.add_argument("--rapl-interval", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zones = discover_package_zones()
    print("[RAPL] reading one zone per package: " + ", ".join(z[0] for z in zones))

    if args.target == "tt":
        try:
            _, preflight_power, preflight_output = read_tt_power_once()
        except Exception as error:
            (output_dir / "tt_smi_preflight_error.txt").write_text(f"{error}\n")
            raise SystemExit(f"TT power preflight failed: {error}")
        (output_dir / "tt_smi_preflight.json").write_text(preflight_output)
        print(f"[Power] tt-smi preflight: {preflight_power:.3f} W")

    print("\n==========================================================")
    print(f"Running Power Profiler for [{args.target.upper()}]")
    print(f"Dataset: {args.dataset} | NList: {args.nlist} | NProbe: {args.nprobe} | Runs: {args.runs}")
    print("==========================================================\n")

    if args.target == "tt":
        cmd = [args.executable, "--dataset", args.dataset, "--nlist", str(args.nlist),
               "--nprobe", str(args.nprobe), "--k", str(args.k),
               "--max_num_queries", str(args.max_num_queries), "--runs", str(args.runs),
               "--result-staging", args.result_staging,
               "--cluster-chunk-blocks", str(args.cluster_chunk_blocks)]
        start_trigger, end_trigger = "ENERGY_MEASUREMENT_START", "ENERGY_MEASUREMENT_END"
    else:
        cmd = ["python3", "-u", FAISS_SCRIPT_PATH, "--dataset", args.dataset + ".hdf5",
               "--nlist", str(args.nlist), "--nprobe", str(args.nprobe),
               "--k", str(args.k), "--runs", str(args.runs)]
        start_trigger, end_trigger = "--- Run 1 /", "Saving per-query recall"

    (output_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    events_path = output_dir / "events.csv"
    events_path.write_text("Event,Timestamp\n")

    def record_event(name, timestamp=None):
        with events_path.open("a") as handle:
            handle.write(f"{name},{(time.time() if timestamp is None else timestamp):.9f}\n")

    profiler = PowerProfiler(args.target, zones, args.rapl_interval)
    profiler.start()
    time.sleep(0.5)  # bracket the start marker with samples on both streams

    start_time = end_time = None
    measured_queries = 0
    status = 1
    try:
        record_event("process_start")
        with (output_dir / "benchmark.log").open("w") as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if start_trigger in line and start_time is None:
                    start_time = time.time()
                    record_event("measurement_start", start_time)
                    marker = re.search(r"runs=(\d+)\s+queries_per_run=(\d+)", line)
                    if marker:
                        measured_queries = int(marker.group(1)) * int(marker.group(2))
                elif end_trigger in line and end_time is None:
                    end_time = time.time()
                    record_event("measurement_end", end_time)
            process.wait()
            status = process.returncode
        record_event("process_end")
    except Exception as error:
        print(f"\n[Error] Failed to run benchmark: {error}")
    time.sleep(0.5)  # bracket the end marker
    profiler.stop()

    if args.target == "tt" and not profiler.tt_samples:
        detail = profiler.last_tt_error or "tt-smi returned no usable samples"
        (output_dir / "tt_smi_sampling_error.txt").write_text(detail + "\n" + profiler.last_tt_output)
        raise SystemExit(f"[Error] No TT power samples were collected: {detail}")
    if start_time is None or end_time is None:
        raise SystemExit("[Error] Measurement markers not found; refusing to guess the window.")

    host_power = profiler.host_power_samples()
    cpu_avg, cpu_energy, cpu_count = counter_energy(profiler.rapl_energy, start_time, end_time)
    tt_avg, tt_energy, tt_count = integrate_power(profiler.tt_samples, start_time, end_time)
    duration = end_time - start_time

    print("\n==========================================================")
    print(f"                    ENERGY SUMMARY [{args.target.upper()}]")
    print("==========================================================")
    print(f"-> Measured interval:  {duration:.3f} s")
    if measured_queries:
        print(f"-> Queries processed:  {measured_queries}")
    if args.target == "tt":
        print(f"\n[Tenstorrent chip]  samples={tt_count}  avg={tt_avg:.2f} W  energy={tt_energy:.2f} J")
    print(f"[Host CPU, {len(zones)} package(s)]  samples={cpu_count}  avg={cpu_avg:.2f} W  energy={cpu_energy:.2f} J")
    if args.target == "tt":
        total = tt_energy + cpu_energy
        print(f"[System]  avg={tt_avg + cpu_avg:.2f} W  energy={total:.2f} J"
              + (f"  ({total * 1000.0 / measured_queries:.4f} mJ/query)" if measured_queries else ""))
    print("==========================================================\n")

    save_power_log(profiler.tt_samples, "tt", args.nlist, args.nprobe, start_time, tt_avg, output_dir)
    save_power_log(host_power, "cpu", args.nlist, args.nprobe, start_time, cpu_avg, output_dir)
    save_rapl_audit(output_dir, zones, profiler.rapl_raw)
    if args.target == "tt":
        save_energy_summary(output_dir / f"energy_summary_{args.nlist}_{args.nprobe}_tt.csv",
                            args, duration, tt_count, tt_avg, tt_energy, measured_queries)
    else:
        save_energy_summary(output_dir / f"energy_summary_{args.nlist}_{args.nprobe}_cpu.csv",
                            args, duration, cpu_count, cpu_avg, cpu_energy, measured_queries)
    if status != 0:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
