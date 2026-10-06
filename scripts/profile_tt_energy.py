#!/usr/bin/env python3
"""Profile Tenstorrent device power and host CPU RAPL energy.

The host CPU interface exposes cumulative microjoule counters. This program
discovers every top-level RAPL package/die domain once, samples each counter
separately, unwraps each domain independently, interpolates the counters at
the benchmark markers, and obtains energy from their differences. It never
integrates cumulative RAPL values as though they were power.

Tenstorrent device power is sampled through ``tt-smi`` and integrated over the
same marker interval with the trapezoidal rule. The benchmark must emit
``ENERGY_MEASUREMENT_START`` and ``ENERGY_MEASUREMENT_END``. Missing markers or
incomplete telemetry cause the run to fail instead of silently falling back to
an ambiguous interval.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


TT_EXECUTABLE_PATH = "./build/programming_examples/ann_ivf_energy"
FAISS_SCRIPT_PATH = "run_faiss_benchmark.py"
DATASET = "glove-100-angular"
K = 10
RAPL_ROOT = Path("/sys/class/powercap")
TOP_LEVEL_RAPL_PATTERN = re.compile(r"^intel-rapl:(?P<index>\d+)$")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


@dataclass(frozen=True)
class RaplZone:
    index: int
    name: str
    energy_path: Path
    maximum_uj: int | None


@dataclass(frozen=True)
class RaplSample:
    timestamp_s: float
    counters_uj: tuple[int, ...]


@dataclass(frozen=True)
class PowerSample:
    timestamp_s: float
    power_w: float


@dataclass(frozen=True)
class EnergyResult:
    duration_s: float
    average_power_w: float
    energy_j: float
    sample_count: int
    maximum_gap_s: float


def midpoint_timestamp(before_s: float, after_s: float) -> float:
    return before_s + (after_s - before_s) / 2.0


def read_integer(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8").strip()
        return int(text) / 2
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read integer from {path}") from error


def discover_rapl_zones(root: Path = RAPL_ROOT) -> tuple[RaplZone, ...]:
    """Discover readable top-level RAPL domains, excluding child domains."""

    zones: list[RaplZone] = []
    for directory in root.glob("intel-rapl:*"):
        match = TOP_LEVEL_RAPL_PATTERN.fullmatch(directory.name)
        if match is None:
            continue

        energy_path = directory / "energy_uj"
        name_path = directory / "name"
        maximum_path = directory / "max_energy_range_uj"
        if not energy_path.exists() or not name_path.exists():
            continue

        try:
            name = name_path.read_text(encoding="utf-8").strip()
            read_integer(energy_path)
        except RuntimeError:
            raise
        except OSError as error:
            raise RuntimeError(f"cannot inspect RAPL domain {directory}") from error

        # The top-level node117 domains are named package-*-die-*.
        if not name.startswith("package-"):
            continue

        maximum_uj = None
        if maximum_path.exists():
            maximum_uj = read_integer(maximum_path)
            if maximum_uj <= 0:
                raise RuntimeError(
                    f"invalid maximum energy range for RAPL domain {name}"
                )

        zones.append(
            RaplZone(
                index=int(match.group("index")),
                name=name,
                energy_path=energy_path,
                maximum_uj=maximum_uj,
            )
        )

    zones.sort(key=lambda zone: zone.index)
    if not zones:
        raise RuntimeError(
            f"no readable top-level RAPL package domains found below {root}"
        )
    return tuple(zones)


def read_rapl_sample(zones: tuple[RaplZone, ...]) -> RaplSample:
    """Read a complete RAPL snapshot or fail without returning partial data."""

    before_s = time.time()
    counters = tuple(read_integer(zone.energy_path) for zone in zones)
    after_s = time.time()
    return RaplSample(midpoint_timestamp(before_s, after_s), counters)


def parse_tt_power(output: str) -> float:
    try:
        document = json.loads(output)
        raw_power = document["device_info"][0]["telemetry"]["power"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError("tt-smi output does not contain device power") from error

    match = NUMBER_PATTERN.search(str(raw_power))
    if match is None:
        raise RuntimeError(f"cannot parse TT power value {raw_power!r}")
    power_w = float(match.group(0))
    if not math.isfinite(power_w) or power_w < 0:
        raise RuntimeError(f"invalid TT power value {power_w!r}")
    return power_w


class PowerProfiler:
    """Run independent TT-power and host-RAPL sampling threads."""

    def __init__(
        self,
        target: str,
        rapl_zones: tuple[RaplZone, ...],
        rapl_interval_s: float,
        tt_interval_s: float,
        tt_smi_timeout_s: float,
    ) -> None:
        self.target = target
        self.rapl_zones = rapl_zones
        self.rapl_interval_s = rapl_interval_s
        self.tt_interval_s = tt_interval_s
        self.tt_smi_timeout_s = tt_smi_timeout_s
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.rapl_samples: list[RaplSample] = []
        self.tt_samples: list[PowerSample] = []
        self.error_counts: dict[str, int] = {"rapl": 0, "tt-smi": 0}
        self.last_errors: dict[str, str] = {}
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        self.stop_event.clear()
        self.threads = [
            threading.Thread(
                target=self._poll_rapl,
                name="rapl-profiler",
                daemon=True,
            )
        ]
        if self.target == "tt":
            self.threads.append(
                threading.Thread(
                    target=self._poll_tt,
                    name="tt-profiler",
                    daemon=True,
                )
            )
        for thread in self.threads:
            thread.start()

    def stop(self) -> tuple[list[PowerSample], list[RaplSample]]:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=self.tt_smi_timeout_s + 2.0)
            if thread.is_alive():
                raise RuntimeError(f"sampler thread did not terminate: {thread.name}")
        with self.lock:
            return list(self.tt_samples), list(self.rapl_samples)

    def wait_for_rapl(self, timeout_s: float) -> bool:
        """Wait only for host telemetry before launching the benchmark.

        On some Tenstorrent systems ``tt-smi`` does not answer while the
        device is idle. Requiring a TT sample here would therefore deadlock
        startup: the benchmark is what makes device telemetry available.
        Device coverage is validated later against both measurement markers.
        """

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self.lock:
                rapl_ready = bool(self.rapl_samples)
            if rapl_ready:
                return True
            self.stop_event.wait(0.05)
        return False

    def _record_error(self, source: str, error: Exception) -> None:
        with self.lock:
            self.error_counts[source] += 1
            self.last_errors[source] = str(error)

    def _poll_rapl(self) -> None:
        while not self.stop_event.is_set():
            try:
                sample = read_rapl_sample(self.rapl_zones)
            except RuntimeError as error:
                self._record_error("rapl", error)
            else:
                with self.lock:
                    self.rapl_samples.append(sample)
            self.stop_event.wait(self.rapl_interval_s)

    def _poll_tt(self) -> None:
        command = ["tt-smi", "-s", "--snapshot_no_tty"]
        while not self.stop_event.is_set():
            before_s = time.time()
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.tt_smi_timeout_s,
                    check=False,
                )
                after_s = time.time()
                if result.returncode != 0:
                    message = result.stderr.strip() or result.stdout.strip()
                    raise RuntimeError(
                        f"tt-smi exited with {result.returncode}: {message}"
                    )
                sample = PowerSample(
                    midpoint_timestamp(before_s, after_s),
                    parse_tt_power(result.stdout),
                )
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                self._record_error("tt-smi", error)
            else:
                with self.lock:
                    self.tt_samples.append(sample)
            self.stop_event.wait(self.tt_interval_s)


def validate_timestamps(timestamps: list[float], context: str) -> None:
    if len(timestamps) < 2:
        raise RuntimeError(f"{context}: fewer than two samples")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise RuntimeError(f"{context}: timestamps are not strictly increasing")


def interpolate(
    timestamps: list[float],
    values: list[float],
    timestamp_s: float,
) -> float:
    if timestamp_s < timestamps[0] or timestamp_s > timestamps[-1]:
        raise RuntimeError("requested boundary is outside the sampled interval")

    position = bisect.bisect_left(timestamps, timestamp_s)
    if position < len(timestamps) and timestamps[position] == timestamp_s:
        return values[position]
    if position == 0 or position == len(timestamps):
        raise RuntimeError("cannot interpolate a telemetry boundary")

    left_t = timestamps[position - 1]
    right_t = timestamps[position]
    fraction = (timestamp_s - left_t) / (right_t - left_t)
    return values[position - 1] + fraction * (
        values[position] - values[position - 1]
    )


def unwrap_counter(
    raw_values_uj: list[int],
    maximum_uj: int | None,
    zone_name: str,
) -> list[float]:
    unwrapped = [float(raw_values_uj[0])]
    offset = 0
    previous_raw = raw_values_uj[0]

    for current_raw in raw_values_uj[1:]:
        if current_raw < previous_raw:
            if maximum_uj is None:
                raise RuntimeError(
                    f"RAPL counter {zone_name} wrapped but its maximum range "
                    "was unavailable"
                )
            offset += maximum_uj
        current_unwrapped = float(current_raw + offset)
        if current_unwrapped < unwrapped[-1]:
            raise RuntimeError(f"could not unwrap RAPL counter {zone_name}")
        unwrapped.append(current_unwrapped)
        previous_raw = current_raw
    return unwrapped


def calculate_rapl_energy(
    samples: list[RaplSample],
    zones: tuple[RaplZone, ...],
    start_time_s: float,
    end_time_s: float,
) -> tuple[EnergyResult, list[PowerSample]]:
    """Difference cumulative counters independently at interpolated markers."""

    samples = sorted(samples, key=lambda sample: sample.timestamp_s)
    timestamps = [sample.timestamp_s for sample in samples]
    validate_timestamps(timestamps, "RAPL")
    if timestamps[0] > start_time_s or timestamps[-1] < end_time_s:
        raise RuntimeError(
            "RAPL samples do not bracket the complete measurement window"
        )

    unwrapped_by_zone: list[list[float]] = []
    total_energy_uj = 0.0
    for zone_position, zone in enumerate(zones):
        raw_values = [sample.counters_uj[zone_position] for sample in samples]
        unwrapped = unwrap_counter(raw_values, zone.maximum_uj, zone.name)
        unwrapped_by_zone.append(unwrapped)
        start_uj = interpolate(timestamps, unwrapped, start_time_s)
        end_uj = interpolate(timestamps, unwrapped, end_time_s)
        delta_uj = end_uj - start_uj
        if delta_uj < 0:
            raise RuntimeError(f"negative RAPL energy for {zone.name}")
        total_energy_uj += delta_uj

    # Derive interval power only for diagnostics and plotting. Energy above is
    # obtained directly from the cumulative counter differences.
    total_cumulative_uj = [
        sum(zone_values[index] for zone_values in unwrapped_by_zone)
        for index in range(len(samples))
    ]
    interval_powers: list[PowerSample] = []
    for index in range(1, len(samples)):
        delta_t = timestamps[index] - timestamps[index - 1]
        delta_j = (
            total_cumulative_uj[index] - total_cumulative_uj[index - 1]
        ) / 1_000_000.0
        if delta_t <= 0 or delta_j < 0:
            raise RuntimeError("invalid interval while deriving RAPL power")
        interval_powers.append(
            PowerSample(
                midpoint_timestamp(timestamps[index - 1], timestamps[index]),
                delta_j / delta_t,
            )
        )

    duration_s = end_time_s - start_time_s
    energy_j = total_energy_uj / 1_000_000.0
    used_samples = sum(start_time_s <= value <= end_time_s for value in timestamps)
    maximum_gap_s = max(
        right - left for left, right in zip(timestamps, timestamps[1:])
    )
    return (
        EnergyResult(
            duration_s=duration_s,
            average_power_w=energy_j / duration_s,
            energy_j=energy_j,
            sample_count=used_samples,
            maximum_gap_s=maximum_gap_s,
        ),
        interval_powers,
    )


def clip_power_samples(
    samples: list[PowerSample],
    start_time_s: float,
    end_time_s: float,
) -> list[PowerSample]:
    samples = sorted(samples, key=lambda sample: sample.timestamp_s)
    timestamps = [sample.timestamp_s for sample in samples]
    powers = [sample.power_w for sample in samples]
    validate_timestamps(timestamps, "power")
    if timestamps[0] > start_time_s or timestamps[-1] < end_time_s:
        raise RuntimeError(
            "power samples do not bracket the complete measurement window"
        )

    clipped = [
        PowerSample(start_time_s, interpolate(timestamps, powers, start_time_s))
    ]
    clipped.extend(
        sample
        for sample in samples
        if start_time_s < sample.timestamp_s < end_time_s
    )
    clipped.append(
        PowerSample(end_time_s, interpolate(timestamps, powers, end_time_s))
    )
    return clipped


def calculate_sampled_power_energy(
    samples: list[PowerSample],
    start_time_s: float,
    end_time_s: float,
) -> EnergyResult:
    clipped = clip_power_samples(samples, start_time_s, end_time_s)
    energy_j = 0.0
    for left, right in zip(clipped, clipped[1:]):
        energy_j += (
            (left.power_w + right.power_w)
            * 0.5
            * (right.timestamp_s - left.timestamp_s)
        )

    original_timestamps = sorted(sample.timestamp_s for sample in samples)
    used_samples = sum(
        start_time_s <= timestamp <= end_time_s
        for timestamp in original_timestamps
    )
    maximum_gap_s = max(
        right - left
        for left, right in zip(original_timestamps, original_timestamps[1:])
    )
    duration_s = end_time_s - start_time_s
    return EnergyResult(
        duration_s=duration_s,
        average_power_w=energy_j / duration_s,
        energy_j=energy_j,
        sample_count=used_samples,
        maximum_gap_s=maximum_gap_s,
    )


def write_rapl_metadata(path: Path, zones: tuple[RaplZone, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Column", "ZoneName", "EnergyPath", "MaxEnergyRange_uJ"]
        )
        for position, zone in enumerate(zones):
            writer.writerow(
                [
                    f"zone{position}_energy_uJ",
                    zone.name,
                    zone.energy_path,
                    zone.maximum_uj if zone.maximum_uj is not None else "",
                ]
            )


def write_rapl_samples(
    path: Path,
    samples: list[RaplSample],
    zones: tuple[RaplZone, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Timestamp", *(f"zone{i}_energy_uJ" for i in range(len(zones)))]
        )
        for sample in samples:
            writer.writerow(
                [f"{sample.timestamp_s:.9f}", *sample.counters_uj]
            )


def write_power_samples(path: Path, samples: list[PowerSample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Timestamp", "Power_W"])
        for sample in samples:
            writer.writerow([f"{sample.timestamp_s:.9f}", f"{sample.power_w:.9f}"])


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0 or len(values) < window:
        return []
    running_sum = sum(values[:window])
    averages = [running_sum / window]
    for index in range(window, len(values)):
        running_sum += values[index] - values[index - window]
        averages.append(running_sum / window)
    return averages


def plot_power(
    samples: list[PowerSample],
    component: str,
    nlist: int,
    nprobe: int,
    start_time_s: float,
    end_time_s: float,
    average_power_w: float,
    output_path: Path,
) -> None:
    if len(samples) < 2:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "WARNING: matplotlib is unavailable; raw telemetry and summary "
            "files were saved, but plots were skipped"
        )
        return

    samples = sorted(samples, key=lambda sample: sample.timestamp_s)
    origin = samples[0].timestamp_s
    relative_times = [sample.timestamp_s - origin for sample in samples]
    powers = [sample.power_w for sample in samples]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        relative_times,
        powers,
        color="#ff7f0e",
        alpha=0.65,
        linewidth=1.4,
        label=f"{component} power",
    )

    window = 5
    smoothed = moving_average(powers, window)
    if smoothed:
        axis.plot(
            relative_times[window - 1 :],
            smoothed,
            color="#d62728",
            linewidth=2,
            label=f"{window}-sample moving average",
        )

    axis.axvline(
        start_time_s - origin,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="Measurement starts",
    )
    axis.axvline(
        end_time_s - origin,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Measurement ends",
    )
    axis.axhline(
        average_power_w,
        color="green",
        linewidth=1.5,
        label=f"Window average: {average_power_w:.1f} W",
    )
    axis.set_title(
        f"{component} power (NList={nlist}, NProbe={nprobe})"
    )
    axis.set_xlabel("Time from first sample (s)")
    axis.set_ylabel("Power (W)")
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_energy_summary(
    path: Path,
    args: argparse.Namespace,
    measured_queries: int,
    host: EnergyResult,
    device: EnergyResult | None,
) -> None:
    device_energy_j = device.energy_j if device is not None else 0.0
    device_power_w = device.average_power_w if device is not None else 0.0
    device_samples = device.sample_count if device is not None else 0
    total_energy_j = host.energy_j + device_energy_j
    total_average_power_w = total_energy_j / host.duration_s
    energy_per_query_j = total_energy_j / measured_queries

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Target",
                "Dataset",
                "NList",
                "NProbe",
                "K",
                "Runs",
                "Queries",
                "Duration_s",
                "RaplZones",
                "RaplSamples",
                "DeviceSamples",
                "HostAveragePower_W",
                "HostEnergy_J",
                "DeviceAveragePower_W",
                "DeviceEnergy_J",
                "SystemAveragePower_W",
                "SystemEnergy_J",
                "SystemEnergyPerQuery_J",
            ]
        )
        writer.writerow(
            [
                args.target,
                args.dataset,
                args.nlist,
                args.nprobe,
                args.k,
                args.runs,
                measured_queries,
                f"{host.duration_s:.9f}",
                args.rapl_zone_count,
                host.sample_count,
                device_samples,
                f"{host.average_power_w:.9f}",
                f"{host.energy_j:.9f}",
                f"{device_power_w:.9f}",
                f"{device_energy_j:.9f}",
                f"{total_average_power_w:.9f}",
                f"{total_energy_j:.9f}",
                f"{energy_per_query_j:.12f}",
            ]
        )


def build_command(args: argparse.Namespace) -> tuple[list[str], str, str]:
    if args.target == "tt":
        return (
            [
                args.executable,
                "--dataset",
                args.dataset,
                "--nlist",
                str(args.nlist),
                "--nprobe",
                str(args.nprobe),
                "--k",
                str(args.k),
                "--max_num_queries",
                str(args.max_num_queries),
                "--runs",
                str(args.runs),
                "--result-staging",
                args.result_staging,
                "--cluster-chunk-blocks",
                str(args.cluster_chunk_blocks),
            ],
            "ENERGY_MEASUREMENT_START",
            "ENERGY_MEASUREMENT_END",
        )

    dataset_path = args.dataset
    if not dataset_path.endswith(".hdf5"):
        dataset_path += ".hdf5"
    return (
        [
            "python3",
            "-u",
            args.faiss_script,
            "--dataset",
            dataset_path,
            "--nlist",
            str(args.nlist),
            "--nprobe",
            str(args.nprobe),
            "--k",
            str(args.k),
            "--num_queries",
            str(args.max_num_queries),
            "--runs",
            str(args.runs),
        ],
        args.cpu_start_trigger,
        args.cpu_end_trigger,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["tt", "cpu"], default="tt")
    parser.add_argument("--nlist", type=int, default=2048)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--max-num-queries", type=int, default=10_000)
    parser.add_argument(
        "--result-staging",
        choices=["dram", "core0-l1"],
        default="dram",
    )
    parser.add_argument("--cluster-chunk-blocks", type=int, default=128)
    parser.add_argument("--executable", default=TT_EXECUTABLE_PATH)
    parser.add_argument("--faiss-script", default=FAISS_SCRIPT_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--rapl-root", type=Path, default=RAPL_ROOT)
    parser.add_argument("--rapl-interval", type=float, default=0.1)
    parser.add_argument("--tt-interval", type=float, default=0.1)
    parser.add_argument("--tt-smi-timeout", type=float, default=2.0)
    parser.add_argument("--sampler-ready-timeout", type=float, default=10.0)
    parser.add_argument("--post-window-delay", type=float, default=1.0)
    parser.add_argument("--cpu-start-trigger", default="--- Run 1 /")
    parser.add_argument("--cpu-end-trigger", default="Saving per-query recall")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="save telemetry and summaries without generating PNG files",
    )
    args = parser.parse_args()

    positive_arguments = {
        "--runs": args.runs,
        "--max-num-queries": args.max_num_queries,
        "--k": args.k,
        "--rapl-interval": args.rapl_interval,
        "--tt-interval": args.tt_interval,
        "--tt-smi-timeout": args.tt_smi_timeout,
        "--sampler-ready-timeout": args.sampler_ready_timeout,
        "--post-window-delay": args.post_window_delay,
    }
    for name, value in positive_arguments.items():
        if value <= 0:
            parser.error(f"{name} must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rapl_zones = discover_rapl_zones(args.rapl_root)
    args.rapl_zone_count = len(rapl_zones)
    print("Discovered top-level RAPL domains:")
    for zone in rapl_zones:
        print(
            f"  {zone.index}: {zone.name} -> {zone.energy_path} "
            f"(max={zone.maximum_uj})"
        )

    command, start_trigger, end_trigger = build_command(args)
    (output_dir / "command.txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )
    write_rapl_metadata(output_dir / "rapl_zones.csv", rapl_zones)

    events_path = output_dir / "events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["Event", "Timestamp"])

    def record_event(name: str, timestamp_s: float | None = None) -> float:
        event_time = time.time() if timestamp_s is None else timestamp_s
        with events_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([name, f"{event_time:.9f}"])
        return event_time

    print("\n==========================================================")
    print(f"Running Power Profiler for [{args.target.upper()}]")
    print(
        f"Dataset: {args.dataset} | NList: {args.nlist} | "
        f"NProbe: {args.nprobe} | Runs: {args.runs}"
    )
    print("==========================================================\n")

    profiler = PowerProfiler(
        target=args.target,
        rapl_zones=rapl_zones,
        rapl_interval_s=args.rapl_interval,
        tt_interval_s=args.tt_interval,
        tt_smi_timeout_s=args.tt_smi_timeout,
    )

    search_start_time: float | None = None
    search_end_time: float | None = None
    measured_queries = args.runs * args.max_num_queries
    benchmark_status = 1
    process: subprocess.Popen[str] | None = None

    record_event("monitor_start")
    profiler.start()
    try:
        if not profiler.wait_for_rapl(args.sampler_ready_timeout):
            raise RuntimeError(
                "RAPL did not produce an initial sample before the benchmark; "
                "inspect RAPL access and rapl_zones.csv"
            )
        record_event("monitor_ready")

        benchmark_log_path = output_dir / "benchmark.log"
        record_event("process_start")
        with benchmark_log_path.open("w", encoding="utf-8") as benchmark_log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("benchmark stdout pipe was not created")

            for line in process.stdout:
                print(line, end="", flush=True)
                benchmark_log.write(line)
                benchmark_log.flush()

                if start_trigger in line and search_start_time is None:
                    search_start_time = record_event("measurement_start")
                    marker = re.search(
                        r"runs=(\d+)\s+queries_per_run=(\d+)",
                        line,
                    )
                    if marker:
                        measured_queries = int(marker.group(1)) * int(
                            marker.group(2)
                        )
                elif end_trigger in line and search_end_time is None:
                    search_end_time = record_event("measurement_end")

            benchmark_status = process.wait()
        record_event("process_end")
    finally:
        # Ensure that a telemetry sample is collected after the end marker so
        # both device power and cumulative counters bracket the exact window.
        if search_end_time is not None:
            time.sleep(args.post_window_delay)
        tt_samples, rapl_samples = profiler.stop()
        record_event("monitor_end")

    write_rapl_samples(output_dir / "host_rapl.csv", rapl_samples, rapl_zones)
    write_power_samples(output_dir / "tt_power.csv", tt_samples)

    for source, count in profiler.error_counts.items():
        if count:
            print(
                f"WARNING: {source} sampling errors={count}; "
                f"last={profiler.last_errors.get(source, 'unknown')}"
            )

    if benchmark_status != 0:
        raise RuntimeError(f"benchmark exited with status {benchmark_status}")
    if search_start_time is None or search_end_time is None:
        raise RuntimeError(
            "the benchmark did not emit both energy measurement markers"
        )
    if search_end_time <= search_start_time:
        raise RuntimeError("the measurement marker interval is not positive")
    if measured_queries <= 0:
        raise RuntimeError("the number of measured queries is not positive")

    host_result, host_power_samples = calculate_rapl_energy(
        rapl_samples,
        rapl_zones,
        search_start_time,
        search_end_time,
    )
    write_power_samples(output_dir / "host_power_derived.csv", host_power_samples)

    device_result = None
    if args.target == "tt":
        device_result = calculate_sampled_power_energy(
            tt_samples,
            search_start_time,
            search_end_time,
        )

    total_energy_j = host_result.energy_j + (
        device_result.energy_j if device_result is not None else 0.0
    )
    total_average_power_w = total_energy_j / host_result.duration_s

    print("\n==========================================================")
    print(f"                    ENERGY SUMMARY [{args.target.upper()}]")
    print("==========================================================")
    print(f"Measurement duration:       {host_result.duration_s:.3f} s")
    print(f"Queries processed:          {measured_queries}")
    print(f"RAPL domains:               {len(rapl_zones)}")
    print(f"Host RAPL samples in window:{host_result.sample_count:>9d}")
    print(f"Host average power:         {host_result.average_power_w:.3f} W")
    print(f"Host energy:                {host_result.energy_j:.3f} J")

    if device_result is not None:
        print(f"TT samples in window:       {device_result.sample_count:>9d}")
        print(f"TT average power:           {device_result.average_power_w:.3f} W")
        print(f"TT energy:                  {device_result.energy_j:.3f} J")

    print(f"System average power:       {total_average_power_w:.3f} W")
    print(f"System energy:              {total_energy_j:.3f} J")
    print(
        "System energy/query:        "
        f"{total_energy_j * 1000.0 / measured_queries:.6f} mJ"
    )
    print("==========================================================\n")

    if not args.no_plots:
        plot_power(
            host_power_samples,
            "Host CPU",
            args.nlist,
            args.nprobe,
            search_start_time,
            search_end_time,
            host_result.average_power_w,
            output_dir / f"power_log_{args.nlist}_{args.nprobe}_cpu_timeline.png",
        )
        if device_result is not None:
            plot_power(
                tt_samples,
                "Tenstorrent",
                args.nlist,
                args.nprobe,
                search_start_time,
                search_end_time,
                device_result.average_power_w,
                output_dir / f"power_log_{args.nlist}_{args.nprobe}_tt_timeline.png",
            )

    summary_path = (
        output_dir
        / f"energy_summary_{args.nlist}_{args.nprobe}_{args.target}.csv"
    )
    write_energy_summary(
        summary_path,
        args,
        measured_queries,
        host_result,
        device_result,
    )
    print(f"Saved energy summary to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f"ERROR: {error}") from error
