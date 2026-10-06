#!/usr/bin/env python3
"""Discover and combine NVIDIA power traces with matching host-RAPL traces.

The script scans the selected directory for pairs named as follows:

    gpu_power_nlist<N>_nprobe<P>_gpu<ID>.log
    gpu_power_nlist<N>_nprobe<P>_cpu.log

The files may also use the ``.csv`` extension. Their expected contents are

    Timestamp,Power_W
    178...,123.4

and

    Timestamp,Total_uJ
    178...,123456789

The analysis uses the timestamp interval covered by both traces. NVIDIA power
is integrated with the trapezoidal rule. RAPL already reports cumulative
energy, so host energy is the counter difference over the same interval; the
counter is linearly interpolated at the interval boundaries when necessary.
Integrating cumulative RAPL values as if they were watts would be
dimensionally incorrect.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawTrace:
    path: Path
    kind: str
    timestamps_s: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ComponentResult:
    source: str
    sample_count: int
    duration_s: float
    energy_j: float
    average_power_w: float
    mean_gap_s: float
    maximum_gap_s: float
    plot_times_s: np.ndarray
    plot_powers_w: np.ndarray


@dataclass(frozen=True)
class PairResult:
    gpu_path: Path
    cpu_path: Path
    start_time_s: float
    end_time_s: float
    gpu: ComponentResult
    cpu: ComponentResult

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def total_energy_j(self) -> float:
        return self.gpu.energy_j + self.cpu.energy_j

    @property
    def average_power_w(self) -> float:
        return self.total_energy_j / self.duration_s


@dataclass(frozen=True)
class PairSpec:
    nlist: int
    nprobe: int
    gpu_path: Path
    cpu_path: Path


def read_trace(path: Path) -> RawTrace:
    """Read a timestamped GPU-power or cumulative-RAPL trace."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header_row = next(reader)
        except StopIteration as error:
            raise ValueError(f"{path}: empty file") from error

        header = [field.strip().lower() for field in header_row]
        if len(header) < 2 or "timestamp" not in header[0]:
            raise ValueError(
                f"{path}: expected Timestamp and one measurement column"
            )

        value_header = header[1]
        if "uj" in value_header or "energy" in value_header:
            kind = "rapl"
        elif "power" in value_header or "watt" in value_header:
            kind = "gpu"
        else:
            raise ValueError(
                f"{path}: cannot infer units from header {header_row!r}"
            )

        samples: list[tuple[float, float]] = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{path}: line {line_number} has fewer than two columns"
                )
            try:
                timestamp = float(row[0])
                value = float(row[1])
            except ValueError as error:
                raise ValueError(
                    f"{path}: line {line_number} is not numeric"
                ) from error
            if not math.isfinite(timestamp) or not math.isfinite(value):
                raise ValueError(
                    f"{path}: line {line_number} contains a non-finite value"
                )
            samples.append((timestamp, value))

    if len(samples) < 2:
        raise ValueError(f"{path}: at least two data samples are required")

    timestamps = np.asarray([sample[0] for sample in samples], dtype=float)
    values = np.asarray([sample[1] for sample in samples], dtype=float)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            f"{path}: timestamps must be strictly increasing in file order"
        )

    return RawTrace(path, kind, timestamps, values)


def unwrap_rapl_counter(values_uj: np.ndarray, maximum_uj: float | None) -> np.ndarray:
    """Return a monotonically increasing RAPL counter sequence."""

    values = values_uj.astype(float, copy=True)
    offset = 0.0
    previous = values[0]
    for index in range(1, len(values)):
        current = values[index]
        if current < previous:
            if maximum_uj is None:
                raise ValueError(
                    "RAPL counter decreased. Supply --rapl-max-uj if this was "
                    "a normal counter wrap; otherwise inspect the raw trace."
                )
            offset += maximum_uj
        values[index] = current + offset
        previous = current
    return values


def clip_with_interpolated_boundaries(
    timestamps: np.ndarray,
    values: np.ndarray,
    start_time_s: float,
    end_time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a trace and insert linearly interpolated boundary samples."""

    if start_time_s < timestamps[0] or end_time_s > timestamps[-1]:
        raise ValueError("requested window lies outside a trace")
    if end_time_s <= start_time_s:
        raise ValueError("integration window must have positive duration")

    interior = (timestamps > start_time_s) & (timestamps < end_time_s)
    clipped_times = np.concatenate(
        ([start_time_s], timestamps[interior], [end_time_s])
    )
    clipped_values = np.concatenate(
        (
            [np.interp(start_time_s, timestamps, values)],
            values[interior],
            [np.interp(end_time_s, timestamps, values)],
        )
    )
    return clipped_times, clipped_values


def trapezoidal_integral(values: np.ndarray, timestamps: np.ndarray) -> float:
    """Integrate sampled values over their actual timestamps."""

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(values, timestamps))
    return float(np.trapz(values, timestamps))


def component_statistics(
    source: str,
    timestamps: np.ndarray,
    powers_w: np.ndarray,
    energy_j: float,
    plot_times_s: np.ndarray,
) -> ComponentResult:
    gaps = np.diff(timestamps)
    duration_s = float(timestamps[-1] - timestamps[0])
    return ComponentResult(
        source=source,
        sample_count=len(timestamps),
        duration_s=duration_s,
        energy_j=energy_j,
        average_power_w=energy_j / duration_s,
        mean_gap_s=float(np.mean(gaps)),
        maximum_gap_s=float(np.max(gaps)),
        plot_times_s=plot_times_s,
        plot_powers_w=powers_w,
    )


def analyse_pair(
    first_path: Path,
    second_path: Path,
    trim_start_s: float,
    trim_end_s: float,
    rapl_maximum_uj: float | None,
) -> PairResult:
    """Align and integrate one GPU trace and one host-RAPL trace."""

    traces = [read_trace(first_path), read_trace(second_path)]
    gpu_traces = [trace for trace in traces if trace.kind == "gpu"]
    rapl_traces = [trace for trace in traces if trace.kind == "rapl"]
    if len(gpu_traces) != 1 or len(rapl_traces) != 1:
        raise ValueError(
            "the two files must contain exactly one Power_W trace and one "
            "Total_uJ trace"
        )

    gpu_trace = gpu_traces[0]
    cpu_trace = rapl_traces[0]
    if np.any(gpu_trace.values < 0):
        raise ValueError("GPU power trace contains a negative value")

    start_time_s = max(
        float(gpu_trace.timestamps_s[0]),
        float(cpu_trace.timestamps_s[0]),
    ) + trim_start_s
    end_time_s = min(
        float(gpu_trace.timestamps_s[-1]),
        float(cpu_trace.timestamps_s[-1]),
    ) - trim_end_s
    if end_time_s <= start_time_s:
        raise ValueError("the common timestamp window is empty after trimming")

    gpu_times, gpu_powers = clip_with_interpolated_boundaries(
        gpu_trace.timestamps_s,
        gpu_trace.values,
        start_time_s,
        end_time_s,
    )
    gpu_energy_j = trapezoidal_integral(gpu_powers, gpu_times)
    gpu_result = component_statistics(
        "GPU board",
        gpu_times,
        gpu_powers,
        gpu_energy_j,
        gpu_times - start_time_s,
    )

    unwrapped_cpu_uj = unwrap_rapl_counter(
        cpu_trace.values,
        rapl_maximum_uj,
    )
    cpu_times, cpu_cumulative_uj = clip_with_interpolated_boundaries(
        cpu_trace.timestamps_s,
        unwrapped_cpu_uj,
        start_time_s,
        end_time_s,
    )
    cpu_gaps = np.diff(cpu_times)
    cpu_differences_j = np.diff(cpu_cumulative_uj) / 1_000_000.0
    if np.any(cpu_differences_j < 0):
        raise ValueError("unwrapped RAPL energy is not monotonic")
    cpu_interval_powers = cpu_differences_j / cpu_gaps
    cpu_energy_j = float(
        (cpu_cumulative_uj[-1] - cpu_cumulative_uj[0]) / 1_000_000.0
    )
    cpu_midpoints = 0.5 * (cpu_times[:-1] + cpu_times[1:])
    cpu_result = component_statistics(
        "Host RAPL",
        cpu_times,
        cpu_interval_powers,
        cpu_energy_j,
        cpu_midpoints - start_time_s,
    )

    return PairResult(
        gpu_path=gpu_trace.path,
        cpu_path=cpu_trace.path,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        gpu=gpu_result,
        cpu=cpu_result,
    )


def discover_pairs(directory: Path) -> tuple[list[PairSpec], list[str]]:
    """Pair CPU and GPU traces using NList/NProbe encoded in each filename."""

    pattern = re.compile(
        r"^gpu_power_nlist(?P<nlist>\d+)_nprobe(?P<nprobe>\d+)_"
        r"(?P<component>cpu|gpu\d+)\.(?P<extension>log|csv)$"
    )
    grouped: dict[tuple[int, int], dict[str, list[Path]]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        key = (int(match.group("nlist")), int(match.group("nprobe")))
        component = "cpu" if match.group("component") == "cpu" else "gpu"
        grouped.setdefault(key, {"cpu": [], "gpu": []})[component].append(path)

    pairs: list[PairSpec] = []
    warnings: list[str] = []
    for (nlist, nprobe), components in sorted(grouped.items()):
        cpu_files = components["cpu"]
        gpu_files = components["gpu"]
        label = f"NList={nlist}, NProbe={nprobe}"
        if len(cpu_files) != 1 or len(gpu_files) != 1:
            warnings.append(
                f"{label}: expected one CPU and one GPU file, found "
                f"{len(cpu_files)} CPU and {len(gpu_files)} GPU files; skipped"
            )
            continue
        pairs.append(
            PairSpec(
                nlist=nlist,
                nprobe=nprobe,
                gpu_path=gpu_files[0],
                cpu_path=cpu_files[0],
            )
        )
    return pairs, warnings


def default_output_prefix(result: PairResult) -> Path:
    """Build a stable output prefix from NList/NProbe when available."""

    pattern = re.compile(r"nlist(?P<nlist>\d+)_nprobe(?P<nprobe>\d+)")
    gpu_match = pattern.search(result.gpu_path.name)
    cpu_match = pattern.search(result.cpu_path.name)
    if gpu_match and cpu_match and gpu_match.groupdict() == cpu_match.groupdict():
        return result.gpu_path.parent / (
            f"gpu_power_nlist{gpu_match.group('nlist')}_"
            f"nprobe{gpu_match.group('nprobe')}_combined"
        )
    return result.gpu_path.parent / "gpu_host_power_combined"


def save_plot(result: PairResult, output_prefix: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required unless --no-plot is used") from error

    figure, axis = plt.subplots(figsize=(10, 5.2))
    axis.plot(
        result.gpu.plot_times_s,
        result.gpu.plot_powers_w,
        color="#59A14F",
        linewidth=1.7,
        marker="o",
        markersize=3,
        label=f"GPU board ({result.gpu.energy_j:.2f} J)",
    )
    axis.step(
        result.cpu.plot_times_s,
        result.cpu.plot_powers_w,
        where="mid",
        color="#4E79A7",
        linewidth=1.5,
        label=f"Host RAPL interval power ({result.cpu.energy_j:.2f} J)",
    )
    axis.axhline(
        result.average_power_w,
        color="#E15759",
        linestyle="--",
        linewidth=1.4,
        label=f"Combined time-average ({result.average_power_w:.1f} W)",
    )
    axis.set_title(
        f"GPU + host power over common window "
        f"({result.total_energy_j:.2f} J, {result.duration_s:.3f} s)"
    )
    axis.set_xlabel("Time from common-window start (s)")
    axis.set_ylabel("Power (W)")
    axis.grid(True, linestyle=":", alpha=0.55)
    axis.legend(frameon=False)
    figure.tight_layout()

    output_path = output_prefix.with_name(
        f"{output_prefix.name}_timeline.png"
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_aggregate_csv(
    analysed_pairs: list[tuple[PairSpec, PairResult]],
    output_path: Path,
    query_count: int,
) -> None:
    """Write one machine-readable table containing every valid pair."""

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "NList",
                "NProbe",
                "GPUFile",
                "CPUFile",
                "WindowStart",
                "WindowEnd",
                "Duration_s",
                "GPUSamples",
                "CPUSamples",
                "GPUEnergy_J",
                "HostEnergy_J",
                "TotalEnergy_J",
                "GPUAvgPower_W",
                "HostAvgPower_W",
                "CombinedAvgPower_W",
                "Queries",
                "EnergyPerQuery_mJ",
            ]
        )
        for spec, result in analysed_pairs:
            writer.writerow(
                [
                    spec.nlist,
                    spec.nprobe,
                    result.gpu_path.name,
                    result.cpu_path.name,
                    f"{result.start_time_s:.9f}",
                    f"{result.end_time_s:.9f}",
                    f"{result.duration_s:.9f}",
                    result.gpu.sample_count,
                    result.cpu.sample_count,
                    f"{result.gpu.energy_j:.9f}",
                    f"{result.cpu.energy_j:.9f}",
                    f"{result.total_energy_j:.9f}",
                    f"{result.gpu.average_power_w:.9f}",
                    f"{result.cpu.average_power_w:.9f}",
                    f"{result.average_power_w:.9f}",
                    query_count,
                    f"{result.total_energy_j / query_count * 1000.0:.9f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="directory containing the trace pairs (default: current directory)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100_000,
        help="queries represented by the common window (default: 100000)",
    )
    parser.add_argument(
        "--trim-start",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="omit seconds from the beginning of the common window",
    )
    parser.add_argument(
        "--trim-end",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="omit seconds from the end of the common window",
    )
    parser.add_argument(
        "--rapl-max-uj",
        type=float,
        default=None,
        help="RAPL max_energy_range_uj, required only when the counter wraps",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="aggregate CSV path (default: DIRECTORY/gpu_host_energy_summary.csv)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.queries <= 0:
        parser.error("--queries must be positive")
    if args.trim_start < 0 or args.trim_end < 0:
        parser.error("trim values cannot be negative")
    if args.rapl_max_uj is not None and args.rapl_max_uj <= 0:
        parser.error("--rapl-max-uj must be positive")
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    try:
        pairs, discovery_warnings = discover_pairs(args.directory)
    except OSError as error:
        parser.error(str(error))

    for warning in discovery_warnings:
        print(f"WARNING: {warning}")
    if not pairs:
        parser.error(
            "no complete trace pairs matched "
            "gpu_power_nlist<N>_nprobe<P>_{gpu<ID>,cpu}.{log,csv}"
        )

    analysed_pairs: list[tuple[PairSpec, PairResult]] = []
    for spec in pairs:
        print()
        print(f"NList={spec.nlist}, NProbe={spec.nprobe}")
        print(f"  GPU file: {spec.gpu_path.name}")
        print(f"  CPU file: {spec.cpu_path.name}")
        try:
            result = analyse_pair(
                spec.gpu_path,
                spec.cpu_path,
                args.trim_start,
                args.trim_end,
                args.rapl_max_uj,
            )
        except (OSError, ValueError) as error:
            print(f"  ERROR: {error}")
            continue

        analysed_pairs.append((spec, result))
        print(f"  common window:       {result.duration_s:.3f} s")
        print(f"  GPU energy:          {result.gpu.energy_j:.3f} J")
        print(f"  host energy:         {result.cpu.energy_j:.3f} J")
        print(f"  combined energy:     {result.total_energy_j:.3f} J")
        print(f"  combined avg power:  {result.average_power_w:.3f} W")
        print(
            f"  energy per query:    "
            f"{result.total_energy_j / args.queries * 1000.0:.6f} mJ"
        )
        print(
            f"  sample gaps (GPU/CPU max): "
            f"{result.gpu.maximum_gap_s:.3f}/"
            f"{result.cpu.maximum_gap_s:.3f} s"
        )

        if not args.no_plot:
            output_prefix = default_output_prefix(result)
            try:
                plot_path = save_plot(result, output_prefix)
            except RuntimeError as error:
                parser.error(str(error))
            print(f"  timeline plot:       {plot_path}")

    if not analysed_pairs:
        parser.error("all discovered pairs failed validation")

    summary_path = args.summary_csv or (
        args.directory / "gpu_host_energy_summary.csv"
    )
    save_aggregate_csv(analysed_pairs, summary_path, args.queries)

    print()
    heading = (
        f"{'NList':>7} {'NProbe':>7} {'Time (s)':>10} "
        f"{'GPU (J)':>11} {'Host (J)':>11} {'Total (J)':>12} "
        f"{'mJ/query':>11}"
    )
    print(heading)
    print("-" * len(heading))
    for spec, result in analysed_pairs:
        print(
            f"{spec.nlist:>7d} {spec.nprobe:>7d} "
            f"{result.duration_s:>10.3f} "
            f"{result.gpu.energy_j:>11.3f} "
            f"{result.cpu.energy_j:>11.3f} "
            f"{result.total_energy_j:>12.3f} "
            f"{result.total_energy_j / args.queries * 1000.0:>11.6f}"
        )
    print(f"\nAggregate summary: {summary_path}")


if __name__ == "__main__":
    main()
