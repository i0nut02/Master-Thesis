#!/usr/bin/env python3
"""Analyse the independent energy sessions produced by the Slurm jobs.

With no arguments, the script reads the newest job directory for each platform
below ``energy_raw``:

    energy_raw/cpu_56/job_<ID>/nlist<N>_nprobe<P>/session_<S>_task_<T>/
    energy_raw/cpu_112/job_<ID>/nlist<N>_nprobe<P>/session_<S>_task_<T>/
    energy_raw/gpu_a100/job_<ID>/nlist<N>_nprobe<P>/session_<S>_task_<T>/

Each session directory must contain ``events.csv``, ``host_rapl.csv``, and
``rapl_zones.csv``. GPU sessions must also contain ``gpu_power.csv``. The exact
``benchmark_start`` and ``benchmark_end`` markers define the integration
window, including process initialization and all 100,000 searches.

RAPL files contain cumulative microjoule counters. Each zone is unwrapped and
its boundary values are interpolated before the energy difference is taken.
GPU power is integrated over the same benchmark window with the trapezoidal
rule and the recorded timestamps. No fixed startup interval is discarded.

Outputs are written to ``energy_raw/analysis`` by default:

* ``energy_sessions.csv``: one row per independent session, including average
  power computed as integrated session energy divided by session duration;
* ``energy_configuration_summary.csv``: median, quartiles, extrema, standard
  deviation, and relative standard deviation for every configuration;
* ``energy_per_query_distribution.png`` and ``energy_rsd.png`` unless plotting
  is disabled.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PLATFORM_LABELS = {
    "cpu_56": "CPU (56 cores)",
    "cpu_112": "CPU (112 cores)",
    "gpu_a100": "GPU (A100) + CPU",
}

PLATFORM_ORDER = {
    "cpu_56": 0,
    "cpu_112": 1,
    "gpu_a100": 2,
}

PLATFORM_COLORS = {
    "cpu_56": "#4E79A7",
    "cpu_112": "#F28E2B",
    "gpu_a100": "#59A14F",
}

CONFIGURATION_ORDER = (
    (2048, 4),
    (2048, 16),
    (1024, 32),
    (512, 64),
)

TARGET_LABELS = {
    (2048, 4): "~60%",
    (2048, 16): "~78%",
    (1024, 32): "~88%",
    (512, 64): "~95%",
}

CONFIG_PATTERN = re.compile(r"^nlist(?P<nlist>\d+)_nprobe(?P<nprobe>\d+)$")
SESSION_PATTERN = re.compile(
    r"^session_(?P<session>\d+)_task_(?P<task>\d+)$"
)
JOB_PATTERN = re.compile(r"^job_(?P<job>\d+)$")


@dataclass(frozen=True)
class RaplZone:
    column: str
    name: str
    path: str
    maximum_uj: float | None


@dataclass(frozen=True)
class SessionSpec:
    platform: str
    job_id: int
    nlist: int
    nprobe: int
    session_number: int
    task_id: int
    directory: Path


@dataclass(frozen=True)
class SessionResult:
    spec: SessionSpec
    start_time_s: float
    end_time_s: float
    duration_s: float
    query_count: int
    host_energy_j: float
    accelerator_energy_j: float
    total_energy_j: float
    average_power_w: float
    energy_per_query_mj: float
    rapl_samples: int
    rapl_maximum_gap_s: float
    gpu_samples: int | None
    gpu_maximum_gap_s: float | None


@dataclass(frozen=True)
class ConfigurationSummary:
    platform: str
    nlist: int
    nprobe: int
    target: str
    session_count: int
    mean_energy_j: float
    standard_deviation_j: float
    rsd_percent: float
    minimum_energy_j: float
    first_quartile_j: float
    median_energy_j: float
    third_quartile_j: float
    maximum_energy_j: float
    median_energy_per_query_mj: float
    median_duration_s: float
    median_average_power_w: float


def parse_finite_float(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{context}: expected a numeric value") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{context}: value is not finite")
    return parsed


def read_events(path: Path) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Event", "Timestamp"]:
            raise ValueError(
                f"{path}: expected the header Event,Timestamp"
            )
        events: dict[str, float] = {}
        for line_number, row in enumerate(reader, start=2):
            event = (row.get("Event") or "").strip()
            if not event:
                raise ValueError(f"{path}: line {line_number} has no event")
            if event in events:
                raise ValueError(f"{path}: duplicate event {event!r}")
            events[event] = parse_finite_float(
                row.get("Timestamp") or "",
                f"{path}: line {line_number}",
            )

    missing = {
        "benchmark_start",
        "benchmark_end",
    }.difference(events)
    if missing:
        raise ValueError(f"{path}: missing events {sorted(missing)}")

    start_time_s = events["benchmark_start"]
    end_time_s = events["benchmark_end"]
    if end_time_s <= start_time_s:
        raise ValueError(f"{path}: benchmark interval is not positive")
    return start_time_s, end_time_s


def read_rapl_metadata(path: Path) -> list[RaplZone]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "Column",
            "ZoneName",
            "EnergyPath",
            "MaxEnergyRange_uJ",
        }
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"{path}: unexpected RAPL metadata header")

        zones: list[RaplZone] = []
        for line_number, row in enumerate(reader, start=2):
            maximum_text = (row.get("MaxEnergyRange_uJ") or "").strip()
            maximum_uj = None
            if maximum_text:
                maximum_uj = parse_finite_float(
                    maximum_text,
                    f"{path}: line {line_number} maximum range",
                )
                if maximum_uj <= 0:
                    raise ValueError(
                        f"{path}: line {line_number} has a non-positive range"
                    )
            zones.append(
                RaplZone(
                    column=(row.get("Column") or "").strip(),
                    name=(row.get("ZoneName") or "").strip(),
                    path=(row.get("EnergyPath") or "").strip(),
                    maximum_uj=maximum_uj,
                )
            )

    if not zones:
        raise ValueError(f"{path}: no RAPL zones were recorded")
    if any(not zone.column for zone in zones):
        raise ValueError(f"{path}: a RAPL zone has no column name")
    return zones


def read_numeric_csv(
    path: Path,
    required_columns: Iterable[str],
) -> dict[str, np.ndarray]:
    columns = list(required_columns)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        values = {column: [] for column in columns}
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not (value or "").strip() for value in row.values()):
                continue
            for column in columns:
                values[column].append(
                    parse_finite_float(
                        row.get(column) or "",
                        f"{path}: line {line_number}, column {column}",
                    )
                )

    arrays = {
        column: np.asarray(column_values, dtype=float)
        for column, column_values in values.items()
    }
    lengths = {len(array) for array in arrays.values()}
    if lengths != {next(iter(lengths), 0)} or not lengths:
        raise ValueError(f"{path}: inconsistent column lengths")
    sample_count = next(iter(lengths))
    if sample_count < 2:
        raise ValueError(f"{path}: fewer than two samples")

    timestamps = arrays["Timestamp"]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    return arrays


def unwrap_rapl_counter(
    raw_values_uj: np.ndarray,
    maximum_uj: float | None,
    context: str,
) -> np.ndarray:
    unwrapped = np.empty_like(raw_values_uj, dtype=float)
    unwrapped[0] = raw_values_uj[0]
    offset = 0.0
    previous_raw = raw_values_uj[0]

    for index in range(1, len(raw_values_uj)):
        current_raw = raw_values_uj[index]
        if current_raw < previous_raw:
            if maximum_uj is None:
                raise ValueError(
                    f"{context}: counter decreased but no maximum range was "
                    "recorded"
                )
            offset += maximum_uj
        unwrapped[index] = current_raw + offset
        previous_raw = current_raw

    if np.any(np.diff(unwrapped) < 0):
        raise ValueError(f"{context}: counter could not be unwrapped")
    return unwrapped


def ensure_window_is_bracketed(
    timestamps: np.ndarray,
    start_time_s: float,
    end_time_s: float,
    context: str,
) -> None:
    if timestamps[0] > start_time_s or timestamps[-1] < end_time_s:
        raise ValueError(
            f"{context}: samples do not bracket the benchmark interval "
            f"[{start_time_s:.6f}, {end_time_s:.6f}]"
        )


def clip_with_boundaries(
    timestamps: np.ndarray,
    values: np.ndarray,
    start_time_s: float,
    end_time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
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


def trapezoidal_integral(
    values: np.ndarray,
    timestamps: np.ndarray,
) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(values, timestamps))
    return float(np.trapz(values, timestamps))


def analyse_host_rapl(
    session_directory: Path,
    start_time_s: float,
    end_time_s: float,
) -> tuple[float, int, float]:
    zones = read_rapl_metadata(session_directory / "rapl_zones.csv")
    columns = ["Timestamp", *(zone.column for zone in zones)]
    arrays = read_numeric_csv(session_directory / "host_rapl.csv", columns)
    timestamps = arrays["Timestamp"]
    ensure_window_is_bracketed(
        timestamps,
        start_time_s,
        end_time_s,
        str(session_directory / "host_rapl.csv"),
    )

    total_energy_j = 0.0
    for zone in zones:
        unwrapped_uj = unwrap_rapl_counter(
            arrays[zone.column],
            zone.maximum_uj,
            f"{session_directory}: {zone.name}",
        )
        _, clipped_uj = clip_with_boundaries(
            timestamps,
            unwrapped_uj,
            start_time_s,
            end_time_s,
        )
        zone_energy_j = (clipped_uj[-1] - clipped_uj[0]) / 1_000_000.0
        if zone_energy_j < 0:
            raise ValueError(
                f"{session_directory}: negative energy for {zone.name}"
            )
        total_energy_j += float(zone_energy_j)

    return total_energy_j, len(timestamps), float(np.max(np.diff(timestamps)))


def analyse_gpu_power(
    session_directory: Path,
    start_time_s: float,
    end_time_s: float,
) -> tuple[float, int, float]:
    path = session_directory / "gpu_power.csv"
    arrays = read_numeric_csv(path, ["Timestamp", "Power_W"])
    timestamps = arrays["Timestamp"]
    powers_w = arrays["Power_W"]
    if np.any(powers_w < 0):
        raise ValueError(f"{path}: negative GPU power sample")
    ensure_window_is_bracketed(
        timestamps,
        start_time_s,
        end_time_s,
        str(path),
    )
    clipped_times, clipped_powers = clip_with_boundaries(
        timestamps,
        powers_w,
        start_time_s,
        end_time_s,
    )
    energy_j = trapezoidal_integral(clipped_powers, clipped_times)
    return energy_j, len(timestamps), float(np.max(np.diff(timestamps)))


def newest_job_directory(platform_directory: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in platform_directory.iterdir():
        if not path.is_dir():
            continue
        match = JOB_PATTERN.match(path.name)
        if match:
            candidates.append((int(match.group("job")), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def discover_sessions(root: Path, all_jobs: bool) -> tuple[list[SessionSpec], list[str]]:
    sessions: list[SessionSpec] = []
    selections: list[str] = []

    for platform in PLATFORM_ORDER:
        platform_directory = root / platform
        if not platform_directory.is_dir():
            continue

        if all_jobs:
            job_directories = sorted(
                (
                    path
                    for path in platform_directory.iterdir()
                    if path.is_dir() and JOB_PATTERN.match(path.name)
                ),
                key=lambda path: int(JOB_PATTERN.match(path.name).group("job")),
            )
        else:
            newest = newest_job_directory(platform_directory)
            job_directories = [newest] if newest is not None else []

        for job_directory in job_directories:
            job_match = JOB_PATTERN.match(job_directory.name)
            if job_match is None:
                continue
            job_id = int(job_match.group("job"))
            selections.append(f"{platform}: job_{job_id}")

            for config_directory in sorted(job_directory.iterdir()):
                if not config_directory.is_dir():
                    continue
                config_match = CONFIG_PATTERN.match(config_directory.name)
                if config_match is None:
                    continue
                nlist = int(config_match.group("nlist"))
                nprobe = int(config_match.group("nprobe"))

                for session_directory in sorted(config_directory.iterdir()):
                    if not session_directory.is_dir():
                        continue
                    session_match = SESSION_PATTERN.match(session_directory.name)
                    if session_match is None:
                        continue
                    sessions.append(
                        SessionSpec(
                            platform=platform,
                            job_id=job_id,
                            nlist=nlist,
                            nprobe=nprobe,
                            session_number=int(session_match.group("session")),
                            task_id=int(session_match.group("task")),
                            directory=session_directory,
                        )
                    )

    sessions.sort(
        key=lambda spec: (
            PLATFORM_ORDER[spec.platform],
            CONFIGURATION_ORDER.index((spec.nlist, spec.nprobe))
            if (spec.nlist, spec.nprobe) in CONFIGURATION_ORDER
            else len(CONFIGURATION_ORDER),
            spec.session_number,
            spec.job_id,
        )
    )
    return sessions, selections


def analyse_session(spec: SessionSpec, query_count: int) -> SessionResult:
    start_time_s, end_time_s = read_events(spec.directory / "events.csv")
    duration_s = end_time_s - start_time_s
    host_energy_j, rapl_samples, rapl_maximum_gap_s = analyse_host_rapl(
        spec.directory,
        start_time_s,
        end_time_s,
    )

    accelerator_energy_j = 0.0
    gpu_samples = None
    gpu_maximum_gap_s = None
    if spec.platform == "gpu_a100":
        (
            accelerator_energy_j,
            gpu_samples,
            gpu_maximum_gap_s,
        ) = analyse_gpu_power(spec.directory, start_time_s, end_time_s)

    total_energy_j = host_energy_j + accelerator_energy_j
    return SessionResult(
        spec=spec,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        duration_s=duration_s,
        query_count=query_count,
        host_energy_j=host_energy_j,
        accelerator_energy_j=accelerator_energy_j,
        total_energy_j=total_energy_j,
        average_power_w=total_energy_j / duration_s,
        energy_per_query_mj=total_energy_j / query_count * 1000.0,
        rapl_samples=rapl_samples,
        rapl_maximum_gap_s=rapl_maximum_gap_s,
        gpu_samples=gpu_samples,
        gpu_maximum_gap_s=gpu_maximum_gap_s,
    )


def group_results(
    results: Iterable[SessionResult],
) -> dict[tuple[str, int, int], list[SessionResult]]:
    grouped: dict[tuple[str, int, int], list[SessionResult]] = {}
    for result in results:
        key = (result.spec.platform, result.spec.nlist, result.spec.nprobe)
        grouped.setdefault(key, []).append(result)
    return grouped


def summarize_configurations(
    results: list[SessionResult],
) -> list[ConfigurationSummary]:
    summaries: list[ConfigurationSummary] = []
    grouped = group_results(results)

    for (platform, nlist, nprobe), group in grouped.items():
        energies = np.asarray(
            [result.total_energy_j for result in group],
            dtype=float,
        )
        durations = np.asarray(
            [result.duration_s for result in group],
            dtype=float,
        )
        powers = np.asarray(
            [result.average_power_w for result in group],
            dtype=float,
        )
        q1, median, q3 = np.percentile(energies, [25, 50, 75])
        mean = float(np.mean(energies))
        if len(energies) >= 2:
            standard_deviation = float(statistics.stdev(energies.tolist()))
        else:
            standard_deviation = math.nan
        rsd = standard_deviation / mean * 100.0 if mean > 0 else math.nan

        summaries.append(
            ConfigurationSummary(
                platform=platform,
                nlist=nlist,
                nprobe=nprobe,
                target=TARGET_LABELS.get((nlist, nprobe), ""),
                session_count=len(group),
                mean_energy_j=mean,
                standard_deviation_j=standard_deviation,
                rsd_percent=rsd,
                minimum_energy_j=float(np.min(energies)),
                first_quartile_j=float(q1),
                median_energy_j=float(median),
                third_quartile_j=float(q3),
                maximum_energy_j=float(np.max(energies)),
                median_energy_per_query_mj=float(median / group[0].query_count * 1000.0),
                median_duration_s=float(np.median(durations)),
                median_average_power_w=float(np.median(powers)),
            )
        )

    summaries.sort(
        key=lambda summary: (
            PLATFORM_ORDER[summary.platform],
            CONFIGURATION_ORDER.index((summary.nlist, summary.nprobe))
            if (summary.nlist, summary.nprobe) in CONFIGURATION_ORDER
            else len(CONFIGURATION_ORDER),
        )
    )
    return summaries


def write_session_csv(
    results: list[SessionResult],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Platform",
                "PlatformLabel",
                "JobID",
                "ArrayTaskID",
                "Session",
                "NList",
                "NProbe",
                "TargetRecall",
                "StartTimestamp",
                "EndTimestamp",
                "Duration_s",
                "Queries",
                "HostEnergy_J",
                "AcceleratorEnergy_J",
                "TotalEnergy_J",
                "AveragePower_W",
                "EnergyPerQuery_mJ",
                "RaplSamples",
                "RaplMaxGap_s",
                "GPUSamples",
                "GPUMaxGap_s",
                "SessionDirectory",
            ]
        )
        for result in results:
            spec = result.spec
            writer.writerow(
                [
                    spec.platform,
                    PLATFORM_LABELS[spec.platform],
                    spec.job_id,
                    spec.task_id,
                    spec.session_number,
                    spec.nlist,
                    spec.nprobe,
                    TARGET_LABELS.get((spec.nlist, spec.nprobe), ""),
                    f"{result.start_time_s:.9f}",
                    f"{result.end_time_s:.9f}",
                    f"{result.duration_s:.9f}",
                    result.query_count,
                    f"{result.host_energy_j:.9f}",
                    f"{result.accelerator_energy_j:.9f}",
                    f"{result.total_energy_j:.9f}",
                    f"{result.average_power_w:.9f}",
                    f"{result.energy_per_query_mj:.9f}",
                    result.rapl_samples,
                    f"{result.rapl_maximum_gap_s:.9f}",
                    result.gpu_samples if result.gpu_samples is not None else "",
                    (
                        f"{result.gpu_maximum_gap_s:.9f}"
                        if result.gpu_maximum_gap_s is not None
                        else ""
                    ),
                    str(spec.directory),
                ]
            )


def write_summary_csv(
    summaries: list[ConfigurationSummary],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Platform",
                "PlatformLabel",
                "NList",
                "NProbe",
                "TargetRecall",
                "SessionCount",
                "MeanEnergy_J",
                "StdEnergy_J",
                "RSD_percent",
                "MinEnergy_J",
                "Q1Energy_J",
                "MedianEnergy_J",
                "Q3Energy_J",
                "MaxEnergy_J",
                "MedianEnergyPerQuery_mJ",
                "MedianDuration_s",
                "MedianAveragePower_W",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.platform,
                    PLATFORM_LABELS[summary.platform],
                    summary.nlist,
                    summary.nprobe,
                    summary.target,
                    summary.session_count,
                    f"{summary.mean_energy_j:.9f}",
                    f"{summary.standard_deviation_j:.9f}",
                    f"{summary.rsd_percent:.9f}",
                    f"{summary.minimum_energy_j:.9f}",
                    f"{summary.first_quartile_j:.9f}",
                    f"{summary.median_energy_j:.9f}",
                    f"{summary.third_quartile_j:.9f}",
                    f"{summary.maximum_energy_j:.9f}",
                    f"{summary.median_energy_per_query_mj:.9f}",
                    f"{summary.median_duration_s:.9f}",
                    f"{summary.median_average_power_w:.9f}",
                ]
            )


def print_summary(summaries: list[ConfigurationSummary]) -> None:
    heading = (
        f"{'Platform':<18} {'Config':<13} {'N':>3} "
        f"{'Median (J)':>12} {'IQR (J)':>23} {'Min/Max (J)':>23} "
        f"{'RSD':>8} {'Time (s)':>10} {'Avg power (W)':>15} "
        f"{'mJ/query':>11}"
    )
    print(heading)
    print("-" * len(heading))
    for summary in summaries:
        configuration = f"{summary.nlist}/{summary.nprobe} {summary.target}"
        iqr = (
            f"{summary.first_quartile_j:.2f}.."
            f"{summary.third_quartile_j:.2f}"
        )
        extrema = (
            f"{summary.minimum_energy_j:.2f}.."
            f"{summary.maximum_energy_j:.2f}"
        )
        rsd = (
            f"{summary.rsd_percent:.2f}%"
            if math.isfinite(summary.rsd_percent)
            else "N/A"
        )
        print(
            f"{PLATFORM_LABELS[summary.platform]:<18} "
            f"{configuration:<13} {summary.session_count:>3d} "
            f"{summary.median_energy_j:>12.2f} {iqr:>23} "
            f"{extrema:>23} {rsd:>8} "
            f"{summary.median_duration_s:>10.3f} "
            f"{summary.median_average_power_w:>15.2f} "
            f"{summary.median_energy_per_query_mj:>11.4f}"
        )


def save_distribution_plot(
    results: list[SessionResult],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as error:
        raise RuntimeError("matplotlib is required for plots") from error

    grouped = group_results(results)
    figure, axis = plt.subplots(figsize=(12.5, 6.2))
    configuration_positions = np.arange(len(CONFIGURATION_ORDER), dtype=float)
    platforms = [
        platform
        for platform in PLATFORM_ORDER
        if any(key[0] == platform for key in grouped)
    ]
    offsets = np.linspace(-0.27, 0.27, len(platforms)) if platforms else []

    for platform, offset in zip(platforms, offsets):
        for config_index, (nlist, nprobe) in enumerate(CONFIGURATION_ORDER):
            group = grouped.get((platform, nlist, nprobe), [])
            if not group:
                continue
            values = [result.energy_per_query_mj for result in group]
            position = configuration_positions[config_index] + offset
            box = axis.boxplot(
                [values],
                positions=[position],
                widths=0.22,
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 5,
                },
                medianprops={"color": "black", "linewidth": 1.4},
                whiskerprops={"linewidth": 1.1},
                capprops={"linewidth": 1.1},
            )
            box["boxes"][0].set_facecolor(PLATFORM_COLORS[platform])
            box["boxes"][0].set_alpha(0.75)
            jitter = np.linspace(-0.025, 0.025, len(values))
            axis.scatter(
                position + jitter,
                values,
                color="black",
                s=12,
                alpha=0.45,
                zorder=3,
            )

    labels = [
        f"{TARGET_LABELS[configuration]}\n"
        f"({configuration[0]}, {configuration[1]})"
        for configuration in CONFIGURATION_ORDER
    ]
    axis.set_xticks(configuration_positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("Nominal Recall@10 target and (NList, NProbe)")
    axis.set_ylabel("Energy per query (mJ), logarithmic scale")
    axis.set_yscale("log")
    axis.set_title("Distribution across independent 100,000-query sessions")
    axis.grid(True, axis="y", which="both", linestyle=":", alpha=0.5)
    axis.legend(
        handles=[
            Patch(
                facecolor=PLATFORM_COLORS[platform],
                alpha=0.75,
                label=PLATFORM_LABELS[platform],
            )
            for platform in platforms
        ],
        frameon=False,
        loc="upper left",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_rsd_plot(
    summaries: list[ConfigurationSummary],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for plots") from error

    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    configuration_positions = np.arange(len(CONFIGURATION_ORDER), dtype=float)
    platforms = [
        platform
        for platform in PLATFORM_ORDER
        if any(summary.platform == platform for summary in summaries)
    ]
    width = 0.8 / max(len(platforms), 1)

    summary_lookup = {
        (summary.platform, summary.nlist, summary.nprobe): summary
        for summary in summaries
    }
    for platform_index, platform in enumerate(platforms):
        offset = (platform_index - (len(platforms) - 1) / 2) * width
        heights = []
        for nlist, nprobe in CONFIGURATION_ORDER:
            summary = summary_lookup.get((platform, nlist, nprobe))
            heights.append(summary.rsd_percent if summary is not None else np.nan)
        axis.bar(
            configuration_positions + offset,
            heights,
            width=width,
            color=PLATFORM_COLORS[platform],
            label=PLATFORM_LABELS[platform],
        )

    labels = [
        f"{TARGET_LABELS[configuration]}\n"
        f"({configuration[0]}, {configuration[1]})"
        for configuration in CONFIGURATION_ORDER
    ]
    axis.set_xticks(configuration_positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("Nominal Recall@10 target and (NList, NProbe)")
    axis.set_ylabel("Energy RSD (%)")
    axis.set_title("Run-to-run energy variability")
    axis.grid(True, axis="y", linestyle=":", alpha=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("energy_raw"),
        help="energy-data root (default: energy_raw)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="output directory (default: ROOT/analysis)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100_000,
        help="queries in each independent session (default: 100000)",
    )
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=10,
        help="expected sessions per configuration (default: 10)",
    )
    parser.add_argument(
        "--all-jobs",
        action="store_true",
        help="combine all job_<ID> directories instead of the newest per platform",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.queries <= 0:
        parser.error("--queries must be positive")
    if args.expected_sessions <= 0:
        parser.error("--expected-sessions must be positive")
    if not args.root.is_dir():
        parser.error(f"energy root does not exist: {args.root}")

    sessions, selections = discover_sessions(args.root, args.all_jobs)
    if not sessions:
        parser.error(f"no session directories were found below {args.root}")

    print("Selected energy jobs:")
    for selection in dict.fromkeys(selections):
        print(f"  {selection}")

    results: list[SessionResult] = []
    failures: list[str] = []
    for spec in sessions:
        try:
            result = analyse_session(spec, args.queries)
        except (OSError, ValueError) as error:
            failures.append(f"{spec.directory}: {error}")
            continue
        results.append(result)

    if failures:
        print("\nRejected sessions:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
    if not results:
        parser.error("all discovered sessions failed validation")

    summaries = summarize_configurations(results)

    if args.output_directory is not None:
        output_directory = args.output_directory
    else:
        output_directory = args.root / "analysis"
    output_directory.mkdir(parents=True, exist_ok=True)

    session_csv = output_directory / "energy_sessions.csv"
    summary_csv = output_directory / "energy_configuration_summary.csv"
    write_session_csv(
        results,
        session_csv,
    )
    write_summary_csv(
        summaries,
        summary_csv,
    )

    print()
    print_summary(summaries)

    incomplete = [
        summary
        for summary in summaries
        if summary.session_count != args.expected_sessions
    ]
    if incomplete:
        print("\nWARNING: unexpected session counts:", file=sys.stderr)
        for summary in incomplete:
            print(
                f"  {PLATFORM_LABELS[summary.platform]} "
                f"({summary.nlist}, {summary.nprobe}): "
                f"{summary.session_count}, expected {args.expected_sessions}",
                file=sys.stderr,
            )

    if not args.no_plots:
        try:
            save_distribution_plot(
                results,
                output_directory / "energy_per_query_distribution.png",
            )
            save_rsd_plot(
                summaries,
                output_directory / "energy_rsd.png",
            )
        except RuntimeError as error:
            parser.error(str(error))

    print()
    print(f"Per-session results: {session_csv}")
    print(f"Configuration summary: {summary_csv}")
    if not args.no_plots:
        print(
            "Distribution plot: "
            f"{output_directory / 'energy_per_query_distribution.png'}"
        )
        print(f"RSD plot: {output_directory / 'energy_rsd.png'}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
