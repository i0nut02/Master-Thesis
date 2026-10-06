#!/usr/bin/env python3
"""Reproduce the thesis's recall-matched energy figures and legacy cost plots.

The input CSV contains the retained trapezoidally integrated session-energy
totals and elapsed times used for the published tables. Every row represents
one complete configuration session containing initialization and 100,000
queries. Average power is derived as integrated energy divided by elapsed time;
it is not used to reconstruct energy from an arithmetic mean of samples.

The throughput-energy figure uses 100,000 / retained elapsed time as a derived
throughput. It must not be confused with the thesis's median QPS from ten
search-only repetitions, whose numerical source data are not in this file.

Run from any directory with Python 3.9+ and Matplotlib installed:

    python3 scripts/plot_energy_cost.py
    python3 scripts/plot_energy_cost.py --output-dir /path/to/figures
    python3 scripts/plot_energy_cost.py --check-only

The default output directory is ./generated_energy_cost. Existing outputs are
not replaced unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


D = Decimal
ROOT = Path(__file__).resolve().parents[1]
ENERGY_INPUT = ROOT / "data" / "energy_session_inputs.csv"
QUERIES_PER_RUN = D("100000")
QUERIES_PER_BILLION = D("1000000000")
JOULES_PER_KWH = D("3600000")
ELECTRICITY_EUR_PER_KWH = D("0.30")
PUE = D("1.20")
EUR_PER_JOULE = ELECTRICITY_EUR_PER_KWH * PUE / JOULES_PER_KWH
ENERGY_PRECISION_J = D("0.01")
QUERY_ENERGY_PRECISION_MJ = D("0.001")
COST_PRECISION_100K_EUR = D("0.00000001")
COST_PRECISION_BILLION_EUR = D("0.0001")

RECALL_TARGETS = ("~60%", "~78%", "~88%", "~95%")
TARGET_95_INDEX = RECALL_TARGETS.index("~95%")
MAX_QUERY_BILLIONS = 1600


@dataclass(frozen=True)
class Measurement:
    elapsed_s: Decimal
    integrated_energy_j: Decimal

    @property
    def energy_j(self) -> Decimal:
        return self.integrated_energy_j.quantize(
            ENERGY_PRECISION_J, rounding=ROUND_HALF_UP
        )

    @property
    def average_power_w(self) -> Decimal:
        return self.integrated_energy_j / self.elapsed_s

    @property
    def energy_mj_per_query(self) -> Decimal:
        return (self.energy_j * D("1000") / QUERIES_PER_RUN).quantize(
            QUERY_ENERGY_PRECISION_MJ, rounding=ROUND_HALF_UP
        )

    @property
    def electricity_eur_per_100k(self) -> Decimal:
        return (self.energy_j * EUR_PER_JOULE).quantize(
            COST_PRECISION_100K_EUR, rounding=ROUND_HALF_UP
        )

    @property
    def electricity_eur_per_billion(self) -> Decimal:
        return (
            self.energy_j * EUR_PER_JOULE * QUERIES_PER_BILLION / QUERIES_PER_RUN
        ).quantize(COST_PRECISION_BILLION_EUR, rounding=ROUND_HALF_UP)

    @property
    def reported_interval_qps(self) -> Decimal:
        # Derived from the retained elapsed-time input, not median search-only QPS.
        return (QUERIES_PER_RUN / self.elapsed_s).quantize(D("0.01"), rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class Platform:
    name: str
    color: str
    fixed_cost_eur: Decimal
    measurements: tuple[Measurement, ...]

    @property
    def at_95(self) -> Measurement:
        return self.measurements[TARGET_95_INDEX]


@dataclass(frozen=True)
class Crossing:
    first: Platform
    second: Platform
    query_billions: Decimal
    cost_eur: Decimal


PLATFORM_METADATA = (
    ("CPU (56 cores)", "#4E79A7", D("8500")),
    ("CPU (112 cores)", "#F28E2B", D("15000")),
    ("GPU (A100) + CPU", "#59A14F", D("20300")),
    ("TT + CPU", "#B07AA1", D("1200")),
)


def load_platforms(path: Path) -> tuple[Platform, ...]:
    """Load and validate the retained session measurements used by every plot."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rows_by_platform: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        platform = row["platform"]
        target = int(row["nominal_recall_pct"])
        if target in rows_by_platform.setdefault(platform, {}):
            raise ValueError(f"Duplicate energy row: {platform}, {target}%")
        rows_by_platform[platform][target] = row

    platforms = []
    for name, color, fixed_cost in PLATFORM_METADATA:
        platform_rows = rows_by_platform.get(name, {})
        measurements = []
        for target in (60, 78, 88, 95):
            if target not in platform_rows:
                raise ValueError(f"Missing energy row: {name}, {target}%")
            row = platform_rows[target]
            measurement = Measurement(
                D(row["elapsed_s"]), D(row["integrated_energy_100k_j"])
            )
            if measurement.energy_mj_per_query != D(row["energy_mj_per_query"]):
                raise ValueError(
                    f"Per-query energy mismatch in input row: {name}, {target}%"
                )
            measurements.append(measurement)
        platforms.append(Platform(name, color, fixed_cost, tuple(measurements)))

    unexpected = set(rows_by_platform) - {item[0] for item in PLATFORM_METADATA}
    if unexpected:
        raise ValueError(f"Unexpected platforms in {path}: {sorted(unexpected)}")
    return tuple(platforms)


# Rows are approximately recall-matched energy sessions, not same-NProbe
# configurations. The exact configurations and recalls are retained in the
# input CSV and in the methodology's operating-point table.
PLATFORMS = load_platforms(ENERGY_INPUT)

FIGURE_NAMES = (
    "energy_comparison_recall",
    "energy_per_query_recall",
    "session_throughput_vs_energy",
    "cost_scaling_recall95",
)
CSV_NAME = "energy_cost_derived.csv"
SHORT_NAMES = {
    "CPU (56 cores)": "CPU (56)",
    "CPU (112 cores)": "CPU (112)",
    "GPU (A100) + CPU": "GPU",
    "TT + CPU": "TT",
}


def validate_inputs() -> None:
    if PUE < 1 or ELECTRICITY_EUR_PER_KWH < 0:
        raise ValueError("PUE must be at least 1 and electricity price non-negative")
    if len({platform.name for platform in PLATFORMS}) != len(PLATFORMS):
        raise ValueError("Platform names must be unique")
    for platform in PLATFORMS:
        if platform.fixed_cost_eur < 0 or len(platform.measurements) != len(RECALL_TARGETS):
            raise ValueError(f"Invalid fixed cost or recall-point count: {platform.name}")
        for point in platform.measurements:
            if point.average_power_w <= 0 or point.elapsed_s <= 0:
                raise ValueError(f"Power and duration must be positive: {platform.name}")


def positive_crossings() -> list[Crossing]:
    """Return positive pairwise intersections of the 95%-recall cost curves."""
    crossings = []
    for index, first in enumerate(PLATFORMS):
        for second in PLATFORMS[index + 1:]:
            slope_difference = (
                first.at_95.electricity_eur_per_billion
                - second.at_95.electricity_eur_per_billion
            )
            if slope_difference == 0:
                continue
            query_billions = (second.fixed_cost_eur - first.fixed_cost_eur) / slope_difference
            if query_billions <= 0:
                continue
            cost = first.fixed_cost_eur + first.at_95.electricity_eur_per_billion * query_billions
            crossings.append(Crossing(first, second, query_billions, cost))
    return sorted(crossings, key=lambda crossing: crossing.query_billions)


def print_summary(crossings: list[Crossing]) -> None:
    print(
        "Integrated session-energy results for 100,000 queries "
        "at approximately matched Recall@10"
    )
    print(f"Electricity: EUR {ELECTRICITY_EUR_PER_KWH}/kWh; PUE={PUE}")
    print("Each row represents one complete monitored energy session.\n")
    print(
        f"{'Platform':<20} {'Recall':>7} {'Energy (J)':>13} "
        f"{'mJ/query':>11} {'Interval QPS':>12} {'EUR/1B':>10}"
    )
    for platform in PLATFORMS:
        for recall, point in zip(RECALL_TARGETS, platform.measurements):
            print(
                f"{platform.name:<20} {recall:>7} {point.energy_j:>13,.2f} "
                f"{point.energy_mj_per_query:>11,.3f} "
                f"{point.reported_interval_qps:>12,.2f} "
                f"{point.electricity_eur_per_billion:>10,.4f}"
            )
    print("\nPositive 95%-recall cost intersections:")
    for crossing in crossings:
        print(
            f"  {crossing.first.name} / {crossing.second.name}: "
            f"{crossing.query_billions:,.3f} billion queries, "
            f"EUR {crossing.cost_eur:,.2f}"
        )


def write_derived_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "platform", "recall_target", "elapsed_s", "integrated_energy_100k_j",
            "derived_average_power_w", "energy_mj_per_query",
            "reported_interval_qps", "electricity_eur_100k",
            "electricity_eur_per_billion", "assumed_fixed_cost_eur",
        ))
        for platform in PLATFORMS:
            for recall, point in zip(RECALL_TARGETS, platform.measurements):
                writer.writerow((
                    platform.name, recall, point.elapsed_s, point.energy_j,
                    point.average_power_w, point.energy_mj_per_query,
                    point.reported_interval_qps,
                    point.electricity_eur_per_100k,
                    point.electricity_eur_per_billion, platform.fixed_cost_eur,
                ))


def save_figure(figure, output_dir: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"{stem}.{extension}", dpi=300,
            bbox_inches="tight", facecolor="white",
        )


def style_axis(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", linestyle=":", color="#aaaaaa", alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_energy_bars(plt, output_dir: Path, *, per_query: bool) -> None:
    rows = list(csv.DictReader(ENERGY_INPUT.open(newline="", encoding="utf-8")))
    iqr = {(r["platform"], int(r["nominal_recall_pct"])): (float(r["energy_q1_j"]), float(r["energy_q3_j"]))
           for r in rows if r.get("energy_q1_j")}
    scale = 1000 / float(QUERIES_PER_RUN) if per_query else 1.0  # J per session -> mJ per query
    figure, axis = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    width = 0.19
    centers = list(range(len(RECALL_TARGETS)))
    for index, platform in enumerate(PLATFORMS):
        x = [center + (index - (len(PLATFORMS) - 1) / 2) * width for center in centers]
        values = [float(point.energy_j) * scale for point in platform.measurements]
        targets = [int(label.strip("~%")) for label in RECALL_TARGETS]
        low = [values[i] - iqr[(platform.name, t)][0] * scale if (platform.name, t) in iqr else 0
               for i, t in enumerate(targets)]
        high = [iqr[(platform.name, t)][1] * scale - values[i] if (platform.name, t) in iqr else 0
                for i, t in enumerate(targets)]
        axis.bar(x, values, width=width, label=platform.name, color=platform.color,
                 edgecolor="white", linewidth=0.6, zorder=3)
        axis.errorbar(x, values, yerr=[low, high], fmt="none", ecolor="#333333",
                      elinewidth=0.9, capsize=2.5, zorder=4)
        for xi, v, h in zip(x, values, high):
            axis.text(xi, v + h + 2.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
                      color="#333333")

    axis.set_xticks(centers)
    axis.set_xticklabels(RECALL_TARGETS)
    axis.set_xlabel("Nominal Recall@10 target", fontsize=10.5)
    if per_query:
        axis.set_ylabel("Workload energy per query (mJ)", fontsize=10.5)
        stem = "energy_per_query_recall"
    else:
        axis.set_ylabel("Workload energy per 100,000-query session (J)", fontsize=10.5)
        stem = "energy_comparison_recall"
    axis.set_ylim(bottom=0)
    axis.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=9.5)
    style_axis(axis)
    save_figure(figure, output_dir, stem)
    plt.close(figure)


def plot_session_tradeoff(plt, output_dir: Path) -> None:
    """Plot two metrics derived from each archived power/time pair."""
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    markers = ("o", "s", "^", "D")
    for platform in PLATFORMS:
        energies = [float(point.energy_mj_per_query) for point in platform.measurements]
        throughputs = [float(point.reported_interval_qps) for point in platform.measurements]
        for index, (energy, throughput) in enumerate(zip(energies, throughputs)):
            axis.scatter(
                energy, throughput, marker=markers[index], s=90,
                color=platform.color, edgecolor="white", linewidth=0.7, zorder=4,
            )

    platform_handles = [
        Line2D([0], [0], color=platform.color, marker="o", linestyle="None",
               label=platform.name)
        for platform in PLATFORMS
    ]
    target_handles = [
        Line2D([0], [0], marker=marker, linestyle="None", color="#444444",
               markersize=8, label=label)
        for marker, label in zip(markers, RECALL_TARGETS)
    ]
    platform_legend = axis.legend(
        handles=platform_handles, title="Platform", loc="upper right",
        frameon=False, fontsize=10.5, title_fontsize=10.5,
    )
    axis.add_artist(platform_legend)
    axis.legend(
        handles=target_handles, title="Recall target", loc="center right",
        frameon=False, fontsize=10.5, title_fontsize=10.5,
    )

    axis.set_xscale("log")
    axis.set_xlim(7, 300)
    axis.set_xticks((10, 20, 50, 100, 200))
    axis.set_xticklabels(("10", "20", "50", "100", "200"))
    axis.set_yscale("log")
    axis.set_ylim(2000, 60000)
    axis.set_yticks((2000, 5000, 10000, 20000, 50000))
    axis.set_yticklabels(("2,000", "5,000", "10,000", "20,000", "50,000"))
    from matplotlib.ticker import NullFormatter
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Workload energy per query (mJ; log scale)", fontsize=12)
    axis.set_ylabel("Workload throughput (queries/s; log scale)", fontsize=12)
    axis.tick_params(axis="both", labelsize=10.5)
    style_axis(axis)
    save_figure(figure, output_dir, "session_throughput_vs_energy")
    plt.close(figure)


def plot_cost_scaling(plt, output_dir: Path, crossings: list[Crossing]) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.3), constrained_layout=True)
    volumes = [MAX_QUERY_BILLIONS * index / 400 for index in range(401)]
    for platform in PLATFORMS:
        fixed = float(platform.fixed_cost_eur)
        slope = float(platform.at_95.electricity_eur_per_billion)
        costs = [fixed + slope * volume for volume in volumes]
        axis.plot(volumes, costs, linewidth=2.3, color=platform.color, label=platform.name)

    visible = [item for item in crossings if item.query_billions <= MAX_QUERY_BILLIONS]
    for index, item in enumerate(visible):
        x, y = float(item.query_billions), float(item.cost_eur)
        axis.plot(x, y, marker="o", markersize=5, color="#222222", zorder=5)
        label = (
            f"{SHORT_NAMES[item.first.name]}–{SHORT_NAMES[item.second.name]}\n"
            f"{item.query_billions:,.0f}B queries"
        )
        offset = (10, 15) if index == 0 else (-10, -15)
        axis.annotate(
            label, (x, y), xytext=offset, textcoords="offset points",
            ha="left" if index == 0 else "right",
            va="bottom" if index == 0 else "top",
            fontsize=8.5, color="#333333",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
        )

    axis.set_xlim(0, MAX_QUERY_BILLIONS)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Cumulative query volume (billions)")
    axis.set_ylabel("Illustrative hardware + electricity cost (EUR)")
    axis.set_title("Illustrative cost scaling at ~95% Recall@10", pad=12)
    axis.legend(ncol=2, loc="upper left", frameon=False, fontsize=9)
    style_axis(axis)
    save_figure(figure, output_dir, "cost_scaling_recall95")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated_energy_cost"))
    parser.add_argument("--overwrite", action="store_true", help="replace existing generated files")
    parser.add_argument("--check-only", action="store_true", help="print calculations without writing files")
    args = parser.parse_args()

    validate_inputs()
    crossings = positive_crossings()
    print_summary(crossings)
    if args.check_only:
        return

    targets = [args.output_dir / CSV_NAME]
    targets.extend(
        args.output_dir / f"{name}.{extension}"
        for name in FIGURE_NAMES for extension in ("png", "pdf")
    )
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Output already exists; choose another --output-dir or use --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    try:
        import matplotlib
    except ImportError as error:
        raise SystemExit("Matplotlib is required to draw the figures") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_derived_csv(args.output_dir / CSV_NAME)
    plot_energy_bars(plt, args.output_dir, per_query=False)
    plot_energy_bars(plt, args.output_dir, per_query=True)
    plot_session_tradeoff(plt, args.output_dir)
    plot_cost_scaling(plt, args.output_dir, crossings)
    print(f"\nSaved CSV, PNG, and PDF files in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
