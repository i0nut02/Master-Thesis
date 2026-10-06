#!/usr/bin/env python3
"""Plot median QPS against Recall@10 from the selected benchmark grid."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "qps_recall_medians.csv"
DEFAULT_OUTPUT = ROOT / "src" / "assets" / "qps_vs_recall"

PLATFORMS = {
    "cpu-batched-56": ("CPU (56 cores)", "#4E79A7"),
    "cpu-batched-112": ("CPU (112 cores)", "#F28E2B"),
    "gpu-batched": ("GPU (A100) + CPU", "#59A14F"),
    "tt-balanced-sort": ("TT + CPU", "#B07AA1"),
}

NLIST_STYLES = {
    512: ("o", "-"),
    1024: ("s", "--"),
    2048: ("^", ":"),
}


def load_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            platform = row["Type"]
            nlist = int(row["NList"])
            nprobe = int(row["NProbe"])
            if platform not in PLATFORMS:
                raise ValueError(f"unknown platform: {platform}")
            if nlist not in NLIST_STYLES:
                raise ValueError(f"unsupported NList: {nlist}")
            if nprobe == 12 and not (
                platform == "tt-balanced-sort" and nlist == 2048
            ):
                raise ValueError(
                    "only the confirmed TT (NList, NProbe)=(2048, 12) "
                    "energy-matching point may use NProbe=12"
                )
            rows.append(
                {
                    "platform": platform,
                    "nlist": nlist,
                    "nprobe": nprobe,
                    "qps": float(row["QPS"]),
                    "recall": 100.0 * float(row["Recall@K"]),
                }
            )
    return rows


def plot(rows: list[dict[str, float | int | str]], output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["platform"], row["nlist"])].append(row)

    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for platform in PLATFORMS:
        for nlist in NLIST_STYLES:
            points = sorted(grouped[(platform, nlist)], key=lambda row: row["nprobe"])
            if not points:
                continue
            _, color = PLATFORMS[platform]
            marker, line_style = NLIST_STYLES[nlist]
            axis.plot(
                [point["recall"] for point in points],
                [point["qps"] for point in points],
                color=color,
                linestyle=line_style,
                linewidth=1.7,
                marker=marker,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                alpha=0.95,
            )

    platform_handles = [
        Line2D([0], [0], color=color, linewidth=2.2, label=label)
        for label, color in PLATFORMS.values()
    ]
    nlist_handles = [
        Line2D(
            [0], [0], color="#444444", linestyle=line_style, marker=marker,
            linewidth=1.5, markersize=5.5, label=rf"$N_{{\mathrm{{list}}}}={nlist}$",
        )
        for nlist, (marker, line_style) in NLIST_STYLES.items()
    ]
    platform_legend = axis.legend(
        handles=platform_handles,
        title="Platform",
        loc="upper right",
        frameon=False,
        fontsize=10,
        title_fontsize=10,
    )
    axis.add_artist(platform_legend)
    axis.legend(
        handles=nlist_handles,
        title="Index size",
        loc="lower left",
        frameon=False,
        fontsize=9.5,
        title_fontsize=10,
    )

    axis.set_yscale("log")
    axis.set_xlim(34, 97)
    axis.set_ylim(2500, 5_000_000)
    axis.set_xlabel("Recall@10 (%)", fontsize=12)
    axis.set_ylabel("Median throughput (queries/s; log scale)", fontsize=12)
    axis.set_title("Throughput–recall trade-off", pad=10, fontsize=13)
    axis.tick_params(axis="both", labelsize=10.5)
    axis.grid(axis="both", which="major", linestyle=":", color="#b8b8b8")
    axis.grid(axis="y", which="minor", linestyle=":", color="#dddddd", alpha=0.7)
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = load_rows(args.input)
    plot(rows, args.output_stem)
    print(f"Plotted {len(rows)} selected median operating points")
    print(f"Saved {args.output_stem}.png and {args.output_stem}.pdf")


if __name__ == "__main__":
    main()
