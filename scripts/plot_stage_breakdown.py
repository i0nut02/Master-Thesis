#!/usr/bin/env python3
"""Plot the Tenstorrent stage decomposition (Figures 5.2 and 5.3) from
data/stage_timing_nlist512.csv (per-stage medians over ten repetitions, in µs)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "stage_timing_nlist512.csv"
OUT = ROOT / "src" / "assets"

STAGES = [  # (csv column, legend label, colour), bottom to top
    ("h2d_median_us", r"$T_{\mathrm{H2D}}$", "#4E79A7"),
    ("cpu_coarse_config_median_us", r"$T_{\mathrm{CPU,coarse}}$", "#F28E2B"),
    ("tt_coarse_search_median_us", r"$T_{\mathrm{TT,coarse}}$", "#59A14F"),
    ("cpu_fine_prep_median_us", r"$T_{\mathrm{CPU,fine}}$", "#17BECF"),
    ("tt_fine_search_median_us", r"$T_{\mathrm{TT,fine}}$", "#B07AA1"),
    ("cpu_output_median_us", r"$T_{\mathrm{CPU,output}}$", "#E15759"),
]

rows = list(csv.DictReader(DATA.open()))
nprobe = [r["nprobe"] for r in rows]
ms = {col: [float(r[col]) / 1000 for r in rows] for col, _, _ in STAGES}


def plot(stages, name, total_label, inside_min_ms, ymax, label_small=True):
    fig, ax = plt.subplots(figsize=(5.0, 4.6), constrained_layout=True)
    x = range(len(rows))
    bottom = [0.0] * len(rows)
    handles = []
    for col, label, colour in stages:
        values = ms[col]
        h = ax.bar(x, values, 0.6, bottom=bottom, color=colour, edgecolor="white",
                   linewidth=0.6, label=label, zorder=3)
        handles.append(h)
        for i, v in enumerate(values):
            if v >= inside_min_ms:
                ax.text(i, bottom[i] + v / 2, f"{v:.2f}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
            elif label_small:
                ax.text(i + 0.32, bottom[i] + v / 2, f"{v:.2f}", ha="left",
                        va="center", fontsize=7.5, color="#333333")
        bottom = [b + v for b, v in zip(bottom, values)]
    for i, total in enumerate(bottom):
        ax.text(i, total + ymax * 0.012, f"{total_label}{total:.2f} ms", ha="center",
                va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(list(x), nprobe)
    ax.set_xlim(-0.55, len(rows) - 0.2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$N_{\mathrm{probe}}$", fontsize=11)
    ax.set_ylabel("Median stage time (ms)", fontsize=11)
    ax.grid(axis="y", linestyle=":", color="#bbbbbb", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles, [s[1] for s in stages], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9.5,
              columnspacing=1.0, handlelength=1.4)
    fig.savefig(OUT / f"{name}.png", dpi=250)
    plt.close(fig)


plot(STAGES, "memcopy_cost_nlist512", "", 30, 2000, label_small=False)
plot([s for s in STAGES if s[0] != "tt_fine_search_median_us"],
     "memcopy_cost_nlist512_without_tt_fine", "", 2.5, 33)
print("wrote", OUT / "memcopy_cost_nlist512.png", "and", OUT / "memcopy_cost_nlist512_without_tt_fine.png")
