#!/usr/bin/env python3
"""Draw the two Tenstorrent device-profiler figures of Results Section 5.3.1
with one shared style:

* data/trace_batches_512_8.csv     -> src/assets/fine_pipeline_horizontal.png
  last three 32-query batches at (N_list, N_probe) = (512, 8), in ms;
* data/trace_single_list_512_1.csv -> src/assets/fine_cluster_bubbles.png
  16 consecutive pages of one list on worker (1,2) at (512, 1), in µs.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "assets"

READER = "#7B5CE0"
QUERY = "#C7B9F5"
COMPUTE = "#F2C500"
WAIT = "#9AA3AD"
WRITER = "#D62728"
ROLE = {"NCRISC": "reader", "TRISC_0": "unpack", "TRISC_1": "math",
        "TRISC_2": "pack", "BRISC": "writer"}
RISCS = ["NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "BRISC"]
MIN_ZONE_US = 1.0  # sub-µs compute markers carry no work and are not drawn


def rows_for(cores, riscs):
    """y position per (core, risc); first core at the top."""
    labels, pos, y = [], {}, 0.0
    for core in reversed(cores):
        for risc in reversed(riscs):
            pos[(core, risc)] = y
            labels.append((y, f"Core({core[0]},{core[1]}) {risc} {ROLE[risc]}"))
            y += 1
        y += 0.6
    return pos, labels


def style_axis(ax, labels, shade_pos):
    ax.set_yticks([y for y, _ in labels], [t for _, t in labels], fontsize=7.5)
    for y in shade_pos:
        ax.axhspan(y - 0.45, y + 0.45, color="#F3F5F7", zorder=0)
    ax.grid(axis="x", color="#AAB0B6", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def legend(fig, ax, entries):
    handles = [Patch(facecolor=c, label=l) for l, c in entries]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=len(handles), frameon=False, fontsize=8)


def bar(ax, y, start, dur, color, label=None, text_color="#202020", fontsize=6.5):
    ax.barh(y, dur, left=start, height=0.7, color=color, edgecolor="white",
            linewidth=0.4, zorder=3)
    if label:
        ax.text(start + dur / 2, y, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, zorder=4, clip_on=True)


def batch_figure():
    rows = list(csv.DictReader((ROOT / "data" / "trace_batches_512_8.csv").open()))
    cores = [(1, 4), (1, 3), (1, 2), (1, 1)]
    pos, labels = rows_for(cores, RISCS)
    fig, ax = plt.subplots(figsize=(10.0, 7.2), constrained_layout=True)
    style_axis(ax, labels, [y for (c, r), y in pos.items() if r in ("NCRISC", "TRISC_1", "BRISC")])

    end_ms = max(float(r["end_us"]) for r in rows) / 1000
    counter = defaultdict(int)
    done = []
    for r in sorted(rows, key=lambda r: float(r["start_us"])):
        core, risc = (int(r["core_x"]), int(r["core_y"])), r["risc"]
        start, dur = float(r["start_us"]) / 1000, float(r["duration_us"]) / 1000
        batch = int(r["batch_id"]) + 1
        phase, y = r["phase"], pos[(core, risc)]
        if phase == "Send result" or phase == "Write result":
            ax.plot([start, start], [y - 0.35, y + 0.35], color=WRITER, lw=1.6, zorder=4)
            continue
        if float(r["duration_us"]) < MIN_ZONE_US and phase == "Cluster compute":
            continue
        if phase in ("Query read",):
            bar(ax, y, start, dur, QUERY)
        elif phase == "Query wait":
            bar(ax, y, start, dur, WAIT)
        elif phase in ("Cluster read", "Gather partials"):
            counter[(core, risc, batch)] += 1
            name = (f"B{batch} gather" if dur >= 0.6 else f"B{batch}") if phase == "Gather partials" \
                else f"B{batch} L{counter[(core, risc, batch)]}"
            bar(ax, y, start, dur, READER, name if dur >= 0.2 else None, "white")
        else:  # Cluster compute / Reduce partials
            counter[(core, risc, batch)] += 1
            name = (f"B{batch} reduce" if dur >= 0.6 else f"B{batch}") if phase == "Reduce partials" \
                else f"B{batch} L{counter[(core, risc, batch)]}"
            bar(ax, y, start, dur, COMPUTE, name if dur >= 0.2 else None)
            if phase == "Reduce partials" and risc == "TRISC_1":
                done.append((start + dur, batch))

    top = max(pos.values()) + 0.8
    for t, b in done:
        ax.axvline(t, color="#555555", linestyle=":", linewidth=0.9, zorder=2)
        ax.text(t, top, f"B{b} done", ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xlim(0, end_ms * 1.03)
    ax.set_ylim(-0.8, top + 0.9)
    ax.set_xlabel("Time since first displayed zone (ms)", fontsize=9)
    legend(fig, ax, [("Reader: query read", QUERY), ("Reader: list read / gather", READER),
                     ("Compute", COMPUTE), ("Query wait", WAIT), ("Writer: send result", WRITER)])
    fig.savefig(OUT / "fine_pipeline_horizontal.png", dpi=250)
    plt.close(fig)


def single_list_figure():
    rows = list(csv.DictReader((ROOT / "data" / "trace_single_list_512_1.csv").open()))
    core = (1, 2)
    pos, labels = rows_for([core], RISCS[:4])
    fig, ax = plt.subplots(figsize=(10.0, 3.6), constrained_layout=True)
    style_axis(ax, labels, [y for (c, r), y in pos.items() if r in ("NCRISC", "TRISC_1")])

    last_label_x = -1e9
    for r in sorted(rows, key=lambda r: float(r["start_us"])):
        risc, zone = r["risc"], r["zone"]
        start, dur = float(r["start_us"]), float(r["duration_us"])
        y = pos[(core, risc)]
        if zone.endswith("Reader.Query"):
            bar(ax, y, start, dur, QUERY, "Q", fontsize=7)
        elif zone.endswith("Reader.Fetch"):
            name = f"P{r['block']}"
            if dur >= 6:
                bar(ax, y, start, dur, READER, name, "white", 7)
            else:
                bar(ax, y, start, dur, READER)
                x = max(start + dur / 2, last_label_x + 3.5)
                last_label_x = x
                ax.text(x, y + 0.4, name, ha="center", va="bottom",
                        fontsize=6.5, rotation=90, color="#202020")
        elif zone.endswith("QueryWait"):
            if dur >= MIN_ZONE_US:
                bar(ax, y, start, dur, WAIT, "wait" if dur >= 5 else None, "white", 6.5)
        else:
            bar(ax, y, start, dur, COMPUTE, f"P{r['block']}" if dur >= 3 else None, fontsize=6.5)

    end = max(float(r["end_us"]) for r in rows)
    ax.set_xlim(0, end * 1.02)
    ax.set_ylim(-0.6, max(pos.values()) + 1.1)
    ax.set_xlabel("Time since first displayed zone (µs)", fontsize=9)
    legend(fig, ax, [("Reader: query read", QUERY), ("Reader: page fetch (vector + ID)", READER),
                     ("Compute", COMPUTE), ("Query wait", WAIT)])
    fig.savefig(OUT / "fine_cluster_bubbles.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    batch_figure()
    single_list_figure()
    print("wrote fine_pipeline_horizontal.png and fine_cluster_bubbles.png")
