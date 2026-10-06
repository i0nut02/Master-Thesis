#!/usr/bin/env python3
"""Reduce repeated ANN benchmark rows to one row per configuration.

The output keeps the original CSV schema. For every unique configuration, it
reports the median QPS across the repeated runs, derives a consistent average
latency from that median, reports the median Recall@K, and appends the QPS
relative standard deviation (sample standard deviation over mean, in percent)
as QPS_RSD_pct. No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median, stdev


FIELDS = (
    "RunID",
    "Type",
    "Dataset",
    "N",
    "D",
    "NList",
    "NProbe",
    "K",
    "BatchSize",
    "AvgLatency_us",
    "QPS",
    "Recall@K",
)

OUTPUT_FIELDS = FIELDS + ("QPS_RSD_pct",)

CONFIG_FIELDS = (
    "Type",
    "Dataset",
    "N",
    "D",
    "NList",
    "NProbe",
    "K",
    "BatchSize",
)


def decimal_value(row: dict[str, str], field: str, line_number: int) -> Decimal:
    try:
        return Decimal(row[field])
    except (InvalidOperation, KeyError) as error:
        raise ValueError(
            f"line {line_number}: invalid {field} value {row.get(field)!r}"
        ) from error


def sort_key(item: tuple[tuple[str, ...], list[dict[str, str]]]) -> tuple:
    key, _ = item
    values = dict(zip(CONFIG_FIELDS, key))
    return (
        values["Type"],
        values["Dataset"],
        int(values["NList"]),
        int(values["NProbe"]),
        int(values["K"]),
        int(values["BatchSize"]),
    )


def reduce_csv(
    input_path: Path,
    output_path: Path,
    expected_runs: int,
    type_label: str | None,
) -> None:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)

    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            decimal_value(row, "QPS", line_number)
            decimal_value(row, "Recall@K", line_number)
            if type_label:
                row["Type"] = type_label
            key = tuple(row[field].strip() for field in CONFIG_FIELDS)
            groups[key].append(row)

    if not groups:
        raise ValueError("the input CSV contains no data rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for key, rows in sorted(groups.items(), key=sort_key):
            qps_values = [Decimal(row["QPS"]) for row in rows]
            qps = median(qps_values)
            rsd = (
                100 * stdev(qps_values) / mean(qps_values)
                if len(qps_values) > 1
                else Decimal(0)
            )
            recall = median(Decimal(row["Recall@K"]) for row in rows)
            latency_us = Decimal("1000000") / qps
            output = dict(zip(CONFIG_FIELDS, key))
            output.update(
                {
                    "RunID": f"median_of_{len(rows)}",
                    "AvgLatency_us": f"{latency_us:.6f}",
                    "QPS": f"{qps:.6f}",
                    "Recall@K": f"{recall:.6f}",
                    "QPS_RSD_pct": f"{rsd:.3f}",
                }
            )
            writer.writerow(output)

            if expected_runs and len(rows) != expected_runs:
                description = ", ".join(
                    f"{field}={value}" for field, value in zip(CONFIG_FIELDS, key)
                )
                print(
                    f"warning: {description} has {len(rows)} rows; "
                    f"expected {expected_runs}",
                    file=sys.stderr,
                )

    print(f"Read {sum(map(len, groups.values()))} rows")
    print(f"Wrote {len(groups)} unique configurations to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one median row per unique benchmark configuration."
    )
    parser.add_argument("input", type=Path, help="input benchmark CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path (default: INPUT_unique.csv)",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=10,
        help="warn when a configuration has a different run count (default: 10)",
    )
    parser.add_argument(
        "--type-label",
        help=(
            "replace Type before grouping, for example cpu-batched-56; useful "
            "when separate files use the generic Type cpu-batched"
        ),
    )
    args = parser.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_unique.csv")
    if output.resolve() == args.input.resolve():
        parser.error("output must differ from input")

    try:
        reduce_csv(args.input, output, args.expected_runs, args.type_label)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
