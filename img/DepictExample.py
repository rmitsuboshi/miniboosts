#!/usr/bin/env python3
"""Plot MiniBoosts Logger CSVs; requires matplotlib.

Time is cumulative boosting time in milliseconds, excluding evaluation and I/O.
Compare runs only when their data, weak learner, and evaluation settings match.
Soft-margin plots additionally require matching nu; equal tolerances alone do not
make stopping rules equivalent. Metadata is recorded by the example runner.
"""

import argparse
import csv
import math
from pathlib import Path
import sys


ALGORITHMS = {
    "adaboost": ("AdaBoost", "#2864A0"),
    "adaboostv": ("AdaBoostV", "#C47B16"),
    "madaboost": ("MadaBoost", "#8064A2"),
    "smoothboost": ("SmoothBoost", "#229487"),
    "totalboost": ("TotalBoost", "#454D59"),
    "softboost": ("SoftBoost", "#BD668C"),
    "mlpboost": ("MLPBoost", "#D65B40"),
    "lpboost": ("LPBoost", "#46A5C1"),
    "erlpboost": ("ERLPBoost", "#43834B"),
    "cerlpboost": ("CERLPBoost", "#978332"),
    "graphsepboost": ("GraphSepBoost", "#A55265"),
}
SOFT_MARGIN = {"softboost", "mlpboost", "lpboost", "erlpboost", "cerlpboost"}
COLUMNS = ("ObjectiveValue", "TrainLoss", "TestLoss", "Time")


def read_csv(path):
    """Reject corrupt/nonfinite measurements instead of drawing plausible curves."""
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {', '.join(COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            try:
                values = {key: float(row[key]) for key in COLUMNS}
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: invalid numeric field") from error
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{path}:{line}: non-finite measurement")
            if values["Time"] < 0 or (rows and values["Time"] < rows[-1]["Time"]):
                raise ValueError(f"{path}:{line}: time must be nonnegative and cumulative")
            if any(not 0 <= values[key] <= 1 for key in ("TrainLoss", "TestLoss")):
                raise ValueError(f"{path}:{line}: 0/1 loss must be between zero and one")
            rows.append(values)
    if not rows:
        raise ValueError(f"{path}: no measurements")
    return rows


def style_axis(axis, ylabel, time=False):
    from matplotlib.ticker import PercentFormatter
    axis.set_facecolor("#FAFBFD")
    axis.spines[["top", "right"]].set_visible(False)
    for spine in ("bottom", "left"):
        axis.spines[spine].set_color("#CCD3DD")
    axis.grid(axis="y", color="#DDE3EB", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#485568", labelsize=10)
    axis.set_ylabel(ylabel, color="#334155")
    axis.set_xlabel("Boosting iteration" if not time else "Boosting time (s)")
    if time:
        # Preserve zero-ms samples while expanding the crowded early-time region.
        axis.set_xscale("symlog", linthresh=0.01, linscale=1)
    if "loss" in ylabel.lower():
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.margins(x=0.025, y=0.08)


def plot(plt, runs, column, ylabel, destination, title, every):
    figure, axes = plt.subplots(2, 1, figsize=(9, 11))
    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.18, top=0.90, hspace=0.35)
    for algorithm, rows in runs.items():
        label, color = ALGORITHMS[algorithm]
        indices = list(range(0, len(rows), every))
        if indices[-1] != len(rows) - 1:
            indices.append(len(rows) - 1)
        values = [rows[index][column] for index in indices]
        for axis, x in zip(axes, ([index + 1 for index in indices],
                                 [rows[index]["Time"] / 1000 for index in indices])):
            axis.plot(x, values, label=label, color=color, linewidth=1.5,
                      linestyle="-", alpha=0.85,
                      marker="o" if len(indices) == 1 else None, markersize=4)
    for axis, heading, time in zip(axes, ("By iteration", "By elapsed time"), (False, True)):
        style_axis(axis, ylabel, time)
        axis.set_title(heading, loc="left", fontsize=12, fontweight="bold", pad=12)
    figure.suptitle(title or ylabel, x=0.11, ha="left", fontsize=19, fontweight="bold", color="#243247")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.53, 0.035),
                  ncol=min(3, len(runs)), frameon=False, fontsize=10, handlelength=3)
    figure.text(0.98, 0.015, "Time axis: linear below 0.01 s, logarithmic above. No smoothing.",
                ha="right", fontsize=9, color="#65748B")
    figure.savefig(destination, dpi=200, facecolor="white")
    plt.close(figure)
    print(destination)

    # Individual scales expose small changes; overview panels retain common scales.
    ncols = 1
    nrows = math.ceil(len(runs) / ncols)
    figure, grid = plt.subplots(nrows, ncols, figsize=(9, 3 * nrows + 0.7),
                               squeeze=False, constrained_layout=True)
    for axis, (algorithm, rows) in zip(grid.flat, runs.items()):
        label, color = ALGORITHMS[algorithm]
        indices = list(range(0, len(rows), every))
        if indices[-1] != len(rows) - 1:
            indices.append(len(rows) - 1)
        axis.plot([i + 1 for i in indices], [rows[i][column] for i in indices],
                  color=color, linewidth=1.3, linestyle="-", marker="o" if len(indices) == 1 else None)
        style_axis(axis, ylabel)
        axis.set_title(label, loc="left", color=color, fontsize=12, fontweight="bold")
    for axis in list(grid.flat)[len(runs):]:
        axis.set_visible(False)
    figure.suptitle(f"{title or ylabel} — individual algorithms (independent scales)", fontsize=12, color="#243247")
    detail = destination.with_name(destination.stem + "-by-algorithm.png")
    figure.savefig(detail, dpi=200, facecolor="white")
    plt.close(figure)
    print(detail)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=directory / "csv",
                        help="directory containing algorithm-named CSVs (default: img/csv)")
    parser.add_argument("--output-dir", type=Path, default=directory / "plots",
                        help="destination for PNG files (default: img/plots)")
    parser.add_argument("--title", default="", help="optional dataset/experiment title")
    parser.add_argument("--every", type=int, default=1,
                        help="draw every Nth measurement and the final point (default: 1)")
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be positive")
    try:
        runs = {name: read_csv(path) for name in ALGORITHMS
                if (path := args.input_dir / f"{name}.csv").is_file()}
        if not runs:
            parser.error(f"no supported algorithm CSVs in {args.input_dir}")
        missing = set(ALGORITHMS) - runs.keys()
        if missing:
            print(f"Skipping missing CSVs: {', '.join(sorted(missing))}", file=sys.stderr)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot(plt, runs, "TrainLoss", "Training 0/1 loss",
             args.output_dir / "training-loss.png", args.title, args.every)
        plot(plt, runs, "TestLoss", "Test 0/1 loss",
             args.output_dir / "test-loss.png", args.title, args.every)
        margins = {name: rows for name, rows in runs.items() if name in SOFT_MARGIN}
        if margins:
            plot(plt, margins, "ObjectiveValue", "Soft-margin objective",
                 args.output_dir / "soft-margin.png", args.title, args.every)
        else:
            print("Skipping soft-margin plot: no supported soft-margin runs", file=sys.stderr)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
