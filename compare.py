"""Compare runs on the plain yardstick: how close did anyone get to the target?

    uv run --project ../ariel python compare.py results/long/seed0 results/improved/seed0

Writes `comparison.png` (next to the first run, or where `--out` says) with two
panels:

1. The shortest distance to the target in each generation, against the number
   of evaluations spent - who learns more per evaluation.
2. The same, against wall-clock minutes - early stopping makes evaluations
   cheaper, so a method can lose on (1) and still win on (2).

The distance is the same measurement in every run, whatever fitness steered
the selection (curriculum or not), which is what makes the runs comparable.
"""

# Standard library
import argparse
import json
from pathlib import Path

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")  # render to a file; no window
import matplotlib.pyplot as plt
import pandas as pd

# Same fixed-order categorical colours as plot.py.
RUN_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
START_DISTANCE = 2.0


def run_label(run: Path) -> str:
    """A short name for a run: its folder plus the options that differ from default."""
    config = json.loads((run / "config.json").read_text())["ea"]
    extras = [
        name
        for name in ("curriculum", "early_stop")
        if config.get(name)  # older runs do not have these keys
    ]
    label = f"{run.parent.name}/{run.name}"
    return f"{label} ({', '.join(extras)})" if extras else label


def style_axis(axis: plt.Axes, xlabel: str) -> None:
    """Recessive grid and axes; only the data carries colour."""
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_xlabel(xlabel, color=TEXT, fontsize=10)
    axis.set_ylabel("shortest distance\nto the target (m)", color=TEXT, fontsize=10)
    axis.axhline(START_DISTANCE, color=MUTED, linewidth=1, linestyle="--")


def main() -> None:
    """Overlay the given runs and print where each ended."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="run folders (…/seedN)")
    parser.add_argument("--out", type=Path, default=None, help="image path")
    args = parser.parse_args()
    if len(args.runs) > len(RUN_COLOURS):
        parser.error(f"at most {len(RUN_COLOURS)} runs per comparison")

    figure, (by_evaluations, by_time) = plt.subplots(2, 1, figsize=(8, 7.5))
    figure.patch.set_facecolor(SURFACE)
    for run, colour in zip(args.runs, RUN_COLOURS, strict=False):
        log = pd.read_csv(run / "log.csv", dtype={"island": str})
        everyone = log[log["island"] == "all"]
        # A run's best-so-far: the shortest distance reached up to each point.
        best_so_far = everyone["best_distance"].cummin()
        label = run_label(run)
        by_evaluations.plot(
            everyone["evaluations"], best_so_far, color=colour, linewidth=2, label=label
        )
        by_time.plot(everyone["seconds"] / 60, best_so_far, color=colour, linewidth=2)
        print(
            f"{label}: best distance {best_so_far.iloc[-1]:.3f} m after "
            f"{everyone['evaluations'].iloc[-1]} evaluations, "
            f"{everyone['seconds'].iloc[-1] / 60:.1f} min"
        )

    style_axis(by_evaluations, "evaluations")
    style_axis(by_time, "wall-clock minutes")
    by_evaluations.legend(frameon=False, fontsize=9, loc="upper right")
    figure.suptitle(
        "Best distance reached so far", color=TEXT, fontsize=12, x=0.1, ha="left"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out or args.runs[0].parent / "comparison.png"
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
