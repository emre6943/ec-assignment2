"""Plot what happened in one run.

    uv run --project ../ariel python plot.py results/best/seed0

Writes `run.png` into the run's folder with four panels, all against the
number of evaluations spent:

1. The whole population: best and mean fitness per generation, and the
   shortest distance to the target anyone reached.
2. Each island's best fitness: do the islands move together or apart?
3. Posture: the population's mean share of the run spent with the core on
   the ground, and upside down. A spider that learns to stand drives these down.
4. Each island's genotype spread: how different its networks still are.
   Migration makes islands more alike, so this is where its effect shows.

Dotted vertical lines mark migration events. The dashed line at 2.0 m is where
every robot starts; anything below it moved towards the target.
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

# Categorical colours in fixed order (checked for colour-blind separation;
# two of them are low-contrast on white, so every panel also has a legend).
ISLAND_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
BEST_COLOUR = "#2a78d6"
MEAN_COLOUR = "#52514e"
DISTANCE_COLOUR = "#eb6834"
GROUND_COLOUR = "#2a78d6"
FLIPPED_COLOUR = "#eb6834"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
START_DISTANCE = 2.0


def style_axis(axis: plt.Axes, ylabel: str) -> None:
    """Recessive grid and axes; only the data carries colour."""
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_ylabel(ylabel, color=TEXT, fontsize=10)


def plot_run(run: Path) -> Path:
    """Draw the four panels for one run folder; return the image path."""
    log = pd.read_csv(run / "log.csv", dtype={"island": str})
    config = json.loads((run / "config.json").read_text())["ea"]
    everyone = log[log["island"] == "all"]
    islands = sorted(i for i in log["island"].unique() if i != "all")
    migrations = everyone.loc[everyone["immigrants"] > 0, "evaluations"]

    figure, axes = plt.subplots(4, 1, figsize=(8, 11.5), sharex=True)
    figure.patch.set_facecolor(SURFACE)
    condition = (
        "random search"
        if config["algorithm"] == "random_search"
        else f"migration policy: {config['policy']}"
    )
    figure.suptitle(
        f"{condition}, seed {config['seed']}", color=TEXT, fontsize=12, x=0.1, ha="left"
    )

    top, middle, posture, bottom = axes
    x = everyone["evaluations"]
    top.plot(x, everyone["best"], color=BEST_COLOUR, linewidth=2, label="best")
    top.plot(x, everyone["mean"], color=MEAN_COLOUR, linewidth=1.5, label="mean")
    top.plot(
        x,
        everyone["best_distance"],
        color=DISTANCE_COLOUR,
        linewidth=1.5,
        label="shortest distance",
    )
    top.axhline(START_DISTANCE, color=MUTED, linewidth=1, linestyle="--")
    top.legend(frameon=False, fontsize=9, loc="lower left")
    style_axis(top, "fitness (m, lower is better)\nwhole population")

    island_lines = []
    for island, colour in zip(islands, ISLAND_COLOURS, strict=False):
        rows = log[log["island"] == island]
        (line,) = middle.plot(
            rows["evaluations"], rows["best"], color=colour, linewidth=1.5
        )
        island_lines.append(line)
        bottom.plot(rows["evaluations"], rows["spread"], color=colour, linewidth=1.5)
    middle.axhline(START_DISTANCE, color=MUTED, linewidth=1, linestyle="--")
    middle.legend(
        island_lines,
        [f"island {i}" for i in islands],
        frameon=False,
        fontsize=9,
        loc="lower left",
        ncols=len(islands),
    )
    style_axis(middle, "fitness (m)\nbest per island")
    posture.plot(
        x,
        everyone["ground_contact"],
        color=GROUND_COLOUR,
        linewidth=1.5,
        label="core on the ground",
    )
    posture.plot(
        x,
        everyone["upside_down"],
        color=FLIPPED_COLOUR,
        linewidth=1.5,
        label="upside down",
    )
    posture.set_ylim(0, 1)
    posture.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    posture.legend(frameon=False, fontsize=9, loc="upper right")
    style_axis(posture, "share of the run\n(population mean)")
    style_axis(bottom, "genotype spread\nper island (colours as above)")
    bottom.set_xlabel("evaluations", color=TEXT, fontsize=10)

    for axis in axes:
        for evaluations in migrations:
            axis.axvline(evaluations, color=GRID, linewidth=1, linestyle=":")

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path = run / "run.png"
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def main() -> None:
    """Plot every run folder given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="run folders (…/seedN)")
    for run in parser.parse_args().runs:
        print(plot_run(run))


if __name__ == "__main__":
    main()
