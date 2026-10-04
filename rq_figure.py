"""Experiment 14's results in one figure (D3); not a figure of the paper.

`paper_figures.py` imports its condition names and labels from here.

    uv run --project ../ariel python rq_figure.py                    # fitness
    uv run --project ../ariel python rq_figure.py --metric distance  # metres

Four panels from the runs in `results/olympic/`, for either measure:

    A  convergence: the mean of the best so far against evaluations
    B  speed: evaluations each run needed to get below a threshold
    C  the value after 12,000 evaluations per seed, lines joining the same seed
       (the same arena)
    D  how likely one condition beats another, from the Bayesian paired t-test
       of `probabilities.py`

`fitness` is the full fitness the EA minimises (distance plus the gait and
speed terms, D15-D20). `distance` is only how close to the target any brain
ended a walk: the shortest distance so far, in metres. Lower is better for
both. Written to `results/olympic/rq_figure.png` / `rq_figure_distance.png`.
"""

# Standard library
import argparse
from pathlib import Path

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")  # render to a file; no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Local libraries
from analyze import (
    CONDITION_COLOURS,
    GRID_POINTS,
    MUTED,
    SURFACE,
    TEXT,
    best_so_far,
    on_grid,
    style_axis,
)
from probabilities import DEFAULT_ROPE, mean_interval, pair_rows
from simulate import TARGET_RADIUS

OLYMPIC = Path(__file__).parent / "results" / "olympic"
CONDITIONS = ("best", "worst", "random", "none", "standard", "random_search")
EAS = CONDITIONS[:-1]  # random search is far behind; it would squash panels B-D
# Diverging: orange (column better) - neutral grey at 50% - blue (row better).
DIVERGING = LinearSegmentedColormap.from_list(
    "better", ["#c4441c", "#efede8", "#1f5fa8"]
)

# Per measure: the curve column, the speed threshold, and the wording.
MEASURES = {
    "fitness": {
        "column": "fitness",
        "threshold": 1.6,
        "unit": "fitness",
        "curve": "best fitness so far (mean of 5 seeds)",
        "final": "best fitness after 12,000 evaluations",
        "matrix": "D  P(row converges faster than column), AUC of the fitness",
        "out": "rq_figure.png",
    },
    "distance": {
        "column": "distance",
        "threshold": 0.5,
        "unit": "m",
        "curve": "closest any brain got to the target so far, m (mean of 5 seeds)",
        "final": "closest to the target after 12,000 evaluations, m",
        "matrix": "D  P(row gets closer than column), after 12,000 evaluations",
        "out": "rq_figure_distance.png",
    },
}


def label(condition: str) -> str:
    """A condition's name as the figure shows it."""
    return {
        "best": "migrate best",
        "worst": "migrate worst",
        "random": "migrate random",
        "none": "no migration",
        "standard": "standard EA",
        "random_search": "random search",
    }[condition]


def first_below(
    curve: pd.DataFrame, column: str, threshold: float, budget: float
) -> float:
    """Evaluations at which the best so far first drops below the threshold.

    Only the common budget counts (as in `analyze.run_metrics`): a run that gets
    there later counts as never, like one that does not get there at all (NaN).
    """
    within = curve[curve["evaluations"] <= budget]
    below = within.loc[within[column] < threshold, "evaluations"]
    return float(below.iloc[0]) if len(below) else np.nan


def main() -> None:
    """Draw the four panels for the chosen measure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metric", choices=tuple(MEASURES), default="fitness")
    measure = MEASURES[parser.parse_args().metric]
    column, threshold = measure["column"], measure["threshold"]

    curves = {
        condition: {
            run.name: best_so_far(run)
            for run in sorted((OLYMPIC / condition).glob("seed*"))
            if (run / "log.csv").exists()
        }
        for condition in CONDITIONS
    }
    missing = [condition for condition, runs in curves.items() if not runs]
    if missing:
        parser.error(f"no runs in {', '.join(str(OLYMPIC / c) for c in missing)}")
    budget = min(
        c["evaluations"].iloc[-1] for runs in curves.values() for c in runs.values()
    )
    grid = np.linspace(0, budget, GRID_POINTS)
    end = grid[-1:]
    table = pd.DataFrame(
        [
            {
                "condition": condition,
                "seed": seed,
                "final": float(on_grid(curve, column, end)[0]),
                "auc": float(on_grid(curve, column, grid).mean()),
            }
            for condition, runs in curves.items()
            for seed, curve in runs.items()
        ]
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    figure.patch.set_facecolor(SURFACE)
    (curve_axis, speed_axis), (final_axis, matrix_axis) = axes

    # A: convergence
    for condition in CONDITIONS:
        values = np.array(
            [on_grid(c, column, grid) for c in curves[condition].values()]
        )
        curve_axis.plot(
            grid,
            values.mean(axis=0),
            color=CONDITION_COLOURS[condition],
            linewidth=2,
            linestyle="--" if condition == "random_search" else "-",
            label=label(condition),
        )
    curve_axis.axhline(threshold, color=MUTED, linewidth=1, linestyle=":")
    curve_axis.text(
        budget,
        threshold * 1.02,
        f"{threshold:g} {measure['unit']}",
        color=MUTED,
        ha="right",
        fontsize=8,
    )
    if column == "distance":
        curve_axis.axhline(TARGET_RADIUS, color=TEXT, linewidth=1, linestyle="--")
        curve_axis.text(
            budget,
            TARGET_RADIUS * 1.15,
            f"target reached ({TARGET_RADIUS:g} m)",
            color=TEXT,
            ha="right",
            fontsize=8,
        )
    style_axis(curve_axis)
    curve_axis.set_xlabel("evaluations", color=TEXT)
    curve_axis.set_ylabel(measure["curve"], color=TEXT)
    curve_axis.set_title("A  Convergence", loc="left", color=TEXT, fontsize=11)
    curve_axis.legend(frameon=False, fontsize=8)

    # B: evaluations to the threshold
    rng = np.random.default_rng(0)
    for x, condition in enumerate(EAS):
        reached = []
        for curve in curves[condition].values():
            evals = first_below(curve, column, threshold, budget)
            jitter = x + rng.uniform(-0.12, 0.12)
            colour = CONDITION_COLOURS[condition]
            if np.isnan(evals):
                speed_axis.scatter(jitter, budget, marker="^", s=40, color=colour)
            else:
                reached.append(evals)
                speed_axis.scatter(jitter, evals, s=40, color=colour, alpha=0.7)
        if len(reached) > 1:
            mean, _, low, high = mean_interval(np.array(reached))
            speed_axis.errorbar(
                x + 0.3,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                color=TEXT,
                markersize=6,
                capsize=3,
                linewidth=1.2,
            )
    speed_axis.set_xticks(range(len(EAS)), [label(c) for c in EAS], fontsize=8)
    speed_axis.set_ylim(bottom=0)  # an interval over few runs can reach below 0
    style_axis(speed_axis)
    speed_axis.set_ylabel(
        f"evaluations to get below {threshold:g} {measure['unit']}", color=TEXT
    )
    speed_axis.set_title(
        "B  Speed (dots: seeds; black: mean of those that got there, 95% interval;"
        " ▲ never)",
        loc="left",
        color=TEXT,
        fontsize=11,
    )

    # C: the value at the budget per seed, paired
    wide = table.pivot(index="seed", columns="condition", values="final")[list(EAS)]
    for _, row in wide.iterrows():
        final_axis.plot(
            range(len(EAS)), row.to_numpy(), color=MUTED, alpha=0.35, linewidth=1
        )
    for x, condition in enumerate(EAS):
        values = wide[condition].dropna().to_numpy()
        mean, _, low, high = mean_interval(values)
        final_axis.scatter(
            [x] * len(values),
            values,
            s=30,
            color=CONDITION_COLOURS[condition],
            zorder=3,
        )
        final_axis.errorbar(
            x + 0.15,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color=TEXT,
            markersize=6,
            capsize=3,
            linewidth=1.2,
            zorder=4,
        )
    if column == "distance":
        final_axis.axhline(TARGET_RADIUS, color=TEXT, linewidth=1, linestyle="--")
    final_axis.set_xticks(range(len(EAS)), [label(c) for c in EAS], fontsize=8)
    style_axis(final_axis)
    final_axis.set_ylabel(measure["final"], color=TEXT)
    final_axis.set_title(
        "C  After 12,000 evaluations (grey lines join the same seed and arena)",
        loc="left",
        color=TEXT,
        fontsize=11,
    )

    # D: P(row better than column): convergence (AUC) for fitness, the final
    # closest distance for distance
    tested = "auc" if column == "fitness" else "final"
    eas_only = table[table["condition"].isin(EAS)]
    pairs = pd.DataFrame(pair_rows(eas_only, tested, DEFAULT_ROPE))
    matrix = np.full((len(EAS), len(EAS)), np.nan)
    for i, a in enumerate(EAS):
        for j, b in enumerate(EAS):
            match = pairs[(pairs["a"] == a) & (pairs["b"] == b)]
            if not match.empty:
                matrix[i, j] = match["p_a_better_at_all"].iloc[0]
    matrix_axis.imshow(matrix, cmap=DIVERGING, vmin=0, vmax=1)
    for i in range(len(EAS)):
        for j in range(len(EAS)):
            value = matrix[i, j]
            text = "—" if np.isnan(value) else f"{100 * value:.0f}%"
            dark = not np.isnan(value) and abs(value - 0.5) > 0.3
            matrix_axis.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if dark else TEXT,
            )
    names = [label(c) for c in EAS]
    matrix_axis.set_xticks(range(len(EAS)), names, fontsize=8, rotation=20)
    matrix_axis.set_yticks(range(len(EAS)), names, fontsize=8)
    for side in matrix_axis.spines.values():
        side.set_visible(False)
    matrix_axis.tick_params(colors=MUTED, length=0)
    matrix_axis.set_title(measure["matrix"], loc="left", color=TEXT, fontsize=11)

    what = "fitness" if column == "fitness" else "distance to the target"
    figure.suptitle(
        f"Emigrant-selection policy on spider_8 / OlympicArena, {what}: "
        "5 seeds × 12,000 evaluations (lower is better)",
        color=TEXT,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    figure.tight_layout()
    out = OLYMPIC / measure["out"]
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
