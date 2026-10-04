"""The report's figures, sized and styled for the ACM sigconf (GECCO) template.

    uv run --project ../ariel python paper_figures.py
    uv run --project ../ariel python paper_figures.py --final results/olympic \\
        --out results/olympic/figures   # experiment 14's, next to its runs

Writes a vector PDF and a PNG preview of each figure into `--out` (default
`report/figures/`). Every figure but one shows the final experiment, 99: the
six condition folders in `--final` (default `results/final/`). The interval
figure shows experiment 26, from `--olympic` (default `results/olympic/`),
since 99 migrates at one interval only. The figures, by their number and
label in `report/main.tex`:

    Figure 2, fig:interval -> interval.pdf
        experiment 26 (the preliminary study, Section 4.1): final fitness and
        unseen-arena distance per seed for each migration interval of the
        best policy, and how often every island holds the same champion.
    Figure 3, fig:convergence -> convergence.pdf (full text width)
        (a) best fitness so far against generations (top axis: evaluations),
            mean ± standard deviation over the seeds of every condition: the
            line plot the spec asks for.
        (b) the research question's paired comparison: each EA's best fitness
            so far minus that of no migration on the same seed, mean over the
            seeds with a 95% t-interval, on the evaluation grid of the AUC
            (`analyze.py`), so its mean over the run is the AUC difference.
            Random search, about 1 above, is left out.
    Figure 4, fig:spread -> final_spread.pdf
        every seed's final fitness and unseen-arena distance for each EA: a
        box (median and quartiles) and a dot per seed.
    (not in the paper) fitness_terms.pdf
        the champions' fitness split into the weighted terms of the paper's
        Equation 1, from each champion's walk on its own arena in
        `unseen.json`. The paper has no room for it; its Limitations quote
        the gait terms' sum (0.18-0.20, column `gait`) that this script prints.

Figure 1 (fig:walk), the walking robot (`walk_filmstrip.png`), is drawn by
`filmstrip.py`, since it walks a robot rather than reading results.

A figure is skipped, with a message, while its inputs are missing: a condition
without runs, or a run without the `unseen.py` or `summary.json` output that
figure reads. `--final results/olympic` draws the same figures from experiment
14, which has the same folder layout.

Every condition makes 80 evaluations in generation 0 and 72 children per
generation after it (4 islands x 18, or 1 x 72 for the standard EA; random
search draws 72 new genotypes), so all of them share one generation axis.
"""

# Standard library
import argparse
import json
from pathlib import Path

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")  # render to a file; no window
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

# Local libraries
from analyze import (
    CONDITION_COLOURS,
    GRID_POINTS,
    best_so_far,
    on_grid,
    warn_about_short_runs,
)
from ea import LOG_ROUNDING
from probabilities import mean_interval, run_table, runs_of
from rq_figure import CONDITIONS, EAS, label
from simulate import Score, SimConfig, saved_sim_config

ROOT = Path(__file__).parent
FINAL = ROOT / "results" / "final"  # experiment 99
OLYMPIC = ROOT / "results" / "olympic"  # experiments 14 and 26
OUT = ROOT / "report" / "figures"

COLUMN = 3.33  # ACM sigconf column width, inches
TEXT_WIDTH = 7.0  # both columns
INK = "#1a1a1a"
MUTED = "#5f5e5a"
GRID = "#e6e5e1"

# Experiment 26: the best policy migrating every 5, 10, 20 and 50 generations
# (10 is experiment 14's `best`), and never (`none`). Ordered by interval, so
# the blues run light (often) to dark (rarely); "never" keeps its own colour.
INTERVALS = {
    "best_int5": ("5", "#9cc3f0"),
    "best": ("10", "#5a9be6"),
    "best_int20": ("20", "#2a78d6"),
    "best_int50": ("50", "#184f96"),
    "none": ("off", CONDITION_COLOURS["none"]),
}
# Panel (b) of the convergence figure: each EA against no migration. Random
# search, about 1 behind, would flatten the EAs' curves into one line at 0.
REFERENCE = "none"
COMPARED = tuple(c for c in EAS if c != REFERENCE)
# Light enough for overlapping bands to keep the means apart: 6 SD bands in
# panel (a), 4 narrower confidence bands in (b).
BAND_ALPHA = {"spread": 0.07, "gap": 0.1}

# The fitness terms in the order of the paper's Equation 1, as the paper names
# them; `{w}` takes the term's weight from the runs' config ("" for 1), `{v}`
# the plain number. The weights are settings (`TERM_WEIGHTS`): a term whose
# weight is 0 is not part of the fitness and is left out (leg imbalance, D18).
# The colours of Equation 1's six terms, in this order, pass every check of the
# palette validator (dataviz skill), the adjacent-pair ones included; the dark
# green of c also keeps its thin segment visible between two light ones.
TERMS = {
    "distance": ("final distance $d_T$", "#4a3aa7"),
    "mean_distance": (r"mean distance ${w}\bar{{d}}$ (speed)", "#1baf7a"),
    "ground_contact": ("core on ground ${w}c$", "#008300"),
    "low_body": (r"low core ${w}\ell$", "#eda100"),
    "upside_down": ("upside down ${w}u$", "#e34948"),
    "work_imbalance": ("leg-work imbalance ${w}w$", "#2a78d6"),
    "leg_imbalance": ("leg-movement imbalance (weight {v})", "#e87ba4"),  # not in Eq. 1
}
TERM_WEIGHTS = {
    "distance": None,  # weight 1, not a setting
    "mean_distance": "speed_weight",
    "ground_contact": "ground_contact_weight",
    "low_body": "low_body_weight",
    "upside_down": "upside_down_weight",
    "work_imbalance": "work_imbalance_weight",
    "leg_imbalance": "leg_imbalance_weight",
}
# A term whose largest mean is below this share of the longest bar is too
# thin to see; its legend entry lists the value instead.
THIN_SEGMENT = 0.025

mpl.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,  # embed TrueType: text stays text in the PDF
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def print_axis(axis: plt.Axes, grid: str = "y") -> None:
    """White, recessive axes for print; only the data carries colour."""
    axis.set_facecolor("white")
    if grid:
        axis.grid(axis=grid, color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=INK)


def panel(axis: plt.Axes, letter: str, x: float = -0.16) -> None:
    """The panel letter, top left, outside the plot area."""
    axis.text(
        x,
        1.04,
        f"({letter})",
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        color=INK,
        va="bottom",
    )


def save(figure: plt.Figure, name: str, out: Path) -> None:
    """The figure as a vector PDF and a PNG preview."""
    out.mkdir(parents=True, exist_ok=True)
    # No creation date: the same data gives the same file, so git sees no change.
    figure.savefig(
        out / f"{name}.pdf", facecolor="white", metadata={"CreationDate": None}
    )
    figure.savefig(out / f"{name}.png", dpi=300, facecolor="white")
    plt.close(figure)
    print(f"written {out / name}.pdf")


def missing_inputs(name: str, folders: list[Path], files: tuple[str, ...] = ()) -> bool:
    """Whether figure `name` lacks inputs; if so, says what is missing.

    Every condition folder needs runs (`seed*/` with a `log.csv`), and every
    run each of `files`.
    """
    for folder in folders:
        if not runs_of(folder):
            print(f"{name}.pdf skipped: no seed*/log.csv in {folder}")
            return True
    runs = [run for folder in folders for run in runs_of(folder)]
    for file in files:
        missing = [run for run in runs if not (run / file).exists()]
        if missing:
            print(
                f"{name}.pdf skipped: no {file} in {len(missing)} of "
                f"{len(runs)} run folders"
            )
            return True
    return False


def seeds_per_condition(folders: list[Path]) -> str:
    """The number of runs per condition as a figure words it: "20", or "19-20"."""
    counts = sorted({len(runs_of(folder)) for folder in folders})
    return str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"


def generation_size(ea: dict[str, object]) -> tuple[int, int]:
    """Evaluations in generation 0 and in every generation after it.

    Generation 0 evaluates every island's population; after it each island
    replaces all but its elites. Random search draws the same number of new
    genotypes per generation (`ea.EA.make_child`).
    """
    islands, size = int(ea["n_islands"]), int(ea["island_size"])
    return islands * size, islands * (size - int(ea["n_elites"]))


def common_generations(folders: list[Path]) -> tuple[npt.NDArray, npt.NDArray]:
    """Generations 0..n every run reached, and their evaluation counts.

    Raises ValueError if the conditions do not share one generation size,
    since then a generation axis would compare unequal budgets.
    """
    sizes, last = set(), []
    for folder in folders:
        for run in runs_of(folder):
            sizes.add(
                generation_size(json.loads((run / "config.json").read_text())["ea"])
            )
            last.append(
                pd.read_csv(run / "log.csv", usecols=["generation"]).max().item()
            )
    if len(sizes) != 1:
        msg = f"conditions differ in generation size {sorted(sizes)}: use evaluations"
        raise ValueError(msg)
    initial, per_generation = sizes.pop()
    generations = np.arange(min(last) + 1)
    return generations, initial + per_generation * generations


def shared_champion_share(log: pd.DataFrame) -> float:
    """Share of the generations in which every island's best is the same.

    With the best policy a shared best fitness means one champion copied to
    every island: the islands have stopped searching apart.
    """
    islands = log[log["island"].astype(str) != "all"]
    distinct = islands.groupby("generation")["best"].nunique()
    return float((distinct == 1).mean())


def weighted_terms(score: Score, sim: SimConfig) -> dict[str, float]:
    """Each fitness term times its weight; they sum to `simulate.fitness`."""
    return {
        term: getattr(score, term) * (1.0 if setting is None else getattr(sim, setting))
        for term, setting in TERM_WEIGHTS.items()
    }


def champion(run: Path) -> tuple[float, Score]:
    """The run's best fitness and the Score it was computed from.

    Read from `unseen.json`, where `unseen.py` walked the run's best network
    on its own arena again. A walk on a saved arena is deterministic (D10)
    on the same kind of processor as the run, so it repeats the champion's
    evaluation exactly: for all 30 runs of experiment 14 it matches the
    champion's record in `database.db` to the last digit. Unlike the database, it is small enough to share, and it is
    complete for a resumed run (D24), whose champion may sit in an older
    `database_part<N>.db`. Raises ValueError unless the run walked one arena
    and the walk scored the run's best fitness (`summary.json`; after a
    resume that can be the log's, rounded to 4 decimals).
    """
    training = json.loads((run / "unseen.json").read_text())["training"]
    if len(training["scores"]) != 1:
        msg = f"{run}: trained on {len(training['scores'])} arenas, not one"
        raise ValueError(msg)
    fitness = float(training["fitness"][0])
    best = json.loads((run / "summary.json").read_text())["best_fitness_seen"]
    if abs(fitness - best) > LOG_ROUNDING:
        msg = f"{run}: its best scored {fitness} in unseen.json, {best} in the run"
        raise ValueError(msg)
    parts = training["scores"][0]
    return fitness, Score(**{k: parts[k] for k in Score.__dataclass_fields__})


def auc_grid(curves: dict[str, dict[str, pd.DataFrame]]) -> npt.NDArray:
    """The evaluation counts at which `analyze.py` reads the AUC.

    `GRID_POINTS` evenly spaced counts from 0 to the end of the shortest run
    (`analyze.main`, `probabilities.run_table`), so a run's AUC is the mean of
    its best-so-far fitness at these counts (`analyze.run_metrics`).
    """
    budget = min(
        curve["evaluations"].iloc[-1]
        for runs in curves.values()
        for curve in runs.values()
    )
    return np.linspace(0, budget, GRID_POINTS)


def paired_gaps(
    runs: dict[str, pd.DataFrame],
    reference: dict[str, pd.DataFrame],
    grid: npt.NDArray,
) -> npt.NDArray:
    """Per seed, the best fitness so far minus the reference's on that seed.

    `runs` and `reference` map seed folder names to `best_so_far` curves. One
    row per seed, one column per grid point; negative is ahead of the
    reference. Raises ValueError unless both have the same seeds, since the
    comparison is paired by seed (the same arena, D10).
    """
    if set(runs) != set(reference):
        msg = f"seeds differ: {sorted(set(runs) ^ set(reference))}"
        raise ValueError(msg)
    return np.array(
        [
            on_grid(runs[seed], "fitness", grid)
            - on_grid(reference[seed], "fitness", grid)
            for seed in sorted(runs)
        ]
    )


def evaluations_axis(axis: plt.Axes, evaluations: npt.NDArray) -> None:
    """A top axis in evaluations over a bottom axis in generations."""
    first, step = evaluations[0], evaluations[1] - evaluations[0]
    top = axis.secondary_xaxis(
        "top",
        functions=(lambda g: first + step * g, lambda e: (e - first) / step),
    )
    top.set_xticks([0, 3000, 6000, 9000, 12000])
    top.set_xticklabels(["0", "3k", "6k", "9k", "12k"])
    top.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=6)
    top.spines["top"].set_color(MUTED)
    top.set_xlabel("evaluations", color=MUTED, fontsize=6.5, labelpad=2)


def convergence(final: Path, out: Path) -> None:
    """(a) Best fitness so far, mean ± std over seeds, for every condition.

    (b) Each EA's best fitness so far minus no migration's on the same seed,
    mean over seeds with its 95% t-interval, on the AUC's evaluation grid:
    its mean over the run is the condition's mean AUC difference from no
    migration, which is checked against `probabilities.run_table`'s AUC.
    """
    folders = [final / condition for condition in CONDITIONS]
    if missing_inputs("convergence", folders):
        return
    generations, evaluations = common_generations(folders)
    curves = {
        folder.name: {run.name: best_so_far(run) for run in runs_of(folder)}
        for folder in folders
    }
    figure, (spread_axis, gap_axis) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.35))

    means = {}
    for condition in CONDITIONS:
        values = np.array(
            [on_grid(c, "fitness", evaluations) for c in curves[condition].values()]
        )
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        means[condition] = mean
        spread_axis.fill_between(
            generations,
            mean - std,
            mean + std,
            color=CONDITION_COLOURS[condition],
            alpha=BAND_ALPHA["spread"],
            lw=0,
        )
    for condition in CONDITIONS:  # every mean on top of every band
        spread_axis.plot(
            generations,
            means[condition],
            color=CONDITION_COLOURS[condition],
            linewidth=1.1,
            linestyle=(0, (4, 2)) if condition == "random_search" else "-",
            label=label(condition),
        )
    spread_axis.set_ylim(bottom=0)
    spread_axis.set_ylabel("best fitness so far")

    grid = auc_grid(curves)
    # The AUC grid in generations, for the shared x axis. Its first point, 0
    # evaluations, holds generation 0's value and falls left of the axis.
    grid_generations = (grid - evaluations[0]) / (evaluations[1] - evaluations[0])
    auc = run_table(folders).set_index(["condition", "seed"])["auc"]
    gaps = {}
    for condition in COMPARED:
        gaps[condition] = paired_gaps(curves[condition], curves[REFERENCE], grid)
        expected = (auc[condition] - auc[REFERENCE]).mean()
        if not np.isclose(gaps[condition].mean(), expected, atol=1e-9):
            msg = (
                f"{condition}: the mean gap {gaps[condition].mean()} is not the "
                f"AUC difference {expected}"
            )
            raise ValueError(msg)
        bounds = np.array([mean_interval(column)[2:] for column in gaps[condition].T])
        gap_axis.fill_between(
            grid_generations,
            bounds[:, 0],
            bounds[:, 1],
            color=CONDITION_COLOURS[condition],
            alpha=BAND_ALPHA["gap"],
            lw=0,
        )
    for condition in COMPARED:
        gap_axis.plot(
            grid_generations,
            gaps[condition].mean(axis=0),
            color=CONDITION_COLOURS[condition],
            linewidth=1.1,
        )
    gap_axis.axhline(0, color=CONDITION_COLOURS[REFERENCE], linewidth=1.1, zorder=1)
    gap_axis.set_ylabel(f"best fitness so far minus\n{label(REFERENCE)} (same seed)")
    gap_axis.text(
        0.99,
        0.98,
        "below 0: ahead of no migration\nrandom search (about +1) not shown",
        transform=gap_axis.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color=MUTED,
    )
    differences = ", ".join(f"{c} {gaps[c].mean():+.3f}" for c in COMPARED)
    print(f"convergence (b): mean over the run = AUC minus no migration: {differences}")

    for axis in (spread_axis, gap_axis):
        print_axis(axis)
        axis.set_xlim(0, generations[-1])
        axis.set_xlabel("generation")
        evaluations_axis(axis, evaluations)
    panel(spread_axis, "a", x=-0.12)
    panel(gap_axis, "b", x=-0.12)
    handles, labels = spread_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncols=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        handlelength=2.2,
        columnspacing=1.4,
    )
    figure.tight_layout(w_pad=2.5)
    save(figure, "convergence", out)


def final_spread(final: Path, out: Path) -> None:
    """Every seed's final fitness and unseen distance: a box per EA, dots on top.

    Random search is left out: far behind every EA, it would squash their
    boxes into a corner.
    """
    folders = [final / condition for condition in EAS]
    if missing_inputs("final_spread", folders, ("unseen.json",)):
        return
    table = run_table(folders)
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.9), sharey=True)
    rng = np.random.default_rng(0)
    measures = (
        ("final_fitness", "final best fitness"),
        ("unseen_distance", "unseen distance (m)"),
    )
    for axis, (metric, xlabel) in zip(axes, measures, strict=True):
        for y, condition in enumerate(EAS):
            values = table.loc[table["condition"] == condition, metric].to_numpy()
            colour = CONDITION_COLOURS[condition]
            line = {"color": colour, "linewidth": 0.8}
            axis.boxplot(
                values,
                positions=[y],
                orientation="horizontal",
                widths=0.62,
                showfliers=False,  # the dots show every seed
                manage_ticks=False,
                patch_artist=True,
                boxprops={
                    "facecolor": mpl.colors.to_rgba(colour, 0.14),
                    "edgecolor": colour,
                    "linewidth": 0.8,
                },
                medianprops={"color": INK, "linewidth": 1.0},
                whiskerprops=line,
                capprops=line,
                zorder=2,
            )
            axis.scatter(
                values,
                y + rng.uniform(-0.2, 0.2, len(values)),
                s=5,
                color=colour,
                alpha=0.85,
                linewidths=0,
                zorder=3,
            )
        print_axis(axis, grid="x")
        axis.set_xlabel(xlabel)
    axes[0].set_yticks(range(len(EAS)), [label(c) for c in EAS])
    axes[0].set_ylim(len(EAS) - 0.5, -0.5)  # the first condition on top
    axes[1].tick_params(axis="y", length=0)
    panel(axes[0], "a", x=-0.12)
    panel(axes[1], "b", x=-0.12)
    figure.tight_layout(w_pad=1.0)
    save(figure, "final_spread", out)


def interval(olympic: Path, out: Path) -> None:
    """Experiment 26: the final fitness, unseen distance and shared champions."""
    folders = [olympic / name for name in INTERVALS]
    if missing_inputs("interval", folders, ("unseen.json",)):
        return
    table = run_table(folders)
    names = list(INTERVALS)
    figure = plt.figure(figsize=(COLUMN, 3.0))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 1), hspace=0.7, wspace=0.5)
    final_axis = figure.add_subplot(grid[0, 0])
    unseen_axis = figure.add_subplot(grid[0, 1])
    shared_axis = figure.add_subplot(grid[1, :])
    rng = np.random.default_rng(0)
    for axis, metric, ylabel in (
        (final_axis, "final_fitness", "final best fitness"),
        (unseen_axis, "unseen_distance", "unseen distance (m)"),
    ):
        for x, name in enumerate(names):
            values = table.loc[table["condition"] == name, metric].dropna().to_numpy()
            colour = INTERVALS[name][1]
            axis.scatter(
                x + rng.uniform(-0.12, 0.12, len(values)),
                values,
                s=7,
                color=colour,
                alpha=0.85,
                linewidths=0,
                zorder=3,
            )
            mean, _, low, high = mean_interval(values)
            axis.errorbar(
                x + 0.3,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                color=INK,
                markersize=2.5,
                capsize=1.5,
                linewidth=0.7,
                zorder=4,
            )
        print_axis(axis)
        axis.set_xticks(range(len(names)), [INTERVALS[n][0] for n in names])
        axis.set_xlim(-0.5, len(names) - 0.3)
        axis.set_xlabel("interval (generations)")
        axis.set_ylabel(ylabel)

    shares = {
        name: [
            shared_champion_share(pd.read_csv(run / "log.csv", dtype={"island": str}))
            for run in runs_of(olympic / name)
        ]
        for name in names
    }
    means = [100 * np.mean(shares[name]) for name in names]
    shared_axis.bar(
        range(len(names)),
        means,
        width=0.55,
        color=[INTERVALS[n][1] for n in names],
        zorder=2,
    )
    for x, name in enumerate(names):
        shared_axis.scatter(
            x + rng.uniform(-0.12, 0.12, len(shares[name])),
            [100 * s for s in shares[name]],
            s=5,
            color=INK,
            alpha=0.6,
            linewidths=0,
            zorder=3,
        )
        shared_axis.text(
            x + 0.32,
            means[x] + 1.5,
            f"{means[x]:.0f}%",
            fontsize=6,
            color=INK,
            ha="left",
            va="bottom",
        )
    print_axis(shared_axis)
    shared_axis.set_xticks(range(len(names)), [INTERVALS[n][0] for n in names])
    shared_axis.set_xlim(-0.5, len(names) - 0.3)
    shared_axis.set_ylim(0, 65)
    shared_axis.set_xlabel("migration interval (generations)")
    shared_axis.set_ylabel("one champion on\nall islands (%)")
    panel(final_axis, "a", x=-0.38)
    panel(unseen_axis, "b", x=-0.38)
    panel(shared_axis, "c", x=-0.155)
    save(figure, "interval", out)


def term_weights(sims: list[SimConfig]) -> dict[str, float]:
    """Each fitness term's weight, the same in every run's config.

    Raises ValueError if the runs weigh a term differently, since one figure
    then cannot name one weight per term.
    """
    weights = {}
    for term, setting in TERM_WEIGHTS.items():
        values = {1.0 if setting is None else getattr(sim, setting) for sim in sims}
        if len(values) != 1:
            msg = f"the runs weigh {term} differently: {sorted(values)}"
            raise ValueError(msg)
        weights[term] = values.pop()
    return weights


def term_label(term: str, weight: float, means: pd.Series, longest: float) -> str:
    """A term's legend entry: its name and weighted symbol from Equation 1.

    A term too thin to see in the bars (below `THIN_SEGMENT` of the longest)
    lists its largest mean over the conditions, rounded up, or "always 0".
    """
    name = TERMS[term][0].format(
        w="" if weight == 1 else f"{weight:g}\\,", v=f"{weight:g}"
    )
    largest = float(means.max())
    if largest >= THIN_SEGMENT * longest:
        return name
    if (means == 0).all():
        return f"{name} (always 0)"
    return f"{name} (\u2264 {np.ceil(100 * largest) / 100:.2f})"


def fitness_terms(final: Path, out: Path) -> None:
    """The champion's fitness split into its weighted terms, mean over seeds.

    The terms are those of the paper's Equation 1, in its order and with the
    weights of the runs' config. Needs every run's `unseen.json` (`champion`)
    and `summary.json`; without them the figure is skipped.
    """
    folders = [final / condition for condition in CONDITIONS]
    if missing_inputs("fitness_terms", folders, ("unseen.json", "summary.json")):
        return
    rows, sims = [], []
    for folder in folders:
        for run in runs_of(folder):
            sim = saved_sim_config(json.loads((run / "config.json").read_text())["sim"])
            fitness, score = champion(run)
            terms = weighted_terms(score, sim)
            if not np.isclose(sum(terms.values()), fitness, atol=1e-6):
                msg = f"{run}: terms sum to {sum(terms.values())}, fitness {fitness}"
                raise ValueError(msg)
            rows.append({"condition": folder.name, **terms})
            sims.append(sim)
    weights = term_weights(sims)
    means = pd.DataFrame(rows).groupby("condition").mean().loc[list(CONDITIONS)]
    used = [term for term in TERMS if weights[term] != 0]
    print("fitness terms of the champions, weighted, mean over the seeds:")
    gait = [term for term in used if term not in ("distance", "mean_distance")]
    table = means[used].assign(gait=means[gait].sum(axis=1))
    print(table.assign(total=means[used].sum(axis=1)).round(3).to_string())
    longest = float(means[used].sum(axis=1).max())
    # Narrower than the column: the condition names stick out to the left.
    figure, axis = plt.subplots(figsize=(COLUMN - 0.13, 1.75))
    left = np.zeros(len(CONDITIONS))
    handles = []
    for term in used:
        values = means[term].to_numpy()
        drawn = values > 0  # a 0-wide bar would still leave a hairline
        # No white edges between the segments: they would hide a thin one.
        axis.barh(
            np.flatnonzero(drawn),
            values[drawn],
            left=left[drawn],
            height=0.62,
            color=TERMS[term][1],
            linewidth=0,
            zorder=2,
        )
        handles.append(
            mpl.patches.Patch(
                color=TERMS[term][1],
                label=term_label(term, weights[term], means[term], longest),
            )
        )
        left += values
    for y, total in enumerate(left):
        axis.text(total + 0.03, y, f"{total:.2f}", va="center", fontsize=6.5, color=INK)
    print_axis(axis, grid="x")
    axis.set_yticks(range(len(CONDITIONS)), [label(c) for c in CONDITIONS])
    axis.invert_yaxis()
    axis.set_xlim(0, max(left) + 0.35)
    axis.set_xlabel(
        f"champion's fitness by term (mean of {seeds_per_condition(folders)} seeds)"
    )
    axis.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.33, -0.27),
        ncols=2,
        handlelength=1.2,
        columnspacing=1.2,
    )
    save(figure, "fitness_terms", out)


def main() -> None:
    """Write every figure whose inputs are there."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--final",
        type=Path,
        default=FINAL,
        help="the final experiment's condition folders (default: results/final)",
    )
    parser.add_argument(
        "--olympic",
        type=Path,
        default=OLYMPIC,
        help="experiment 26's condition folders, with experiment 14's best and "
        "none, for the interval figure (default: results/olympic)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="where the PDFs and PNGs go (default: report/figures)",
    )
    args = parser.parse_args()

    warn_about_short_runs({c: runs_of(args.final / c) for c in CONDITIONS})
    convergence(args.final, args.out)
    final_spread(args.final, args.out)
    fitness_terms(args.final, args.out)
    interval(args.olympic, args.out)


if __name__ == "__main__":
    main()
