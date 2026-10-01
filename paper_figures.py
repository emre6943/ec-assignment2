"""The report's figures, sized and styled for the ACM sigconf (GECCO) template.

    uv run --project ../ariel python paper_figures.py

Reads the runs of experiments 14 and 26 in `results/olympic/` and writes a
vector PDF and a PNG preview of each figure into `report/figures/`:

    convergence     best fitness and closest distance so far against
                    generations: mean ± standard deviation over the 5 seeds of
                    every condition of experiment 14 (the line plot the spec
                    asks for)
    interval        experiment 26: final fitness and unseen-arena distance per
                    seed for each migration interval of the best policy, and
                    how often every island holds the same champion
    longer_walks    trained on 15 s walks: the brains that reach the target on
                    their own arena with more time, and on 20 unseen arenas
    probabilities   P(row converges faster than column): the Bayesian paired
                    t-test on the AUC (`probabilities.py`)
    fitness_terms   what each fitness term contributes to the champions'
                    fitness, from the champion's record in `database.db`

Every condition makes 80 evaluations in generation 0 and 72 children per
generation after it (4 islands x 18, or 1 x 72 for the standard EA; random
search draws 72 new genotypes), so all of them share one generation axis.
"""

# Standard library
import json
import sqlite3
from pathlib import Path

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")  # render to a file; no window
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

# Local libraries
from analyze import CONDITION_COLOURS, best_so_far, on_grid
from longer_walks import LONG_TEST
from probabilities import DEFAULT_ROPE, mean_interval, pair_rows, run_table, runs_of
from rq_figure import CONDITIONS, DIVERGING, EAS, label
from simulate import TARGET_RADIUS, Score, SimConfig, saved_sim_config

ROOT = Path(__file__).parent
OLYMPIC = ROOT / "results" / "olympic"
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
LONG_WALKS = ("15", "20", "30", "60")  # seconds, as `longer_walks.json` keys them

# The fitness terms (D15-D20) in stacking order: what the walk achieved, then
# the gait terms. Weights come from each run's config; leg imbalance is 0 and
# no champion is ever upside down, so those two are left out of the figure.
# The colours of the five drawn, in this order, pass the palette validator's
# adjacent-pair checks.
TERMS = {
    "distance": ("distance", "#4a3aa7"),
    "mean_distance": ("mean distance", "#1baf7a"),
    "low_body": ("low body", "#eda100"),
    "ground_contact": ("on the ground", "#e34948"),
    "upside_down": ("upside down", "#e87ba4"),
    "work_imbalance": ("work imbalance", "#2a78d6"),
    "leg_imbalance": ("leg imbalance", "#008300"),
}
TERM_WEIGHTS = {
    "distance": None,  # weight 1, not a setting
    "mean_distance": "speed_weight",
    "low_body": "low_body_weight",
    "ground_contact": "ground_contact_weight",
    "upside_down": "upside_down_weight",
    "work_imbalance": "work_imbalance_weight",
    "leg_imbalance": "leg_imbalance_weight",
}

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


def save(figure: plt.Figure, name: str) -> None:
    """The figure as a vector PDF and a PNG preview."""
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / f"{name}.pdf", facecolor="white")
    figure.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    plt.close(figure)
    print(f"written {OUT / name}.pdf")


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
    """The run's best fitness and the Score it was computed from."""
    with sqlite3.connect(run / "database.db") as database:
        fitness, tags = database.execute(
            "SELECT fitness_, tags_ FROM individual "
            "WHERE fitness_ IS NOT NULL ORDER BY fitness_ LIMIT 1"
        ).fetchone()
    parts = json.loads(tags)
    return float(fitness), Score(**{k: parts[k] for k in Score.__dataclass_fields__})


def convergence() -> None:
    """Best fitness and closest distance so far, mean ± std over seeds."""
    folders = [OLYMPIC / condition for condition in CONDITIONS]
    generations, evaluations = common_generations(folders)
    curves = {f.name: [best_so_far(run) for run in runs_of(f)] for f in folders}
    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.35))
    measures = (
        ("fitness", "best fitness so far"),
        ("distance", "closest to the target so far (m)"),
    )
    for axis, (column, ylabel) in zip(axes, measures, strict=True):
        for condition in CONDITIONS:
            values = np.array(
                [on_grid(c, column, evaluations) for c in curves[condition]]
            )
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            colour = CONDITION_COLOURS[condition]
            axis.fill_between(
                generations, mean - std, mean + std, color=colour, alpha=0.13, lw=0
            )
            axis.plot(
                generations,
                mean,
                color=colour,
                linewidth=1.1,
                linestyle=(0, (4, 2)) if condition == "random_search" else "-",
                label=f"{label(condition)}",
            )
        print_axis(axis)
        axis.set_xlim(0, generations[-1])
        axis.set_ylim(bottom=0)
        axis.set_xlabel("generation")
        axis.set_ylabel(ylabel)
        top = axis.secondary_xaxis(
            "top",
            functions=(
                lambda g: evaluations[0] + (evaluations[1] - evaluations[0]) * g,
                lambda e: (e - evaluations[0]) / (evaluations[1] - evaluations[0]),
            ),
        )
        top.set_xticks([0, 3000, 6000, 9000, 12000])
        top.set_xticklabels(["0", "3k", "6k", "9k", "12k"])
        top.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=6)
        top.spines["top"].set_color(MUTED)
        top.set_xlabel("evaluations", color=MUTED, fontsize=6.5, labelpad=2)
    axes[1].axhline(TARGET_RADIUS, color=INK, linewidth=0.7, linestyle=(0, (2, 2)))
    axes[1].text(
        generations[-1],
        TARGET_RADIUS + 0.03,
        f"target reached (< {TARGET_RADIUS:g} m)",
        ha="right",
        va="bottom",
        fontsize=6,
        color=INK,
    )
    panel(axes[0], "a", x=-0.12)
    panel(axes[1], "b", x=-0.12)
    handles, labels = axes[0].get_legend_handles_labels()
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
    save(figure, "convergence")


def interval() -> None:
    """Experiment 26: the final fitness, unseen distance and shared champions."""
    folders = [OLYMPIC / name for name in INTERVALS]
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
            for run in runs_of(OLYMPIC / name)
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
    save(figure, "interval")


def reach_shares(runs: list[Path]) -> dict[str, tuple[float, float]]:
    """Own arena (brains of the runs) and unseen (% of walks) reaching the target,
    at the training length and at LONG_TEST seconds."""
    long_key = f"{LONG_TEST:g}"
    own = [
        sum(
            json.loads((run / "longer_walks.json").read_text())[key]["arrived_at"]
            is not None
            for run in runs
        )
        for key in ("15", long_key)
    ]
    unseen = []
    for name in ("unseen.json", f"unseen_{long_key}s.json"):
        values = [
            json.loads((run / name).read_text())["unseen"]["reached"]
            for run in runs
            if (run / name).exists()
        ]
        unseen.append(100 * float(np.mean(values)))
    return {"own": (own[0], own[1]), "unseen": (unseen[0], unseen[1])}


def longer_walks() -> None:
    """Brains that reach the target at 15 s and with 30 s, own arena and unseen."""
    figure, (own_axis, unseen_axis) = plt.subplots(
        1, 2, figsize=(COLUMN, 1.85), sharey=True
    )
    for y, condition in enumerate(CONDITIONS):
        shares = reach_shares(runs_of(OLYMPIC / condition))
        colour = CONDITION_COLOURS[condition]
        for axis, key in ((own_axis, "own"), (unseen_axis, "unseen")):
            short, long = shares[key]
            axis.plot([short, long], [y, y], color=colour, linewidth=1.0, zorder=2)
            axis.scatter(
                short, y, s=14, facecolor="white", edgecolor=colour, lw=0.9, zorder=3
            )
            axis.scatter(long, y, s=14, color=colour, zorder=3)
    for axis in (own_axis, unseen_axis):
        print_axis(axis, grid="x")
    own_axis.set_yticks(range(len(CONDITIONS)), [label(c) for c in CONDITIONS])
    own_axis.invert_yaxis()
    own_axis.set_xlim(-0.3, 5.3)
    own_axis.set_xticks(range(6))
    own_axis.set_xlabel("own arena:\nbrains reaching it (of 5)")
    unseen_axis.set_xlim(-0.8, 15)
    unseen_axis.set_xticks([0, 5, 10, 15])
    unseen_axis.set_xlabel("20 unseen arenas:\nwalks reaching it (%)")
    unseen_axis.tick_params(axis="y", length=0)
    unseen_axis.scatter([], [], s=14, facecolor="white", edgecolor=MUTED, label="15 s")
    unseen_axis.scatter([], [], s=14, color=MUTED, label=f"{LONG_TEST:g} s")
    unseen_axis.legend(
        frameon=False, loc="lower right", handletextpad=0.1, borderaxespad=0.0
    )
    panel(own_axis, "a", x=-0.12)
    panel(unseen_axis, "b", x=-0.12)
    figure.tight_layout(w_pad=1.0)
    save(figure, "longer_walks")


def probabilities() -> None:
    """P(row converges faster than column), from the AUC, paired by seed."""
    table = run_table([OLYMPIC / condition for condition in EAS])
    pairs = pd.DataFrame(pair_rows(table, "auc", DEFAULT_ROPE))
    matrix = np.full((len(EAS), len(EAS)), np.nan)
    for i, a in enumerate(EAS):
        for j, b in enumerate(EAS):
            match = pairs[(pairs["a"] == a) & (pairs["b"] == b)]
            if not match.empty:
                matrix[i, j] = match["p_a_better_at_all"].iloc[0]
    figure, axis = plt.subplots(figsize=(COLUMN, 2.3))
    axis.imshow(matrix, cmap=DIVERGING, vmin=0, vmax=1)
    for i in range(len(EAS)):
        for j in range(len(EAS)):
            value = matrix[i, j]
            dark = not np.isnan(value) and abs(value - 0.5) > 0.3
            axis.text(
                j,
                i,
                "—" if np.isnan(value) else f"{100 * value:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if dark else INK,
            )
    names = [label(c) for c in EAS]
    axis.set_xticks(range(len(EAS)), names, rotation=25, ha="right")
    axis.set_yticks(range(len(EAS)), names)
    axis.set_xlabel("column")
    axis.set_ylabel("row")
    for side in axis.spines.values():
        side.set_visible(False)
    axis.tick_params(length=0, labelcolor=INK)
    save(figure, "probabilities")


def fitness_terms() -> None:
    """The champion's fitness split into its weighted terms, mean over seeds.

    Needs every run's `database.db`, the one file that is too big to share;
    without them the figure is skipped.
    """
    runs = {condition: runs_of(OLYMPIC / condition) for condition in CONDITIONS}
    missing = [
        r for rs in runs.values() for r in rs if not (r / "database.db").exists()
    ]
    if missing:
        print(
            f"fitness_terms.pdf skipped: no database.db in {len(missing)} run folders"
        )
        return
    rows = []
    for condition in CONDITIONS:
        for run in runs[condition]:
            sim = saved_sim_config(json.loads((run / "config.json").read_text())["sim"])
            fitness, score = champion(run)
            terms = weighted_terms(score, sim)
            if not np.isclose(sum(terms.values()), fitness, atol=1e-6):
                msg = f"{run}: terms sum to {sum(terms.values())}, fitness {fitness}"
                raise ValueError(msg)
            rows.append({"condition": condition, **terms})
    means = pd.DataFrame(rows).groupby("condition").mean().loc[list(CONDITIONS)]
    used = [term for term in TERMS if means[term].abs().max() > 0]
    figure, axis = plt.subplots(figsize=(COLUMN, 2.0))
    left = np.zeros(len(CONDITIONS))
    for term in used:
        name, colour = TERMS[term]
        axis.barh(
            range(len(CONDITIONS)),
            means[term],
            left=left,
            height=0.62,
            color=colour,
            edgecolor="white",
            linewidth=0.5,
            label=name,
            zorder=2,
        )
        left += means[term].to_numpy()
    for y, total in enumerate(left):
        axis.text(total + 0.03, y, f"{total:.2f}", va="center", fontsize=6.5, color=INK)
    print_axis(axis, grid="x")
    axis.set_yticks(range(len(CONDITIONS)), [label(c) for c in CONDITIONS])
    axis.invert_yaxis()
    axis.set_xlim(0, max(left) + 0.35)
    axis.set_xlabel("champion's fitness by term (mean of 5 seeds)")
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.42, -0.24),
        ncols=3,
        handlelength=1.2,
        columnspacing=0.9,
    )
    save(figure, "fitness_terms")


def main() -> None:
    """Write every figure."""
    convergence()
    interval()
    longer_walks()
    probabilities()
    fitness_terms()


if __name__ == "__main__":
    main()
