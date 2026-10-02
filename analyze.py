"""Compare conditions across seeds: the figures and numbers for the report.

    uv run --project ../ariel python analyze.py results/best results/worst \\
        results/random results/none results/random_search

Each argument is a condition folder holding `seed*/` run folders. Writes into
`--out` (default `results/analysis/`):

    convergence.png   mean ± std across seeds of the best fitness found so far,
                      against evaluations - the line plot the spec asks for
    summary.csv       one row per run: best at the budget, evaluations to
                      threshold, area under the curve, unseen-terrain distance
                      if `unseen.py` has been run
    summary.md        per condition: mean ± std of those numbers
    stats.md          the statistical tests (see below)
    paired_tests.csv  the all-pairs tests of point 3, one row per pair and metric

How each number is defined (decision D3):

- best so far       the lowest fitness any individual reached up to that point
- evals to threshold  the first evaluation count at which best so far drops
                    below `--threshold`; runs that never do count as "not reached"
- AUC               the mean of best so far over the evaluation budget; lower
                    means good solutions were found earlier

The statistics (decision D3), on the best fitness at the budget and on the AUC:

1. A Friedman test across all conditions, blocked by seed. Conditions with the
   same seed walk the same terrain, so the runs form matched blocks.
2. Planned comparisons: every condition against `--reference` (default
   `none`, the no-migration control) with a two-sided Mann-Whitney U test,
   Holm-corrected. With 5 seeds per condition the smallest possible
   Mann-Whitney p is 0.008, so only a few planned comparisons - not all
   pairs - can ever reach significance after correction.
3. All pairs, paired by seed (for the final experiment's 20 seeds): every
   pair of conditions with a two-sided Wilcoxon signed-rank test on the
   per-seed differences, Holm-corrected across the pairs of each metric, with
   the median difference and the matched-pairs rank-biserial correlation as
   effect sizes. Also on the unseen-terrain distance when it is there. With
   n seeds the smallest possible p is 2 / 2^n: 0.0625 for experiment 14's 5,
   so nothing can be significant there.
"""

# Standard library
import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")  # render to a file; no window
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata, wilcoxon

# The four migration policies in fixed categorical order (checked for
# colour-blind separation); random search is the neutral baseline.
CONDITION_COLOURS = {
    "best": "#2a78d6",
    "worst": "#eb6834",
    "random": "#1baf7a",
    "none": "#eda100",
    "random_search": "#52514e",
    "standard": "#e87ba4",
}
FALLBACK_COLOURS = ("#4a3aa7", "#008300", "#e34948")
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
GRID_POINTS = 200
TESTED_METRICS = ("final_fitness", "auc")
PAIRED_METRICS = (*TESTED_METRICS, "unseen_distance")
ALPHA = 0.05
# The logs keep 4 decimals; rounding the differences stops float noise in the
# subtraction from splitting what are really tied differences (scipy's advice).
DIFFERENCE_DECIMALS = 9
EXACT_UP_TO = 50  # nonzero differences; 2^50 still counts exactly in a float
INCOMPLETE_RUN_SHARE = 0.95  # a run that stopped below this share of its budget


class PairedTest(NamedTuple):
    """One pair of conditions on one metric, paired by seed.

    The differences are A - B and lower is better, so a negative median
    difference or rank-biserial correlation means A did better.
    """

    metric: str
    a: str
    b: str
    pairs: int
    median_difference: float
    rank_biserial: float
    p: float
    p_holm: float


def best_so_far(run: Path) -> pd.DataFrame:
    """Evaluations and the best fitness / distance found up to each generation."""
    log = pd.read_csv(run / "log.csv", dtype={"island": str})
    everyone = log[log["island"] == "all"]
    # `best_final` excludes a curriculum's movement reward; older logs lack it.
    column = "best_final" if "best_final" in everyone else "best"
    return pd.DataFrame(
        {
            "evaluations": everyone["evaluations"].to_numpy(),
            "fitness": everyone[column].cummin().to_numpy(),
            "distance": everyone["best_distance"].cummin().to_numpy(),
        }
    )


def on_grid(curve: pd.DataFrame, column: str, grid: npt.NDArray) -> npt.NDArray:
    """The curve's value at each grid point: the last value logged at or before it."""
    index = np.searchsorted(curve["evaluations"].to_numpy(), grid, side="right") - 1
    values = curve[column].to_numpy()
    return values[np.clip(index, 0, len(values) - 1)]


def run_metrics(
    curve: pd.DataFrame, grid: npt.NDArray, threshold: float
) -> dict[str, float]:
    """Best at the budget, evaluations to threshold and AUC for one run.

    Everything is read within the common budget (the grid's last point), so a
    run that happened to go on longer gains nothing from it.
    """
    within = curve[curve["evaluations"] <= grid[-1]]
    reached = within.loc[within["fitness"] < threshold, "evaluations"]
    end = grid[-1:]
    return {
        "final_fitness": float(on_grid(curve, "fitness", end)[0]),
        "final_distance": float(on_grid(curve, "distance", end)[0]),
        "evals_to_threshold": float(reached.iloc[0]) if len(reached) else np.nan,
        "auc": float(on_grid(curve, "fitness", grid).mean()),
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the original order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def complete_blocks(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per seed and one column per condition, for complete seeds only.

    A seed missing from any condition (a run still going, an unseen test not yet
    done) is dropped from all of them, so every test compares the same seeds.
    The columns keep the order in which the conditions were given.
    """
    wide = table.pivot(index="run", columns="condition", values=metric)
    return wide[list(dict.fromkeys(table["condition"]))].dropna()


def friedman(table: pd.DataFrame, metric: str) -> tuple[float, float, int] | None:
    """Friedman test across conditions, one block per seed: (statistic, p, blocks).

    Only seeds present in every condition are used. None when there are fewer
    than 3 conditions or 2 complete seeds, where the test is undefined.
    """
    blocks = complete_blocks(table, metric)
    if blocks.shape[1] < 3 or blocks.shape[0] < 2:
        return None
    statistic, p = friedmanchisquare(*(blocks[c] for c in blocks.columns))
    return float(statistic), float(p), int(blocks.shape[0])


def planned_comparisons(
    table: pd.DataFrame, metric: str, reference: str
) -> list[tuple[str, float, float]]:
    """Every condition against `reference`: (condition, p, Holm-adjusted p)."""
    others = [c for c in table["condition"].unique() if c != reference]
    base = table.loc[table["condition"] == reference, metric]
    p_values = [
        float(
            mannwhitneyu(
                table.loc[table["condition"] == other, metric],
                base,
                alternative="two-sided",
            ).pvalue
        )
        for other in others
    ]
    return list(zip(others, p_values, holm(p_values), strict=True))


def rank_biserial(differences: npt.NDArray) -> float:
    """Matched-pairs rank-biserial correlation of the differences A - B, -1 to 1.

    Rank the seeds by the size of their difference; r is the rank sum of the
    seeds where A was higher minus that of the seeds where A was lower, over
    all ranks (Kerby 2014). So -1 means A was lower (better) on every seed and
    0 no tendency either way. It is the effect size that belongs to the
    Wilcoxon test, built from the same signed ranks: 0 exactly when the test
    statistic sits at its null mean. A12 (probabilities.py) would ignore the
    pairing these tests rest on. A zero difference adds its rank to both sides,
    as the test does, so ties pull r towards 0.
    """
    ranks = rankdata(np.abs(differences))
    return float((np.sign(differences) * ranks).sum() / ranks.sum())


def signed_rank_p(differences: npt.NDArray) -> float:
    """Two-sided p of the Wilcoxon signed-rank test, exact even with ties.

    The seeds are ranked by the size of their difference (midranks for equal
    sizes). Under the null hypothesis every nonzero difference is as likely
    positive as negative, so the exact p is the share of all 2^m sign patterns
    of the m nonzero differences whose signed rank sum is at least as far from
    0 as the observed one. The distribution of that sum is counted exactly over
    doubled ranks, which are whole numbers. A zero difference keeps its rank
    and adds nothing to the sum, which is scipy's "zsplit". scipy's own exact
    method allows no ties and otherwise falls back to a normal approximation:
    with 20 seeds all favouring A and two of them tied, it gives 8.8e-5
    instead of the exact 1.9e-6. Above 50 nonzero differences (counts beyond
    float precision) that approximation is used here too. If every seed ties,
    p is 1.
    """
    ranks = rankdata(np.abs(differences))
    signs = np.sign(differences)
    weights = np.rint(2 * ranks[signs != 0]).astype(int)
    if len(weights) == 0:
        return 1.0
    if len(weights) > EXACT_UP_TO:
        return float(wilcoxon(differences, zero_method="zsplit").pvalue)
    observed = abs(int(np.rint(2 * (signs * ranks).sum())))
    total = int(weights.sum())
    counts = np.zeros(2 * total + 1)  # how many sign patterns give each sum
    counts[total] = 1.0
    for weight in weights:
        counts = np.roll(counts, weight) + np.roll(counts, -weight)
    distance = np.abs(np.arange(-total, total + 1))
    return float(counts[distance >= observed].sum() / counts.sum())


def paired_comparisons(table: pd.DataFrame, metric: str) -> list[PairedTest]:
    """Every pair of conditions on one metric: Wilcoxon signed-rank tests by seed.

    Each test is two-sided on the per-seed differences A - B over the complete
    seeds only (`complete_blocks`), Holm-corrected across all the pairs of this
    metric, with exact p-values (`signed_rank_p`). A seed on which A and B tie
    exactly keeps its rank, split between the two signs (scipy's "zsplit"): a
    tie is evidence of no difference, which the textbook rule of dropping it
    would throw away. Demšar (2006) also keeps ties, though he drops one zero
    when their number is odd. A condition with no values for this metric (say,
    no unseen test yet) is left out rather than emptying the whole table. Empty
    when there are fewer than 2 conditions or 2 complete seeds.
    """
    if metric not in table:
        return []
    blocks = complete_blocks(table.dropna(subset=[metric]), metric)
    if blocks.shape[1] < 2 or blocks.shape[0] < 2:
        return []
    pairs = list(combinations(blocks.columns, 2))
    differences = [
        np.round((blocks[a] - blocks[b]).to_numpy(), DIFFERENCE_DECIMALS)
        for a, b in pairs
    ]
    p_values = [signed_rank_p(d) for d in differences]
    return [
        PairedTest(
            metric, a, b, len(blocks), float(np.median(d)), rank_biserial(d), p, p_holm
        )
        for (a, b), d, p, p_holm in zip(
            pairs, differences, p_values, holm(p_values), strict=True
        )
    ]


def format_p(p: float) -> str:
    """A p-value to 4 decimals, or in scientific notation when smaller."""
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def paired_report(table: pd.DataFrame) -> list[str]:
    """The all-pairs Wilcoxon tests as Markdown lines, one table per metric."""
    lines = [
        "## All pairs, paired by seed",
        "",
        "Two-sided Wilcoxon signed-rank tests on the per-seed differences A − B, with",
        "exact p-values, Holm-corrected across the pairs of each metric. A seed missing",
        "from any condition is dropped from all, so every pair compares the same seeds;",
        "a seed on which A and B tie counts half for each side. Lower is better: a",
        "negative median difference or r means A did better. r is the matched-pairs",
        "rank-biserial correlation (−1: A better on every seed). **Bold**: below",
        f"{ALPHA} after Holm.",
    ]
    for metric in PAIRED_METRICS:
        if metric not in table or table[metric].isna().all():
            continue
        tests = paired_comparisons(table, metric)
        blocks = complete_blocks(table.dropna(subset=[metric]), metric)
        lines += ["", f"### {metric}: {len(blocks)} seeds in every condition", ""]
        left_out = [c for c in dict.fromkeys(table["condition"]) if c not in blocks]
        if left_out:
            lines += [
                f"Left out, no values for this metric: {', '.join(left_out)}.",
                "",
            ]
        dropped = sorted(set(table["run"]) - set(blocks.index))
        if dropped:
            lines += [
                f"Dropped, missing from some condition: {', '.join(dropped)}.",
                "",
            ]
        if not tests:
            lines.append("Needs ≥ 2 conditions and ≥ 2 seeds present in every one.")
            continue
        # With n seeds the most extreme outcome, every seed on one side and no
        # ties, has p = 2 / 2^n; Holm multiplies the smallest p by the pair count.
        smallest = 2.0 / 2 ** len(blocks)
        after_holm = min(1.0, len(tests) * smallest)
        verdict = ", so no pair can be significant here" if after_holm >= ALPHA else ""
        lines += [
            f"With {len(blocks)} seeds the smallest possible p is 2 / 2^{len(blocks)} "
            f"= {format_p(smallest)}, {format_p(after_holm)} after Holm over "
            f"{len(tests)} pairs{verdict}.",
            "",
            "| A vs B | median A − B | r | p | p (Holm) |",
            "|---|---|---|---|---|",
        ]
        for test in tests:
            p_holm = format_p(test.p_holm)
            if test.p_holm < ALPHA:
                p_holm = f"**{p_holm}**"
            lines.append(
                f"| {test.a} vs {test.b} | {test.median_difference:+.3f} | "
                f"{test.rank_biserial:+.2f} | {format_p(test.p)} | {p_holm} |"
            )
    return lines


def stats_report(table: pd.DataFrame, reference: str) -> str:
    """The Friedman tests, planned comparisons and all-pairs tests, as Markdown."""
    lines = [
        "## Friedman test across all conditions (blocked by seed)",
        "",
        "| metric | chi² | p | seeds |",
        "|---|---|---|---|",
    ]
    for metric in TESTED_METRICS:
        result = friedman(table, metric)
        if result is None:
            lines.append(f"| {metric} | – | – | needs ≥ 3 conditions, ≥ 2 seeds |")
        else:
            statistic, p, blocks = result
            lines.append(f"| {metric} | {statistic:.2f} | {p:.4f} | {blocks} |")

    lines += ["", f"## Planned comparisons against `{reference}`", ""]
    if reference not in set(table["condition"]):
        lines.append(f"No condition named `{reference}`; pass `--reference`.")
    else:
        lines += [
            "Two-sided Mann-Whitney U, Holm-corrected within each metric.",
            "",
            "| metric | condition | p | p (Holm) |",
            "|---|---|---|---|",
        ]
        for metric in TESTED_METRICS:
            for other, p, p_holm in planned_comparisons(table, metric, reference):
                lines.append(
                    f"| {metric} | {other} vs {reference} | {p:.4f} | {p_holm:.4f} |"
                )
    lines += ["", *paired_report(table)]
    return "\n".join(lines) + "\n"


def summary_markdown(table: pd.DataFrame, threshold: float, budget: float) -> str:
    """Per condition: mean ± sample std of every per-run number."""
    numeric = [c for c in table.columns if c not in {"condition", "run"}]
    lines = ["| condition | runs | " + " | ".join(numeric) + " |"]
    lines.append("|---" * (len(numeric) + 2) + "|")
    for condition, group in table.groupby("condition", sort=False):
        cells = []
        for column in numeric:
            values = group[column].dropna()
            if column == "evals_to_threshold":
                cells.append(
                    f"{values.mean():.0f} ± {values.std():.0f} "
                    f"({len(values)}/{len(group)} reached)"
                    if len(values)
                    else f"not reached (0/{len(group)})"
                )
            elif len(values):
                cells.append(f"{values.mean():.3f} ± {values.std():.3f}")
            else:
                cells.append("–")
        lines.append(f"| {condition} | {len(group)} | " + " | ".join(cells) + " |")
    return (
        f"Mean ± sample standard deviation across seeds. Threshold for 'evals to "
        f"threshold': fitness < {threshold}. Budget compared: {budget:.0f} "
        f"evaluations.\n\n" + "\n".join(lines) + "\n"
    )


def warn_about_short_runs(runs_of: dict[str, list[Path]]) -> None:
    """Point out runs that stopped well short of their own evaluation budget.

    The comparison is cut to the shortest run, so one crashed or stopped run
    would otherwise shrink every condition's comparison without a word.
    """
    for runs in runs_of.values():
        for run in runs:
            budget = json.loads((run / "config.json").read_text())["ea"][
                "max_evaluations"
            ]
            done = best_so_far(run)["evaluations"].iloc[-1]
            if done < INCOMPLETE_RUN_SHARE * budget:
                print(
                    f"WARNING: {run} stopped at {done} of {budget} evaluations; "
                    "every condition is compared only up to the shortest run."
                )


def style_axis(axis: plt.Axes) -> None:
    """Recessive grid and axes; only the data carries colour."""
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=9)


def draw_convergence(
    curves: dict[str, list[pd.DataFrame]],
    grid: npt.NDArray,
    threshold: float,
    path: Path,
) -> bool:
    """The mean ± std figure. Returns False if there are too many conditions."""
    extra = [c for c in curves if c not in CONDITION_COLOURS]
    if len(extra) > len(FALLBACK_COLOURS):
        return False  # more lines than distinguishable colours: never cycle them
    figure, axis = plt.subplots(figsize=(8, 5))
    figure.patch.set_facecolor(SURFACE)
    fallback = iter(FALLBACK_COLOURS)
    for condition, runs in curves.items():
        colour = CONDITION_COLOURS.get(condition) or next(fallback)
        values = np.array([on_grid(curve, "fitness", grid) for curve in runs])
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
        axis.fill_between(grid, mean - std, mean + std, color=colour, alpha=0.15, lw=0)
        axis.plot(
            grid, mean, color=colour, linewidth=2, label=f"{condition} (n={len(runs)})"
        )
    style_axis(axis)
    axis.axhline(threshold, color=MUTED, linewidth=1, linestyle=":")
    axis.set_xlabel("evaluations", color=TEXT, fontsize=10)
    axis.set_ylabel("best fitness so far (lower is better)", color=TEXT, fontsize=10)
    axis.legend(frameon=False, fontsize=9, loc="upper right")
    figure.suptitle(
        "Convergence per condition: mean ± sample std across seeds",
        color=TEXT,
        fontsize=12,
        x=0.1,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return True


def main() -> None:
    """Aggregate every condition and write the figure, tables and tests."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("conditions", type=Path, nargs="+")
    parser.add_argument("--threshold", type=float, default=1.6)
    parser.add_argument("--reference", default="none", help="control condition")
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    names = [folder.name for folder in args.conditions]
    if len(set(names)) != len(names):
        parser.error(f"condition folders must have different names, got {names}")

    runs_of: dict[str, list[Path]] = {}
    curves: dict[str, list[pd.DataFrame]] = {}
    for folder in args.conditions:
        runs = sorted(p.parent for p in folder.glob("seed*/log.csv"))
        if runs:
            runs_of[folder.name] = runs
            curves[folder.name] = [best_so_far(run) for run in runs]
    if not curves:
        parser.error("no seed*/log.csv found in the given folders")
    args.out.mkdir(parents=True, exist_ok=True)
    warn_about_short_runs(runs_of)

    budget = min(c["evaluations"].iloc[-1] for runs in curves.values() for c in runs)
    grid = np.linspace(0, budget, GRID_POINTS)

    rows = []
    for condition, runs in curves.items():
        for run_folder, curve in zip(runs_of[condition], runs, strict=True):
            row = {"condition": condition, "run": run_folder.name}
            row.update(run_metrics(curve, grid, args.threshold))
            unseen = run_folder / "unseen.json"
            if unseen.exists():
                result = json.loads(unseen.read_text())["unseen"]
                row["unseen_distance"] = result["distance_mean"]
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summary.csv", index=False)
    summary = summary_markdown(table, args.threshold, budget)
    (args.out / "summary.md").write_text(summary)
    (args.out / "stats.md").write_text(stats_report(table, args.reference))
    paired = [test for m in PAIRED_METRICS for test in paired_comparisons(table, m)]
    pd.DataFrame(paired, columns=PairedTest._fields).to_csv(
        args.out / "paired_tests.csv", index=False
    )

    drawn = draw_convergence(curves, grid, args.threshold, args.out / "convergence.png")
    print(summary)
    if not drawn:
        print(
            f"{len(curves)} conditions are too many for one chart; tables written, "
            "figure skipped. Pass fewer folders for a figure."
        )
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()
