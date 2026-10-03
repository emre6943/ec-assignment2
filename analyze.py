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
    normality_<metric>.png  normal Q-Q plots of every pair's differences, the
                      check that picks the test of point 3

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
   pair of conditions with one two-sided test on the per-seed differences,
   Holm-corrected across the pairs of each metric. Also on the unseen-terrain
   distance when it is there. The test is chosen per metric by a rule fixed
   before the final experiment's results existed (decision D25): a
   Shapiro-Wilk test on each pair's differences, Holm-corrected across the
   pairs. If no pair rejects normality, every pair gets the paired t-test,
   with the mean difference, its 95% interval and Cohen's d_z as effect
   sizes; otherwise every pair gets the Wilcoxon signed-rank test, with the
   median difference and the matched-pairs rank-biserial correlation. With
   n seeds the smallest possible Wilcoxon p is 2 / 2^n: 0.0625 for experiment
   14's 5, so nothing can be significant there with that test.
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
from scipy.stats import (
    friedmanchisquare,
    mannwhitneyu,
    probplot,
    rankdata,
    shapiro,
    ttest_1samp,
    wilcoxon,
)

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
REJECTED = "#e34948"  # a pair whose differences fail the normality check
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
T_TEST = "paired t-test"
WILCOXON = "Wilcoxon signed-rank"
QQ_COLUMNS = 5  # panels per row of the normality figure: 15 pairs in 3 rows


class PairedTest(NamedTuple):
    """One pair of conditions on one metric, paired by seed.

    The differences are A - B and lower is better, so a negative difference,
    d_z or rank-biserial correlation means A did better. `test` is the test
    D25's rule chose for this metric (the same for all its pairs), `p` its
    p-value and `p_holm` that p Holm-corrected across the metric's pairs.
    `p_other_test` is the other test's p, uncorrected, kept only to show
    whether the choice mattered. Both tests' effect sizes are always filled in:
    the mean difference, its 95% interval and d_z go with the t-test, the
    median difference and r with the Wilcoxon test.
    """

    metric: str
    a: str
    b: str
    pairs: int
    test: str
    mean_difference: float
    ci_low: float
    ci_high: float
    cohen_dz: float
    median_difference: float
    rank_biserial: float
    p: float
    p_holm: float
    p_other_test: float


class NormalityCheck(NamedTuple):
    """Shapiro-Wilk on one pair's per-seed differences A - B.

    `w`, `p` and `p_holm` are NaN for a pair that could not be checked, and
    `unchecked` then says why; such a pair takes no part in the decision.
    """

    a: str
    b: str
    w: float
    p: float
    p_holm: float
    unchecked: str = ""


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


def paired_t(differences: npt.NDArray) -> tuple[float, float, float, float]:
    """Two-sided paired t-test on the differences A - B: (p, CI low, CI high, d_z).

    The test from the lecture: is the mean difference zero? scipy's ttest_rel
    on A and B subtracts them and runs exactly this one-sample test on the
    differences. The confidence interval is the 95% t interval of the mean
    difference, and d_z (Cohen's d for paired data; Lakens 2013) is the mean
    difference over the standard deviation of the differences: how many
    seed-to-seed spreads apart the two conditions lie. If every seed has the
    same difference, the spread is zero and the t statistic undefined; the
    limits are used instead: p = 1 and d_z = 0 when that difference is 0
    (identical conditions), p = 0 and an infinite d_z otherwise.
    """
    mean = float(differences.mean())
    if np.ptp(differences) == 0:
        if differences[0] == 0:
            return 1.0, mean, mean, 0.0
        return 0.0, mean, mean, float(np.copysign(np.inf, differences[0]))
    result = ttest_1samp(differences, 0.0)
    interval = result.confidence_interval(confidence_level=1 - ALPHA)
    d_z = mean / float(differences.std(ddof=1))
    return float(result.pvalue), float(interval.low), float(interval.high), d_z


def pair_differences(
    table: pd.DataFrame, metric: str
) -> dict[tuple[str, str], npt.NDArray]:
    """Every pair's per-seed differences A - B on one metric, by pair (A, B).

    Over the complete seeds only (`complete_blocks`), in the order the
    conditions were given, so the normality check, the tests and the figure all
    see the same numbers. A condition with no values for this metric (say, no
    unseen test yet) is left out rather than emptying the whole table. Empty
    when there are fewer than 2 conditions or 2 complete seeds.
    """
    if metric not in table:
        return {}
    blocks = complete_blocks(table.dropna(subset=[metric]), metric)
    if blocks.shape[1] < 2 or blocks.shape[0] < 2:
        return {}
    return {
        (a, b): np.round((blocks[a] - blocks[b]).to_numpy(), DIFFERENCE_DECIMALS)
        for a, b in combinations(blocks.columns, 2)
    }


def normality_checks(
    differences: dict[tuple[str, str], npt.NDArray],
) -> list[NormalityCheck]:
    """Shapiro-Wilk on every pair's differences, Holm-corrected across the pairs.

    The paired t-test assumes that the differences A - B, not the conditions'
    own values, are bell-shaped, so that is what is checked. Shapiro-Wilk
    (Shapiro & Wilk 1965) is the usual normality test for samples this small.
    Holm keeps the chance that any check rejects a truly bell-shaped pair by
    accident at 5% at most, so only clear evidence switches the metric to the
    Wilcoxon test. Two kinds of pair cannot be checked and are left out of the
    correction and the decision: one with the same difference on every seed
    (it has no shape; the differences are rounded, so float noise does not
    count as spread, and scipy would return W = 1, p = 1 with a warning), and
    any pair when there are fewer than 3 seeds, the least Shapiro-Wilk needs.
    """
    reasons = {}
    for pair, d in differences.items():
        if len(d) < 3:
            reasons[pair] = "needs ≥ 3 seeds"
        elif np.ptp(d) == 0:
            reasons[pair] = "the same difference on every seed"
    checked = [pair for pair in differences if pair not in reasons]
    results = {pair: shapiro(differences[pair]) for pair in checked}
    p_values = [float(results[pair].pvalue) for pair in checked]
    adjusted = dict(zip(checked, holm(p_values), strict=True))
    checks = []
    for a, b in differences:
        if (a, b) in reasons:
            checks.append(NormalityCheck(a, b, np.nan, np.nan, np.nan, reasons[a, b]))
        else:
            w, p = (float(value) for value in results[a, b])
            checks.append(NormalityCheck(a, b, w, p, adjusted[a, b]))
    return checks


def bell_shaped(checks: list[NormalityCheck]) -> bool:
    """D25's rule: no checked pair rejects normality after Holm at `ALPHA`.

    It decides for the whole metric at once, never pair by pair, so the test
    cannot be picked to suit a pair. With no pair checked there is no sign of a
    bell shape, and the Wilcoxon test, which needs none, is kept.
    """
    checked = [check for check in checks if not check.unchecked]
    return bool(checked) and all(check.p_holm >= ALPHA for check in checked)


def paired_comparisons(table: pd.DataFrame, metric: str) -> list[PairedTest]:
    """Every pair of conditions on one metric, paired by seed, with D25's test.

    Each test is two-sided on the per-seed differences A - B
    (`pair_differences`). One test serves every pair of the metric: the paired
    t-test (`paired_t`) when the differences are bell-shaped (`bell_shaped`),
    otherwise the Wilcoxon signed-rank test with exact p-values
    (`signed_rank_p`). The chosen test's p-values are Holm-corrected across all
    the pairs of this metric. In the Wilcoxon test a seed on which A and B tie
    exactly keeps its rank, split between the two signs (scipy's "zsplit"): a
    tie is evidence of no difference, which the textbook rule of dropping it
    would throw away. Demšar (2006) also keeps ties, though he drops one zero
    when their number is odd. Empty when there are fewer than 2 conditions or 2
    complete seeds.
    """
    differences = pair_differences(table, metric)
    if not differences:
        return []
    use_t = bell_shaped(normality_checks(differences))
    t_results = [paired_t(d) for d in differences.values()]
    t_p = [p for p, *_ in t_results]
    wilcoxon_p = [signed_rank_p(d) for d in differences.values()]
    chosen, other = (t_p, wilcoxon_p) if use_t else (wilcoxon_p, t_p)
    return [
        PairedTest(
            metric,
            a,
            b,
            len(d),
            T_TEST if use_t else WILCOXON,
            float(d.mean()),
            low,
            high,
            d_z,
            float(np.median(d)),
            rank_biserial(d),
            p,
            p_holm,
            p_other,
        )
        for ((a, b), d), (_, low, high, d_z), p, p_holm, p_other in zip(
            differences.items(), t_results, chosen, holm(chosen), other, strict=True
        )
    ]


def format_p(p: float) -> str:
    """A p-value to 4 decimals, or in scientific notation when smaller."""
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def bold_below_alpha(p: float) -> str:
    """A Holm-adjusted p for a table, in bold when it is below `ALPHA`."""
    return f"**{format_p(p)}**" if p < ALPHA else format_p(p)


def normality_report(checks: list[NormalityCheck]) -> list[str]:
    """One metric's normality checks and the decision they give, as Markdown."""
    lines = [
        "Normality of the differences: Shapiro-Wilk per pair, Holm-corrected across",
        "the pairs checked (bold: rejected, the differences are not bell-shaped).",
        "",
        "| A vs B | Shapiro W | p | p (Holm) |",
        "|---|---|---|---|",
    ]
    for check in checks:
        if check.unchecked:
            lines.append(f"| {check.a} vs {check.b} | – | – | {check.unchecked} |")
        else:
            lines.append(
                f"| {check.a} vs {check.b} | {check.w:.3f} | {format_p(check.p)} | "
                f"{bold_below_alpha(check.p_holm)} |"
            )
    checked = [check for check in checks if not check.unchecked]
    rejected = [f"{c.a} vs {c.b}" for c in checked if c.p_holm < ALPHA]
    if not checked:
        decision = (
            "no pair could be checked, so every pair gets the Wilcoxon signed-rank "
            "test, which needs no bell shape."
        )
    elif rejected:
        decision = (
            f"normality is rejected for {', '.join(rejected)}, so every pair gets "
            "the Wilcoxon signed-rank test."
        )
    else:
        decision = (
            f"no pair rejects normality ({len(checked)} of {len(checks)} checked), "
            "so every pair gets the paired t-test."
        )
    lines += ["", f"**Decision:** {decision}"]
    if checked and len(checked) < len(checks):
        lines.append("Pairs that could not be checked take no part in the decision.")
    return lines


def t_test_report(tests: list[PairedTest]) -> list[str]:
    """One metric's paired t-tests as a Markdown table."""
    lines = [
        "Paired t-tests: the mean difference with its 95% confidence interval, and",
        "d_z, the mean difference over the standard deviation of the differences.",
        "",
        "| A vs B | mean A − B | 95% CI | d_z | p | p (Holm) |",
        "|---|---|---|---|---|---|",
    ]
    for test in tests:
        lines.append(
            f"| {test.a} vs {test.b} | {test.mean_difference:+.3f} | "
            f"[{test.ci_low:+.3f}, {test.ci_high:+.3f}] | {test.cohen_dz:+.2f} | "
            f"{format_p(test.p)} | {bold_below_alpha(test.p_holm)} |"
        )
    return lines


def wilcoxon_report(tests: list[PairedTest], seeds: int) -> list[str]:
    """One metric's Wilcoxon signed-rank tests as a Markdown table."""
    # With n seeds the most extreme outcome, every seed on one side and no
    # ties, has p = 2 / 2^n; Holm multiplies the smallest p by the pair count.
    smallest = 2.0 / 2**seeds
    after_holm = min(1.0, len(tests) * smallest)
    verdict = ", so no pair can be significant here" if after_holm >= ALPHA else ""
    lines = [
        "Wilcoxon signed-rank tests with exact p-values; a seed on which A and B tie",
        "counts half for each side. r is the matched-pairs rank-biserial correlation",
        "(−1: A better on every seed).",
        f"With {seeds} seeds the smallest possible p is 2 / 2^{seeds} "
        f"= {format_p(smallest)}, {format_p(after_holm)} after Holm over "
        f"{len(tests)} pairs{verdict}.",
        "",
        "| A vs B | median A − B | r | p | p (Holm) |",
        "|---|---|---|---|---|",
    ]
    for test in tests:
        lines.append(
            f"| {test.a} vs {test.b} | {test.median_difference:+.3f} | "
            f"{test.rank_biserial:+.2f} | {format_p(test.p)} | "
            f"{bold_below_alpha(test.p_holm)} |"
        )
    return lines


def paired_report(table: pd.DataFrame) -> list[str]:
    """The all-pairs tests as Markdown lines, per metric: normality, decision, test."""
    lines = [
        "## All pairs, paired by seed",
        "",
        "Every pair of conditions is compared on the per-seed differences A − B with",
        "a two-sided test, Holm-corrected across the pairs of each metric. The test",
        "follows a rule fixed before the final experiment's results existed (D25):",
        "if no pair's differences reject normality (Shapiro-Wilk, Holm-corrected,",
        f"α = {ALPHA}), every pair of the metric gets the paired t-test; otherwise",
        "every pair gets the Wilcoxon signed-rank test. A seed missing from any",
        "condition is dropped from all, so every pair compares the same seeds. Lower",
        "is better: a negative difference or effect size means A did better.",
        f"**Bold**: below {ALPHA} after Holm.",
    ]
    for metric in PAIRED_METRICS:
        if metric not in table or table[metric].isna().all():
            continue
        differences = pair_differences(table, metric)
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
        if not differences:
            lines.append("Needs ≥ 2 conditions and ≥ 2 seeds present in every one.")
            continue
        tests = paired_comparisons(table, metric)
        lines += [*normality_report(normality_checks(differences)), ""]
        if tests[0].test == T_TEST:
            lines += t_test_report(tests)
        else:
            lines += wilcoxon_report(tests, len(blocks))
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


def draw_normality(
    differences: dict[tuple[str, str], npt.NDArray],
    checks: list[NormalityCheck],
    metric: str,
    path: Path,
) -> None:
    """Normal Q-Q plots of every pair's differences A - B, one small panel each.

    Each panel puts the sorted differences against the quantiles a normal
    distribution would give at the same ranks. The dashed line is where they
    would lie if the differences were exactly normal with their own mean and
    standard deviation, so a bell shape shows as points along the line, a skew
    as a curve and an outlying seed as a lone point far off it. It is the
    picture behind the Shapiro-Wilk numbers that pick the test (D25), so a
    reader can see what a rejection rests on. Pairs whose check rejected
    normality after Holm are drawn in red and say so in words.
    """
    columns = min(QQ_COLUMNS, len(differences))
    rows = -(-len(differences) // columns)  # ceiling division
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.7 * columns, 2.3 * rows + 0.9), squeeze=False
    )
    figure.patch.set_facecolor(SURFACE)
    for axis, ((a, b), d), check in zip(
        axes.flat, differences.items(), checks, strict=False
    ):
        quantiles, ordered = probplot(d, fit=False)
        rejected = not check.unchecked and check.p_holm < ALPHA
        colour = REJECTED if rejected else TEXT
        axis.plot(
            quantiles,
            d.mean() + d.std(ddof=1) * quantiles,
            color=MUTED,
            linewidth=1,
            linestyle="--",
        )
        axis.scatter(quantiles, ordered, s=12, color=colour, zorder=3)
        style_axis(axis)
        axis.tick_params(labelsize=7)
        axis.set_title(f"{a} vs {b}", color=TEXT, fontsize=9)
        if check.unchecked:
            note = f"not checked:\n{check.unchecked}"
        else:
            note = f"W = {check.w:.3f}\np (Holm) = {format_p(check.p_holm)}"
            if rejected:
                note += "\nnot bell-shaped"
        axis.text(
            0.04,
            0.96,
            note,
            transform=axis.transAxes,
            va="top",
            fontsize=7,
            color=colour,
        )
    for axis in axes.flat[len(differences) :]:
        axis.set_visible(False)
    figure.supxlabel("normal quantiles", color=TEXT, fontsize=10)
    figure.supylabel(
        "difference A − B, sorted (lower: A better)", color=TEXT, fontsize=10
    )
    figure.suptitle(
        f"{metric}: normal Q-Q plots of the per-seed differences "
        f"({len(next(iter(differences.values())))} seeds)",
        color=TEXT,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


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
    for metric in PAIRED_METRICS:
        differences = pair_differences(table, metric)
        if differences:
            draw_normality(
                differences,
                normality_checks(differences),
                metric,
                args.out / f"normality_{metric}.png",
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
