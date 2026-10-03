"""Which algorithm ends best if compute is no concern? (experiment 99, exploratory)

    uv run --project ../ariel python end_results.py results/final/best \\
        results/final/worst results/final/random results/final/none \\
        results/final/standard --out results/final/end_results.md

The pre-registered tests of experiment 99 (D3, D25; `analyze.py`) cover the
final fitness, the AUC and the unseen distance, so they mix speed and outcome.
This script looks only at where the runs END, with reaching the target first,
as Emre asked after seeing the results. It is therefore exploratory: the
measures were chosen after the results were known, and a reader should weigh
its p-values accordingly (D26). Random search is left out: it never reaches
the target and is far behind on every measure (`analyze.py`).

Per run (one per seed), from files the experiment already wrote:

    final_fitness       best fitness on the training arena (log.csv)
    own_reached_15      did the best brain reach its own target in 15 s, the
                        training length? (yes/no, longer_walks.json)
    own_reached_30      ... within 30 s?
    unseen_reached_30   share of the 20 unseen arenas whose target it reached
                        within 30 s (unseen_30s.json); none arrives in 15 s
    unseen_fitness      mean fitness on the 20 unseen arenas, 15 s (unseen.json)
    unseen_failed_30    unseen arenas on which the robot fell off the arena
                        within 30 s (shown, not tested), of unseen_walks_30

The distance left after 30 s is not used: a walk whose robot falls off the
arena is scored as 10 m away (`simulate.FAILED_SCORE`), and in 30 s many
brains walk on past the strip and fall, so that mean mostly counts falls. The
unseen fitness charges a fall too (fitness 18.5), but falls within 15 s are rare
(4 of 2,000 walks of the five EAs).

Tests, all paired by seed and Holm-corrected across the 10 pairs of the five
EAs: the yes/no measures get Cochran's Q (do the arrival rates differ at all?)
and exact McNemar tests per pair; the others get `analyze.paired_comparisons`,
D25's rule (paired t-test if the differences are bell-shaped, else Wilcoxon).
Reaching is tested as the share of arenas NOT reached, so that lower is better
everywhere, as in `analyze.py`. The ranking gives every condition its mean rank
per seed (1 = best of the five) and how often it was the best and the worst.
"""

# Standard library
import argparse
import json
from itertools import combinations
from pathlib import Path

# Third-party libraries
import pandas as pd
from scipy.stats import binomtest, chi2

# Local libraries
from analyze import (
    T_TEST,
    best_so_far,
    format_p,
    holm,
    normality_checks,
    normality_report,
    pair_differences,
    paired_comparisons,
    t_test_report,
    wilcoxon_report,
)
from simulate import FAILED_SCORE

BINARY = ("own_reached_15", "own_reached_30")
# Lower is better for every tested measure; reaching is tested as "missed".
CONTINUOUS = ("final_fitness", "unseen_missed_30", "unseen_fitness")
LABELS = {
    "final_fitness": "final fitness",
    "own_reached_15": "own target in 15 s",
    "own_reached_30": "own target in 30 s",
    "unseen_reached_30": "unseen arenas reached in 30 s",
    "unseen_missed_30": "unseen arenas missed in 30 s",
    "unseen_fitness": "unseen fitness (15 s)",
    "unseen_failed_30": "unseen walks that fell off, 30 s",
}


def run_measures(run: Path) -> dict[str, float]:
    """The end-result measures of one run (see the module docstring)."""
    walks = json.loads((run / "longer_walks.json").read_text())
    unseen = json.loads((run / "unseen.json").read_text())["unseen"]
    unseen_30 = json.loads((run / "unseen_30s.json").read_text())["unseen"]
    reached_30 = float(unseen_30["reached"])
    return {
        # Every run of experiment 99 has the same budget, so its best at the
        # end of the curve is its best at the budget.
        "final_fitness": float(best_so_far(run)["fitness"].iloc[-1]),
        "own_reached_15": float(walks["15"]["arrived_at"] is not None),
        "own_reached_30": float(walks["30"]["arrived_at"] is not None),
        "unseen_reached_30": reached_30,
        "unseen_missed_30": 1.0 - reached_30,
        "unseen_fitness": float(unseen["fitness_mean"]),
        "unseen_failed_30": float(
            sum(distance >= FAILED_SCORE.distance for distance in unseen_30["distance"])
        ),
        "unseen_walks_30": float(len(unseen_30["distance"])),
    }


def cochran_q(blocks: pd.DataFrame) -> tuple[float, float]:
    """Cochran's Q test for k paired yes/no outcomes: (Q, p).

    `blocks` has one row per seed and one 0/1 column per condition. Q compares
    the conditions' success counts, using only the seeds where they disagree;
    under the null hypothesis it follows chi-squared with k - 1 degrees of
    freedom. p is 1 when every seed has the same outcome in every condition.
    """
    k = blocks.shape[1]
    column_totals = blocks.sum(axis=0).to_numpy()
    row_totals = blocks.sum(axis=1).to_numpy()
    total = row_totals.sum()
    denominator = k * total - (row_totals**2).sum()
    if denominator == 0:
        return 0.0, 1.0
    q = (k - 1) * (k * (column_totals**2).sum() - total**2) / denominator
    return float(q), float(chi2.sf(q, k - 1))


def mcnemar_exact(a: pd.Series, b: pd.Series) -> float:
    """Two-sided exact McNemar test of two paired yes/no outcomes.

    Only the seeds where exactly one of the two reached the target count; under
    the null hypothesis each such seed is equally likely to favour either, so
    the count favouring A is binomial(n, 1/2).
    """
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    if a_only + b_only == 0:
        return 1.0
    return float(binomtest(a_only, a_only + b_only, 0.5).pvalue)


def ranking(table: pd.DataFrame, measure: str, higher_is_better: bool) -> pd.DataFrame:
    """Per condition: mean rank per seed (1 = best), and seeds best and worst on."""
    wide = table.pivot(index="run", columns="condition", values=measure)
    wide = wide[list(dict.fromkeys(table["condition"]))].dropna()
    ranks = (-wide if higher_is_better else wide).rank(axis=1, method="average")
    worst = len(wide.columns)
    return pd.DataFrame(
        {
            "mean rank": ranks.mean(),
            "best on": (ranks == 1).sum(),
            "worst on": (ranks == worst).sum(),
        }
    )


def report(table: pd.DataFrame) -> str:
    """The end-result analysis as Markdown."""
    conditions = list(dict.fromkeys(table["condition"]))
    seeds = table["run"].nunique()
    groups = table.groupby("condition", sort=False)
    lines = [
        "# End results: which algorithm is best if compute is no concern?",
        "",
        "Written by `end_results.py`. **Exploratory** (D26): these measures were",
        "chosen after experiment 99's results were known. Lower is better except for",
        "the reaching measures. Paired by seed, Holm-corrected across the "
        f"{len(conditions) * (len(conditions) - 1) // 2} pairs.",
        "",
        "## Per condition",
        "",
        "| Condition | Final fitness | Own target, 15 s | Own target, 30 s "
        "| Unseen reached, 30 s | Unseen fitness | Unseen walks fell off, 30 s |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition, group in groups:
        lines.append(
            f"| {condition} "
            f"| {group['final_fitness'].mean():.3f} ± {group['final_fitness'].std():.3f} "
            f"| {int(group['own_reached_15'].sum())} of {len(group)} "
            f"| {int(group['own_reached_30'].sum())} of {len(group)} "
            f"| {group['unseen_reached_30'].mean():.1%} "
            f"(best seed {group['unseen_reached_30'].max():.0%}) "
            f"| {group['unseen_fitness'].mean():.3f} ± {group['unseen_fitness'].std():.3f} "
            f"| {int(group['unseen_failed_30'].sum())} of "
            f"{int(group['unseen_walks_30'].sum())} |"
        )

    lines += ["", "## Ranking per seed (1 = best of the five)", ""]
    measures = (
        ("final_fitness", False),
        ("unseen_reached_30", True),
        ("unseen_fitness", False),
    )
    header = "| Condition | " + " | ".join(LABELS[m] for m, _ in measures) + " |"
    lines += [header, "|---|" + "---|" * len(measures)]
    ranks = {m: ranking(table, m, higher) for m, higher in measures}
    for condition in conditions:
        cells = [
            f"{ranks[m].loc[condition, 'mean rank']:.2f} "
            f"(best {int(ranks[m].loc[condition, 'best on'])}, "
            f"worst {int(ranks[m].loc[condition, 'worst on'])})"
            for m, _ in measures
        ]
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Cells: mean rank over the {seeds} seeds (ties share a rank), then on how "
        "many seeds the condition was the best and the worst of the five.",
    ]

    lines += ["", "## Reaching the own target (yes/no per seed)", ""]
    for measure in BINARY:
        blocks = table.pivot(index="run", columns="condition", values=measure)
        blocks = blocks[conditions].dropna()
        q, p = cochran_q(blocks)
        pairs = list(combinations(conditions, 2))
        p_values = [mcnemar_exact(blocks[a], blocks[b]) for a, b in pairs]
        lines += [
            f"### {LABELS[measure]}",
            "",
            f"Cochran's Q = {q:.2f}, p = {format_p(p)} (do the arrival rates differ "
            f"at all? {len(blocks)} seeds). Exact McNemar per pair:",
            "",
            "| A vs B | A only | B only | p | p (Holm) |",
            "|---|---|---|---|---|",
        ]
        for (a, b), p_pair, p_holm in zip(pairs, p_values, holm(p_values), strict=True):
            a_only = int(((blocks[a] == 1) & (blocks[b] == 0)).sum())
            b_only = int(((blocks[a] == 0) & (blocks[b] == 1)).sum())
            lines.append(
                f"| {a} vs {b} | {a_only} | {b_only} | {format_p(p_pair)} "
                f"| {format_p(p_holm)} |"
            )
        lines.append("")

    lines += ["## Continuous measures (D25's rule)", ""]
    for measure in CONTINUOUS:
        tests = paired_comparisons(table, measure)
        if not tests:
            continue
        lines += [f"### {LABELS[measure]}", ""]
        lines += [*normality_report(normality_checks(pair_differences(table, measure)))]
        lines.append("")
        if tests[0].test == T_TEST:
            lines += t_test_report(tests)
        else:
            lines += wilcoxon_report(tests, seeds)
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Read every given condition's runs and write the end-result report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("conditions", type=Path, nargs="+")
    parser.add_argument(
        "--out", type=Path, default=Path("results/final/end_results.md")
    )
    args = parser.parse_args()
    rows = [
        {"condition": folder.name, "run": run.name, **run_measures(run)}
        for folder in args.conditions
        for run in sorted(folder.glob("seed*"))
        if (run / "summary.json").exists()
    ]
    text = report(pd.DataFrame(rows))
    args.out.write_text(text)
    print(text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
