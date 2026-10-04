"""The paper's numbers that no other script prints.

    uv run --project ../ariel python paper_numbers.py

Every other number in the report comes from a Markdown file written by
`analyze.py`, `probabilities.py`, `longer_walks.py` or `end_results.py`, or is
printed by `paper_figures.py` (the fitness-term means). These few are
computed here from the same per-run files:

    the closest random search came to the target, on its best seed
    how often one champion held all four islands, per policy (final experiment)
    how many runs still improved after 9,000 evaluations (the plateau limitation)
    the standard EA's early lag behind no migration (Discussion)
    how many of the EAs' unseen-arena walks arrive within 15 s (Table 4)
    the power of a paired t-test at the Holm-corrected level for d_z = 0.6
    how much of the preliminary study's final-fitness variance the seed (the
        arena) and the condition explain (why D25 tests normality first)
"""

# Standard library
import argparse
import json
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
from scipy import stats

# Local libraries
from analyze import best_so_far
from paper_figures import shared_champion_share
from simulate import TARGET_RADIUS

EAS = ("best", "worst", "random", "none", "standard")
POLICIES = ("best", "worst", "random", "none")
PAIRS = 15  # pairs of the six conditions, the Holm family of each metric
LATE = 9_000  # evaluations after which the plateau limitation counts improvements
SECONDS = 15.0  # the walk length of the unseen test


def runs(folder: Path) -> list[Path]:
    """The finished runs of one condition."""
    return sorted(p.parent for p in folder.glob("seed*/summary.json"))


def paired_t_power(n: int, d_z: float, alpha: float) -> float:
    """Power of a two-sided paired t-test with `n` pairs and effect size `d_z`."""
    df = n - 1
    noncentrality = d_z * np.sqrt(n)
    critical = stats.t.ppf(1 - alpha / 2, df)
    return float(
        stats.nct.sf(critical, df, noncentrality)
        + stats.nct.cdf(-critical, df, noncentrality)
    )


def variance_shares(table: pd.DataFrame) -> tuple[float, float]:
    """Shares of the total sum of squares explained by seed and by condition.

    `table` has one row per seed and one column per condition (a complete
    two-way layout without replication).
    """
    grand = table.to_numpy().mean()
    total = ((table - grand) ** 2).to_numpy().sum()
    seeds = table.shape[1] * ((table.mean(axis=1) - grand) ** 2).sum()
    conditions = table.shape[0] * ((table.mean(axis=0) - grand) ** 2).sum()
    return float(seeds / total), float(conditions / total)


def final_fitness_table(folder: Path) -> pd.DataFrame:
    """Final best fitness, one row per seed, one column per EA condition."""
    return pd.DataFrame(
        {
            condition: {
                run.name: best_so_far(run)["fitness"].iloc[-1]
                for run in runs(folder / condition)
            }
            for condition in EAS
        }
    ).dropna()


def early_lag(final: Path, until: int = 20) -> tuple[int, int, int, int, int]:
    """The standard EA against no migration before the first migration.

    Returns: the seeds on which the standard EA's best fitness so far trails
    no migration's on average over generations 0-`until`; the number of
    seeds; the number of no-migration islands whose best at generation 10
    is worse than the standard EA's whole population's; the number of
    islands compared; and the generation from which the mean gap over the
    seeds stays at or below zero (the standard EA has caught up).
    """
    behind = seeds = worse_islands = islands = 0
    gaps = []
    for run in runs(final / "standard"):
        twin = final / "none" / run.name
        if not (twin / "summary.json").exists():
            continue
        standard = pd.read_csv(run / "log.csv", dtype={"island": str})
        none = pd.read_csv(twin / "log.csv", dtype={"island": str})
        gap = best_so_far(run)["fitness"] - best_so_far(twin)["fitness"]
        gaps.append(gap.to_numpy())
        seeds += 1
        behind += gap.iloc[: until + 1].mean() > 0
        at_ten = standard[
            (standard["island"] == "all") & (standard["generation"] == 10)
        ]
        rivals = none[(none["island"] != "all") & (none["generation"] == 10)]
        worse_islands += int((rivals["best"] > at_ten["best"].iloc[0]).sum())
        islands += len(rivals)
    mean_gap = np.mean(gaps, axis=0)
    ahead = np.flatnonzero(mean_gap > 0)
    caught_up = int(ahead[-1]) + 1 if len(ahead) else 0
    return behind, seeds, worse_islands, islands, caught_up


def early_unseen_arrivals(final: Path) -> tuple[int, int]:
    """The EAs' unseen-arena walks of 15 s that reach the target, and all of them."""
    arrived = walks = 0
    for condition in EAS:
        for run in runs(final / condition):
            scores = json.loads((run / "unseen.json").read_text())["unseen"]["scores"]
            # A walk ends early on arrival or when the robot falls off (then
            # it is scored as FAILED_SCORE, 10 m away); only arrivals count.
            arrived += sum(
                score["seconds"] < SECONDS and score["distance"] < TARGET_RADIUS
                for score in scores
            )
            walks += len(scores)
    return arrived, walks


def main() -> None:
    """Print the numbers, each with the place in the paper that uses it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", type=Path, default=Path("results/final"))
    parser.add_argument("--olympic", type=Path, default=Path("results/olympic"))
    args = parser.parse_args()

    closest = [
        best_so_far(run)["distance"].iloc[-1]
        for run in runs(args.final / "random_search")
    ]
    print(
        f"random search, closest to the target: mean {np.mean(closest):.2f} m, "
        f"best seed {min(closest):.2f} m (Section 4.2)"
    )

    print("one champion on all four islands, share of generations (Discussion):")
    for policy in POLICIES:
        shares = [
            shared_champion_share(pd.read_csv(run / "log.csv"))
            for run in runs(args.final / policy)
        ]
        print(f"    {policy:8s} {100 * np.mean(shares):.1f}%")

    print(f"runs still improving after {LATE:,} evaluations (Limitations):")
    for condition in EAS:
        improved = 0
        for run in runs(args.final / condition):
            curve = best_so_far(run)
            before = curve.loc[curve["evaluations"] <= LATE, "fitness"].iloc[-1]
            improved += curve["fitness"].iloc[-1] < before
        print(f"    {condition:8s} {improved} of {len(runs(args.final / condition))}")

    behind, seeds, worse, islands, caught_up = early_lag(args.final)
    print(
        f"standard EA behind no migration over generations 0-20 on {behind} of "
        f"{seeds} seeds; at generation 10 its population beat {worse} of "
        f"{islands} no-migration islands; level or ahead on average from "
        f"generation {caught_up} on (Discussion)"
    )

    arrived, walks = early_unseen_arrivals(args.final)
    print(
        f"unseen walks of the EAs that arrive within 15 s: {arrived} of {walks} (Table 4)"
    )

    seeds = len(runs(args.final / "best"))
    alpha = 0.05 / PAIRS
    needed = next(n for n in range(5, 500) if paired_t_power(n, 0.6, alpha) >= 0.8)
    print(
        f"power for d_z = 0.6 at alpha = 0.05/{PAIRS}: "
        f"{paired_t_power(seeds, 0.6, alpha):.0%} with {seeds} seeds; "
        f"80% needs {needed} (Limitations)"
    )

    seed_share, condition_share = variance_shares(final_fitness_table(args.olympic))
    print(
        f"preliminary study, final fitness of the five EAs: the seed explains "
        f"{seed_share:.0%} of the variance, the condition {condition_share:.0%} "
        f"(Section 3, Statistics)"
    )


if __name__ == "__main__":
    main()
