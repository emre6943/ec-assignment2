"""How sure are we that one condition beats another? Numbers for the report (D3).

    uv run --project ../ariel python probabilities.py results/olympic/best \\
        results/olympic/worst ... --out results/olympic/probabilities.md

Each argument is a condition folder holding `seed*/` run folders. For three
metrics - the best fitness at the common budget, the area under the curve
(AUC, lower means good solutions were found earlier) and the distance left on
unseen arenas (when `unseen.py` has run) - it reports:

- per condition: mean ± standard deviation, and a 95% interval for the mean;
- for every pair of conditions A and B, paired by seed (a seed walks the same
  arena in every condition, D10):
    - the mean difference A - B with its 95% interval;
    - P(A better): the probability that A's true mean is lower than B's, from
      a Bayesian paired t-test with a flat prior. The posterior of the mean
      difference is then a Student t with n - 1 degrees of freedom, centred
      on the observed mean difference with its standard error as the scale
      (Benavoli et al. 2017). It is split three ways, with a region of
      practical equivalence of +-`--rope`: A better, practically equal, B
      better;
    - A12 (Vargha & Delaney 2000): the chance that a random run of A beats a
      random run of B, ignoring the pairing;
    - the seeds A won out of n.

Lower is better for all three metrics. The probabilities are per pair and not
corrected for the number of pairs: six conditions make 15 pairs, so some high
values arise by chance alone.
"""

# Standard library
import argparse
import json
from itertools import permutations
from pathlib import Path

# Third-party libraries
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import t as student_t

# Local libraries
from analyze import GRID_POINTS, best_so_far, run_metrics, warn_about_short_runs

METRICS = {
    "final_fitness": "best fitness at the budget",
    "auc": "AUC (mean best-so-far fitness over the budget)",
    "unseen_distance": "distance left on 20 unseen arenas (m)",
}
DEFAULT_ROPE = 0.05  # fitness units, about metres: 5 cm counts as "the same"


def run_table(conditions: list[Path]) -> pd.DataFrame:
    """One row per run: condition, seed and every metric, at a common budget."""
    curves = {
        folder.name: {run.name: best_so_far(run) for run in runs_of(folder)}
        for folder in conditions
    }
    curves = {name: runs for name, runs in curves.items() if runs}
    budget = min(
        c["evaluations"].iloc[-1] for runs in curves.values() for c in runs.values()
    )
    grid = np.linspace(0, budget, GRID_POINTS)
    rows = []
    for folder in conditions:
        for seed, curve in curves.get(folder.name, {}).items():
            metrics = run_metrics(curve, grid, threshold=float("-inf"))
            row = {"condition": folder.name, "seed": seed}
            row.update({key: metrics[key] for key in ("final_fitness", "auc")})
            unseen = folder / seed / "unseen.json"
            if unseen.exists():
                result = json.loads(unseen.read_text())["unseen"]
                row["unseen_distance"] = result["distance_mean"]
            rows.append(row)
    return pd.DataFrame(rows)


def runs_of(folder: Path) -> list[Path]:
    """The run folders (`seed*/` with a `log.csv`) of one condition."""
    return sorted(log.parent for log in folder.glob("seed*/log.csv"))


def mean_interval(values: npt.NDArray) -> tuple[float, float, float, float]:
    """Mean, standard deviation and the 95% t-interval of the mean.

    With a single value the standard deviation is 0 and the interval undefined
    (NaN at both ends).
    """
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, 0.0, np.nan, np.nan
    sd = float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return mean, sd, mean - half, mean + half


def paired_posterior(differences: npt.NDArray, rope: float) -> dict[str, float]:
    """Bayesian paired t-test on A - B (lower is better): the posterior of the
    mean difference is Student t(n - 1, mean, sd / sqrt(n)) under a flat prior.
    """
    n = len(differences)
    mean = float(differences.mean())
    scale = float(differences.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if scale == 0.0:  # every seed gave the same difference: no spread to model
        # The limit of the t posterior as its scale shrinks to 0: certain on
        # either side of 0, and a coin flip for an exact tie.
        a_better = float(mean < -rope)
        b_better = float(mean > rope)
        return {
            "mean_difference": mean,
            "low": mean,
            "high": mean,
            "p_a_better_at_all": 0.5 if mean == 0.0 else float(mean < 0),
            "p_a_better": a_better,
            "p_equal": 1.0 - a_better - b_better,
            "p_b_better": b_better,
        }
    posterior = student_t(n - 1, loc=mean, scale=scale)
    a_better = float(posterior.cdf(-rope))
    b_better = float(posterior.sf(rope))
    return {
        "mean_difference": mean,
        "low": float(posterior.ppf(0.025)),
        "high": float(posterior.ppf(0.975)),
        "p_a_better_at_all": float(posterior.cdf(0.0)),
        "p_a_better": a_better,
        "p_equal": 1.0 - a_better - b_better,
        "p_b_better": b_better,
    }


def a12(a: npt.NDArray, b: npt.NDArray) -> float:
    """Vargha-Delaney A12 for "lower is better": P(a random a < a random b)."""
    wins = (a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    return float(wins / (len(a) * len(b)))


def pair_rows(table: pd.DataFrame, metric: str, rope: float) -> list[dict[str, object]]:
    """Every ordered pair of conditions on one metric, paired by seed.

    Only the seeds both conditions have a value for count (an unseen test may be
    missing for some runs); a pair with fewer than 2 such seeds is left out. The
    pairs come in the order the conditions first appear in `table`.
    """
    rows = []
    wide = table.pivot(index="seed", columns="condition", values=metric)
    wide = wide[list(dict.fromkeys(table["condition"]))]
    for a, b in permutations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        if len(both) < 2:
            continue
        differences = (both[a] - both[b]).to_numpy()
        row = {"metric": metric, "a": a, "b": b, "seeds": len(both)}
        row.update(paired_posterior(differences, rope))
        row["a12"] = a12(both[a].to_numpy(), both[b].to_numpy())
        row["seeds_a_won"] = int((differences < 0).sum())
        rows.append(row)
    return rows


def percent(value: float) -> str:
    """A probability as a whole percentage."""
    return f"{100 * value:.0f}%"


def report(table: pd.DataFrame, pairs: pd.DataFrame, rope: float) -> str:
    """The markdown for the report: summaries, probability matrices, pair details."""
    lines = [
        "# How sure are we?",
        "",
        "Written by `probabilities.py`. Lower is better for every metric. Seeds are",
        "paired (same arena in every condition). P(A better) comes from a Bayesian paired",
        "t-test with a flat prior (Benavoli et al. 2017); 'practically equal' means the",
        f"mean difference lies within ±{rope:g}. A12 is the chance that a random run of A",
        "beats a random run of B (Vargha & Delaney 2000). The probabilities are per pair",
        "and not corrected for the number of pairs.",
    ]
    conditions = list(dict.fromkeys(table["condition"]))
    for metric, title in METRICS.items():
        if metric not in table or table[metric].isna().all():
            continue
        lines += [
            "",
            f"## {title}",
            "",
            "| Condition | Runs | Mean ± sd | 95% interval of the mean |",
        ]
        lines.append("|---|---|---|---|")
        for condition in conditions:
            values = (
                table.loc[table["condition"] == condition, metric].dropna().to_numpy()
            )
            if len(values) == 0:
                continue
            mean, sd, low, high = mean_interval(values)
            lines.append(
                f"| {condition} | {len(values)} | {mean:.3f} ± {sd:.3f} | {low:.3f} to {high:.3f} |"
            )

        chosen = pairs[pairs["metric"] == metric]
        present = [c for c in conditions if c in set(chosen["a"])]
        lines += [
            "",
            "Probability that the **row** has a lower (better) true mean than the **column**:",
            "",
            "| | " + " | ".join(present) + " |",
            "|---|" + "---|" * len(present),
        ]
        for a in present:
            cells = []
            for b in present:
                match = chosen[(chosen["a"] == a) & (chosen["b"] == b)]
                cells.append(
                    "—" if match.empty else percent(match["p_a_better_at_all"].iloc[0])
                )
            lines.append(f"| **{a}** | " + " | ".join(cells) + " |")

        lines += [
            "",
            "| A vs B | Mean difference A − B (95% interval) | A better | Practically equal | B better | A12 | Seeds A won |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, row in chosen.iterrows():
            if conditions.index(row["a"]) > conditions.index(row["b"]):
                continue  # each pair once; the reverse is the mirror image
            lines.append(
                f"| {row['a']} vs {row['b']} | {row['mean_difference']:+.3f} "
                f"({row['low']:+.3f} to {row['high']:+.3f}) | {percent(row['p_a_better'])} | "
                f"{percent(row['p_equal'])} | {percent(row['p_b_better'])} | "
                f"{row['a12']:.2f} | {row['seeds_a_won']} of {row['seeds']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the probabilities for the given conditions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("conditions", type=Path, nargs="+")
    parser.add_argument("--rope", type=float, default=DEFAULT_ROPE)
    parser.add_argument("--out", type=Path, default=Path("results/probabilities.md"))
    args = parser.parse_args()
    if args.rope < 0:
        parser.error("--rope must be 0 or more")
    names = [folder.name for folder in args.conditions]
    if len(set(names)) != len(names):
        parser.error(f"condition folders must have different names, got {names}")

    table = run_table(args.conditions)
    if table.empty:
        parser.error("no seed*/log.csv found in the given folders")
    warn_about_short_runs({folder.name: runs_of(folder) for folder in args.conditions})
    pairs = pd.DataFrame(
        [
            row
            for metric in METRICS
            if metric in table and not table[metric].isna().all()
            for row in pair_rows(table, metric, args.rope)
        ]
    )
    if pairs.empty:
        parser.error("no two conditions share at least 2 seeds; nothing to compare")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report(table, pairs, args.rope))
    table.to_csv(args.out.with_suffix(".runs.csv"), index=False)
    pairs.to_csv(args.out.with_suffix(".pairs.csv"), index=False)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
