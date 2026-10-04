"""Give the evolved brains more time: do they reach the target? (decision D12)

    uv run --project ../ariel python longer_walks.py results/final/best \\
        results/final/none ... --out results/final/longer_walks

The brains are trained on 15 s walks. This walks each run's best brain on its
own training arena for every `--durations` length (default 15, 20, 30 and 60
s), ending a walk when the target is reached, and records whether and when it
arrived. It writes `longer_walks.json` into each run folder, and with `--out`
a markdown summary and a figure:

    A  brains that reach the target on their own arena, against walk length
    B  the share of walks that reach the target on the 20 unseen arenas, at
       the training length and at 30 s, from `unseen.json` and
       `unseen_30s.json` (run `unseen.py --duration 30` first)

Each argument is a condition folder holding `seed*/` run folders.
"""

# Standard library
import argparse
import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")  # render to a file; no window
import matplotlib.pyplot as plt
import numpy as np

# Local libraries
from analyze import CONDITION_COLOURS, FALLBACK_COLOURS, SURFACE, TEXT, style_axis
from simulate import TARGET_RADIUS, Score, evaluate_task, final_sim_config

DEFAULT_DURATIONS = (15.0, 20.0, 30.0, 60.0)
LONG_TEST = 30.0  # the unseen-arena test length shown next to the training length


def arrival(score: Score) -> float | None:
    """When the walk reached the target (s), or None if it never did.

    With `stop_at_target` the walk ends on arrival, so the last distance is
    within TARGET_RADIUS exactly when the target was reached.
    """
    return score.seconds if score.distance < TARGET_RADIUS else None


def walk_longer(
    runs: list[Path], durations: tuple[float, ...], workers: int
) -> dict[Path, dict[str, object]]:
    """Every run's best brain on its own arena, once per duration.

    A run whose `longer_walks.json` already holds every duration is not walked
    again: the simulation is deterministic, so the result would be the same.
    """
    results: dict[Path, dict[str, object]] = {}
    tasks, keys = [], []
    for run in runs:
        saved = run / "longer_walks.json"
        if saved.exists():
            previous = json.loads(saved.read_text())
            if all(f"{d:g}" in previous for d in durations):
                results[run] = previous
                continue
        results[run] = {}
        config = final_sim_config(run)
        terrains = tuple(json.loads((run / "config.json").read_text())["terrains"])
        genotype = np.load(run / "best_genotype.npy").tolist()
        for duration in durations:
            walk = replace(config, duration=duration, stop_at_target=True)
            tasks.append((genotype, terrains, walk, None))
            keys.append((run, duration))
    scores = []
    if tasks:
        with mp.get_context("spawn").Pool(workers) as pool:
            scores = pool.map(evaluate_task, tasks)
    for (run, duration), score in zip(keys, scores, strict=True):
        results[run][f"{duration:g}"] = {
            "distance": score.distance,
            "seconds": score.seconds,
            "arrived_at": arrival(score),
        }
    return results


def unseen_reached(run: Path, name: str) -> float | None:
    """Share of the unseen walks that reached the target, if that test ran."""
    path = run / name
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["unseen"]["reached"])


def summary(
    conditions: dict[str, list[Path]],
    results: dict[Path, dict[str, object]],
    durations: tuple[float, ...],
) -> str:
    """Markdown: per condition, arrivals per walk length and on unseen arenas."""
    keys = [f"{d:g}" for d in durations]
    lines = [
        "# Longer walks",
        "",
        "Written by `longer_walks.py`. The brains were trained on 15 s walks; here each",
        "run's best brain walks its own arena for longer, ending on arrival (within",
        f"{TARGET_RADIUS:g} m of the target). Unseen: the share of the walks on the 20 unseen",
        "arenas that reached the target, at 15 s and at 30 s.",
        "",
        "| Condition | "
        + " | ".join(f"Reached in {k} s" for k in keys)
        + " | Unseen, 15 s | Unseen, 30 s |",
        "|---|" + "---|" * (len(keys) + 2),
    ]
    totals = dict.fromkeys(keys, 0)
    count = 0
    for condition, runs in conditions.items():
        cells = []
        for key in keys:
            arrived = sum(results[run][key]["arrived_at"] is not None for run in runs)
            totals[key] += arrived
            cells.append(f"{arrived} of {len(runs)}")
        count += len(runs)
        unseen = []
        for name in ("unseen.json", f"unseen_{LONG_TEST:g}s.json"):
            shares = [unseen_reached(run, name) for run in runs]
            known = [s for s in shares if s is not None]
            unseen.append(f"{100 * np.mean(known):.1f}%" if known else "—")
        lines.append(f"| {condition} | " + " | ".join(cells + unseen) + " |")
    lines.append(
        "| **all** | "
        + " | ".join(f"**{totals[k]} of {count}**" for k in keys)
        + " | | |"
    )
    lines += [
        "",
        "Arrival times (s) of the brains that reach the target within the longest walk:",
        "",
    ]
    for condition, runs in conditions.items():
        times = [
            results[run][keys[-1]]["arrived_at"]
            for run in runs
            if results[run][keys[-1]]["arrived_at"] is not None
        ]
        listed = ", ".join(f"{t:.1f}" for t in sorted(times)) or "none"
        lines.append(f"- {condition}: {listed}")
    return "\n".join(lines) + "\n"


def figure(
    conditions: dict[str, list[Path]],
    results: dict[Path, dict[str, object]],
    durations: tuple[float, ...],
    path: Path,
) -> None:
    """Arrivals against walk length (A) and on unseen arenas (B)."""
    keys = [f"{d:g}" for d in durations]
    fallback = iter(FALLBACK_COLOURS)
    colours = {c: CONDITION_COLOURS.get(c) or next(fallback) for c in conditions}
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(SURFACE)
    # A: grouped bars, one group per walk length, one bar per condition
    names = list(conditions)
    bar = 0.8 / len(names)
    totals = np.zeros(len(keys), dtype=int)
    for i, condition in enumerate(names):
        arrived = np.array(
            [
                sum(
                    results[run][key]["arrived_at"] is not None
                    for run in conditions[condition]
                )
                for key in keys
            ]
        )
        totals += arrived
        left.bar(
            np.arange(len(keys)) + (i - (len(names) - 1) / 2) * bar,
            arrived,
            width=bar * 0.92,
            color=colours[condition],
            label=condition,
        )
    runs_total = sum(len(runs) for runs in conditions.values())
    style_axis(left)
    left.set_xticks(
        range(len(keys)),
        [
            f"{key} s"
            + (" (training)" if float(key) == 15 else "")
            + f"\nall: {total} of {runs_total}"
            for key, total in zip(keys, totals, strict=True)
        ],
    )
    left.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    left.set_ylim(0, max(len(r) for r in conditions.values()) + 0.8)
    left.set_xlabel("walk length", color=TEXT)
    left.set_ylabel("brains that reach the target on their own arena", color=TEXT)
    left.set_title("A  More time on the training arena", loc="left", color=TEXT)
    left.legend(frameon=False, fontsize=8, ncols=3, loc="upper left")

    width = 0.38
    for i, name in enumerate(("unseen.json", f"unseen_{LONG_TEST:g}s.json")):
        shares = []
        for condition in names:
            known = [
                s
                for s in (unseen_reached(r, name) for r in conditions[condition])
                if s is not None
            ]
            shares.append(100 * np.mean(known) if known else 0.0)
        right.bar(
            np.arange(len(names)) + (i - 0.5) * width,
            shares,
            width=width * 0.95,
            color=[colours[c] for c in names],
            alpha=0.45 if i == 0 else 1.0,
            label="15 s (training length)" if i == 0 else f"{LONG_TEST:g} s",
        )
    style_axis(right)
    right.set_xticks(range(len(names)), names, fontsize=8)
    right.set_ylabel("walks that reach the target, %", color=TEXT)
    right.set_title(
        "B  20 unseen arenas: light 15 s, dark 30 s", loc="left", color=TEXT
    )
    fig.suptitle(
        "Trained on 15 s walks, tested with more time", color=TEXT, x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    """Walk every given run longer; optionally write the summary and figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("conditions", type=Path, nargs="+")
    parser.add_argument(
        "--durations", type=float, nargs="+", default=list(DEFAULT_DURATIONS)
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, help="summary .md and figure .png stem")
    args = parser.parse_args()
    if any(d <= 0 for d in args.durations):
        parser.error("--durations must be positive")
    durations = tuple(sorted(args.durations))

    conditions = {
        folder.name: sorted(
            r for r in folder.glob("seed*") if (r / "best_genotype.npy").exists()
        )
        for folder in args.conditions
    }
    conditions = {name: runs for name, runs in conditions.items() if runs}
    if not conditions:
        parser.error("no seed*/best_genotype.npy found in the given folders")
    runs = [run for runs in conditions.values() for run in runs]
    results = walk_longer(runs, durations, args.workers)
    for run, by_duration in results.items():
        (run / "longer_walks.json").write_text(json.dumps(by_duration, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".md").write_text(summary(conditions, results, durations))
        figure(conditions, results, durations, args.out.with_suffix(".png"))
        print(f"written to {args.out.with_suffix('.md')} and .png")


if __name__ == "__main__":
    main()
