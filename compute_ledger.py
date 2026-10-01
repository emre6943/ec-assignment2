"""Count the compute every experiment used, for the report.

    uv run --project ../ariel python compute_ledger.py

Scans the run folders in `results/` (a run folder holds a `log.csv`), sorts
them into the experiments of `experiments/README.md`, and writes
`docs/compute.md`: per phase and per experiment, the number of runs, fitness
evaluations, walks (an evaluation walks every terrain of its run once) and
simulated and wall-clock hours, plus the walks of the unseen-terrain tests.

Run it again at any time; the numbers only grow as experiments are added.
What it cannot see: runs that were deleted or overwritten by a rerun, smoke
tests, and small diagnostics run from scratch scripts. So every number is a
lower bound.
"""

# Standard library
import csv
import json
from dataclasses import dataclass
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "docs" / "compute.md"

# (experiment, phase, folders under results/ that hold its runs). Phases:
# "development" builds and debugs the problem (world, body, fitness, what the
# brain sees); "tuning" chooses the EA's and the brain's settings by
# controlled comparisons; "final" are the research-question experiments (14,
# and 26's follow-up on the migration interval).
EXPERIMENTS: list[tuple[str, str, list[str]]] = [
    ("1", "development", ["best_per_generation_terrain"]),
    ("2", "development", ["debug_flat"]),
    ("3", "development", ["best_before_posture"]),
    ("4", "development", ["old/best_exp04"]),
    ("5", "development", ["long"]),
    ("6", "development", ["improved"]),
    ("7", "tuning", ["pilot"]),
    (
        "8-9",
        "development",
        ["best", "worst", "random", "none", "standard", "random_search"],
    ),
    ("10", "development", ["overnight"]),
    ("11", "development", ["bodies"]),
    ("12", "tuning", ["no_crossover"]),
    ("13", "development", ["walking"]),
    (
        "14",
        "final",
        [
            f"olympic/{name}"
            for name in ("best", "worst", "random", "none", "standard", "random_search")
        ],
    ),
    ("15", "development", ["olympic_long"]),
    ("16", "development", ["gait_pilot"]),
    ("17", "development", ["gait_pilot_2"]),
    ("18", "development", ["inputs_pilot"]),
    (
        "19",
        "tuning",
        [
            f"tuning/{name}"
            for name in (
                "base",
                "sigma_0.02",
                "sigma_0.1",
                "pop_4x10",
                "pop_4x40",
                "tour_2",
                "tour_5",
            )
        ],
    ),
    (
        "19b",
        "tuning",
        [
            f"tuning/{name}"
            for name in ("elites_1", "elites_5", "sparse_mut", "xover_0", "xover_0.9")
        ],
    ),
    ("20", "development", ["long_walk"]),
    ("21", "development", ["robustness"]),
    ("22", "tuning", ["tuning/xover_0.9_nostall"]),
    ("23", "tuning", [f"tuning/{name}" for name in ("rays3", "rays1", "rays0")]),
    (
        "24",
        "tuning",
        [f"tuning/{name}" for name in ("h8", "h32", "h8_8", "h16_16", "h8_4")],
    ),
    (
        "25",
        "tuning",
        [f"tuning/{name}" for name in ("h8_4_weight", "h8_4_blx", "h8_4_headless")],
    ),
    ("26", "final", [f"olympic/best_int{interval}" for interval in (5, 20, 50)]),
    (
        "29",
        "tuning",
        [
            f"olympic/pos_{name}"
            for name in ("best", "worst", "random", "none", "standard")
        ],
    ),
]
PHASES = ("development", "tuning", "final")


@dataclass
class Tally:
    """Compute of a group of runs."""

    runs: int = 0
    unfinished: int = 0
    evaluations: int = 0
    walks: int = 0
    simulated_s: float = 0.0
    wall_s: float = 0.0
    test_walks: int = 0
    test_simulated_s: float = 0.0

    def add(self, other: "Tally") -> None:
        """Add another tally's numbers to this one."""
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


def run_tally(run: Path) -> Tally:
    """One run's compute, from its config, log and summary.

    Walks and simulated time are counted per generation, so a run whose walks
    got longer halfway (D21) is counted right. The simulated time is nominal:
    walks that ended early at the target were shorter.
    """
    config = json.loads((run / "config.json").read_text())
    n_terrains = int(config["ea"].get("n_terrains", 1))
    default_duration = float(config["sim"].get("duration", 15.0))
    with (run / "log.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["island"] == "all"]
    tally = Tally(runs=1)
    previous = 0
    for row in rows:
        evaluations = int(row["evaluations"])
        duration = float(row.get("duration") or default_duration)
        tally.walks += (evaluations - previous) * n_terrains
        tally.simulated_s += (evaluations - previous) * n_terrains * duration
        previous = evaluations
    tally.evaluations = previous
    summary = run / "summary.json"
    if summary.exists():
        tally.wall_s = float(json.loads(summary.read_text())["seconds"])
    else:
        tally.unfinished = 1
        tally.wall_s = float(rows[-1]["seconds"]) if rows else 0.0
    longer = run / "longer_walks.json"  # longer_walks.py: one walk per length
    if longer.exists():
        walks = json.loads(longer.read_text()).values()
        tally.test_walks += len(walks)
        tally.test_simulated_s += sum(float(walk["seconds"]) for walk in walks)
    for test in run.glob("unseen*.json"):
        result = json.loads(test.read_text())
        walks = len(result["training"]["distance"]) + len(result["unseen"]["distance"])
        tally.test_walks += walks
        tally.test_simulated_s += walks * float(
            result.get("duration", default_duration)
        )
    return tally


def runs_in(folder: Path) -> list[Path]:
    """Every run folder under `folder`, skipping saved terrains."""
    return sorted(
        log.parent
        for log in folder.rglob("log.csv")
        if "terrains" not in log.parts and (log.parent / "config.json").exists()
    )


def hours(seconds: float) -> str:
    """Seconds as hours with one decimal, e.g. "1,234.5"."""
    return f"{seconds / 3600:,.1f}"


def table(rows: list[tuple[str, Tally]], first: str) -> list[str]:
    """A Markdown table of tallies, one row per name; `first` heads the names."""
    lines = [
        f"| {first} | Runs | Evaluations | Walks | Simulated h | Wall-clock h "
        "| Unseen-test walks |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, t in rows:
        runs = f"{t.runs}" + (f" ({t.unfinished} unfinished)" if t.unfinished else "")
        lines.append(
            f"| {name} | {runs} | {t.evaluations:,} | {t.walks:,} | "
            f"{hours(t.simulated_s + t.test_simulated_s)} | {hours(t.wall_s)} | "
            f"{t.test_walks:,} |"
        )
    return lines


def main() -> None:
    """Tally every experiment and write docs/compute.md."""
    if not runs_in(RESULTS):
        # Never overwrite the committed ledger with an empty one.
        raise SystemExit(f"no run folders in {RESULTS}; {OUT} left unchanged")
    per_experiment: list[tuple[str, str, Tally]] = []
    counted: set[Path] = set()
    for experiment, phase, folders in EXPERIMENTS:
        tally = Tally()
        for folder in folders:
            for run in runs_in(RESULTS / folder):
                if run not in counted:
                    counted.add(run)
                    tally.add(run_tally(run))
        if tally.runs:
            per_experiment.append((experiment, phase, tally))
    unassigned = [run for run in runs_in(RESULTS) if run not in counted]

    per_phase = {phase: Tally() for phase in PHASES}
    for _, phase, tally in per_experiment:
        per_phase[phase].add(tally)
    total = Tally()
    for tally in per_phase.values():
        total.add(tally)
    phase_table = table(
        [(phase, per_phase[phase]) for phase in PHASES] + [("**total**", total)],
        "Phase",
    )

    lines = [
        "# Compute used",
        "",
        "Written by `compute_ledger.py` from the run folders in `results/`; rerun it",
        "to update. Lower bounds: runs deleted or overwritten by a rerun, smoke tests",
        "and scratch diagnostics are not counted. An evaluation walks every terrain of",
        "its run once (3 in experiment 21's multi-arena runs, else 1). Simulated hours",
        "are nominal (walks that stopped at the target were shorter) and include the",
        "unseen-terrain tests; wall-clock hours add up the runs, some of which ran in",
        "parallel or were paused.",
        "",
        "## By phase",
        "",
        *phase_table,
        "",
        "- **development**: building and debugging the problem (world, body, fitness,",
        "  inputs, walk length): experiments 1-6, 8-11, 13, 15-18, 20, 21.",
        "- **tuning**: choosing the EA's and the brain's settings by controlled",
        "  comparisons: experiments 7, 12, 19, 19b, 22-25, 29.",
        "- **final**: the research-question experiment 14, and 26's follow-up on the",
        "  migration interval.",
        "",
        "## By experiment",
        "",
        *table([(f"{e} ({p})", t) for e, p, t in per_experiment], "Experiment"),
    ]
    if unassigned:
        lines += ["", "Run folders not assigned to an experiment:", ""]
        lines += [f"- `{run.relative_to(RESULTS)}`" for run in unassigned]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(phase_table))
    if unassigned:
        print(f"{len(unassigned)} run folders are not assigned to an experiment")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
