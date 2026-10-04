"""Test each run's best network on terrain it never saw (decision D14).

    uv run --project ../ariel python unseen.py results/final/best/seed10 ...
    uv run --project ../ariel python unseen.py RUN --duration 30
    uv run --project ../ariel python unseen.py RUN --yaws 0 30 -30 --duration 20

Every run evolves on its seed's training terrain. This asks the robustness
question: does the evolved brain still walk on new ground? For each world and
body, the first call builds `N_TEST_TERRAINS` fresh copies of the run's own
world (a new random terrain, or a new rugged strip for OlympicArena) into
`results/terrains/<world>/test/<body>/` (spider_16 on rugged keeps
`results/terrains/rugged/test/`); every later call reuses them, so all runs
of a body and world are tested on exactly the same unseen ground. Runs on
SimpleFlatWorld are skipped: it is the same on every build.

`--duration` overrides the walk length, so brains trained with different walk
lengths can be tested on the same one; the final experiment also tests at
30 s. `--yaws` (experiment 21 only, not in the paper) also starts the robot
turned away from the target (decision D23): each turn gets its own
`N_TEST_TERRAINS` test arenas, so `--yaws 0 30 -30` means 60 unseen walks.

For each run it writes `unseen.json` next to the run's other files, or, with
`--yaws` or `--duration`, a file named after them (`unseen_yaws0_30_20s.json`),
so a special test never overwrites the standard one that `analyze.py` reads:

    training   the score on the run's own training terrain(s)
    unseen     the score on each test terrain, and their mean and spread
    yaws, duration   the turns (one per test terrain) and walk length used

Each part also holds `reached`: the share of walks that ended at the target.
"""

# Standard library
import argparse
import json
import multiprocessing as mp
from dataclasses import asdict, replace
from pathlib import Path

# Third-party libraries
import numpy as np

# Local libraries (ARIEL)
from ariel.simulation.environments import RuggedTerrainWorld, SimpleFlatWorld

# Local libraries
from bodies import FIRST_BODY
from run import WORLDS
from simulate import (
    TARGET_RADIUS,
    Score,
    SimConfig,
    evaluate_task,
    final_sim_config,
    fitness,
    run_terrains,
)

TERRAIN_DIR = Path(__file__).parent / "results" / "terrains"
N_TEST_TERRAINS = 20


def test_terrain_dir(world: str, body: str) -> Path:
    """Where the unseen terrains of one world (a WORLDS key) and body live."""
    folder = TERRAIN_DIR / world / "test"
    if world == "rugged" and body == FIRST_BODY:
        return folder  # where experiments 7-12 keep theirs
    return folder / body


def load_sim_config(run: Path) -> SimConfig | None:
    """The SimConfig of the run's last walks, or None for an older, incompatible run."""
    config_file = run / "config.json"
    if not config_file.exists():
        return None
    try:
        return final_sim_config(run)
    except (KeyError, ValueError):
        return None


def result_file_name(yaws: list[float], duration: float | None) -> str:
    """`unseen.json` for the standard test (D14), else a name that says which."""
    parts = []
    if yaws != [0.0]:
        parts.append("yaws" + "_".join(f"{yaw:g}" for yaw in yaws))
    if duration is not None:
        parts.append(f"{duration:g}s")
    return "_".join(["unseen", *parts]) + ".json"


def summarise(scores: list[Score], config: SimConfig) -> dict[str, object]:
    """Distances and fitness per terrain, plus their mean and standard deviation.

    A walk reached the target if it ended within `TARGET_RADIUS` of it: with
    `stop_at_target` the walk ends on arrival, so its last distance says so.
    """
    distances = [score.distance for score in scores]
    fitnesses = [fitness(score, config) for score in scores]
    return {
        "distance": distances,
        "fitness": fitnesses,
        "reached": float(np.mean([distance < TARGET_RADIUS for distance in distances])),
        "distance_mean": float(np.mean(distances)),
        "distance_std": float(np.std(distances)),
        "fitness_mean": float(np.mean(fitnesses)),
        "fitness_std": float(np.std(fitnesses)),
        "scores": [asdict(score) for score in scores],
    }


def main() -> None:
    """Walk every given run's best genotype on the training and the test terrains."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="run folders (…/seedN)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--duration",
        type=float,
        help="seconds to walk (default: the run's walks; the paper also uses 30)",
    )
    earlier = parser.add_argument_group("earlier experiments only (not in the paper)")
    earlier.add_argument(
        "--yaws",
        type=float,
        nargs="+",
        default=[0.0],
        help="degrees the robot starts turned by; each gets its own test arenas "
        "(experiment 21, D23)",
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if not all(np.isfinite(yaw) and abs(yaw) <= 180 for yaw in args.yaws):
        parser.error("--yaws must lie between -180 and 180 degrees")
    result_name = result_file_name(args.yaws, args.duration)

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for run in args.runs:
            config = load_sim_config(run)
            if config is None or not (run / "best_genotype.npy").exists():
                print(f"{run}: skipped (no config, older format, or no best genotype)")
                continue
            if args.duration is not None:
                config = replace(config, duration=args.duration)
            genotype = np.load(run / "best_genotype.npy").tolist()
            saved = json.loads((run / "config.json").read_text())
            world_class = saved.get("world", RuggedTerrainWorld.__name__)
            if world_class == SimpleFlatWorld.__name__:
                print(f"{run}: skipped (flat ground is the same on every build)")
                continue
            world = next(
                (
                    key
                    for key, factory in WORLDS.items()
                    if factory.__name__ == world_class
                ),
                None,
            )
            if world is None:
                print(f"{run}: skipped (unknown world {world_class})")
                continue
            training = saved["terrains"]
            if not training:
                print(f"{run}: skipped (no fixed training terrain: per_generation run)")
                continue

            test_terrains: list[str] = []
            yaws: list[float] = []
            for yaw in args.yaws:
                test_terrains += run_terrains(
                    test_terrain_dir(world, config.body),
                    N_TEST_TERRAINS,
                    WORLDS[world],
                    config.body,
                    (yaw,),
                    saved.get("ea", {}).get("ariel_spawn", False),
                )
                yaws += [yaw] * N_TEST_TERRAINS
            tasks = [(genotype, (path,), config, None) for path in training]
            tasks += [(genotype, (path,), config, None) for path in test_terrains]
            scores = pool.map(evaluate_task, tasks)

            result = {
                "training": summarise(scores[: len(training)], config),
                "unseen": summarise(scores[len(training) :], config),
                "test_terrains": test_terrains,
                "yaws": yaws,
                "duration": config.duration,
            }
            (run / result_name).write_text(json.dumps(result, indent=2))
            unseen = result["unseen"]
            print(
                f"{run}: training distance "
                f"{result['training']['distance_mean']:.3f} m, unseen "
                f"{unseen['distance_mean']:.3f} ± {unseen['distance_std']:.3f} m, "
                f"reached {unseen['reached']:.0%}"
            )


if __name__ == "__main__":
    main()
