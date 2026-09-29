"""Test each run's best network on terrain it never saw (decision D14).

    uv run --project ../ariel python unseen.py results/best/seed0 results/none/seed0 ...

Every run evolves on its seed's training terrain. This asks the robustness
question: does the evolved brain still walk on new ground? The first call
generates `N_TEST_TERRAINS` plain `RuggedTerrainWorld()` terrains into
`results/terrains/rugged/test/` (per body for bodies other than spider_16,
since a saved terrain includes the robot); every later call reuses them, so
all runs of a body are tested on exactly the same unseen ground. Runs trained
on another world are skipped: only RuggedTerrainWorld changes between builds.

For each run it writes `unseen.json` next to the run's other files:

    training   the score on the run's own training terrain(s)
    unseen     the score on each test terrain, and their mean and spread
"""

# Standard library
import argparse
import json
import multiprocessing as mp
from dataclasses import asdict, fields
from pathlib import Path

# Third-party libraries
import numpy as np

# Local libraries (ARIEL)
from ariel.simulation.environments import RuggedTerrainWorld

# Local libraries
from bodies import DEFAULT_BODY
from simulate import Score, SimConfig, evaluate_task, fitness, run_terrains

TEST_TERRAIN_DIR = Path(__file__).parent / "results" / "terrains" / "rugged" / "test"
N_TEST_TERRAINS = 20


def load_sim_config(run: Path) -> SimConfig | None:
    """The run's SimConfig, or None if it was made by an older, incompatible version."""
    config_file = run / "config.json"
    if not config_file.exists():
        return None
    saved = json.loads(config_file.read_text())["sim"]
    known = {field.name for field in fields(SimConfig)}
    if set(saved) - known:
        return None
    return SimConfig(**saved)


def summarise(scores: list[Score], config: SimConfig) -> dict[str, object]:
    """Distances and fitness per terrain, plus their mean and standard deviation."""
    distances = [score.distance for score in scores]
    fitnesses = [fitness(score, config) for score in scores]
    return {
        "distance": distances,
        "fitness": fitnesses,
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
    args = parser.parse_args()

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for run in args.runs:
            config = load_sim_config(run)
            if config is None or not (run / "best_genotype.npy").exists():
                print(f"{run}: skipped (no config, older format, or no best genotype)")
                continue
            genotype = np.load(run / "best_genotype.npy").tolist()
            saved = json.loads((run / "config.json").read_text())
            if saved.get("world", RuggedTerrainWorld.__name__) != (
                RuggedTerrainWorld.__name__
            ):
                # The test terrains are fresh RuggedTerrainWorld builds, so they
                # are only "unseen versions of the training world" for rugged runs.
                print(f"{run}: skipped (trained on {saved['world']}, not rugged)")
                continue
            training = saved["terrains"]
            if not training:
                print(f"{run}: skipped (no fixed training terrain: per_generation run)")
                continue

            test_dir = TEST_TERRAIN_DIR
            if config.body != DEFAULT_BODY:
                test_dir = TEST_TERRAIN_DIR / config.body
            test_terrains = run_terrains(
                test_dir, N_TEST_TERRAINS, RuggedTerrainWorld, config.body
            )
            tasks = [(genotype, (path,), config, None) for path in training]
            tasks += [(genotype, (path,), config, None) for path in test_terrains]
            scores = pool.map(evaluate_task, tasks)

            result = {
                "training": summarise(scores[: len(training)], config),
                "unseen": summarise(scores[len(training) :], config),
                "test_terrains": list(test_terrains),
            }
            (run / "unseen.json").write_text(json.dumps(result, indent=2))
            unseen = result["unseen"]
            print(
                f"{run}: training distance "
                f"{result['training']['distance_mean']:.3f} m, unseen "
                f"{unseen['distance_mean']:.3f} ± {unseen['distance_std']:.3f} m"
            )


if __name__ == "__main__":
    main()
