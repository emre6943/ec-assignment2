"""Run one configuration for one or more seeds.

    uv run --project ../ariel python run.py --policy best --seeds 0 1 2 3 4
    uv run --project ../ariel python run.py --algorithm random_search --seeds 0 1 2

Each run writes to results/<condition>/seed<S>/:

    config.json        every setting of the run
    log.csv            per generation: fitness statistics per island and overall
    database.db        ariel.ec's record of every individual that ever lived
    best_genotype.npy  the best network so far (saved on every improvement)
    summary.json       generations, evaluations, best fitness, wall time

For a quick smoke test, shrink everything:

    uv run --project ../ariel python run.py --policy best --seeds 0 \\
        --max-evaluations 150 --island-size 6 --n-elites 1 --n-migrants 1 \\
        --migration-interval 2 --duration 3 --out results/smoke
"""

# Standard library
import argparse
import multiprocessing as mp
import random
from dataclasses import fields
from pathlib import Path

# Third-party libraries
import numpy as np

# Local libraries (ARIEL)
from ariel import console
from ariel.ec import set_seed
from ariel.simulation.environments import RuggedTerrainWorld, SimpleFlatWorld

# Local libraries
from ea import EAConfig, Experiment
from migration import POLICIES
from simulate import SimConfig

RESULTS_DIR = Path(__file__).parent / "results"

# The experiment world is rugged. "flat" is ARIEL's SimpleFlatWorld, for
# debugging only: it has no terrain noise, so it shows whether the EA itself
# can learn.
WORLDS = {"rugged": RuggedTerrainWorld, "flat": SimpleFlatWorld}


def build_parser() -> argparse.ArgumentParser:
    """Every EAConfig / SimConfig field becomes an optional flag."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--algorithm", choices=("island", "random_search"), default="island"
    )
    parser.add_argument("--policy", choices=POLICIES, default="best")
    parser.add_argument(
        "--world", choices=tuple(WORLDS), default="rugged", help="flat = debug only"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: results/<condition>"
    )

    for config_class in (EAConfig, SimConfig):
        for field in fields(config_class):
            if field.name in {"algorithm", "policy", "seed"}:
                continue
            flag = f"--{field.name.replace('_', '-')}"
            if field.name == "terrain_mode":
                parser.add_argument(
                    flag, choices=("per_run", "per_generation"), default=field.default
                )
            elif isinstance(field.default, bool):
                # --vision / --no-vision; type=bool would read "False" as True.
                parser.add_argument(
                    flag, action=argparse.BooleanOptionalAction, default=field.default
                )
            else:
                parser.add_argument(
                    flag, type=type(field.default), default=field.default
                )
    return parser


def configs_from_args(
    args: argparse.Namespace, seed: int
) -> tuple[EAConfig, SimConfig]:
    """Split the parsed flags into the two config objects."""
    values = vars(args)
    ea_config = EAConfig(
        algorithm=args.algorithm,
        policy="none" if args.algorithm == "random_search" else args.policy,
        seed=seed,
        **{
            f.name: values[f.name]
            for f in fields(EAConfig)
            if f.name not in {"algorithm", "policy", "seed"}
        },
    )
    sim_config = SimConfig(**{f.name: values[f.name] for f in fields(SimConfig)})
    return ea_config, sim_config


def seed_everything(seed: int) -> None:
    """Seed every RNG anything in the run might draw from.

    Our own operators use the `np.random.Generator` inside `Experiment`;
    ariel.ec's mutation uses its package-level RNG (`set_seed`); ariel's
    `Population.sample/shuffle` use the stdlib `random` module.
    """
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def main() -> None:
    """Run every requested seed with one shared pool of worker processes."""
    args = build_parser().parse_args()
    condition = "random_search" if args.algorithm == "random_search" else args.policy
    out_root = args.out or RESULTS_DIR / condition

    # "spawn" starts clean worker processes; MuJoCo and forked processes do
    # not mix well, and it is the macOS default anyway.
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for seed in args.seeds:
            ea_config, sim_config = configs_from_args(args, seed)
            out = out_root / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            seed_everything(seed)
            console.rule(f"[bold]{condition}  seed {seed}  ->  {out}")
            # Every condition with this seed walks the same terrain(s).
            terrain_dir = out_root.parent / "terrains" / args.world / f"seed{seed}"
            Experiment(
                ea_config,
                sim_config,
                out,
                pool,
                world_factory=WORLDS[args.world],
                terrain_dir=terrain_dir,
            ).evolve()


if __name__ == "__main__":
    main()
