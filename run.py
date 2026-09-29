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
import json
import multiprocessing as mp
import random
from dataclasses import fields, replace
from pathlib import Path

# Third-party libraries
import numpy as np

# Local libraries (ARIEL)
from ariel import console
from ariel.ec import set_seed
from ariel.simulation.environments import (
    AmphitheatreTerrainWorld,
    CraterTerrainWorld,
    OlympicArena,
    RuggedTerrainWorld,
    SimpleFlatWorld,
)

# Local libraries
from bodies import BODIES, FIRST_BODY
from ea import EAConfig, Experiment
from migration import POLICIES
from simulate import SimConfig, saved_sim_config

RESULTS_DIR = Path(__file__).parent / "results"

# Every world is ARIEL's, used with its default settings. "olympic"
# (OlympicArena: a flat start, then a gentle rugged strip with the target) is
# the final world (decision D2); "rugged" was the world of experiments 1-12.
# RuggedTerrainWorld and OlympicArena's rugged strip are random on every
# build, so each seed's world is built once and saved (D10).
WORLDS = {
    "rugged": RuggedTerrainWorld,
    "flat": SimpleFlatWorld,
    "olympic": OlympicArena,
    "amphitheatre": AmphitheatreTerrainWorld,
    "crater": CraterTerrainWorld,
}


def build_parser() -> argparse.ArgumentParser:
    """Every EAConfig / SimConfig field becomes an optional flag."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--algorithm", choices=("island", "random_search"), default="island"
    )
    parser.add_argument("--policy", choices=POLICIES, default="best")
    parser.add_argument(
        "--standard",
        action="store_true",
        help="a standard EA: one population of n_islands x island_size, no migration",
    )
    parser.add_argument(
        "--world", choices=tuple(WORLDS), default="olympic", help="an ARIEL world"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="skip seeds that already finished with exactly these settings",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="default: results/<condition>"
    )

    for config_class in (EAConfig, SimConfig):
        for field in fields(config_class):
            if field.name in {"algorithm", "policy", "seed"}:
                continue
            flag = f"--{field.name.replace('_', '-')}"
            if field.name == "body":
                parser.add_argument(flag, choices=BODIES, default=field.default)
            elif field.name == "terrain_mode":
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
    if args.standard:
        ea_config = as_standard(ea_config)
    sim_config = SimConfig(**{f.name: values[f.name] for f in fields(SimConfig)})
    return ea_config, sim_config


def as_standard(config: EAConfig) -> EAConfig:
    """The same EA as ONE population: no islands, no migration.

    The total population and the share of elites stay the same (4 islands of 20
    with 2 elites each become 1 population of 80 with 8 elites), so the only
    difference from the island model is the population structure.
    """
    return replace(
        config,
        policy="none",
        n_islands=1,
        island_size=config.n_islands * config.island_size,
        n_elites=config.n_islands * config.n_elites,
        n_migrants=0,
    )


def terrain_folder(results: Path, world: str, body: str, seed: int) -> Path:
    """Where a seed's terrain lives. The saved terrain includes the robot, so
    each body gets its own folder; spider_16 keeps the original location.
    """
    folder = results / "terrains" / world
    if body != FIRST_BODY:
        folder = folder / body
    return folder / f"seed{seed}"


def finished_with(
    out: Path, ea_config: EAConfig, sim_config: SimConfig, world: str
) -> bool:
    """True if `out` holds a finished run made with exactly these settings."""
    if not (out / "summary.json").exists() or not (out / "config.json").exists():
        return False
    saved = json.loads((out / "config.json").read_text())
    try:
        # Settings a run did not record get the values it ran with (D1, D12).
        saved_ea = EAConfig(**saved["ea"])
        saved_sim = saved_sim_config(saved["sim"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        saved_ea == ea_config
        and saved_sim == sim_config
        and saved.get("world") == WORLDS[world].__name__
    )


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
    parser = build_parser()
    args = parser.parse_args()
    if args.standard and args.algorithm == "random_search":
        parser.error("--standard only applies to the island EA, not random search")
    if args.algorithm == "random_search":
        condition = "random_search"
    elif args.standard:
        condition = "standard"
    else:
        condition = args.policy
    out_root = args.out or RESULTS_DIR / condition

    # "spawn" starts clean worker processes; MuJoCo and forked processes do
    # not mix well, and it is the macOS default anyway.
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for seed in args.seeds:
            ea_config, sim_config = configs_from_args(args, seed)
            out = out_root / f"seed{seed}"
            if args.skip_done and finished_with(out, ea_config, sim_config, args.world):
                console.print(f"{condition} seed {seed}: already done, skipped")
                continue
            out.mkdir(parents=True, exist_ok=True)
            seed_everything(seed)
            console.rule(f"[bold]{condition}  seed {seed}  ->  {out}")
            # Every condition with this seed walks the same terrain(s).
            terrain_dir = terrain_folder(out_root.parent, args.world, args.body, seed)
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
