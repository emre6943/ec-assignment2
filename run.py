"""Run one configuration for one or more seeds.

The defaults are the settings of the early experiments, not the paper's (see
`ea.EAConfig` and `simulate.SimConfig`). The paper's 120 runs come from
`experiments/99_final_experiment.sh`, which passes the final settings as flags.
One of its runs (migrate best, seed 10) by hand, into a new folder:

    uv run --project ../ariel python run.py --policy best --seeds 10 \\
        --world olympic --body spider_8 --duration 15 --no-evolve-tempo \\
        --no-vision --clock-boost 1 --hidden-layers 8,4 \\
        --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04 \\
        --work-imbalance-weight 0.5 --leg-imbalance-weight 0 \\
        --speed-weight 0.5 --stop-at-target \\
        --crossover-probability 0.9 --ariel-spawn --max-evaluations 12000 \\
        --migration-interval 20 --out results/example/best

The other conditions swap `--policy best` for `--policy worst|random|none`,
`--standard` or `--algorithm random_search`. A new results folder builds new
arenas; the paper's are in results/final/terrains/. `--help` lists every
option, with those that only earlier experiments used in a group of their own.

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
from ea import CROSSOVERS, EAConfig, Experiment
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


# The EAConfig / SimConfig fields that only earlier experiments changed, and
# the experiments that did. The final experiment (99) leaves them at their
# defaults, which switch them off; `--help` lists them in a group of their own.
EARLIER_ONLY: dict[str, str] = {
    "terrain_mode": "per_generation in experiments 1-3 (D10)",
    "n_terrains": "experiment 21 (D23)",
    "spawn_yaws": "experiment 21 (D23)",
    "curriculum": "experiment 6 (D16)",
    "curriculum_movement_weight": "with --curriculum, experiment 6 (D16)",
    "curriculum_generations": "with --curriculum or --early-stop, experiment 6",
    "early_stop": "experiment 6 (D16)",
    "early_stop_progress_start": "with --early-stop, experiment 6 (D16)",
    "early_stop_progress_end": "with --early-stop, experiment 6 (D16)",
    "early_stop_time": "with --early-stop, experiment 6 (D16)",
    "final_duration": "experiment 20 (D21)",
    "final_duration_from": "with --final-duration, experiment 20 (D21)",
    "stall_generations": "experiments 16-22 (D19)",
    "max_sigma": "with --stall-generations, experiments 16-22 (D19)",
    "init_from": "experiments/x_best_walk.sh",
    "position": "experiment 29",
    "vision_rays": "with vision on, experiments 18-23 (D5)",
}
# The paper's value of every final-experiment setting that differs from its
# default or was set otherwise in some experiment, as 99_final_experiment.sh
# passes it (tests/test_final_config.py checks that its flags give these).
NOTES: dict[str, str] = {
    "migration_interval": "20 in the paper (D11)",
    "crossover_probability": "0.9 in the paper (D7, D22)",
    "crossover": "neuron in the paper; experiment 25 tried the others (D7)",
    "clock_boost": "1 in the paper; 3 in experiments 13 and 15-18 (D17)",
    "ariel_spawn": "on in the paper (D2a)",
    "vision": "off in the paper (D5)",
    "hidden_layers": "8,4 in the paper (D6)",
    "evolve_tempo": "off in the paper; on in experiment 13 (D17)",
    "ground_contact_weight": "1.0 in the paper (D18)",
    "low_body_weight": "1.0 in the paper (D18)",
    "carry_height": "0.04 in the paper (D18)",
    "work_imbalance_weight": "0.5 in the paper (D18)",
    "speed_weight": "0.5 in the paper (D20)",
    "stop_at_target": "on in the paper (D20)",
}


def build_parser() -> argparse.ArgumentParser:
    """Every EAConfig / SimConfig field becomes an optional flag.

    The flags are grouped for `--help`: what a run is, the settings the
    final experiment (99) uses, and those only earlier experiments used.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    runs = parser.add_argument_group("which runs, and where")
    runs.add_argument(
        "--algorithm",
        choices=("island", "random_search"),
        default="island",
        help="the island EA, or the random-search baseline",
    )
    runs.add_argument(
        "--policy",
        choices=POLICIES,
        default="best",
        help="the emigrant selection: the research question's conditions",
    )
    runs.add_argument(
        "--standard",
        action="store_true",
        help="a standard EA: one population of n_islands x island_size, no migration",
    )
    runs.add_argument(
        "--world",
        choices=tuple(WORLDS),
        default="olympic",
        help="an ARIEL world: olympic in the paper (D2), rugged in experiments "
        "1-13, flat for debugging; amphitheatre and crater were never used",
    )
    runs.add_argument("--seeds", type=int, nargs="+", default=[0], help="run seeds")
    runs.add_argument("--workers", type=int, default=10, help="parallel walks")
    runs.add_argument(
        "--skip-done",
        action="store_true",
        help="skip seeds that already finished with exactly these settings",
    )
    runs.add_argument(
        "--out",
        type=Path,
        default=None,
        help="the folder for the seed<S>/ folders; None means results/<condition>",
    )
    runs.add_argument(
        "--resume",
        action="store_true",
        help="continue an unfinished run (crash, power cut) from its database",
    )

    final = parser.add_argument_group(
        "settings used by the final experiment (99)",
        "Several defaults are the early experiments' values, not the paper's; "
        "the paper's are the flags in experiments/99_final_experiment.sh.",
    )
    earlier = parser.add_argument_group(
        "earlier experiments only (not in the paper)",
        "The final experiment leaves these at their defaults (off).",
    )
    for config_class in (EAConfig, SimConfig):
        for field in fields(config_class):
            if field.name in {"algorithm", "policy", "seed"}:
                continue
            group = earlier if field.name in EARLIER_ONLY else final
            flag = f"--{field.name.replace('_', '-')}"
            # The formatter adds the default to a note; a bare default needs
            # a help text of its own to be shown at all.
            note = EARLIER_ONLY.get(field.name) or NOTES.get(field.name)
            options: dict[str, object] = {
                "default": field.default,
                "help": note or "(default: %(default)s)",
            }
            if field.name == "body":
                options["choices"] = BODIES
            elif field.name == "terrain_mode":
                options["choices"] = ("per_run", "per_generation")
            elif field.name == "crossover":
                options["choices"] = CROSSOVERS
            elif isinstance(field.default, bool):
                # --vision / --no-vision; type=bool would read "False" as True.
                options["action"] = argparse.BooleanOptionalAction
            else:
                options["type"] = type(field.default)
            group.add_argument(flag, **options)
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
            experiment = Experiment(
                ea_config,
                sim_config,
                out,
                pool,
                world_factory=WORLDS[args.world],
                terrain_dir=terrain_dir,
                resume=args.resume,
            )
            if experiment.resumed is not None:
                # Fresh random streams, not a replay of the run's first ones
                # (+ 1: a resume after generation 0 must not reuse the seed).
                seed_everything(seed + 1009 * (experiment.generation + 1))
            experiment.evolve()


if __name__ == "__main__":
    main()
