"""The island-model EA, built on ariel.ec.

How one generation works (the operations run in this order):

    reproduce   per island: keep the elites, fill the rest with children
                (tournament -> crossover -> Gaussian mutation)
    evaluate    everyone new walks the run's terrain(s), in parallel
    migrate     every `migration_interval` generations, each island sends
                copies of `n_migrants` individuals to the next island in the
                ring, chosen by the migration POLICY (the research question)
    log         one CSV row per island + one for the whole population

ariel.ec provides the data model (`Individual`, `Population`), the generation
loop (`EA.step`), persistence of every individual to SQLite, and the Gaussian
mutation operator. Islands exist as an "island" tag on each individual; ariel
knows nothing about them. Selection, crossover and migration are ours.

ariel.ec traps this file works around (verified):

- Genotypes are stored as JSON: assign a list, never a numpy array.
- Changing a stored genotype or tag IN PLACE is silently lost; always assign
  a new value through the setter.
- Individuals are detached from the database after each commit, so none are
  kept between generations; only plain lists and numbers are.
- `from __future__ import annotations` breaks `EAOperation`'s signature check,
  so this file must not use it.
"""

# Standard library
import csv
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Literal

# Third-party libraries
import numpy as np
import numpy.typing as npt
from sqlalchemy import func, or_
from sqlalchemy.exc import DatabaseError
from sqlmodel import Session, col, create_engine, select

# Local libraries (ARIEL)
from ariel import console
from ariel.ec import EA, EAOperation, FloatMutator, Individual, Population
from ariel.simulation.environments import BaseWorld, RuggedTerrainWorld

# Local libraries
from genome import crossover, new_genotype, split
from migration import Policy, plan_migration
from network import Genotype
from operators import tournament_select
from sensors import clock_rows
from simulate import (
    FAILED_SCORE,
    Score,
    SimConfig,
    build_model,
    evaluate_task,
    parse_yaws,
    run_terrains,
    save_terrains,
    saved_sim_config,
    terrain_yaw,
)
from simulate import fitness as compute_fitness

type Algorithm = Literal["island", "random_search"]
type TerrainMode = Literal["per_run", "per_generation"]
type CrossoverKind = Literal["neuron", "weight", "blx", "headless"]
CROSSOVERS: tuple[str, ...] = ("neuron", "weight", "blx", "headless")

# Outputs of an earlier run in the same folder. A fresh start removes them, or
# they would survive and be read as this run's (by analyze.py, longer_walks.py,
# or a later --resume). ariel would delete database.db itself, but only once
# generation 0 has been walked.
STALE_OUTPUTS: tuple[str, ...] = (
    "summary.json",
    "unseen*.json",
    "longer_walks.json",
    "best_genotype*.npy",
    "database*.db",
    "database*.db-journal",
    "resumes.json",
)
LOG_ROUNDING = 5e-5  # the log keeps fitness to 4 decimals

# The parts of a Score stored as tags on every individual, next to its fitness.
SCORE_PARTS: tuple[str, ...] = tuple(Score.__dataclass_fields__)

LOG_COLUMNS: tuple[str, ...] = (
    "generation",
    "evaluations",
    "island",
    "best",
    "mean",
    "worst",
    "best_final",
    "best_distance",
    "best_mean_distance",
    "ground_contact",
    "upside_down",
    "low_body",
    "leg_imbalance",
    "work_imbalance",
    "mean_seconds",
    "duration",
    "spread",
    "sigma",
    "immigrants",
    "seconds",
)


@dataclass(frozen=True)
class EAConfig:
    """Every EA setting. Only `policy` differs between research conditions.

    The defaults are the settings of the EARLY experiments, not the paper's.
    They stay as they are because every saved run's config.json is read back
    against them (`run.finished_with`, `Experiment.restore`), and experiments
    14 and 19-26 rely on them. The paper's settings (experiment 99) are the
    flags in `experiments/99_final_experiment.sh`; a field whose paper value
    differs from its default says so in a `# paper:` comment. The options
    that only earlier experiments used are grouped as such in `run.py --help`.
    """

    algorithm: Algorithm = "island"
    policy: Policy = "best"
    seed: int = 0

    # Islands and migration (decision D11)
    n_islands: int = 4
    island_size: int = 20
    migration_interval: int = 10  # paper: 20 (D11, experiment 26)
    n_migrants: int = 2

    # Selection and variation (decisions D7-D9)
    n_elites: int = 2
    tournament_size: int = 3
    crossover_probability: float = 0.5  # paper: 0.9 (D7, D22)
    # "neuron" (D7) keeps hidden neurons whole; "weight", "blx" and "headless"
    # (a random genotype as the second parent) are experiment 25's alternatives.
    crossover: CrossoverKind = "neuron"
    mutation_sigma: float = 0.05  # chosen by the pilot (D8)
    # Stagnation rule (decision D19); off at 0. When an island's best has not
    # improved for `stall_generations` generations, its sigma doubles (up to
    # `max_sigma`); it drops back to `mutation_sigma` when the island improves.
    stall_generations: int = 0
    max_sigma: float = 0.4
    mutation_rate: float = 1.0
    init_scale: float = 0.5
    clock_boost: float = 1.0  # clock-input weights x this in new random genotypes (D17)
    # Start from saved networks (experiment X): an .npy file of genotypes, one
    # per row (a single best_genotype.npy works too). They take the first
    # slots of the initial population, dealt round-robin over the islands;
    # random networks fill the rest. "" starts fully random, as every run did.
    init_from: str = ""

    # Terrain (decision D10). "per_run": one set of terrains per seed, kept for
    # every generation and shared by every condition with that seed.
    # "per_generation": a fresh set every generation (noisy; for robustness).
    terrain_mode: TerrainMode = "per_run"
    n_terrains: int = 1
    # Degrees the robot starts turned by, cycled over the terrains (decision
    # D23): "0,30,-30" with 3 terrains starts one walk facing the target and
    # two turned away from it, so a brain has to steer. On the command line, a
    # list that starts with a minus needs "=": --spawn-yaws=-30,30.
    spawn_yaws: str = "0"
    # Place the robot with ARIEL's own floor correction, exactly as the
    # template does, instead of 2 cm above the real ground (decision D2a).
    ariel_spawn: bool = False  # paper: True (D2a)

    # Curriculum and early stopping (decision D16); both off by default. Over
    # the first `curriculum_generations` generations, the reward for moving at
    # all fades from `curriculum_movement_weight` to 0, and the progress a walk
    # must show by `SimConfig.early_stop_time` rises from start to end.
    curriculum: bool = False
    curriculum_movement_weight: float = 0.5
    early_stop: bool = False
    early_stop_progress_start: float = 0.02
    early_stop_progress_end: float = 0.10
    curriculum_generations: int = 50

    # Longer walks later in the run (decision D21); off at 0. From the first
    # generation after `final_duration_from` evaluations on, every walk lasts
    # `final_duration` seconds instead of `SimConfig.duration`.
    final_duration: float = 0.0
    final_duration_from: int = 6000

    # Budget (decision D12): a run stops after this many evaluations.
    max_evaluations: int = 12000


def check_config(config: "EAConfig") -> None:
    """Reject settings that would fail later with a confusing error."""
    problems = []
    if config.n_elites >= config.island_size:
        problems.append("n_elites must be smaller than island_size")
    if not 1 <= config.tournament_size <= config.island_size:
        problems.append("tournament_size must be between 1 and island_size")
    if config.n_migrants > config.island_size:
        problems.append("n_migrants cannot exceed island_size")
    if config.curriculum_generations < 1:
        problems.append("curriculum_generations must be at least 1")
    if config.migration_interval < 1:
        problems.append("migration_interval must be at least 1")
    if config.final_duration < 0:
        problems.append("final_duration must be 0 (off) or positive")
    if config.final_duration_from < 0:
        problems.append("final_duration_from must be 0 or more")
    if config.crossover not in CROSSOVERS:
        problems.append(f"crossover must be one of {', '.join(CROSSOVERS)}")
    if config.init_from and config.algorithm == "random_search":
        problems.append("init_from only applies to the island EA, not random search")
    try:
        yaws = parse_yaws(config.spawn_yaws)
    except ValueError:
        problems.append(
            f"spawn_yaws must be comma-separated degrees, got {config.spawn_yaws!r}"
        )
    else:
        if len(yaws) > config.n_terrains:
            problems.append("spawn_yaws has more values than there are terrains")
        if not all(np.isfinite(yaw) and abs(yaw) <= 180 for yaw in yaws):
            problems.append("spawn_yaws must lie between -180 and 180 degrees")
    if config.stall_generations < 0:
        problems.append("stall_generations must be 0 (off) or more")
    if config.stall_generations > 0:
        if config.max_sigma < config.mutation_sigma:
            problems.append("max_sigma must be at least mutation_sigma")
        # The rule compares an island's best across generations, which needs a
        # fitness that means the same every generation (D19).
        if config.curriculum or config.early_stop:
            problems.append(
                "stall_generations cannot be combined with a curriculum or early stop"
            )
        if config.terrain_mode != "per_run":
            problems.append("stall_generations needs terrain_mode per_run")
    if problems:
        raise ValueError("; ".join(problems))


def finite_or_failed(score: Score) -> Score:
    """`score`, or `FAILED_SCORE` if any part of it is NaN or infinite.

    A NaN anywhere in the score makes the fitness NaN, and `np.argmin` (the
    tournament, the run's best) picks a NaN over every real number, so a
    walk that blew up could win.
    """
    if all(np.isfinite(getattr(score, part)) for part in SCORE_PARTS):
        return score
    return FAILED_SCORE


def operator_rng(seed: int) -> np.random.Generator:
    """The random stream for our own operators (initialisation, selection,
    crossover, migration).

    ARIEL's mutation draws from its own generator, which `ariel.ec.set_seed`
    seeds with the plain run seed. Seeding ours with the same number would
    make both produce the SAME numbers: the first mutation noise would be an
    exact scaled copy of the first initial network. Deriving ours from
    (seed, 1) keeps the two streams independent and still reproducible.
    """
    return np.random.default_rng((seed, 1))


def next_sigma(
    config: EAConfig, sigma: float, stall: int, improved: bool
) -> tuple[float, int]:
    """One generation of the stagnation rule (D19): (new sigma, generations stalled).

    Off (`stall_generations` 0): sigma never changes. An improvement resets
    sigma to `mutation_sigma`; every `stall_generations` generations without
    one double it, up to `max_sigma`.
    """
    if config.stall_generations == 0:
        return sigma, stall
    if improved:
        return config.mutation_sigma, 0
    stall += 1
    if stall >= config.stall_generations:
        return min(2.0 * sigma, config.max_sigma), 0
    return sigma, stall


def new_individual(
    genotype: Genotype | list[float], island: int, **tags: object
) -> Individual:
    """A fresh ariel `Individual` on `island`, still to be evaluated."""
    individual = Individual()
    individual.genotype = [float(gene) for gene in genotype]
    individual.tags = {"island": island, **tags}
    return individual


def saved_genotypes(path: str, length: int, room: int) -> list[Genotype]:
    """The genotypes in `path` (`EAConfig.init_from`); none for an empty path.

    `length` is this run's genotype length and `room` its population size;
    a file that does not fit is rejected before anything runs.
    """
    if not path:
        return []
    genotypes = np.atleast_2d(np.load(path))
    if genotypes.shape[1] != length:
        msg = (
            f"init_from: {path} holds genotypes of {genotypes.shape[1]} weights, "
            f"but this run's network has {length}"
        )
        raise ValueError(msg)
    if len(genotypes) > room:
        msg = (
            f"init_from: {path} holds {len(genotypes)} genotypes, more than "
            f"the population of {room}"
        )
        raise ValueError(msg)
    return list(genotypes)


def save_durably(path: Path, array: npt.NDArray[np.float64]) -> None:
    """`np.save`, but the file is complete on disk before it replaces the old one.

    After a power cut the old file or the new one is there, never half of one;
    `Experiment.restore` reads it.
    """
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def last_generation(database: Path) -> int | None:
    """The last generation a run's database holds; None if it holds none."""
    if not database.exists():
        return None
    engine = create_engine(f"sqlite:///{database}")
    try:
        with Session(engine) as session:
            return session.exec(select(func.max(Individual.time_of_death))).one()
    except DatabaseError:  # no table yet: the run stopped while creating it
        return None
    finally:
        engine.dispose()


def part_number(database: Path) -> int:
    """N of `database_part<N>.db`."""
    return int(database.stem.removeprefix("database_part"))


def island_members(population: Population, island: int) -> list[Individual]:
    """The living individuals on one island."""
    return [
        individual
        for individual in population
        if individual.alive and individual.tags["island"] == island
    ]


class Experiment:
    """One run: one configuration, one seed."""

    def __init__(
        self,
        config: EAConfig,
        sim: SimConfig,
        out: Path,
        pool: Pool,
        world_factory: Callable[[], BaseWorld] = RuggedTerrainWorld,
        terrain_dir: Path | None = None,
        resume: bool = False,
    ) -> None:
        """`terrain_dir` holds this seed's terrains in "per_run" mode; pass the
        same directory to every condition with the same seed so they share
        the ground. It defaults to a folder inside `out`.

        With `resume`, an unfinished run in `out` (a database, no summary)
        continues where its database ends (see `restore`); otherwise the run
        starts from scratch.
        """
        check_config(config)
        self.config = config
        self.sim = sim
        self.init_genotypes: list[Genotype] = []  # `init_from`, read on a fresh start
        self.out = out
        self.pool = pool
        self.world_factory = world_factory  # replaced only in tests
        self.terrain_dir = terrain_dir or out / "terrains"
        # "per_generation" terrains live in a scratch folder that is emptied
        # after every generation - never the shared per-seed folder above.
        self.generation_terrain_dir = out / "generation_terrains"
        self.fixed_terrain = config.terrain_mode == "per_run"
        self.terrain_paths: tuple[str, ...] = ()
        if self.fixed_terrain:
            self.terrain_paths = run_terrains(
                self.terrain_dir,
                config.n_terrains,
                world_factory,
                sim.body,
                parse_yaws(config.spawn_yaws),
                config.ariel_spawn,
            )
        self.rng = operator_rng(config.seed)

        self.generation = 0
        self.evaluations = 0
        # Per-island mutation step and stagnation bookkeeping (decision D19).
        self.island_sigma = [config.mutation_sigma] * config.n_islands
        self.island_best = [float("inf")] * config.n_islands
        self.island_stall = [0] * config.n_islands
        self.best_genotype: list[float] = []
        self.best_fitness = float("inf")
        self.started_at = time.time()

        self.log_path = out / "log.csv"
        self.resumed: Population | None = None
        source = self.resumable_database() if resume else None
        if source is not None:
            self.resumed = self.restore(source, world_factory.__name__)
        else:
            # Checked before anything is removed; a resumed run never needs it.
            self.init_genotypes = saved_genotypes(
                config.init_from,
                sim.genotype_length,
                config.n_islands * config.island_size,
            )
            for pattern in STALE_OUTPUTS:
                for stale in out.glob(pattern):
                    stale.unlink()
            with self.log_path.open("w", newline="") as handle:
                csv.writer(handle).writerow(LOG_COLUMNS)
        (out / "config.json").write_text(
            json.dumps(
                {
                    "ea": asdict(config),
                    "sim": asdict(sim),
                    "world": world_factory.__name__,
                    "terrains": list(self.terrain_paths),
                },
                indent=2,
            )
        )

    def resumable_database(self) -> Path | None:
        """The database an unfinished run continues from; None to start afresh.

        Normally `database.db`. If the run stopped during a resume - after the
        old database was set aside as `database_part<N>.db`, before ariel saved
        a generation in the new one - it is the newest part.
        """
        if (self.out / "summary.json").exists():
            return None
        parts = sorted(self.out.glob("database_part*.db"), key=part_number)
        for database in (self.out / "database.db", *reversed(parts)):
            if last_generation(database) is not None:
                return database
        return None

    def restore(self, database: Path, world: str) -> Population:
        """Pick up an unfinished run (a crash, a power cut) where it stopped.

        ariel.ec saves every finished generation in the database, and `log`
        puts that generation's log rows (and any new best network) on disk
        just before, so after a cut the log ends on the database's last
        generation or on the one after it, which was lost. The run continues
        from the last generation both hold; the lost one simply runs again.
        (A log written before it was put on disk at every generation may end
        a few generations early; the run then continues from the log's end,
        so the evaluation count stays right.)

        The population is that generation's individuals: born in it, and
        alive at its end (not replaced by immigrants). ariel's own
        `EA(restart=...)` is not used because it also brings back the
        generation's dead parents and the individuals replaced by migration.
        The best network so far is the better of the logged best and the
        database's. Everything is read and checked before anything changes, so
        a refused or failed resume leaves the run as it was. Then the old
        database is kept as `database_part<N>.db`, a new `database.db`
        records the rest, and `resumes.json` lists every resume.

        The settings must be the run's own (only `max_evaluations` may grow).
        Only an unfinished run resumes: a finished one (with a summary.json)
        started again with a larger budget starts over. A curriculum, longer
        walks (D21) or the stagnation rule (D19) keep state the database does
        not hold, so such runs cannot resume. The random streams cannot be
        restored either: a resumed run is not identical to one that never
        stopped.
        """
        config = self.config
        if config.curriculum or config.final_duration > 0 or config.stall_generations:
            msg = (
                f"resume: {self.out} uses a curriculum, longer walks or the "
                "stagnation rule, whose state the database does not hold"
            )
            raise ValueError(msg)
        saved = json.loads((self.out / "config.json").read_text())
        saved_ea = replace(
            EAConfig(**saved["ea"]), max_evaluations=config.max_evaluations
        )
        if (
            saved_ea != config
            or saved_sim_config(saved["sim"]) != self.sim
            or saved.get("world") != world
        ):
            msg = f"resume: {self.out} was run with other settings"
            raise ValueError(msg)

        logged = self.read_log()
        everyone = [row for row in logged if row["island"] == "all"]
        saved_last = last_generation(database)
        if not everyone or saved_last is None:
            msg = f"resume: {self.out} holds no finished generation"
            raise ValueError(msg)
        generation = min(saved_last, max(int(row["generation"]) for row in everyone))

        engine = create_engine(f"sqlite:///{database}")
        with Session(engine) as session:
            rows = session.exec(
                select(Individual)
                .where(Individual.time_of_birth == generation)
                .where(or_(Individual.time_of_death > generation, Individual.alive))
            ).all()
            population = []
            for row in rows:
                tags = dict(row.tags)
                individual = new_individual(
                    row.genotype, int(tags.pop("island")), **tags
                )
                individual.fitness = row.fitness
                population.append(individual)
            champion = session.exec(
                select(Individual)
                .where(col(Individual.fitness_).is_not(None))
                .order_by(col(Individual.fitness_))
                .limit(1)
            ).one()
            saved_best = (float(champion.fitness), list(champion.genotype))
        engine.dispose()
        if len(population) != config.n_islands * config.island_size:
            msg = f"resume: {database} does not hold all of generation {generation}"
            raise ValueError(msg)

        # The database's best is exact. Only a logged best that beats it by
        # more than the log's rounding is better: it comes from the lost
        # generation, whose network `log` saved before writing the row.
        best_fitness, best_genotype = saved_best
        logged_best = min(float(row["best_final"]) for row in everyone)
        if logged_best < best_fitness - LOG_ROUNDING:
            try:
                best_genotype = np.load(self.out / "best_genotype.npy").tolist()
                best_fitness = logged_best
            except (OSError, ValueError):
                best_fitness, best_genotype = saved_best
        kept = [row for row in logged if int(row["generation"]) <= generation]
        last = next(
            row
            for row in kept
            if row["island"] == "all" and int(row["generation"]) == generation
        )

        # Everything checked: only now change the run's files.
        self.write_log(kept)
        save_durably(self.out / "best_genotype.npy", np.asarray(best_genotype))
        if database.name == "database.db":
            parts = [part_number(part) for part in self.out.glob("database_part*.db")]
            database.rename(self.out / f"database_part{max(parts, default=0) + 1}.db")
        else:  # resuming from a part: a database.db left here holds nothing
            (self.out / "database.db").unlink(missing_ok=True)
        resumes_file = self.out / "resumes.json"
        resumes = json.loads(resumes_file.read_text()) if resumes_file.exists() else []
        resumes.append(
            {
                "after_generation": generation,
                "evaluations": int(last["evaluations"]),
                "from": database.name,
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        resumes_file.write_text(json.dumps(resumes, indent=2))

        self.generation = generation
        self.evaluations = int(last["evaluations"])
        self.best_fitness = best_fitness
        self.best_genotype = best_genotype
        self.started_at = time.time() - float(last["seconds"])
        self.rng = np.random.default_rng((config.seed, 1, generation))
        console.print(
            f"resuming {self.out} after generation {generation} "
            f"({self.evaluations} evaluations, best {self.best_fitness:.3f})",
            highlight=False,
        )
        return Population(population)

    def read_log(self) -> list[dict[str, str]]:
        """The log's rows, without a last line that a crash cut short."""
        with self.log_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [row for row in rows if None not in row and None not in row.values()]

    def write_log(self, rows: list[dict[str, str]]) -> None:
        """Replace the log with these rows, all at once (see `save_durably`)."""
        temporary = self.log_path.with_name(self.log_path.name + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.log_path)

    # -- Operations -------------------------------------------------------- #

    def initial_population(self) -> Population:
        """`island_size` networks per island, weights from N(0, init_scale^2).

        With `init_from`, saved network k takes slot k // n_islands of island
        k % n_islands, so each island gets its share. Only the other slots
        draw a random network, in the same order as a run without saved ones.
        """
        n_islands = self.config.n_islands
        population = []
        for island in range(n_islands):
            for slot in range(self.config.island_size):
                k = slot * n_islands + island
                if k < len(self.init_genotypes):
                    genotype = self.init_genotypes[k]
                else:
                    genotype = self.random_individual()
                population.append(new_individual(genotype, island))
        return Population(population)

    def random_individual(self) -> Genotype:
        """A random genotype: the initial population, and random search.

        `clock_boost` applies to every random genotype, so in random search it
        applies to every sample - the same starting distribution as the EA.
        """
        return new_genotype(
            self.sim.shape,
            self.rng,
            evolve_tempo=self.sim.evolve_tempo,
            clock_rows=clock_rows(self.sim.hinges),
            init_scale=self.config.init_scale,
            clock_boost=self.config.clock_boost,
        )

    def update_sigma(self, island: int, best: float) -> None:
        """The stagnation rule (D19): widen a stuck island's mutation step.

        The island's elites are always kept, so a wider step cannot lose the
        best gait found so far; it only makes the children search further away.
        """
        if self.config.algorithm == "random_search":
            return  # random search never mutates
        improved = best < self.island_best[island]
        self.island_best[island] = min(best, self.island_best[island])
        self.island_sigma[island], self.island_stall[island] = next_sigma(
            self.config, self.island_sigma[island], self.island_stall[island], improved
        )

    def make_child(
        self,
        members: list[Individual],
        fitness: npt.NDArray[np.float64],
        sigma: float,
    ) -> Genotype:
        """Tournament -> (maybe) crossover -> Gaussian mutation.

        A "headless chicken" crossover (Jones 1995; experiment 25) crosses the
        first parent with a fresh random genotype instead of a second parent:
        a control for whether crossover only acts as a large mutation.
        """
        if self.config.algorithm == "random_search":
            return self.random_individual()

        k = self.config.tournament_size
        parent_a = np.asarray(members[tournament_select(fitness, k, self.rng)].genotype)
        if self.rng.random() < self.config.crossover_probability:
            if self.config.crossover == "headless":
                parent_b = self.random_individual()
            else:
                parent_b = np.asarray(
                    members[tournament_select(fitness, k, self.rng)].genotype
                )
            child = crossover(
                parent_a,
                parent_b,
                self.sim.shape,
                self.sim.evolve_tempo,
                self.rng,
                self.config.crossover,
            )
        else:
            child = parent_a
        mutated = FloatMutator.gaussian(
            child.tolist(),
            std=sigma,
            mutation_probability=self.config.mutation_rate,
        )
        return np.asarray(mutated, dtype=np.float64)

    def reproduce(self, population: Population) -> Population:
        """Generational replacement with elitism, separately on every island.

        The island's `n_elites` best are copied into the next generation
        unchanged; the rest of the next generation are children. On a fixed
        terrain an elite keeps its measurements (the simulation is
        deterministic). It is re-evaluated when the terrain changes every
        generation - its old measurements came from different ground (D10) -
        while the early-stop bar is still rising, because a walk measured
        under a more lenient bar is not comparable (D16), and when the walks
        get longer (D21).
        """
        self.generation += 1
        lengthen = self.lengthen_due()
        next_generation = []
        for island in range(self.config.n_islands):
            members = island_members(population, island)
            fitness = np.array([member.fitness for member in members])
            self.update_sigma(island, float(fitness.min()))
            for rank, index in enumerate(np.argsort(fitness)[: self.config.n_elites]):
                elite = members[index]
                if (
                    self.fixed_terrain
                    and not self.early_stop_bar_changing()
                    and not lengthen
                ):
                    copy = new_individual(
                        elite.genotype,
                        island,
                        elite=rank,
                        **{part: elite.tags[part] for part in SCORE_PARTS},
                    )
                    copy.fitness = elite.fitness
                else:
                    copy = new_individual(elite.genotype, island, elite=rank)
                next_generation.append(copy)
            for _ in range(self.config.island_size - self.config.n_elites):
                child = self.make_child(members, fitness, self.island_sigma[island])
                next_generation.append(new_individual(child, island))
        if lengthen:
            self.lengthen_walks()

        for individual in population:
            individual.alive = False
        population.extend(next_generation)
        return population

    # -- Schedules (decision D16) ------------------------------------------ #

    def schedule(self) -> float:
        """How far through the curriculum we are: 0 at generation 0, 1 at the end."""
        return min(1.0, self.generation / self.config.curriculum_generations)

    def movement_weight(self) -> float:
        """This generation's reward per metre moved in any direction."""
        if not self.config.curriculum:
            return 0.0
        return self.config.curriculum_movement_weight * (1.0 - self.schedule())

    def min_progress(self) -> float | None:
        """Progress towards the target a walk needs by the early-stop check."""
        if not self.config.early_stop:
            return None
        start = self.config.early_stop_progress_start
        end = self.config.early_stop_progress_end
        return start + (end - start) * self.schedule()

    def early_stop_bar_changing(self) -> bool:
        """True while the early-stop bar is still rising."""
        return self.config.early_stop and self.schedule() < 1.0

    def lengthen_due(self) -> bool:
        """True when the walks should switch to `final_duration` now (D21)."""
        return (
            self.config.final_duration > 0
            and self.sim.duration != self.config.final_duration
            and self.evaluations >= self.config.final_duration_from
        )

    def lengthen_walks(self) -> None:
        """Make every walk from now on last `final_duration` seconds (D21).

        A fitness measured on the shorter walks does not compare with one on
        the longer walks, so everything that compares fitness across
        generations starts afresh: the elites walk again (`reproduce`), each
        island's stagnation record restarts (its next generation counts as an
        improvement, which resets its sigma), and so does the run's best. The
        best network of the shorter walks is kept as `best_genotype_short.npy`.
        """
        if self.best_genotype:
            np.save(
                self.out / "best_genotype_short.npy", np.asarray(self.best_genotype)
            )
        self.sim = replace(self.sim, duration=self.config.final_duration)
        self.island_best = [float("inf")] * self.config.n_islands
        self.island_stall = [0] * self.config.n_islands
        self.best_fitness = float("inf")

    # -- Operations, continued ------------------------------------------- #

    def evaluate(self, population: Population) -> Population:
        """Let everyone that needs a fitness walk the terrains.

        Everybody - on every island - walks the same terrains, so their
        fitness values are directly comparable (decision D10). In "per_run"
        mode (the final experiment) these are the seed's fixed terrains; in
        "per_generation" mode (experiments 1-3) `n_terrains` new copies of the
        run's world are built for this generation and deleted afterwards.
        """
        if self.fixed_terrain:
            paths = self.terrain_paths
        else:
            yaws = parse_yaws(self.config.spawn_yaws)
            models = [
                build_model(
                    self.world_factory,
                    self.sim.body,
                    terrain_yaw(yaws, i),
                    self.config.ariel_spawn,
                )
                for i in range(self.config.n_terrains)
            ]
            paths = save_terrains(models, self.generation_terrain_dir, self.generation)

        todo = [ind for ind in population if ind.alive and ind.requires_eval]
        min_progress = self.min_progress()
        tasks = [(list(ind.genotype), paths, self.sim, min_progress) for ind in todo]
        for individual, walked in zip(
            todo, self.pool.map(evaluate_task, tasks), strict=True
        ):
            score = finite_or_failed(walked)
            individual.tags = {
                **individual.tags,
                **{part: getattr(score, part) for part in SCORE_PARTS},
            }
        self.evaluations += len(todo)

        # Score everyone alive - elites and immigrants included - with THIS
        # generation's fitness, recomputed from their stored measurements, so
        # a changing curriculum never compares old and new fitness values.
        weight = self.movement_weight()
        for individual in population:
            if individual.alive:
                individual.fitness = compute_fitness(
                    self.stored_score(individual), self.sim, weight
                )

        if not self.fixed_terrain:
            shutil.rmtree(self.generation_terrain_dir)
        return population

    @staticmethod
    def stored_score(individual: Individual) -> Score:
        """The measurements saved in an individual's tags, as a `Score`."""
        return Score(**{part: individual.tags[part] for part in SCORE_PARTS})

    def migrate(self, population: Population) -> Population:
        """Every `migration_interval` generations, run one ring migration event.

        Immigrants keep the fitness they earned at home: every island walked
        the same terrains this generation, so it is valid on arrival too.
        """
        interval = self.config.migration_interval
        if self.config.algorithm != "island" or self.generation % interval != 0:
            return population

        islands = [island_members(population, i) for i in range(self.config.n_islands)]
        fitness = [
            np.array([member.fitness for member in island]) for island in islands
        ]
        plan = plan_migration(
            fitness, self.config.n_migrants, self.config.policy, self.rng
        )

        arrivals = []
        for transfer in plan:
            for index in transfer.emigrants:
                emigrant = islands[transfer.source][index]
                immigrant = new_individual(
                    emigrant.genotype,
                    transfer.target,
                    migrant_from=transfer.source,
                    migrated_at=self.generation,
                    **{part: emigrant.tags[part] for part in SCORE_PARTS},
                )
                immigrant.fitness = emigrant.fitness
                arrivals.append(immigrant)
        for transfer in plan:
            for index in transfer.replaced:
                islands[transfer.target][index].alive = False
        population.extend(arrivals)
        return population

    def log(self, population: Population) -> Population:
        """One CSV row per island and one for everyone (island = "all")."""
        rows = []
        everyone: list[Individual] = []
        for island in range(self.config.n_islands):
            members = island_members(population, island)
            everyone.extend(members)
            rows.append(self.stats_row(str(island), members, self.island_sigma[island]))
        rows.append(self.stats_row("all", everyone, float(np.mean(self.island_sigma))))

        # The run's best is always judged by the FINAL fitness (no movement
        # reward), so a curriculum stage can never crown a different "best".
        fitness = np.array(
            [compute_fitness(self.stored_score(m), self.sim) for m in everyone]
        )
        best = int(np.argmin(fitness))
        if fitness[best] < self.best_fitness:
            self.best_fitness = float(fitness[best])
            self.best_genotype = list(everyone[best].genotype)
            # Saved on every improvement, so a stopped run keeps its best.
            save_durably(self.out / "best_genotype.npy", np.asarray(self.best_genotype))

        # On disk after the best network and before ariel saves this generation
        # in the database: a logged best always has its network saved, and the
        # log never falls behind the database (`restore` relies on both).
        with self.log_path.open("a", newline="") as handle:
            csv.writer(handle).writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())

        champion = everyone[best].tags
        console.print(
            f"gen {self.generation:>3}  evals {self.evaluations:>5}  "
            f"best {fitness[best]:.3f} (distance {champion['distance']:.2f}, "
            f"on ground {champion['ground_contact']:.0%}, "
            f"flipped {champion['upside_down']:.0%})  "
            f"mean {fitness.mean():.3f}  {time.time() - self.started_at:.0f}s",
            highlight=False,
        )
        return population

    def stats_row(
        self, island: str, members: list[Individual], sigma: float
    ) -> list[object]:
        """Fitness statistics and genotype spread for a group of individuals.

        `best`, `mean` and `worst` use this generation's fitness (which in a
        curriculum run still includes the movement reward); `best_final` is
        the best FINAL fitness (no movement reward), comparable across runs.
        `best_distance` is the shortest distance to the target in the group,
        `best_mean_distance` the lowest walk-averaged distance (D20);
        `ground_contact` and `upside_down` are the group's MEAN fractions of
        the run spent with the core on the ground and upside down;
        `low_body`, `leg_imbalance` and `work_imbalance` are the group's means
        of those measurements (D18). `duration` is this generation's walk length
        (D21). `sigma` is the island's current mutation step (the mean over
        islands for "all"; D19).

        `spread` is the mean Euclidean distance of the genotypes to their
        centroid: how different the group's networks are. Comparing islands'
        spreads over time shows how quickly migration makes them alike.
        Only the weights count; the tempo gene (D17) is on a different scale.
        """
        fitness = np.array([member.fitness for member in members])
        final = [compute_fitness(self.stored_score(m), self.sim) for m in members]
        parts = {
            part: np.array([member.tags[part] for member in members])
            for part in SCORE_PARTS
        }
        genotypes = np.array(
            [
                split(
                    np.asarray(member.genotype), self.sim.shape, self.sim.evolve_tempo
                )[0]
                for member in members
            ]
        )
        spread = np.linalg.norm(genotypes - genotypes.mean(axis=0), axis=1).mean()
        immigrants = sum(
            member.tags.get("migrated_at") == self.generation for member in members
        )
        return [
            self.generation,
            self.evaluations,
            island,
            f"{fitness.min():.4f}",
            f"{fitness.mean():.4f}",
            f"{fitness.max():.4f}",
            f"{min(final):.4f}",
            f"{parts['distance'].min():.4f}",
            f"{parts['mean_distance'].min():.4f}",
            f"{parts['ground_contact'].mean():.4f}",
            f"{parts['upside_down'].mean():.4f}",
            f"{parts['low_body'].mean():.4f}",
            f"{parts['leg_imbalance'].mean():.4f}",
            f"{parts['work_imbalance'].mean():.4f}",
            f"{parts['seconds'].mean():.2f}",
            f"{self.sim.duration:g}",
            f"{spread:.4f}",
            f"{sigma:.4f}",
            immigrants,
            f"{time.time() - self.started_at:.1f}",
        ]

    # -- Running ------------------------------------------------------------ #

    def evolve(self) -> None:
        """Evaluate generation 0 (or take the resumed population), then step
        until the evaluation budget is spent."""
        if self.resumed is None:
            population = self.evaluate(self.initial_population())
            self.log(population)
        else:
            population = self.resumed

        database = self.out / "database.db"
        ea = EA(
            population,
            operations=[
                EAOperation(self.reproduce),
                EAOperation(self.evaluate),
                EAOperation(self.migrate),
                EAOperation(self.log),
            ],
            is_maximisation=False,
            db_file_path=database,
            quiet=True,
            first_generation_id=self.generation,
        )

        while self.evaluations < self.config.max_evaluations:
            ea.step()

        summary: dict[str, object] = {
            "generations": self.generation,
            "evaluations": self.evaluations,
            "best_fitness_seen": self.best_fitness,
            "seconds": round(time.time() - self.started_at, 1),
        }
        resumes = self.out / "resumes.json"
        if resumes.exists():
            # A resumed run cannot be reproduced exactly; say so where it ends.
            summary["resumed_after"] = [
                resume["after_generation"] for resume in json.loads(resumes.read_text())
            ]
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2))
