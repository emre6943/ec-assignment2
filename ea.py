"""The island-model EA, built on ariel.ec.

How one generation works (the operations run in this order):

    reproduce   per island: keep the elites, fill the rest with children
                (tournament -> neuron crossover -> Gaussian mutation)
    evaluate    everyone new walks the generation's terrains, in parallel
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
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Literal

# Third-party libraries
import numpy as np
import numpy.typing as npt

# Local libraries (ARIEL)
from ariel import console
from ariel.ec import EA, EAOperation, FloatMutator, Individual, Population
from ariel.simulation.environments import BaseWorld, RuggedTerrainWorld

# Local libraries
from migration import Policy, plan_migration
from network import Genotype, random_genotype
from operators import neuron_crossover, tournament_select
from simulate import (
    FAILED_SCORE,
    Score,
    SimConfig,
    build_model,
    evaluate_task,
    run_terrains,
    save_terrains,
)
from simulate import fitness as compute_fitness

type Algorithm = Literal["island", "random_search"]
type TerrainMode = Literal["per_run", "per_generation"]

# The parts of a Score stored as tags on every individual, next to its fitness.
SCORE_PARTS: tuple[str, ...] = tuple(Score.__dataclass_fields__)

LOG_COLUMNS: tuple[str, ...] = (
    "generation",
    "evaluations",
    "island",
    "best",
    "mean",
    "worst",
    "best_distance",
    "ground_contact",
    "upside_down",
    "mean_seconds",
    "spread",
    "immigrants",
    "seconds",
)


@dataclass(frozen=True)
class EAConfig:
    """Every EA setting. Only `policy` differs between research conditions."""

    algorithm: Algorithm = "island"
    policy: Policy = "best"
    seed: int = 0

    # Islands and migration (decision D11)
    n_islands: int = 4
    island_size: int = 20
    migration_interval: int = 10
    n_migrants: int = 2

    # Selection and variation (decisions D7-D9)
    n_elites: int = 2
    tournament_size: int = 3
    crossover_probability: float = 0.5
    mutation_sigma: float = 0.1
    mutation_rate: float = 1.0
    init_scale: float = 0.5

    # Terrain (decision D10). "per_run": one set of terrains per seed, kept for
    # every generation and shared by every condition with that seed.
    # "per_generation": a fresh set every generation (noisy; for robustness).
    terrain_mode: TerrainMode = "per_run"
    n_terrains: int = 1

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

    # Budget and stopping (decision D12)
    max_evaluations: int = 3000
    plateau_window: int = 20
    plateau_tolerance: float = 0.01


def plateaued(best_per_generation: list[float], window: int, tolerance: float) -> bool:
    """True when the last `window` generations did not beat the `window` before.

    Compares the MEAN best fitness of two consecutive windows rather than two
    single generations, because each generation walks different terrain and a
    single generation's best is noisy.
    """
    if len(best_per_generation) < 2 * window:
        return False
    recent = np.mean(best_per_generation[-window:])
    before = np.mean(best_per_generation[-2 * window : -window])
    return bool(recent > before - tolerance)


def new_individual(
    genotype: Genotype | list[float], island: int, **tags: object
) -> Individual:
    """A fresh ariel `Individual` on `island`, still to be evaluated."""
    individual = Individual()
    individual.genotype = [float(gene) for gene in genotype]
    individual.tags = {"island": island, **tags}
    return individual


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
    ) -> None:
        """`terrain_dir` holds this seed's terrains in "per_run" mode; pass the
        same directory to every condition with the same seed so they share
        the ground. It defaults to a folder inside `out`."""
        self.config = config
        self.sim = sim
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
                self.terrain_dir, config.n_terrains, world_factory
            )
        self.rng = np.random.default_rng(config.seed)

        self.generation = 0
        self.evaluations = 0
        self.best_per_generation: list[float] = []
        self.best_genotype: list[float] = []
        self.best_fitness = float("inf")
        self.started_at = time.time()

        self.log_path = out / "log.csv"
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

    # -- Operations -------------------------------------------------------- #

    def initial_population(self) -> Population:
        """`island_size` random networks per island, weights from N(0, init_scale^2)."""
        return Population(
            [
                new_individual(
                    random_genotype(self.sim.shape, self.rng, self.config.init_scale),
                    island,
                )
                for island in range(self.config.n_islands)
                for _ in range(self.config.island_size)
            ]
        )

    def make_child(
        self, members: list[Individual], fitness: npt.NDArray[np.float64]
    ) -> Genotype:
        """Tournament -> (maybe) neuron crossover -> Gaussian mutation."""
        if self.config.algorithm == "random_search":
            return random_genotype(self.sim.shape, self.rng, self.config.init_scale)

        k = self.config.tournament_size
        parent_a = np.asarray(members[tournament_select(fitness, k, self.rng)].genotype)
        if self.rng.random() < self.config.crossover_probability:
            parent_b = np.asarray(
                members[tournament_select(fitness, k, self.rng)].genotype
            )
            child = neuron_crossover(parent_a, parent_b, self.sim.shape, self.rng)
        else:
            child = parent_a
        mutated = FloatMutator.gaussian(
            child.tolist(),
            std=self.config.mutation_sigma,
            mutation_probability=self.config.mutation_rate,
        )
        return np.asarray(mutated, dtype=np.float64)

    def reproduce(self, population: Population) -> Population:
        """Generational replacement with elitism, separately on every island.

        The island's `n_elites` best are copied into the next generation
        unchanged; the rest of the next generation are children. On a fixed
        terrain an elite keeps its fitness (the simulation is deterministic);
        when the terrain changes every generation it is re-evaluated, because
        its old fitness came from different ground (D10).
        """
        self.generation += 1
        next_generation = []
        for island in range(self.config.n_islands):
            members = island_members(population, island)
            fitness = np.array([member.fitness for member in members])
            for rank, index in enumerate(np.argsort(fitness)[: self.config.n_elites]):
                elite = members[index]
                if self.fixed_terrain:
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
                next_generation.append(
                    new_individual(self.make_child(members, fitness), island)
                )

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

    def curriculum_active(self) -> bool:
        """True while either schedule is still changing."""
        uses_schedule = self.config.curriculum or self.config.early_stop
        return uses_schedule and self.schedule() < 1.0

    # -- Operations, continued ------------------------------------------- #

    def evaluate(self, population: Population) -> Population:
        """Let everyone that needs a fitness walk the terrains.

        Everybody - on every island - walks the same terrains, so their
        fitness values are directly comparable (decision D10). In "per_run"
        mode these are the seed's fixed terrains; in "per_generation" mode
        `n_terrains` brand-new `RuggedTerrainWorld()` terrains are generated
        for this generation and deleted afterwards.
        """
        if self.fixed_terrain:
            paths = self.terrain_paths
        else:
            models = [
                build_model(self.world_factory) for _ in range(self.config.n_terrains)
            ]
            paths = save_terrains(models, self.generation_terrain_dir, self.generation)

        todo = [ind for ind in population if ind.alive and ind.requires_eval]
        min_progress = self.min_progress()
        tasks = [(list(ind.genotype), paths, self.sim, min_progress) for ind in todo]
        for individual, score in zip(
            todo, self.pool.map(evaluate_task, tasks), strict=True
        ):
            if not np.isfinite(score.distance):
                score = FAILED_SCORE
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
            rows.append(self.stats_row(str(island), members))
        rows.append(self.stats_row("all", everyone))

        with self.log_path.open("a", newline="") as handle:
            csv.writer(handle).writerows(rows)

        # The run's best is always judged by the FINAL fitness (no movement
        # reward), so a curriculum stage can never crown a different "best".
        fitness = np.array(
            [compute_fitness(self.stored_score(m), self.sim) for m in everyone]
        )
        best = int(np.argmin(fitness))
        self.best_per_generation.append(float(fitness[best]))
        if fitness[best] < self.best_fitness:
            self.best_fitness = float(fitness[best])
            self.best_genotype = list(everyone[best].genotype)

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

    def stats_row(self, island: str, members: list[Individual]) -> list[object]:
        """Fitness statistics and genotype spread for a group of individuals.

        `best_distance` is the shortest distance to the target in the group;
        `ground_contact` and `upside_down` are the group's MEAN fractions of
        the run spent with the core on the ground and upside down.

        `spread` is the mean Euclidean distance of the genotypes to their
        centroid: how different the group's networks are. Comparing islands'
        spreads over time shows how quickly migration makes them alike.
        """
        fitness = np.array([member.fitness for member in members])
        parts = {
            part: np.array([member.tags[part] for member in members])
            for part in SCORE_PARTS
        }
        genotypes = np.array([member.genotype for member in members])
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
            f"{parts['distance'].min():.4f}",
            f"{parts['ground_contact'].mean():.4f}",
            f"{parts['upside_down'].mean():.4f}",
            f"{parts['seconds'].mean():.2f}",
            f"{spread:.4f}",
            immigrants,
            f"{time.time() - self.started_at:.1f}",
        ]

    # -- Running ------------------------------------------------------------ #

    def evolve(self) -> None:
        """Evaluate generation 0, then step until the budget or a plateau."""
        population = self.evaluate(self.initial_population())
        self.log(population)

        database = self.out / "database.db"
        database.unlink(missing_ok=True)
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
            db_handling="halt",
            quiet=True,
        )

        config = self.config
        while self.evaluations < config.max_evaluations:
            ea.step()
            if not self.curriculum_active() and plateaued(
                self.best_per_generation,
                config.plateau_window,
                config.plateau_tolerance,
            ):
                console.print(f"plateau reached at generation {self.generation}")
                break

        np.save(self.out / "best_genotype.npy", np.asarray(self.best_genotype))
        (self.out / "summary.json").write_text(
            json.dumps(
                {
                    "generations": self.generation,
                    "evaluations": self.evaluations,
                    "best_fitness_seen": self.best_fitness,
                    "stopped_by": (
                        "budget"
                        if self.evaluations >= config.max_evaluations
                        else "plateau"
                    ),
                    "seconds": round(time.time() - self.started_at, 1),
                },
                indent=2,
            )
        )
