"""The plateau stopping rule, and a tiny end-to-end run of the island EA.

The end-to-end runs use ARIEL's `SimpleFlatWorld`: rugged terrain is random on
every construction, so only a flat world makes two runs comparable exactly.
"""

import csv
import multiprocessing as mp
import random
from collections.abc import Iterator
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import pytest
from ariel.ec import set_seed
from ariel.simulation.environments import SimpleFlatWorld

from ea import EAConfig, Experiment, plateaued
from simulate import SimConfig

TINY_SIM = SimConfig(duration=0.3, n_hidden=4)


def tiny_config(policy: str, seed: int = 0) -> EAConfig:
    return EAConfig(
        policy=policy,
        seed=seed,
        n_islands=3,
        island_size=4,
        n_elites=1,
        n_migrants=1,
        migration_interval=2,
        n_terrains=1,
        max_evaluations=60,
    )


@pytest.fixture(scope="module")
def pool() -> Iterator[Pool]:
    with mp.get_context("spawn").Pool(2) as worker_pool:
        yield worker_pool


def run_tiny(config: EAConfig, out: Path, pool: Pool) -> list[dict[str, str]]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    set_seed(config.seed)
    out.mkdir(parents=True)
    Experiment(config, TINY_SIM, out, pool, world_factory=SimpleFlatWorld).evolve()
    with (out / "log.csv").open() as handle:
        return list(csv.DictReader(handle))


def test_no_plateau_before_two_full_windows() -> None:
    assert not plateaued([1.0] * 39, window=20, tolerance=0.01)


def test_flat_history_is_a_plateau() -> None:
    assert plateaued([1.0] * 40, window=20, tolerance=0.01)


def test_steady_improvement_is_not_a_plateau() -> None:
    history = list(np.linspace(2.0, 1.0, 40))
    assert not plateaued(history, window=20, tolerance=0.01)


def test_best_policy_moves_individuals_and_none_does_not(
    tmp_path: Path, pool: Pool
) -> None:
    best = run_tiny(tiny_config("best"), tmp_path / "best", pool)
    none = run_tiny(tiny_config("none"), tmp_path / "none", pool)

    def immigrants(rows: list[dict[str, str]]) -> int:
        return sum(int(row["immigrants"]) for row in rows if row["island"] == "all")

    assert immigrants(best) > 0
    assert immigrants(none) == 0


def test_same_seed_gives_the_same_run(tmp_path: Path, pool: Pool) -> None:
    first = run_tiny(tiny_config("random"), tmp_path / "a", pool)
    second = run_tiny(tiny_config("random"), tmp_path / "b", pool)
    for row in first + second:
        del row["seconds"]
    assert first == second


def test_run_writes_its_outputs(tmp_path: Path, pool: Pool) -> None:
    run_tiny(tiny_config("worst"), tmp_path / "run", pool)
    for name in ("config.json", "log.csv", "database.db", "best_genotype.npy"):
        assert (tmp_path / "run" / name).exists()
    assert (tmp_path / "run" / "terrains" / "terrain0.mjb").exists()
    assert not (tmp_path / "run" / "generation_terrains").exists()
    genotype = np.load(tmp_path / "run" / "best_genotype.npy")
    assert genotype.shape == (TINY_SIM.shape.n_weights,)


def test_fixed_terrain_elites_are_not_re_evaluated(tmp_path: Path, pool: Pool) -> None:
    """On a fixed terrain only the children are evaluated after generation 0."""
    config = tiny_config("best")
    rows = run_tiny(config, tmp_path / "run", pool)
    evaluations = [int(r["evaluations"]) for r in rows if r["island"] == "all"]
    per_generation = config.n_islands * (config.island_size - config.n_elites)
    assert evaluations[0] == config.n_islands * config.island_size
    assert all(b - a == per_generation for a, b in zip(evaluations, evaluations[1:]))


def test_conditions_with_the_same_seed_share_the_terrain(
    tmp_path: Path, pool: Pool
) -> None:
    shared = tmp_path / "terrains"
    for policy in ("best", "none"):
        config = tiny_config(policy)
        out = tmp_path / policy
        out.mkdir()
        Experiment(
            config,
            TINY_SIM,
            out,
            pool,
            world_factory=SimpleFlatWorld,
            terrain_dir=shared,
        ).evolve()
    first = (tmp_path / "best" / "config.json").read_text()
    second = (tmp_path / "none" / "config.json").read_text()
    assert str(shared / "terrain0.mjb") in first
    assert str(shared / "terrain0.mjb") in second
    assert sorted(p.name for p in shared.iterdir()) == ["terrain0.mjb"]
