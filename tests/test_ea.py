"""A tiny end-to-end run of the island EA, and its settings and random streams.

The end-to-end runs use ARIEL's `SimpleFlatWorld`: rugged terrain is random on
every construction, so only a flat world makes two runs comparable exactly.
"""

import csv
import multiprocessing as mp
import random
from collections.abc import Iterator
from dataclasses import replace
from itertools import pairwise
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import pytest
from ariel.ec import FloatMutator, set_seed
from ariel.simulation.environments import SimpleFlatWorld

from ea import EAConfig, Experiment, check_config, operator_rng
from simulate import SimConfig

TINY_SIM = SimConfig(duration=0.3, hidden_layers="4")


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
    assert all(b - a == per_generation for a, b in pairwise(evaluations))


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


def test_operator_stream_is_independent_of_ariels_mutation_stream() -> None:
    """Same run seed, but our operators and ARIEL's mutation draw different numbers."""
    set_seed(0)
    mutation_noise = np.asarray(
        FloatMutator.gaussian([0.0] * 50, std=1.0, mutation_probability=1.0)
    )
    ours = operator_rng(0).normal(size=50)
    assert abs(np.corrcoef(mutation_noise, ours)[0, 1]) < 0.5
    np.testing.assert_array_equal(
        operator_rng(0).normal(size=5), operator_rng(0).normal(size=5)
    )


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("n_elites", 20),
        ("tournament_size", 0),
        ("n_migrants", 21),
        ("curriculum_generations", 0),
        ("migration_interval", 0),
    ],
)
def test_impossible_settings_are_rejected(setting: str, value: int) -> None:
    with pytest.raises(ValueError, match=setting):
        check_config(replace(EAConfig(), **{setting: value}))


def test_log_has_best_final_and_best_is_monotone(tmp_path: Path, pool: Pool) -> None:
    """Without a curriculum, best_final equals best, and the best never gets worse."""
    rows = run_tiny(tiny_config("random"), tmp_path / "run", pool)
    everyone = [r for r in rows if r["island"] == "all"]
    assert all(float(r["best_final"]) == float(r["best"]) for r in everyone)
    best = [float(r["best_final"]) for r in everyone]
    assert all(later <= earlier + 1e-12 for earlier, later in pairwise(best))


def test_a_rerun_removes_the_previous_runs_outputs(tmp_path: Path, pool: Pool) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "unseen.json").write_text("{}")  # left over from an earlier run
    Experiment(
        tiny_config("none"), TINY_SIM, out, pool, world_factory=SimpleFlatWorld
    ).evolve()
    assert not (out / "unseen.json").exists()
    assert (out / "summary.json").exists()
