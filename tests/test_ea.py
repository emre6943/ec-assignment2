"""A tiny end-to-end run of the island EA, and its settings and random streams.

The end-to-end runs use ARIEL's `SimpleFlatWorld`: rugged terrain is random on
every construction, so only a flat world makes two runs comparable exactly.
"""

import csv
import json
import multiprocessing as mp
import random
from collections.abc import Iterator
from dataclasses import replace
from itertools import pairwise
from multiprocessing.pool import Pool
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest
from ariel.ec import FloatMutator, set_seed
from ariel.simulation.environments import SimpleFlatWorld

from ea import EAConfig, Experiment, check_config, next_sigma, operator_rng
from replay import make_controller
from run import as_standard, finished_with
from simulate import SimConfig, build_model, final_sim_config

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
        ("final_duration", -1),
        ("final_duration_from", -1),
        ("spawn_yaws", "north"),
        ("spawn_yaws", "0,30"),  # more turns than the one terrain
        ("spawn_yaws", "270"),
        ("spawn_yaws", "nan"),
    ],
)
def test_impossible_settings_are_rejected(setting: str, value: int | str) -> None:
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


def test_standard_ea_is_one_population_of_the_same_size() -> None:
    island = EAConfig()
    standard = as_standard(island)
    assert standard.n_islands == 1
    assert standard.island_size == island.n_islands * island.island_size
    assert standard.n_elites == island.n_islands * island.n_elites
    assert standard.policy == "none"
    check_config(standard)


def test_skip_done_only_skips_identical_finished_runs(
    tmp_path: Path, pool: Pool
) -> None:
    config = tiny_config("best")
    run_tiny(config, tmp_path / "run", pool)
    assert finished_with(tmp_path / "run", config, TINY_SIM, "flat")
    assert not finished_with(
        tmp_path / "run", replace(config, n_migrants=2), TINY_SIM, "flat"
    )
    assert not finished_with(tmp_path / "run", config, TINY_SIM, "rugged")
    assert not finished_with(tmp_path / "missing", config, TINY_SIM, "flat")


def test_a_run_with_another_body_and_the_rhythm_options(
    tmp_path: Path, pool: Pool
) -> None:
    """spider_8 with an evolved tempo: the genotype carries one extra gene."""
    sim = replace(TINY_SIM, body="spider_8", evolve_tempo=True)
    config = replace(tiny_config("best"), clock_boost=3.0)
    random.seed(0)
    np.random.seed(0)
    set_seed(0)
    out = tmp_path / "run"
    out.mkdir()
    Experiment(config, sim, out, pool, world_factory=SimpleFlatWorld).evolve()
    genotype = np.load(out / "best_genotype.npy")
    assert genotype.shape == (sim.shape.n_weights + 1,)
    assert sim.shape.n_outputs == 8

    model = build_model(SimpleFlatWorld, "spider_8")
    control = make_controller(genotype, sim, model)
    data = mj.MjData(model)
    mj.set_mjcb_control(control)
    try:
        mj.mj_step(model, data, nstep=10)
    finally:
        mj.set_mjcb_control(None)
    assert np.all(np.isfinite(data.ctrl)) and np.any(data.ctrl != 0.0)


def test_stagnation_rule_widens_then_resets_sigma() -> None:
    config = EAConfig(mutation_sigma=0.05, stall_generations=3, max_sigma=0.15)
    sigma, stall = 0.05, 0
    history = []
    for _ in range(9):
        sigma, stall = next_sigma(config, sigma, stall, improved=False)
        history.append(sigma)
    assert history == [0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15]
    assert next_sigma(config, sigma, stall, improved=True) == (0.05, 0)
    assert next_sigma(EAConfig(), 0.05, 7, improved=False) == (0.05, 7)  # off


def test_a_run_logs_the_gait_terms_and_sigma(tmp_path: Path, pool: Pool) -> None:
    config = replace(tiny_config("best"), stall_generations=1)
    rows = run_tiny(config, tmp_path / "run", pool)
    for row in rows:
        assert 0.0 <= float(row["low_body"]) <= 1.0
        assert 0.0 <= float(row["leg_imbalance"]) <= 1.0
        assert config.mutation_sigma <= float(row["sigma"]) <= config.max_sigma
    # With a one-generation stall limit, some island must have widened sigma.
    assert any(float(row["sigma"]) > config.mutation_sigma for row in rows)


def test_stagnation_rule_needs_a_fixed_fitness() -> None:
    with pytest.raises(ValueError, match="stall_generations"):
        check_config(EAConfig(stall_generations=15, curriculum=True))
    with pytest.raises(ValueError, match="stall_generations"):
        check_config(EAConfig(stall_generations=15, terrain_mode="per_generation"))
    check_config(EAConfig(mutation_sigma=0.5))  # max_sigma only matters when on


def test_skip_done_reads_configs_from_before_new_settings(
    tmp_path: Path, pool: Pool
) -> None:
    """A run whose config predates a setting still counts as done (D1)."""
    config = tiny_config("best")
    run_tiny(config, tmp_path / "run", pool)
    saved_file = tmp_path / "run" / "config.json"
    saved = json.loads(saved_file.read_text())
    del saved["ea"]["stall_generations"], saved["sim"]["vision_rays"]
    saved_file.write_text(json.dumps(saved))
    assert finished_with(tmp_path / "run", config, TINY_SIM, "flat")


def test_walks_get_longer_once_and_everyone_walks_again(
    tmp_path: Path, pool: Pool
) -> None:
    """D21: one switch, after `final_duration_from` evaluations; elites re-walk."""
    config = replace(
        tiny_config("best"),
        final_duration=0.5,
        final_duration_from=20,
        stall_generations=1,
    )
    rows = [r for r in run_tiny(config, tmp_path / "run", pool) if r["island"] == "all"]
    durations = [float(row["duration"]) for row in rows]
    evaluations = [int(row["evaluations"]) for row in rows]
    switch = durations.index(0.5)
    assert set(durations[:switch]) == {TINY_SIM.duration}
    assert set(durations[switch:]) == {0.5}
    assert evaluations[switch - 2] < 20 <= evaluations[switch - 1]
    assert evaluations[switch] - evaluations[switch - 1] == (
        config.n_islands * config.island_size
    )
    assert float(rows[switch]["mean_seconds"]) > TINY_SIM.duration
    assert (tmp_path / "run" / "best_genotype_short.npy").exists()
    assert final_sim_config(tmp_path / "run").duration == 0.5  # replay, unseen


def test_replay_and_unseen_use_the_walks_a_run_ended_with(
    tmp_path: Path, pool: Pool
) -> None:
    """Before the switch (or before D21), a run's walks are the short ones."""
    config = replace(tiny_config("best"), final_duration=0.5, final_duration_from=10**6)
    run_tiny(config, tmp_path / "run", pool)
    assert final_sim_config(tmp_path / "run").duration == TINY_SIM.duration
    log_file = tmp_path / "run" / "log.csv"
    with log_file.open(newline="") as file:
        rows = list(csv.DictReader(file))
    fieldnames = [name for name in rows[0] if name != "duration"]
    with log_file.open("w", newline="") as file:  # a log from before D21
        writer = csv.DictWriter(file, fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    assert final_sim_config(tmp_path / "run").duration == TINY_SIM.duration
