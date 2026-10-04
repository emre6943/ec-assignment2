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
from ariel.ec import FloatMutator, Population, set_seed
from ariel.simulation.environments import SimpleFlatWorld

from ea import (
    SCORE_PARTS,
    EAConfig,
    Experiment,
    check_config,
    finite_or_failed,
    new_individual,
    next_sigma,
    operator_rng,
)
from replay import make_controller
from run import as_standard, finished_with
from simulate import (
    FAILED_SCORE,
    Score,
    SimConfig,
    build_model,
    final_sim_config,
    fitness,
)

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
        ("crossover", "two-point"),
    ],
)
def test_impossible_settings_are_rejected(setting: str, value: int | str) -> None:
    with pytest.raises(ValueError, match=setting):
        check_config(replace(EAConfig(), **{setting: value}))


GOOD_SCORE = Score(
    distance=0.1, displacement=1.9, ground_contact=0.0, upside_down=0.0, seconds=15.0
)


@pytest.mark.parametrize("part", SCORE_PARTS)
def test_a_score_with_any_non_finite_part_counts_as_failed(part: str) -> None:
    assert finite_or_failed(GOOD_SCORE) == GOOD_SCORE
    for bad in (np.nan, np.inf, -np.inf):
        assert finite_or_failed(replace(GOOD_SCORE, **{part: bad})) == FAILED_SCORE


class NaNPool:
    """Stands in for the worker pool: every walk ends near the target, but its
    ground contact is NaN."""

    def map(self, function: object, tasks: list[object]) -> list[Score]:
        return [replace(GOOD_SCORE, ground_contact=np.nan) for _ in tasks]


def test_a_walk_with_a_nan_part_cannot_win(tmp_path: Path) -> None:
    """A NaN fitness would win every `np.argmin`; it must get the failed one."""
    experiment = Experiment(
        tiny_config("best"),
        TINY_SIM,
        tmp_path,
        NaNPool(),
        world_factory=SimpleFlatWorld,
    )
    population = experiment.evaluate(experiment.initial_population())
    expected = fitness(FAILED_SCORE, TINY_SIM)
    assert all(individual.fitness == expected for individual in population)


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
    del saved["ea"]["crossover"]
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


@pytest.mark.parametrize("kind", ["weight", "blx", "headless"])
def test_every_crossover_kind_runs(kind: str, tmp_path: Path, pool: Pool) -> None:
    """Experiment 25's operators plug into the same EA."""
    config = replace(tiny_config("best"), crossover=kind, crossover_probability=1.0)
    rows = run_tiny(config, tmp_path / "run", pool)
    assert int(rows[-1]["evaluations"]) >= config.max_evaluations


@pytest.mark.parametrize(
    ("kind", "brings_new_genes"), [("neuron", False), ("headless", True)]
)
def test_headless_crosses_with_a_random_genotype(
    kind: str, brings_new_genes: bool, tmp_path: Path, pool: Pool
) -> None:
    """Between all-zero parents, only headless crossover brings in non-zero genes,
    and its random partner has the full length, tempo gene included (D7)."""
    sim = replace(TINY_SIM, evolve_tempo=True)
    config = replace(tiny_config("best"), crossover=kind, crossover_probability=1.0)
    experiment = Experiment(config, sim, tmp_path, pool, world_factory=SimpleFlatWorld)
    length = sim.shape.n_weights + 1
    members = [new_individual(np.zeros(length), 0) for _ in range(config.island_size)]
    child = experiment.make_child(members, np.zeros(config.island_size), sigma=0.0)
    assert child.shape == (length,)
    assert bool(np.any(child != 0.0)) == brings_new_genes


def test_a_run_can_start_from_saved_genotypes(tmp_path: Path, pool: Pool) -> None:
    """`init_from` deals saved networks round-robin over the islands, so each
    island gets its share; random networks fill every other slot (experiment X)."""
    saved = np.random.default_rng(5).normal(size=(4, TINY_SIM.genotype_length))
    np.save(tmp_path / "saved.npy", saved)
    config = replace(tiny_config("best"), init_from=str(tmp_path / "saved.npy"))
    experiment = Experiment(
        config, TINY_SIM, tmp_path, pool, world_factory=SimpleFlatWorld
    )
    population = experiment.initial_population()
    assert len(population) == config.n_islands * config.island_size
    for k, genotype in enumerate(saved):
        island = population_of(population, k % config.n_islands)
        assert island[k // config.n_islands] == genotype.tolist()
    from_file = [ind.genotype for ind in population if ind.genotype in saved.tolist()]
    assert len(from_file) == len(saved)


def test_without_init_from_the_first_population_is_unchanged(
    tmp_path: Path, pool: Pool
) -> None:
    """Every slot is a random draw, in island order, as before `init_from`
    existed: earlier runs still reproduce exactly."""
    config = tiny_config("best")
    population = Experiment(
        config, TINY_SIM, tmp_path, pool, world_factory=SimpleFlatWorld
    ).initial_population()
    fresh = Experiment(config, TINY_SIM, tmp_path, pool, world_factory=SimpleFlatWorld)
    expected = [
        fresh.random_individual().tolist()
        for _ in range(config.n_islands * config.island_size)
    ]
    assert [ind.genotype for ind in population] == expected


@pytest.mark.parametrize(
    ("rows", "length", "message"),
    [
        (2, TINY_SIM.genotype_length + 1, "weights"),
        (13, TINY_SIM.genotype_length, "population"),
    ],
)
def test_init_from_rejects_genotypes_that_do_not_fit(
    rows: int, length: int, message: str, tmp_path: Path, pool: Pool
) -> None:
    np.save(tmp_path / "saved.npy", np.zeros((rows, length)))
    config = replace(tiny_config("best"), init_from=str(tmp_path / "saved.npy"))
    with pytest.raises(ValueError, match=message):
        Experiment(config, TINY_SIM, tmp_path, pool, world_factory=SimpleFlatWorld)


def test_init_from_is_for_the_island_ea_only() -> None:
    config = replace(EAConfig(), algorithm="random_search", init_from="saved.npy")
    with pytest.raises(ValueError, match="init_from"):
        check_config(config)


def population_of(population: Population, island: int) -> list[list[float]]:
    """The genotypes on one island, in population order."""
    return [ind.genotype for ind in population if ind.tags["island"] == island]


def test_an_interrupted_run_resumes_where_its_database_ends(
    tmp_path: Path, pool: Pool
) -> None:
    """`resume` continues from the last saved generation: the log goes on
    without a gap or a repeat, the best brain is kept, and the old database
    is kept beside the new one."""
    out = tmp_path / "run"
    config = replace(tiny_config("best"), max_evaluations=30)  # generations 0-2
    run_tiny(config, out, pool)
    best_before = np.load(out / "best_genotype.npy")
    (out / "summary.json").unlink()  # as if the power went out after generation 2

    longer = replace(config, max_evaluations=48)  # resuming may extend the budget
    Experiment(
        longer, TINY_SIM, out, pool, world_factory=SimpleFlatWorld, resume=True
    ).evolve()

    with (out / "log.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["island"] == "all"]
    assert [int(row["generation"]) for row in rows] == [0, 1, 2, 3, 4]
    assert [int(row["evaluations"]) for row in rows] == [12, 21, 30, 39, 48]
    best = [float(row["best_final"]) for row in rows]
    assert all(later <= earlier for earlier, later in pairwise(best))
    if best[-1] == best[2]:
        np.testing.assert_array_equal(np.load(out / "best_genotype.npy"), best_before)
    assert (out / "database_part1.db").exists() and (out / "database.db").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["generations"] == 4 and summary["evaluations"] == 48


def test_resume_refuses_a_run_with_other_settings(tmp_path: Path, pool: Pool) -> None:
    out = tmp_path / "run"
    run_tiny(tiny_config("best"), out, pool)
    (out / "summary.json").unlink()
    with pytest.raises(ValueError, match="other settings"):
        Experiment(
            tiny_config("worst"),
            TINY_SIM,
            out,
            pool,
            world_factory=SimpleFlatWorld,
            resume=True,
        )


def test_resume_without_a_database_starts_fresh(tmp_path: Path, pool: Pool) -> None:
    out = tmp_path / "run"
    out.mkdir()
    Experiment(
        tiny_config("best"),
        TINY_SIM,
        out,
        pool,
        world_factory=SimpleFlatWorld,
        resume=True,
    ).evolve()
    assert (out / "summary.json").exists()


def interrupted_run(tmp_path: Path, pool: Pool) -> tuple[Path, EAConfig]:
    """A tiny finished run (generations 0-2), made to look cut off."""
    out = tmp_path / "run"
    config = replace(tiny_config("best"), max_evaluations=30)
    run_tiny(config, out, pool)
    (out / "summary.json").unlink()
    return out, replace(config, max_evaluations=48)


def logged_generations(out: Path) -> list[int]:
    with (out / "log.csv").open() as handle:
        rows = csv.DictReader(handle)
        return [int(row["generation"]) for row in rows if row["island"] == "all"]


def resume(config: EAConfig, out: Path, pool: Pool) -> Experiment:
    return Experiment(
        config, TINY_SIM, out, pool, world_factory=SimpleFlatWorld, resume=True
    )


def test_a_log_that_fell_behind_the_database_sets_where_to_resume(
    tmp_path: Path, pool: Pool
) -> None:
    """A power cut can lose log lines the database kept (logs written before
    they were put on disk): the run continues from the log's last generation,
    so the evaluation count, and with it the budget, stays right."""
    out, longer = interrupted_run(tmp_path, pool)
    lines = (out / "log.csv").read_text().splitlines(keepends=True)
    rows_per_generation = tiny_config("best").n_islands + 1
    (out / "log.csv").write_text("".join(lines[: 1 + 2 * rows_per_generation]))
    experiment = resume(longer, out, pool)
    assert (experiment.generation, experiment.evaluations) == (1, 21)
    assert len(experiment.resumed or []) == 12
    experiment.evolve()
    assert logged_generations(out) == [0, 1, 2, 3, 4]
    assert json.loads((out / "summary.json").read_text())["evaluations"] == 48


def test_a_log_ahead_of_the_database_is_cut_back(tmp_path: Path, pool: Pool) -> None:
    """The lost generation's log rows go; the run picks up after generation 2."""
    out, longer = interrupted_run(tmp_path, pool)
    lines = (out / "log.csv").read_text().splitlines(keepends=True)
    ahead = [line.replace("2,30,", "3,39,", 1) for line in lines[-4:]]
    (out / "log.csv").write_text("".join(lines + ahead))
    experiment = resume(longer, out, pool)
    assert (experiment.generation, experiment.evaluations) == (2, 30)
    assert logged_generations(out) == [0, 1, 2]


def test_a_refused_resume_changes_nothing(tmp_path: Path, pool: Pool) -> None:
    out, longer = interrupted_run(tmp_path, pool)
    header = (out / "log.csv").read_text().splitlines(keepends=True)[0]
    (out / "log.csv").write_text(header)  # e.g. a log lost entirely
    with pytest.raises(ValueError, match="no finished generation"):
        resume(longer, out, pool)
    assert (out / "database.db").exists()
    assert not list(out.glob("database_part*.db"))


def test_a_cut_during_a_resume_continues_from_the_set_aside_database(
    tmp_path: Path, pool: Pool
) -> None:
    """Stopped after the old database became a part, before a new one held a
    generation: the next resume reads the part instead of starting over."""
    out, longer = interrupted_run(tmp_path, pool)
    (out / "database.db").rename(out / "database_part1.db")
    experiment = resume(longer, out, pool)
    assert experiment.generation == 2
    experiment.evolve()
    assert logged_generations(out) == [0, 1, 2, 3, 4]
    assert sorted(p.name for p in out.glob("database*.db")) == [
        "database.db",
        "database_part1.db",
    ]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["resumed_after"] == [2]


@pytest.mark.parametrize(
    ("setting", "value"),
    [("stall_generations", 3), ("final_duration", 0.6), ("curriculum", True)],
)
def test_runs_with_state_outside_the_database_cannot_resume(
    setting: str, value: object, tmp_path: Path, pool: Pool
) -> None:
    out = tmp_path / "run"
    config = replace(tiny_config("best"), max_evaluations=30, **{setting: value})
    run_tiny(config, out, pool)
    (out / "summary.json").unlink()
    with pytest.raises(ValueError, match="state the database does not hold"):
        Experiment(
            config, TINY_SIM, out, pool, world_factory=SimpleFlatWorld, resume=True
        )
    assert (out / "database.db").exists()


def test_a_fresh_start_removes_every_output_of_an_earlier_run(
    tmp_path: Path, pool: Pool
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    for stale in ("longer_walks.json", "unseen_30s.json", "database_part1.db"):
        (out / stale).write_text("{}")
    Experiment(tiny_config("none"), TINY_SIM, out, pool, world_factory=SimpleFlatWorld)
    assert not any(
        (out / stale).exists()
        for stale in ("longer_walks.json", "unseen_30s.json", "database_part1.db")
    )


def test_a_resume_does_not_need_the_init_from_file(tmp_path: Path, pool: Pool) -> None:
    saved = tmp_path / "saved.npy"
    np.save(saved, np.zeros((2, TINY_SIM.genotype_length)))
    out = tmp_path / "run"
    config = replace(tiny_config("best"), max_evaluations=30, init_from=str(saved))
    run_tiny(config, out, pool)
    (out / "summary.json").unlink()
    saved.unlink()  # moved or deleted since the run started
    assert resume(config, out, pool).generation == 2


def test_a_resumed_best_keeps_its_exact_fitness(tmp_path: Path, pool: Pool) -> None:
    """The log rounds to 4 decimals; the database's best is exact."""
    out, longer = interrupted_run(tmp_path, pool)
    experiment = resume(longer, out, pool)
    with (out / "log.csv").open() as handle:
        logged = [float(r["best_final"]) for r in csv.DictReader(handle)]
    assert experiment.best_fitness == pytest.approx(min(logged), abs=5e-5)
    assert experiment.best_fitness != round(experiment.best_fitness, 4)
    np.testing.assert_array_equal(
        np.load(out / "best_genotype.npy"), experiment.best_genotype
    )


def test_a_fresh_start_removes_an_old_database_journal(
    tmp_path: Path, pool: Pool
) -> None:
    """A crash during a save leaves a journal; a new database must not replay it."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "database.db-journal").write_text("old")
    Experiment(tiny_config("none"), TINY_SIM, out, pool, world_factory=SimpleFlatWorld)
    assert not (out / "database.db-journal").exists()
