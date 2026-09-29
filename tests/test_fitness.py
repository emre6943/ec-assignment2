"""The fitness: distance to the target plus the two posture penalties (D15)."""

from pathlib import Path

import mujoco as mj
import numpy as np
import pytest
from ariel.simulation.environments import SimpleFlatWorld

from bodies import body_info
from sensors import CORE_BODY
from simulate import (
    CARRY_HEIGHT,
    Score,
    SimConfig,
    build_model,
    carry_shortfall,
    core_touches_ground,
    fitness,
    is_upside_down,
    leg_imbalance,
    run_terrains,
    walk,
    work_imbalance,
)
from terrain import CORE_ABOVE_LOWEST_POINT, ground_geoms

CONFIG = SimConfig(duration=2.0)


@pytest.fixture(scope="module")
def flat_model() -> mj.MjModel:
    return build_model(SimpleFlatWorld)


def test_a_limp_robot_lies_on_the_ground(flat_model: mj.MjModel) -> None:
    """All-zero weights hold every hinge straight: the core rests on the floor."""
    score = walk(np.zeros(CONFIG.shape.n_weights), flat_model, CONFIG)
    assert score.ground_contact > 0.8
    assert score.upside_down == 0.0
    assert score.distance == pytest.approx(2.0, abs=0.05)
    expected = score.distance + CONFIG.ground_contact_weight * score.ground_contact
    assert fitness(score, CONFIG) == pytest.approx(expected)
    assert score.seconds == pytest.approx(CONFIG.duration, abs=0.03)


def test_movement_reward_lowers_the_fitness(flat_model: mj.MjModel) -> None:
    score = walk(np.zeros(CONFIG.shape.n_weights), flat_model, CONFIG)
    rewarded = fitness(score, CONFIG, movement_weight=0.5)
    assert rewarded == pytest.approx(fitness(score, CONFIG) - 0.5 * score.displacement)


def test_hopeless_walk_is_stopped_early(flat_model: mj.MjModel) -> None:
    """A limp robot makes no progress, so it is cut at the check time."""
    config = SimConfig(duration=6.0, early_stop_time=1.0)
    limp = np.zeros(config.shape.n_weights)
    cut = walk(limp, flat_model, config, min_progress=0.05)
    full = walk(limp, flat_model, config)
    assert cut.seconds == pytest.approx(1.0, abs=0.03)
    assert full.seconds == pytest.approx(6.0, abs=0.03)
    assert cut.distance == pytest.approx(full.distance, abs=0.01)


def test_ground_contact_is_detected_only_when_touching(flat_model: mj.MjModel) -> None:
    data = mj.MjData(flat_model)
    core = flat_model.geom(CORE_BODY).id
    ground = ground_geoms(flat_model)

    mj.mj_resetData(flat_model, data)
    data.qpos[2] = 1.0  # hold the robot a metre up
    mj.mj_forward(flat_model, data)
    assert not core_touches_ground(data, core, ground)

    mj.mj_resetData(flat_model, data)
    while data.time < 1.0:  # let it drop and settle
        mj.mj_step(flat_model, data)
    assert core_touches_ground(data, core, ground)


def test_upside_down_is_detected(flat_model: mj.MjModel) -> None:
    data = mj.MjData(flat_model)
    core_id = flat_model.body(CORE_BODY).id

    mj.mj_resetData(flat_model, data)
    mj.mj_forward(flat_model, data)
    assert not is_upside_down(data, core_id)

    data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # half a turn about x
    mj.mj_forward(flat_model, data)
    assert is_upside_down(data, core_id)

    data.qpos[3:7] = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0, 0.0]  # 60° tilt
    mj.mj_forward(flat_model, data)
    assert not is_upside_down(data, core_id)


def test_run_terrains_reuses_existing_files(tmp_path: Path) -> None:
    first = run_terrains(tmp_path, 1, SimpleFlatWorld)
    stamp = (tmp_path / "terrain0.mjb").stat().st_mtime_ns
    second = run_terrains(tmp_path, 1, SimpleFlatWorld)
    assert first == second
    assert (tmp_path / "terrain0.mjb").stat().st_mtime_ns == stamp
    assert sorted(p.name for p in tmp_path.iterdir()) == ["terrain0.mjb"]


def test_carry_shortfall_is_lenient_and_bounded() -> None:
    """0 once the core is CARRY_HEIGHT up, 1 lying down, linear in between (D18)."""
    lying = CORE_ABOVE_LOWEST_POINT
    assert carry_shortfall(lying) == 1.0
    assert carry_shortfall(lying - 0.01) == 1.0  # pressed into the ground
    assert carry_shortfall(lying + CARRY_HEIGHT / 2) == pytest.approx(0.5)
    assert carry_shortfall(lying + CARRY_HEIGHT) == 0.0
    assert carry_shortfall(lying + 0.06) == 0.0


def test_leg_imbalance_punishes_an_unused_leg() -> None:
    legs = ((0, 1), (2, 3), (4, 5), (6, 7))
    t = np.linspace(0, 10, 500)[:, None]
    all_move = np.sin(t) * np.ones((1, 8))
    assert leg_imbalance(all_move, legs) == pytest.approx(0.0)
    one_dead = all_move.copy()
    one_dead[:, 0:2] = 1.2  # leg 0 held still
    assert leg_imbalance(one_dead, legs) == pytest.approx(1.0)
    half = all_move.copy()
    half[:, 0:2] *= 0.5  # leg 0 moves half as much as the others
    assert 0.0 < leg_imbalance(half, legs) < 1.0
    assert leg_imbalance(np.zeros((500, 8)), legs) == 0.0  # nothing moves at all


def test_spider_8_has_four_legs_of_two_hinges() -> None:
    assert body_info("spider_8").legs == ((0, 1), (2, 3), (4, 5), (6, 7))


def test_new_terms_only_count_when_weighted() -> None:
    score = Score(1.0, 0.0, 0.0, 0.0, 10.0, low_body=0.5, leg_imbalance=0.4)
    assert fitness(score, SimConfig()) == pytest.approx(1.0)
    weighted = SimConfig(low_body_weight=0.5, leg_imbalance_weight=0.5)
    assert fitness(score, weighted) == pytest.approx(1.0 + 0.25 + 0.2)


def test_mean_distance_of_a_robot_that_stays_put(flat_model: mj.MjModel) -> None:
    """Standing still, the walk-averaged distance equals the final distance (D20)."""
    config = SimConfig(duration=1.0, hidden_layers="4")
    score = walk(np.zeros(config.genotype_length), flat_model, config)
    assert score.mean_distance == pytest.approx(score.distance, abs=0.01)


def test_arriving_counts_the_rest_of_the_walk_as_zero(
    flat_model: mj.MjModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A robot that starts on the target arrives at once: average distance 0."""
    monkeypatch.setattr("simulate.TARGET_XY", np.array([0.0, 0.0]))
    config = SimConfig(duration=1.0, hidden_layers="4", stop_at_target=True)
    score = walk(np.zeros(config.genotype_length), flat_model, config)
    assert score.seconds < 0.1
    assert score.mean_distance == pytest.approx(0.0, abs=1e-6)


def test_speed_term_prefers_the_faster_of_two_equal_walks() -> None:
    """1 m closed in 10 s beats 1 m closed in 15 s (their final distance is equal)."""
    config = SimConfig(speed_weight=0.5)
    fast = Score(1.0, 1.0, 0.0, 0.0, 15.0, mean_distance=(1.5 * 10 + 1.0 * 5) / 15)
    slow = Score(1.0, 1.0, 0.0, 0.0, 15.0, mean_distance=1.5)
    assert fitness(fast, config) < fitness(slow, config)


def test_work_imbalance_finds_the_lazy_leg() -> None:
    legs = ((0, 1), (2, 3), (4, 5), (6, 7))
    assert work_imbalance(np.full(8, 5.0), legs) == pytest.approx(0.0)
    lazy = np.full(8, 5.0)
    lazy[4:6] = 0.0  # leg 2's motors never drive it
    assert work_imbalance(lazy, legs) == pytest.approx(1.0)
    assert work_imbalance(np.zeros(8), legs) == 0.0


def test_carry_height_is_configurable() -> None:
    lying = CORE_ABOVE_LOWEST_POINT
    assert carry_shortfall(lying + 0.02, carry_height=0.04) == pytest.approx(0.5)
    assert carry_shortfall(lying + 0.04, carry_height=0.04) == pytest.approx(
        0.0, abs=1e-12
    )


def test_a_walk_measures_motor_work(flat_model: mj.MjModel) -> None:
    config = SimConfig(duration=1.0, hidden_layers="4")
    rng = np.random.default_rng(0)
    score = walk(rng.normal(0, 1, config.genotype_length), flat_model, config)
    assert 0.0 <= score.work_imbalance <= 1.0
