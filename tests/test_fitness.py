"""The fitness: distance to the target plus the two posture penalties (D15)."""

from pathlib import Path

import mujoco as mj
import numpy as np
import pytest
from ariel.simulation.environments import SimpleFlatWorld

from sensors import CORE_BODY
from simulate import (
    SimConfig,
    build_model,
    core_touches_ground,
    fitness,
    is_upside_down,
    run_terrains,
    walk,
)
from terrain import ground_geoms

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
