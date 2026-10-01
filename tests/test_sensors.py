"""The vision rays, checked on ARIEL's flat world where the answers are known."""

import mujoco as mj
import numpy as np
import pytest
from ariel.simulation.environments import SimpleFlatWorld

from sensors import (
    CORE_BODY,
    LOOK_AHEAD_ANGLE,
    LOOK_DOWN_ANGLE,
    RAY_MAX_RANGE,
    RAY_ORIGIN,
    RAYS,
    absolute_position,
    n_inputs,
    read_inputs,
    vision,
)
from simulate import TARGET_XY, build_model
from terrain import ground_geoms

DOWN, UP = 0, 1
AHEAD = slice(2, 6)
STEEP = slice(6, 10)


@pytest.fixture(scope="module")
def flat() -> tuple[mj.MjModel, mj.MjData]:
    model = build_model(SimpleFlatWorld)
    return model, mj.MjData(model)


def read(model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    """Ray distances in metres."""
    mj.mj_forward(model, data)
    readings = vision(model, data, model.body(CORE_BODY).id, ground_geoms(model))
    return readings * RAY_MAX_RANGE


def test_ten_rays_make_34_inputs() -> None:
    assert len(RAYS) == 10
    assert n_inputs(vision=True) == 34
    assert n_inputs(vision=False) == 24


def test_upright_on_flat_ground(flat: tuple[mj.MjModel, mj.MjData]) -> None:
    model, data = flat
    mj.mj_resetData(model, data)
    distances = read(model, data)
    height = data.qpos[2] + RAY_ORIGIN[2] - data.geom_xpos[model.geom("floor").id, 2]

    assert distances[DOWN] == pytest.approx(height, abs=1e-6)
    assert distances[UP] == RAY_MAX_RANGE  # nothing above an upright robot
    ahead = height / np.sin(np.deg2rad(LOOK_AHEAD_ANGLE))
    steep = height / np.sin(np.deg2rad(LOOK_DOWN_ANGLE))
    np.testing.assert_allclose(distances[AHEAD], ahead, atol=1e-6)
    np.testing.assert_allclose(distances[STEEP], steep, atol=1e-6)


def test_upside_down_only_the_up_ray_sees_the_ground(
    flat: tuple[mj.MjModel, mj.MjData],
) -> None:
    model, data = flat
    mj.mj_resetData(model, data)
    data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # half a turn about x
    data.qpos[2] = 0.6
    distances = read(model, data)

    assert distances[UP] == pytest.approx(0.6 - RAY_ORIGIN[2], abs=0.01)
    assert np.all(np.delete(distances, UP) == RAY_MAX_RANGE)


def test_rays_turn_with_the_body(flat: tuple[mj.MjModel, mj.MjData]) -> None:
    """On flat ground a quarter turn about z must not change any reading."""
    model, data = flat
    mj.mj_resetData(model, data)
    level = read(model, data)
    data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    np.testing.assert_allclose(read(model, data), level, atol=1e-6)


def test_near_rays_are_down_plus_the_four_steep_ones(
    flat: tuple[mj.MjModel, mj.MjData],
) -> None:
    model, data = flat
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    core = model.body(CORE_BODY).id
    every = vision(model, data, core, ground_geoms(model))
    near = vision(model, data, core, ground_geoms(model), rays="near")
    np.testing.assert_array_equal(near, every[[DOWN, 6, 7, 8, 9]])
    assert n_inputs(vision=True, hinges=8, rays="near") == 8 + 8 + 5
    near3 = vision(model, data, core, ground_geoms(model), rays="near3")
    np.testing.assert_array_equal(near3, every[[DOWN, 6, 8]])  # ahead, behind
    down = vision(model, data, core, ground_geoms(model), rays="down")
    np.testing.assert_array_equal(down, every[[DOWN]])
    assert n_inputs(vision=True, hinges=8, rays="near3") == 8 + 8 + 3
    assert n_inputs(vision=False, hinges=8, rays="near") == 8 + 8
    with pytest.raises(ValueError, match="unknown ray set"):
        n_inputs(vision=True, rays="many")


def test_position_is_the_core_world_xy_over_two(
    flat: tuple[mj.MjModel, mj.MjData],
) -> None:
    """Experiment 29: the core's absolute (x, y), scaled like the target distance."""
    model, data = flat
    mj.mj_resetData(model, data)
    data.qpos[0:2] = [1.0, -0.5]
    mj.mj_forward(model, data)
    core = model.body(CORE_BODY).id
    expected = data.xpos[core, :2] / 2.0
    np.testing.assert_allclose(absolute_position(data, core), expected)
    assert expected[0] == pytest.approx(0.5, abs=0.01)


def test_position_adds_two_inputs_at_the_end(
    flat: tuple[mj.MjModel, mj.MjData],
) -> None:
    model, data = flat
    mj.mj_resetData(model, data)
    data.qpos[0:2] = [0.4, 0.3]
    mj.mj_forward(model, data)
    core = model.body(CORE_BODY).id
    without = read_inputs(model, data, core, TARGET_XY, None)
    with_position = read_inputs(model, data, core, TARGET_XY, None, position=True)
    assert n_inputs(vision=False, hinges=model.nu, position=True) == len(without) + 2
    assert len(with_position) == n_inputs(vision=False, hinges=model.nu, position=True)
    np.testing.assert_array_equal(with_position[:-2], without)
    np.testing.assert_allclose(with_position[-2:], absolute_position(data, core))
