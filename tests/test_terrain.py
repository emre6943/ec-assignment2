"""The ground of every world: one `floor` geom, or several pieces (OlympicArena)."""

import mujoco as mj
import numpy as np
import pytest
from ariel.simulation.environments import OlympicArena, SimpleFlatWorld

from sensors import CORE_BODY, RAY_MAX_RANGE, vision
from simulate import build_model
from terrain import (
    CORE_ABOVE_LOWEST_POINT,
    SPAWN_CLEARANCE,
    ground_geoms,
    ground_height,
)


def test_a_flat_world_has_one_ground_geom() -> None:
    model = build_model(SimpleFlatWorld, "spider_8")
    assert ground_geoms(model) == (model.geom("floor").id,)


@pytest.fixture(scope="module")
def olympic() -> tuple[mj.MjModel, mj.MjData]:
    model = build_model(OlympicArena, "spider_8")
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


def test_olympic_ground_includes_the_rugged_strip(
    olympic: tuple[mj.MjModel, mj.MjData],
) -> None:
    """The target (x = 2) lies on a heightfield that is not the `floor` geom."""
    model, data = olympic
    ground = ground_geoms(model)
    assert len(ground) > 1
    strips = [g for g in ground if model.geom_type[g] == mj.mjtGeom.mjGEOM_HFIELD]
    assert len(strips) == 1 and strips[0] != model.geom("floor").id
    robot = {model.body(CORE_BODY).id}
    assert not any(model.geom_bodyid[g] in robot for g in ground)
    # The strip is bumpy (a few cm) and there is ground under the target.
    heights = [ground_height(model, data, ground, x, 0.0) for x in (1.0, 1.5, 2.0)]
    assert all(np.isfinite(heights)) and max(map(abs, heights)) < 0.1


def test_olympic_spawn_and_vision_use_the_real_ground(
    olympic: tuple[mj.MjModel, mj.MjData],
) -> None:
    model, data = olympic
    start = ground_height(model, data, ground_geoms(model), 0.0, 0.0)
    assert data.qpos[2] == pytest.approx(
        start + CORE_ABOVE_LOWEST_POINT + SPAWN_CLEARANCE
    )
    readings = vision(model, data, model.body(CORE_BODY).id, ground_geoms(model))
    assert 0.0 < readings[0] * RAY_MAX_RANGE < 1.0  # the down ray hits the start
