"""filmstrip.py: frame times, the replay against simulate.walk, the camera."""

import json
import platform

import mujoco as mj
import numpy as np
import pytest
from ariel.simulation.environments import SimpleFlatWorld

from filmstrip import (
    RUN,
    check_against_walk,
    frame_updates,
    over_ground,
    parse_times,
    replay,
    time_label,
    top_camera,
)
from simulate import SimConfig, build_model, final_sim_config, walk


def test_frame_times_are_seconds_then_the_end() -> None:
    assert parse_times("0,3.5, 7,end") == (0.0, 3.5, 7.0, None)
    assert parse_times("2") == (2.0,)


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        ("0,end,5", "last"),
        ("0,soon", "seconds"),
        ("3,2", "rise"),
        ("1,1", "rise"),
        ("-1", "not negative"),
        ("nan", "finite"),
    ],
)
def test_bad_frame_times_are_refused(text: str, complaint: str) -> None:
    with pytest.raises(ValueError, match=complaint):
        parse_times(text)


def test_a_frame_falls_on_the_nearest_network_update() -> None:
    # 50 updates a second for 15 s: update 750 is the walk's full length.
    assert frame_updates((0.0, 3.5, 3.511, None), 0.02, 750) == [0, 175, 176, 750]
    assert frame_updates((15.0,), 0.02, 750) == [750]
    with pytest.raises(ValueError, match="after the walk"):
        frame_updates((15.02,), 0.02, 750)
    with pytest.raises(ValueError, match="same network update"):
        frame_updates((1.0, 1.005), 0.02, 750)


def test_times_are_labelled_as_the_caption_gives_them() -> None:
    assert [time_label(t) for t in (0.0, 3.5, 13.840000000001291)] == [
        "0 s",
        "3.5 s",
        "13.84 s",
    ]


def test_the_replay_walks_exactly_as_simulate_walk() -> None:
    """Same end time and place, and a frame at every requested time."""
    model = build_model(SimpleFlatWorld)
    config = SimConfig(duration=1.0)
    genotype = np.random.default_rng(0).normal(0, 0.5, config.genotype_length)
    result = replay(genotype, model, config, (0.0, 0.5, None))
    score = walk(genotype, model, config)
    assert (result.ended, result.distance) == (score.seconds, score.distance)
    check_against_walk(result, genotype, model, config)  # does not raise
    assert [round(time, 6) for time, _ in result.poses] == [0.0, 0.5, 1.0]
    assert not result.arrived
    x_low, x_high, y_low, y_high = result.box
    assert x_low < 0 < x_high
    assert y_low < 0 < y_high


@pytest.mark.skipif(
    not (RUN / "best_genotype.npy").exists(), reason="the final experiment's runs"
)
def test_the_default_run_arrives_as_recorded() -> None:
    """Seed 25 ran on ARM; only there does its walk repeat to the last digit."""
    model = mj.MjModel.from_binary_path(
        json.loads((RUN / "config.json").read_text())["terrains"][0]
    )
    config = final_sim_config(RUN)
    genotype = np.load(RUN / "best_genotype.npy")
    result = replay(genotype, model, config, (0.0, None))
    check_against_walk(result, genotype, model, config)
    if platform.machine() in ("arm64", "aarch64"):
        # On another processor the walk drifts and may not arrive at all.
        assert result.arrived
        recorded = json.loads((RUN / "longer_walks.json").read_text())["15"]
        assert result.ended == recorded["arrived_at"]
    with pytest.raises(ValueError, match="after the walk ended"):
        replay(genotype, model, config, (14.5,))


def test_a_raised_mark_covers_its_ground_point() -> None:
    camera = mj.MjvCamera()
    camera.lookat[:] = [1.0, 0.5, 0.0]
    camera.distance = 4.0
    ground = np.array([2.0, 0.0])
    mark, shrink = over_ground(camera, ground, 1.0)
    assert shrink == pytest.approx(0.75)
    eye = np.array([1.0, 0.5, 4.0])
    # The mark lies on the line from the camera to the ground point.
    direction = np.array([*ground, 0.0]) - eye
    assert np.cross(mark - eye, direction) == pytest.approx(np.zeros(3))


TINY_WORLD = """
<mujoco>
  <visual><global offwidth="64" offheight="64"/></visual>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1" rgba="1 1 1 1"/>
    <geom type="box" pos="1 0 0.05" size="0.2 0.2 0.05" rgba="1 0 0 1"/>
    <geom type="box" pos="0 1 0.05" size="0.2 0.2 0.05" rgba="0 0 1 1"/>
  </worldbody>
</mujoco>
"""


def test_the_camera_looks_down_with_x_up_and_y_left() -> None:
    model = mj.MjModel.from_xml_string(TINY_WORLD)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    try:
        renderer = mj.Renderer(model, 64, 64)
    except Exception as error:  # no OpenGL here
        pytest.skip(f"no offscreen rendering: {error}")
    with renderer:
        renderer.update_scene(data, top_camera(model, (-1.5, 1.5, -1.5, 1.5)))
        image = renderer.render().astype(int)
    rows, columns = np.indices(image.shape[:2])
    red = (image[..., 0] > 150) & (image[..., 1] < 80) & (image[..., 2] < 80)
    blue = (image[..., 2] > 150) & (image[..., 0] < 80) & (image[..., 1] < 80)
    assert red.any() and blue.any()
    # +x (red) above the centre, +y (blue) left of it.
    assert rows[red].mean() < 32 - 10
    assert abs(columns[red].mean() - 32) < 3
    assert columns[blue].mean() < 32 - 10
    assert abs(rows[blue].mean() - 32) < 3
