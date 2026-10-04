"""The paper's Figure 1 (fig:walk): an evolved controller walking to the target.

    uv run --project ../ariel python filmstrip.py
    uv run --project ../ariel python filmstrip.py results/final/best/seed21 \\
        --times 0,5,10,end --out walk.png

Walks a run's best genotype (`best_genotype.npy`) on the run's own training
arena (`config.json` "terrains"[0], saved with the robot spawned on it) with
the run's final walk settings (`simulate.final_sim_config`), and draws the
robot from straight above at the given times, side by side, into
`report/figures/walk_filmstrip.png` (`--out`). +x, the way to the target,
points up in every frame; each frame shows everything the robot covered
during the walk, so the flat start and the bumpy strip from x = 0.5 m. A red
circle marks the target: its centre (2, 0) m and the 0.1 m within which it
counts as reached. Each frame is labelled with its time, and the frame times
and the arrival time are printed for the caption.

`--times` takes seconds, and "end" for the moment the walk ends: on arrival
(the walk stops there), or after the walk's full length. The default,
0,3.5,7,10.5,end, puts the last frame at the arrival of the default run.

The walk is exactly the experiment's: the replay below repeats
`simulate.walk`'s control timing step for step (the network reads its inputs
after `mj_step1` of every `control_every`-th physics step, its output is held
for the steps in between, and the walk stops when the core comes within
`TARGET_RADIUS` of the target), and before drawing anything the script checks
that its replay ends at the same time and place as `simulate.walk` itself.
Frames are drawn from a copy of the simulation state, so drawing cannot
change the walk. The target is drawn into the renderer's scene only (user
geoms); the model and ARIEL are not changed.

The default run is the standard EA's seed 25 rather than the seed 11 of an
earlier version of this figure: MuJoCo's results differ between x86 and ARM
processors, and the final experiment ran seeds 10-19 on an Intel i7 (x86) and
seeds 20-29 on an Apple M3 Pro (ARM). On an ARM machine only seeds 20-29 repeat
their recorded walks; seed 11, which arrived after 10.3 s on its own machine,
does not arrive here at all. The script compares its walk with the run's
`longer_walks.json` and warns if they differ.
"""

# Standard library
import argparse
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

# Local libraries (ARIEL)
from ariel.simulation.tasks.targeted_locomotion import distance_to_target

# Local libraries
from genome import split
from network import Genotype, layers, run_layers
from paper_figures import COLUMN
from sensors import CORE_BODY, HALF_PI, read_inputs
from simulate import TARGET_RADIUS, TARGET_XY, SimConfig, final_sim_config, walk
from terrain import ground_geoms

ROOT = Path(__file__).parent
RUN = ROOT / "results" / "final" / "standard" / "seed25"
OUT = ROOT / "report" / "figures" / "walk_filmstrip.png"
TIMES = "0,3.5,7,10.5,end"
END = "end"

DPI = 300  # at the paper's column width, where it is shown
GAP = 6  # white pixels between frames: 0.5 mm in print
SUPERSAMPLE = 2  # render at twice the output size, then shrink: smooth edges
MARGIN = 0.05  # metres of ground around everything the robot covered
LABEL_POINTS = 7  # the time above each frame, as the other figures' text
# The target: a red circle of TARGET_RADIUS and a red dot at its centre, each
# on a wider white outline that keeps them visible on the bumps. They float
# above the robot, so nothing hides them, shrunk towards the camera so that
# they cover exactly the target's circle on the ground (`over_ground`).
TARGET_RED = np.array([0.85, 0.05, 0.05, 1.0], dtype=np.float32)
OUTLINE_WHITE = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
TARGET_HEIGHT = 0.3  # metres above the ground's zero: above the robot
RING_WIDTH = 0.022  # metres
DOT_RADIUS = 0.022  # metres
OUTLINE = 0.008  # metres of white on each side
RING_SEGMENTS = 64


@dataclass(frozen=True)
class Replay:
    """A walk, with the robot's pose at each requested frame.

    `poses` holds (time, qpos) per frame; `ended` is when the walk ended,
    `arrived` whether that was on arrival; `distance` is the final distance
    to the target; `box` the (x min, x max, y min, y max) the robot's geoms
    covered, including their size.
    """

    poses: list[tuple[float, npt.NDArray[np.float64]]]
    ended: float
    arrived: bool
    distance: float
    box: tuple[float, float, float, float]


def parse_times(text: str) -> tuple[float | None, ...]:
    """Frame times from text like "0,3.5,7,end": seconds, None for "end".

    Raises ValueError unless the times are finite, not negative and rising,
    with "end" (the moment the walk ends) only last.
    """
    parts = [part.strip() for part in text.split(",")]
    times: list[float | None] = []
    for index, part in enumerate(parts):
        if part == END:
            if index != len(parts) - 1:
                msg = f'"{END}" must be the last frame time, got {text!r}'
                raise ValueError(msg)
            times.append(None)
            continue
        try:
            seconds = float(part)
        except ValueError:
            msg = f'frame times are seconds or "{END}", got {part!r}'
            raise ValueError(msg) from None
        if not np.isfinite(seconds) or seconds < 0:
            msg = f"frame times must be finite and not negative, got {part!r}"
            raise ValueError(msg)
        if times and seconds <= times[-1]:
            msg = f"frame times must rise, got {text!r}"
            raise ValueError(msg)
        times.append(seconds)
    return tuple(times)


def frame_updates(
    times: tuple[float | None, ...], update_period: float, n_updates: int
) -> list[int]:
    """The network update at which each frame is taken; n_updates for "end".

    A time falls on the nearest update (every `update_period` seconds); the
    walk's full length is update `n_updates`. Raises ValueError for a time
    beyond the walk or two times on the same update.
    """
    updates = [
        n_updates if time is None else round(time / update_period) for time in times
    ]
    length = n_updates * update_period
    for time, update in zip(times, updates, strict=True):
        if update > n_updates:
            msg = f"frame time {time} s is after the walk's {length:g} s"
            raise ValueError(msg)
    if len(set(updates)) != len(updates):
        msg = f"frame times {times} fall on the same network update"
        raise ValueError(msg)
    return updates


def robot_geoms(model: mj.MjModel) -> npt.NDArray[np.int_]:
    """The geoms that move with the robot (not welded to the world)."""
    return np.array(
        [g for g in range(model.ngeom) if model.body_weldid[model.geom_bodyid[g]] != 0]
    )


def replay(
    genotype: Genotype,
    model: mj.MjModel,
    config: SimConfig,
    times: tuple[float | None, ...],
) -> Replay:
    """Walk as `simulate.walk` does, keeping the robot's pose at `times`.

    The loop is `simulate.walk`'s, without what only measures: the same calls
    in the same order on the same state, so the same walk. A frame time
    after the walk ended on arrival raises ValueError.
    """
    shape = config.shape
    weights, clock_hz = split(genotype, shape, config.evolve_tempo)
    network = layers(weights, shape)
    core_id = model.body(CORE_BODY).id
    vision_ground = ground_geoms(model) if config.vision else None
    period = model.opt.timestep * config.control_every
    n_updates = round(config.duration / period)
    wanted = frame_updates(times, period, n_updates)
    moving = robot_geoms(model)
    reach = model.geom_rbound[moving]
    low, high = np.full(2, np.inf), np.full(2, -np.inf)

    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    poses: dict[int, tuple[float, npt.NDArray[np.float64]]] = {}
    arrived = False
    for update in range(n_updates):
        mj.mj_step1(model, data)
        xy = data.geom_xpos[moving, :2]
        low = np.minimum(low, (xy - reach[:, None]).min(axis=0))
        high = np.maximum(high, (xy + reach[:, None]).max(axis=0))
        if update in wanted:
            poses[update] = (float(data.time), data.qpos.copy())
        if config.stop_at_target and (
            distance_to_target(data.qpos[0:2], TARGET_XY) < TARGET_RADIUS
        ):
            arrived = True
            break
        inputs = read_inputs(
            model,
            data,
            core_id,
            TARGET_XY,
            vision_ground,
            clock_hz,
            config.vision_rays,
            config.position,
        )
        data.ctrl[:] = run_layers(network, inputs) * HALF_PI
        mj.mj_step2(model, data)
        mj.mj_step(model, data, nstep=config.control_every - 1)
    ended = float(data.time)
    end_pose = (ended, data.qpos.copy())
    frames = []
    for time, update in zip(times, wanted, strict=True):
        if update in poses:
            frames.append(poses[update])
        elif time is None or (not arrived and update == n_updates):
            frames.append(end_pose)  # the state the walk ended in
        else:
            msg = f"frame time {time} s is after the walk ended at {ended:.2f} s"
            raise ValueError(msg)
    return Replay(
        poses=frames,
        ended=ended,
        arrived=arrived,
        distance=distance_to_target(np.asarray(data.qpos[0:2]), TARGET_XY),
        box=(float(low[0]), float(high[0]), float(low[1]), float(high[1])),
    )


def check_against_walk(
    result: Replay, genotype: Genotype, model: mj.MjModel, config: SimConfig
) -> None:
    """Raise RuntimeError unless `simulate.walk` ends where the replay ended."""
    score = walk(genotype, model, config)
    if (score.seconds, score.distance) != (result.ended, result.distance):
        msg = (
            f"the replay ended at {result.ended} s, {result.distance} m from the "
            f"target; simulate.walk at {score.seconds} s, {score.distance} m"
        )
        raise RuntimeError(msg)


def view_window(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """The ground every frame shows: the robot's whole walk and the target.

    (x min, x max, y min, y max) in metres, `MARGIN` around both.
    """
    x_low, x_high, y_low, y_high = box
    target = TARGET_RADIUS + RING_WIDTH / 2 + OUTLINE
    return (
        min(x_low, TARGET_XY[0] - target) - MARGIN,
        max(x_high, TARGET_XY[0] + target) + MARGIN,
        min(y_low, TARGET_XY[1] - target) - MARGIN,
        max(y_high, TARGET_XY[1] + target) + MARGIN,
    )


def top_camera(
    model: mj.MjModel, window: tuple[float, float, float, float]
) -> mj.MjvCamera:
    """A camera straight above the window, +x up and +y left in the image.

    MuJoCo's free camera, at the model's field of view (`vis.global_.fovy`),
    shows a height of 2 x distance x tan(fovy / 2) of the ground (z = 0)
    under it; the distance is chosen to make that the window's length in x.
    The image's width then sets how much of y it shows (`frame_size`).
    """
    x_low, x_high, y_low, y_high = window
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [(x_low + x_high) / 2, (y_low + y_high) / 2, 0.0]
    half_angle = np.radians(model.vis.global_.fovy) / 2
    camera.distance = (x_high - x_low) / (2 * np.tan(half_angle))
    camera.azimuth = 0.0  # looking along +x, so +x is up once looking down
    camera.elevation = -90.0  # straight down
    return camera


def over_ground(
    camera: mj.MjvCamera, xy: npt.NDArray[np.float64], height: float
) -> tuple[npt.NDArray[np.float64], float]:
    """Where a mark at `height` hides the ground point `xy` (z = 0) from `camera`.

    The camera sees in perspective from straight above, so a mark raised
    towards it must move towards the point under the camera, and shrink, by
    the share of the camera's height it rose. Returns the mark's position and
    that shrink factor.
    """
    above = camera.lookat[2] + camera.distance  # the camera's height
    shrink = (above - height) / above
    centre = np.asarray(camera.lookat[:2])
    return np.array([*(centre + (xy - centre) * shrink), height]), shrink


def add_geom(
    scene: mj.MjvScene,
    kind: mj.mjtGeom,
    size: npt.NDArray[np.float64],
    position: npt.NDArray[np.float64],
    rgba: npt.NDArray[np.float32],
) -> mj.MjvGeom:
    """A new geom in the renderer's scene (not the model); RuntimeError if full."""
    if scene.ngeom >= scene.maxgeom:
        msg = f"the scene has no room for the target ({scene.maxgeom} geoms)"
        raise RuntimeError(msg)
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(geom, kind, size, position, np.eye(3).ravel(), rgba)
    scene.ngeom += 1
    return geom


def add_target(scene: mj.MjvScene, camera: mj.MjvCamera) -> None:
    """Draw the target into the scene: its circle of TARGET_RADIUS and centre."""
    angles = np.linspace(0.0, 2 * np.pi, RING_SEGMENTS + 1)
    for colour, widen, height in (
        (OUTLINE_WHITE, OUTLINE, TARGET_HEIGHT - 0.01),  # under the red
        (TARGET_RED, 0.0, TARGET_HEIGHT),
    ):
        _, shrink = over_ground(camera, TARGET_XY, height)
        points = [
            over_ground(
                camera,
                TARGET_XY + TARGET_RADIUS * np.array([np.cos(a), np.sin(a)]),
                height,
            )[0]
            for a in angles
        ]
        for start, end in pairwise(points):
            geom = add_geom(
                scene, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), colour
            )
            width = shrink * (RING_WIDTH / 2 + widen)
            mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, width, start, end)
        add_geom(
            scene,
            mj.mjtGeom.mjGEOM_SPHERE,
            np.full(3, shrink * (DOT_RADIUS + widen)),
            over_ground(camera, TARGET_XY, height)[0],
            colour,
        )


def render_frames(
    model: mj.MjModel,
    poses: list[tuple[float, npt.NDArray[np.float64]]],
    window: tuple[float, float, float, float],
    size: tuple[int, int],
) -> list[npt.NDArray[np.uint8]]:
    """Each pose seen from `top_camera`, as (height, width, 3) images of `size`.

    `size` is (width, height) in pixels; the frames are rendered up to
    SUPERSAMPLE times larger (as far as the model's offscreen buffer allows)
    and shrunk to it.
    """
    width, height = size
    scale = min(
        SUPERSAMPLE,
        model.vis.global_.offwidth / width,
        model.vis.global_.offheight / height,
    )
    camera = top_camera(model, window)
    view = mj.MjData(model)  # a copy to draw from: the walk's state is untouched
    frames = []
    with mj.Renderer(model, round(height * scale), round(width * scale)) as renderer:
        for time, qpos in poses:
            view.time, view.qpos[:] = time, qpos
            mj.mj_forward(model, view)
            renderer.update_scene(view, camera)
            add_target(renderer.scene, camera)
            image = Image.fromarray(renderer.render())
            frames.append(np.asarray(image.resize(size, Image.Resampling.LANCZOS)))
    return frames


def time_label(seconds: float) -> str:
    """A frame's time as the caption gives it: "0 s", "3.5 s", "13.84 s"."""
    return f"{round(seconds, 2):g} s"


def filmstrip(frames: list[npt.NDArray[np.uint8]], labels: list[str]) -> Image.Image:
    """The frames side by side with white gaps, each labelled above."""
    font = ImageFont.truetype(
        font_manager.findfont("DejaVu Sans"), round(LABEL_POINTS / 72 * DPI)
    )
    height, width = frames[0].shape[:2]
    band = round(1.5 * font.size)
    strip = Image.new(
        "RGB", (len(frames) * width + (len(frames) - 1) * GAP, band + height), "white"
    )
    draw = ImageDraw.Draw(strip)
    for index, (frame, text) in enumerate(zip(frames, labels, strict=True)):
        left = index * (width + GAP)
        strip.paste(Image.fromarray(frame), (left, band))
        draw.text(
            (left + width / 2, band / 2),
            text,
            fill=(26, 26, 26),
            font=font,
            anchor="mm",
        )
    return strip


def frame_size(
    n_frames: int, window: tuple[float, float, float, float]
) -> tuple[int, int]:
    """(width, height) of one frame in pixels: the strip fills the column."""
    total = round(COLUMN * DPI)
    width = (total - (n_frames - 1) * GAP) // n_frames
    x_low, x_high, y_low, y_high = window
    return width, round(width * (x_high - x_low) / (y_high - y_low))


def recorded_arrival(run: Path) -> float | None:
    """When the run's best brain arrived in `longer_walks.py`'s 15 s walk."""
    path = run / "longer_walks.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("15", {}).get("arrived_at")


def main() -> None:
    """Walk the run's best genotype, draw the frames, write the strip."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run",
        type=Path,
        nargs="?",
        default=RUN,
        help="a run folder (default: results/final/standard/seed25)",
    )
    parser.add_argument(
        "--times",
        default=TIMES,
        help=f'frame times: seconds, "{END}" for when the walk ends (default: {TIMES})',
    )
    parser.add_argument("--out", type=Path, default=OUT, help="the PNG to write")
    args = parser.parse_args()
    try:
        times = parse_times(args.times)
    except ValueError as error:
        parser.error(str(error))

    config = final_sim_config(args.run)
    terrains = json.loads((args.run / "config.json").read_text())["terrains"]
    model = mj.MjModel.from_binary_path(terrains[0])
    genotype = np.load(args.run / "best_genotype.npy")
    try:
        result = replay(genotype, model, config, times)
    except ValueError as error:
        parser.error(str(error))
    check_against_walk(result, genotype, model, config)

    window = view_window(result.box)
    size = frame_size(len(times), window)
    frames = render_frames(model, result.poses, window, size)
    labels = [time_label(time) for time, _ in result.poses]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    filmstrip(frames, labels).save(args.out, dpi=(DPI, DPI))

    left = f"{result.distance:.3f} m from the target"
    print(f"{args.run}: frames at {', '.join(labels)}")
    if result.arrived:
        print(f"arrived at {result.ended:.2f} s, {left}")
    else:
        print(f"did not arrive: {left} after {result.ended:g} s")
    recorded = recorded_arrival(args.run)
    if recorded != (result.ended if result.arrived else None):
        print(
            f"WARNING: longer_walks.json records arrival at {recorded}; this walk "
            "differs, as MuJoCo does between x86 and ARM machines"
        )
    x_low, x_high, y_low, y_high = window
    print(f"ground shown: x {x_low:.2f} to {x_high:.2f} m, ", end="")
    print(f"y {y_low:.2f} to {y_high:.2f} m")
    print(f"written {args.out} ({size[0]} x {size[1]} px per frame)")


if __name__ == "__main__":
    main()
