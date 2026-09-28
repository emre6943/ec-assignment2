"""What the controller network is told about the world (decision D5).

The input vector, 34 values, each scaled to roughly [-1, 1]:

    hinge angles        16   where each joint is now (proprioception)
    clock                2   sin/cos of a 1 Hz beat to drive rhythmic gaits
    target               3   distance, sin(bearing), cos(bearing) in the
                             robot's own frame - "where is the goal from here"
    gravity              3   the world's up-direction seen from the core -
                             how the body is tilted on the rough ground
    vision              10   distance to whatever each of 10 rays hits -
                             the ground ahead and below on every side, and
                             (via the upward ray) whether the robot has flipped

Vision can be switched off (`SimConfig.vision`), which leaves 24 inputs.

The robot's absolute position is deliberately NOT an input: the target vector
carries the useful part of it in a form that means the same everywhere.

spider_16 layout (measured): qpos[0:3] core position, qpos[3:7] core
orientation quaternion, qpos[7:23] the 16 hinge angles; actuator i drives
hinge i. "Forward" is defined as the core body's +x axis, so at spawn the
target at (2, 0) lies straight ahead (bearing 0). A positive bearing means
the target is to the robot's left.
"""

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt

# Local libraries
from terrain import ground_height

type FloatArray = npt.NDArray[np.float64]

HALF_PI: float = np.pi / 2
CORE_BODY: str = "robot1_core"  # the core's name after world.spawn() prefixes it
CLOCK_HZ: float = 1.0
TARGET_DISTANCE_SCALE: float = 2.0  # the starting distance, so the input starts at 1

# Vision: 10 rays cast from a point 0.25 m above the core's centre, fixed to
# the body, so they tilt and turn with it. Each ray reports how far away the
# TERRAIN is along it. Rays are given as (direction in the robot's horizontal
# plane, angle below horizontal) in degrees; a negative angle points upwards.
# The four faces of the core are +x, +y, -x, -y; the legs stick out of them.
#
# The rays are cast against the terrain only (`mj_rayHfield`), so the robot's
# own legs never block them - the leg along each face would otherwise hide
# that face's steep ray whenever the leg lies flat. This is our own code: no
# sensor or marker is added to the robot, and nothing in ARIEL is touched.
RAY_ORIGIN: npt.NDArray[np.float64] = np.array([0.0, 0.075, 0.25])  # body frame
FACES: tuple[float, ...] = (0.0, 90.0, 180.0, -90.0)
LOOK_AHEAD_ANGLE: float = 15.0  # slightly down: ground about 1.3 m ahead on flat
LOOK_DOWN_ANGLE: float = 45.0  # more steeply down: ground about 0.35 m ahead
RAYS: tuple[tuple[float, float], ...] = (
    (0.0, 90.0),  # straight down: height above the ground
    (0.0, -90.0),  # straight up: sees nothing unless the robot has flipped
    *((face, LOOK_AHEAD_ANGLE) for face in FACES),
    *((face, LOOK_DOWN_ANGLE) for face in FACES),
)
RAY_MAX_RANGE: float = 3.0  # metres; a ray that hits nothing reads as this


def _ray_direction(azimuth_deg: float, depression_deg: float) -> FloatArray:
    """Unit vector in the body frame for (azimuth, angle below horizontal)."""
    azimuth, depression = np.deg2rad(azimuth_deg), np.deg2rad(depression_deg)
    return np.array(
        [
            np.cos(depression) * np.cos(azimuth),
            np.cos(depression) * np.sin(azimuth),
            -np.sin(depression),
        ]
    )


RAY_DIRECTIONS: npt.NDArray[np.float64] = np.array(
    [_ray_direction(azimuth, depression) for azimuth, depression in RAYS]
)

N_BASE_INPUTS: int = 16 + 2 + 3 + 3


def n_inputs(vision: bool) -> int:
    """Length of the input vector."""
    return N_BASE_INPUTS + (len(RAYS) if vision else 0)


def hinge_angles(data: mj.MjData) -> FloatArray:
    """The 16 hinge angles, scaled from [-pi/2, pi/2] to [-1, 1]."""
    return np.asarray(data.qpos[7:], dtype=np.float64) / HALF_PI


def clock(time: float, freq_hz: float = CLOCK_HZ) -> FloatArray:
    """[sin(2 pi f t), cos(2 pi f t)]: a beat the network can walk to.

    A feed-forward network has no memory, so without some time signal it can
    only produce rhythm through feedback from its own body. This is an input
    signal, not a CPG: nothing about it is evolved or coupled.
    """
    phase = 2.0 * np.pi * freq_hz * time
    return np.array([np.sin(phase), np.cos(phase)])


def heading(data: mj.MjData, core_id: int) -> float:
    """Yaw of the core's +x axis in the world's xy-plane, in radians."""
    rotation = data.xmat[core_id].reshape(3, 3)  # columns: body axes in world
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def target_in_body_frame(
    data: mj.MjData,
    core_id: int,
    target_xy: FloatArray,
) -> FloatArray:
    """[distance / 2, sin(bearing), cos(bearing)] of the target, seen from the robot.

    sin/cos rather than the raw angle, so there is no jump when the bearing
    wraps around from +pi to -pi.
    """
    delta = target_xy - data.xpos[core_id, :2]
    distance = float(np.hypot(delta[0], delta[1]))
    bearing = np.arctan2(delta[1], delta[0]) - heading(data, core_id)
    return np.array(
        [distance / TARGET_DISTANCE_SCALE, np.sin(bearing), np.cos(bearing)]
    )


def gravity_in_body_frame(data: mj.MjData, core_id: int) -> FloatArray:
    """The world's up-axis expressed in the core's frame.

    [0, 0, 1] when level; the first two components grow with pitch and roll;
    the last one goes negative when the robot is upside down.
    """
    rotation = data.xmat[core_id].reshape(3, 3)
    return rotation[2, :].copy()


def _distance_to_floor(
    model: mj.MjModel,
    data: mj.MjData,
    floor_id: int,
    origin: FloatArray,
    direction: FloatArray,
) -> float:
    """Distance along the ray to the floor geom only; -1 if it misses."""
    if model.geom_type[floor_id] == mj.mjtGeom.mjGEOM_HFIELD:
        return float(mj.mj_rayHfield(model, data, floor_id, origin, direction, None))
    # A flat plane (used by the tests): intersect with z = floor height.
    if direction[2] > -1e-9:
        return -1.0
    return float((origin[2] - data.geom_xpos[floor_id, 2]) / -direction[2])


def vision(
    model: mj.MjModel,
    data: mj.MjData,
    core_id: int,
    floor_id: int,
) -> FloatArray:
    """The 10 ray distances to the terrain, scaled to [0, 1].

    A ray that hits nothing reads 1 (maximum range): always the upward ray
    while the robot is upright. If the ray origin itself ends up below the
    ground - the robot is upside down and pressed into it - every ray reads 0.
    """
    rotation = data.xmat[core_id].reshape(3, 3)  # body axes as columns
    origin = data.xpos[core_id] + rotation @ RAY_ORIGIN
    if origin[2] < ground_height(model, origin[0], origin[1]):
        return np.zeros(len(RAYS))

    distances = np.array(
        [
            _distance_to_floor(model, data, floor_id, origin, rotation @ direction)
            for direction in RAY_DIRECTIONS
        ]
    )
    distances = np.where(distances < 0.0, RAY_MAX_RANGE, distances)
    return np.minimum(distances, RAY_MAX_RANGE) / RAY_MAX_RANGE


def read_inputs(
    model: mj.MjModel,
    data: mj.MjData,
    core_id: int,
    target_xy: FloatArray,
    floor_id: int | None,
) -> FloatArray:
    """Assemble the network's input vector (without vision if `floor_id` is None)."""
    parts = [
        hinge_angles(data),
        clock(data.time),
        target_in_body_frame(data, core_id, target_xy),
        gravity_in_body_frame(data, core_id),
    ]
    if floor_id is not None:
        parts.append(vision(model, data, core_id, floor_id))
    return np.concatenate(parts)
