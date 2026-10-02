"""The ground of a world, and where to spawn the robot on it.

The worlds are ARIEL's, used exactly as shipped with their default settings;
nothing in ARIEL is changed or re-implemented here. This file only reads the
compiled world.

**The ground.** Most worlds have one ground geom, the heightfield or plane
called `floor`. ARIEL's OlympicArena is built from several pieces instead: a
flat start, a rugged heightfield strip (where the target lies), an incline and
a finish. `ground_geoms()` therefore collects every geom that is fixed to the
world and can collide. Vision, the ground-contact penalty and the spawn height
all use that set, so every world is handled the same way.

**The spawn.** The template spawns the robot at z = 0.1 with
`correct_collision_with_floor=True`, which lifts the robot so its lowest point
sits just above z = 0. It ignores the terrain. The rugged ground under the
spawn point is typically 0.1-0.5 m higher, so the robot starts INSIDE the
terrain and MuJoCo violently pushes it out, which flips it or throws it
around before the controller has done anything.

`spawn_height()` picks a spawn height instead: just above the highest ground
within the robot's leg span. The spawn position is ours to choose (the
template's `SPAWN_POS`); `world.spawn()` is called through its normal API.
"""

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt

# The core cube's centre sits this far above the robot's lowest point at rest
# (true for every John Set body except iguana, whose core ARIEL places 1.6 cm
# higher; it still spawns about 4 mm clear of the ground).
CORE_ABOVE_LOWEST_POINT: float = 0.075
REACH_MARGIN: float = 0.06  # beyond the outermost module centre
SPAWN_CLEARANCE: float = 0.02  # gap between the robot and the highest ground under it
RAY_START_Z: float = 100.0  # vertical rays for the ground height start this high
DOWN: npt.NDArray[np.float64] = np.array([0.0, 0.0, -1.0])  # vertical rays
DOWN.flags.writeable = False


def ground_geoms(model: mj.MjModel) -> tuple[int, ...]:
    """Every geom fixed to the world that can collide: the world's ground.

    The robot's geoms move with it (they are not welded to the world), so they
    are never included. Purely visual geoms cannot collide and are skipped.
    """
    return tuple(
        geom
        for geom in range(model.ngeom)
        if model.body_weldid[model.geom_bodyid[geom]] == 0
        and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    )


def ray_to_geom(
    model: mj.MjModel,
    data: mj.MjData,
    geom: int,
    origin: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Distance along a ray (unit `direction`) to one geom; -1 if it misses."""
    kind = model.geom_type[geom]
    if kind == mj.mjtGeom.mjGEOM_HFIELD:
        return float(mj.mj_rayHfield(model, data, geom, origin, direction, None))
    if kind == mj.mjtGeom.mjGEOM_MESH:
        return float(mj.mj_rayMesh(model, data, geom, origin, direction, None))
    if kind == mj.mjtGeom.mjGEOM_PLANE:
        # A level plane, taken as unbounded: intersect with z = its height.
        if direction[2] > -1e-9:
            return -1.0
        return float((origin[2] - data.geom_xpos[geom, 2]) / -direction[2])
    return float(
        mj.mju_rayGeom(
            data.geom_xpos[geom],
            data.geom_xmat[geom],
            model.geom_size[geom],
            origin,
            direction,
            kind,
            None,
        )
    )


def ray_to_ground(
    model: mj.MjModel,
    data: mj.MjData,
    ground: tuple[int, ...],
    origin: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Distance along a ray to the nearest ground geom; -1 if it misses them all."""
    hits = [
        distance
        for geom in ground
        if (distance := ray_to_geom(model, data, geom, origin, direction)) >= 0.0
    ]
    return min(hits, default=-1.0)


def _heightfield_height(
    model: mj.MjModel, data: mj.MjData, geom: int, x: float, y: float
) -> float | None:
    """Height of a level heightfield geom at world (x, y), None outside it.

    Bilinear interpolation of the heightfield samples.
    """
    field = model.geom_dataid[geom]
    size_x, size_y, size_z, _ = model.hfield_size[field]
    centre = data.geom_xpos[geom]
    local_x, local_y = x - centre[0], y - centre[1]
    if abs(local_x) > size_x or abs(local_y) > size_y:
        return None
    n_rows, n_cols = int(model.hfield_nrow[field]), int(model.hfield_ncol[field])
    start = int(model.hfield_adr[field])
    heights = model.hfield_data[start : start + n_rows * n_cols].reshape(n_rows, n_cols)

    col = (local_x + size_x) / (2 * size_x) * (n_cols - 1)
    row = (local_y + size_y) / (2 * size_y) * (n_rows - 1)
    c0 = int(np.clip(np.floor(col), 0, n_cols - 2))
    r0 = int(np.clip(np.floor(row), 0, n_rows - 2))
    fc, fr = col - c0, row - r0
    value = (
        heights[r0, c0] * (1 - fc) * (1 - fr)
        + heights[r0, c0 + 1] * fc * (1 - fr)
        + heights[r0 + 1, c0] * (1 - fc) * fr
        + heights[r0 + 1, c0 + 1] * fc * fr
    )
    return float(centre[2]) + float(value) * size_z


def mesh_boxes(
    model: mj.MjModel, data: mj.MjData, ground: tuple[int, ...]
) -> dict[int, tuple[float, float, float, float]]:
    """The world (x min, x max, y min, y max) around every mesh ground piece.

    A vertical ray outside a piece's box cannot hit it, so `ground_height`
    can skip it there. The ground never moves, so a walk computes the boxes
    once (after `mj_forward`) and reuses them at every step. Each box is the
    piece's own bounding box turned into the world frame, padded by 1 µm.
    """
    boxes = {}
    for geom in ground:
        if model.geom_type[geom] != mj.mjtGeom.mjGEOM_MESH:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        centre = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
        half = np.abs(rotation) @ model.geom_aabb[geom, 3:] + 1e-6
        boxes[geom] = (
            float(centre[0] - half[0]),
            float(centre[0] + half[0]),
            float(centre[1] - half[1]),
            float(centre[1] + half[1]),
        )
    return boxes


def ground_height(
    model: mj.MjModel,
    data: mj.MjData,
    ground: tuple[int, ...],
    x: float,
    y: float,
    boxes: dict[int, tuple[float, float, float, float]] | None = None,
) -> float:
    """Height of the highest ground at world (x, y); -inf if there is none.

    Heightfields are interpolated from their samples, level planes are taken
    as unbounded, and any other ground piece is found with a vertical ray.
    Used for the spawn height and to tell when the vision rays start below
    the ground - never as a controller input, since a real robot would not
    have a map of the ground. With `boxes` (from `mesh_boxes`), mesh pieces
    whose box does not contain (x, y) are skipped: same result, fewer rays.
    """
    heights = []
    for geom in ground:
        kind = model.geom_type[geom]
        if kind == mj.mjtGeom.mjGEOM_HFIELD:
            height = _heightfield_height(model, data, geom, x, y)
        elif kind == mj.mjtGeom.mjGEOM_PLANE:
            height = float(data.geom_xpos[geom, 2])
        else:
            box = boxes.get(geom) if boxes is not None else None
            if box is not None and not (
                box[0] <= x <= box[1] and box[2] <= y <= box[3]
            ):
                continue
            origin = np.array([x, y, RAY_START_Z])
            distance = ray_to_geom(model, data, geom, origin, DOWN)
            height = RAY_START_Z - distance if distance >= 0.0 else None
        if height is not None:
            heights.append(height)
    return max(heights, default=-np.inf)


def spawn_height(
    world_spec: mj.MjSpec, reach: float, x: float = 0.0, y: float = 0.0
) -> float:
    """Core height that puts the robot just above the ground under it.

    Compiles the bare world once (about 10-20 ms), samples the ground on a
    grid covering the body's `reach` around (x, y), and returns a core height that
    leaves `SPAWN_CLEARANCE` between the robot's lowest point and the highest
    sample. Spawn with `correct_collision_with_floor=False` afterwards.
    """
    bare = world_spec.compile()
    data = mj.MjData(bare)
    mj.mj_kinematics(bare, data)
    ground = ground_geoms(bare)
    radius = reach + REACH_MARGIN
    offsets = np.linspace(-radius, radius, 29)
    highest = max(
        ground_height(bare, data, ground, x + dx, y + dy)
        for dx in offsets
        for dy in offsets
        if dx * dx + dy * dy <= radius**2
    )
    if not np.isfinite(highest):
        msg = f"no ground under the spawn point ({x}, {y})"
        raise ValueError(msg)
    return highest + CORE_ABOVE_LOWEST_POINT + SPAWN_CLEARANCE
