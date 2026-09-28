"""Where to spawn the robot on rugged terrain.

The world itself is ARIEL's `RuggedTerrainWorld()`, used exactly as shipped:
default settings, a new random terrain every time it is constructed. Nothing
in ARIEL is changed or re-implemented here.

The problem this file solves: the template spawns the robot at z = 0.1 with
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

# spider_16 geometry, measured on flat ground.
CORE_ABOVE_LOWEST_POINT: float = 0.075  # core origin above the robot's lowest point
LEG_SPAN_RADIUS: float = 0.70  # leg tips reach about 0.64 m from the core
SPAWN_CLEARANCE: float = 0.02  # gap between the robot and the highest ground under it


def ground_height(model: mj.MjModel, x: float, y: float) -> float:
    """Terrain height at world (x, y), interpolated from the heightfield.

    Used only to choose the spawn height - never as a controller input, since
    a real robot would not have a map of the ground.
    """
    if model.nhfield == 0:
        return 0.0
    size_x, size_y, size_z, _ = model.hfield_size[0]
    n_rows, n_cols = int(model.hfield_nrow[0]), int(model.hfield_ncol[0])
    heights = model.hfield_data[: n_rows * n_cols].reshape(n_rows, n_cols)

    col = (x + size_x) / (2 * size_x) * (n_cols - 1)
    row = (y + size_y) / (2 * size_y) * (n_rows - 1)
    c0 = int(np.clip(np.floor(col), 0, n_cols - 2))
    r0 = int(np.clip(np.floor(row), 0, n_rows - 2))
    fc, fr = col - c0, row - r0
    value = (
        heights[r0, c0] * (1 - fc) * (1 - fr)
        + heights[r0, c0 + 1] * fc * (1 - fr)
        + heights[r0 + 1, c0] * (1 - fc) * fr
        + heights[r0 + 1, c0 + 1] * fc * fr
    )
    return float(model.geom("floor").pos[2]) + float(value) * size_z


def spawn_height(world_spec: mj.MjSpec, x: float = 0.0, y: float = 0.0) -> float:
    """Core height that puts the robot just above the ground under its legs.

    Compiles the bare world once (about 10-20 ms), samples the terrain on a
    grid covering the leg span around (x, y), and returns a core height that
    leaves `SPAWN_CLEARANCE` between the robot's lowest point and the highest
    sample. Spawn with `correct_collision_with_floor=False` afterwards.
    """
    bare = world_spec.compile()
    offsets = np.linspace(-LEG_SPAN_RADIUS, LEG_SPAN_RADIUS, 29)
    highest = max(
        ground_height(bare, x + dx, y + dy)
        for dx in offsets
        for dy in offsets
        if dx * dx + dy * dy <= LEG_SPAN_RADIUS**2
    )
    return highest + CORE_ABOVE_LOWEST_POINT + SPAWN_CLEARANCE
