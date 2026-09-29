"""The robot bodies we can evolve a brain for: ARIEL's John Set, used as shipped.

Each body is built by its own `john_set` function. What the rest of the code
needs to know about a body is measured once from the compiled robot:

- `hinges`   the number of motors, hence the network's outputs and the number
             of joint-angle inputs
- `reach`    how far (in metres, in the ground plane) the robot extends from its
             core at rest - the area the spawn height must clear
- `legs`     which hinges belong to which limb: every limb is one branch of the
             body tree hanging off the core (spider_8: four legs of two hinges)

Why the body is a setting (decision D1): spider_16's motors cannot lift its
body. Every John Set motor is capped at 0.66 N·m, and no static pose raises
spider_16's core above its resting height, so it can only shuffle. A snake
never has to lift itself, which is why experiment 11 compares bodies.
"""

# Standard library
from dataclasses import dataclass
from functools import cache

# Third-party libraries
import mujoco as mj
import numpy as np

# Local libraries (ARIEL)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.simulation.environments import SimpleFlatWorld

BODIES: tuple[str, ...] = (
    "baby_a",
    "baby_b",
    "gecko",
    "linkin_modified",
    "snake",
    "turtle",
    "iguana",
    "spider_8",
    "spider_12",
    "spider_16",
    "centipede_3",
    "centipede_4",
    "centipede_5",
)
DEFAULT_BODY: str = "spider_8"  # the final body (decision D1, since 2026-09-29)
# The body of experiments 1-12. Its saved terrains keep their original folders
# (results/terrains/<world>/seed<S>/); every other body gets a subfolder.
FIRST_BODY: str = "spider_16"
CORE_BODY: str = "robot1_core"  # the core's name after world.spawn() prefixes it


@dataclass(frozen=True)
class BodyInfo:
    """What the code needs to know about one body."""

    hinges: int
    reach: float
    legs: tuple[tuple[int, ...], ...]  # hinge indices (into qpos[7:]) per limb


def build_body(name: str) -> CoreModule:
    """A fresh instance of the John Set body called `name`."""
    if name not in BODIES:
        msg = f"unknown body {name!r}; choose one of {', '.join(BODIES)}"
        raise ValueError(msg)
    return getattr(john_set, name)()


@cache
def body_info(name: str) -> BodyInfo:
    """Measure the body once: its motor count and how far it reaches."""
    world = SimpleFlatWorld()
    world.spawn(build_body(name).spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    core = model.body(CORE_BODY).id
    robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0]
    offsets = data.geom_xpos[robot_geoms, :2] - data.xpos[core, :2]
    reach = float(np.linalg.norm(offsets, axis=1).max())
    return BodyInfo(hinges=int(model.nu), reach=reach, legs=limbs(model, core))


def limbs(model: mj.MjModel, core: int) -> tuple[tuple[int, ...], ...]:
    """The hinges of every limb, grouped by the core's child they hang from."""
    groups: dict[int, list[int]] = {}
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mj.mjtJoint.mjJNT_HINGE:
            continue
        branch = model.jnt_bodyid[joint]
        while model.body_parentid[branch] != core:
            branch = model.body_parentid[branch]
            if branch == 0:
                msg = f"hinge {model.joint(joint).name} is not below the core"
                raise ValueError(msg)
        # qpos[0:7] is the core's free joint; hinge angles follow it.
        groups.setdefault(branch, []).append(int(model.jnt_qposadr[joint]) - 7)
    return tuple(tuple(hinges) for hinges in groups.values())
