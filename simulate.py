"""One fitness evaluation: let the network walk on the given terrains, measure.

    fitness = distance to the target at the end            (walk there)
            + 0.5 x fraction of the run the core touches the ground   (stand)
            + 1.0 x fraction of the run the robot is upside down      (stay upright)
            [ - a fading reward for moving at all, in curriculum runs (D16) ]

averaged over the terrains the run walks. LOWER IS BETTER. `walk` only measures
(`Score`); `fitness` turns the measurements into the number the EA minimises.
The distance term is ARIEL's own `distance_to_target`; the two posture terms
are ours (decision D15). A spider carries its body on its legs: at rest
spider_16's core lies on the ground, so it would have to push itself up to
avoid the ground penalty - which its motors turn out to be too weak for
(see bodies.py).

The pieces:

- `build_model`   the body on a fresh `RuggedTerrainWorld()`, spawned above it
- `walk`          run one episode with one controller; return its `Score`
- `evaluate`      average `walk` over several terrains
- `fitness`       combine a `Score` into the number the EA minimises

Terrains are built in the MAIN process and saved to files - once per seed
(`run_terrains`, the default) or once per generation (`save_terrains`) - and
loaded by the worker processes (`evaluate_task`). Workers never build a world
themselves: `RuggedTerrainWorld()` draws a new random terrain on every
construction, so each worker would otherwise get different ground, and
individuals would no longer be compared fairly (D10).
"""

# Standard library
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt

# Local libraries (ARIEL)
from ariel.simulation.environments import BaseWorld, RuggedTerrainWorld
from ariel.simulation.tasks.targeted_locomotion import distance_to_target

# Local libraries
from bodies import DEFAULT_BODY, body_info, build_body
from genome import genotype_length, split
from network import Genotype, NetworkShape, forward, parse_hidden
from sensors import CORE_BODY, HALF_PI, n_inputs, read_inputs
from terrain import ground_geoms, spawn_height

SPAWN_XY: tuple[float, float] = (0.0, 0.0)
TARGET_XY: npt.NDArray[np.float64] = np.array([2.0, 0.0])

# Given to a controller that blew up (NaN) or fell out of the world. The robot
# starts 2 m from the target, so no genuine run comes anywhere near this.
FAILED_FITNESS: float = 10.0
FELL_OUT_OF_WORLD_Z: float = -1.0
TARGET_RADIUS: float = 0.1  # within this many metres the target counts as reached


@dataclass(frozen=True)
class SimConfig:
    """Everything that defines an evaluation. Identical across all conditions."""

    duration: float = 15.0  # seconds of simulated time per episode (D12)
    control_every: int = 10  # physics steps per network update (10 x 2 ms = 50 Hz)
    vision: bool = True  # the 10 terrain-sensing rays as extra inputs
    hidden_layers: str = "16"  # neurons per hidden layer, e.g. "16" or "8,8" (D6)
    body: str = DEFAULT_BODY  # a John Set body (bodies.BODIES, decision D1)
    evolve_tempo: bool = False  # a tempo gene sets the clock (genome.py, D17)
    ground_contact_weight: float = 0.5  # metres-equivalent for lying down all run
    upside_down_weight: float = 1.0  # metres-equivalent for being flipped all run

    # Ending a walk early (decision D16); all off by default.
    stop_at_target: bool = False  # end the walk once the target is reached
    early_stop_time: float = 5.0  # when a hopeless walk is judged (if enabled)

    @property
    def hinges(self) -> int:
        """Motors of the body: the network's outputs."""
        return body_info(self.body).hinges

    @property
    def genotype_length(self) -> int:
        """Genes per individual: the weights, plus the tempo gene if evolved."""
        return genotype_length(self.shape, self.evolve_tempo)

    @property
    def shape(self) -> NetworkShape:
        """The network shape, hence the genotype length."""
        return NetworkShape(
            n_inputs(self.vision, self.hinges),
            parse_hidden(self.hidden_layers),
            self.hinges,
        )


def build_model(
    world_factory: Callable[[], BaseWorld] = RuggedTerrainWorld,
    body: str = DEFAULT_BODY,
) -> mj.MjModel:
    """Compile `body` on a newly generated world (a new random terrain).

    `world_factory` is only replaced in tests (with a flat world, whose
    results are deterministic).
    """
    mj.set_mjcb_control(None)  # MuJoCo's control callback is global; keep it off
    world = world_factory()
    robot = build_body(body)
    spawn_z = spawn_height(world.spec, body_info(body).reach, *SPAWN_XY)
    world.spawn(
        robot.spec,
        position=[*SPAWN_XY, spawn_z],
        correct_collision_with_floor=False,
    )
    model = world.spec.compile()
    if model.nu != body_info(body).hinges:
        msg = f"{body} has {model.nu} actuators, expected {body_info(body).hinges}"
        raise RuntimeError(msg)
    return model


@dataclass(frozen=True)
class Score:
    """What happened in one evaluation. The fitness is computed from these."""

    distance: float  # metres from the target at the end
    displacement: float  # metres travelled from the spawn point, in any direction
    ground_contact: float  # share of the walk the core touched the ground
    upside_down: float  # share of the walk the robot was upside down
    seconds: float  # simulated seconds actually walked (less if stopped early)


FAILED_SCORE = Score(FAILED_FITNESS, 0.0, 1.0, 1.0, 0.0)


def fitness(score: Score, config: SimConfig, movement_weight: float = 0.0) -> float:
    """The number the EA minimises (decision D15, and D16 for `movement_weight`).

        distance + ground_contact_weight x ground_contact
                 + upside_down_weight x upside_down
                 - movement_weight x displacement

    `movement_weight` is 0 except early in a curriculum run, where it rewards
    moving at all - in any direction - before walking towards the target
    matters (`ea.EAConfig.curriculum`).
    """
    return (
        score.distance
        + config.ground_contact_weight * score.ground_contact
        + config.upside_down_weight * score.upside_down
        - movement_weight * score.displacement
    )


def core_touches_ground(
    data: mj.MjData, core_geom: int, ground: tuple[int, ...]
) -> bool:
    """True if any current contact is between the core and a ground geom."""
    pairs = data.contact.geom[: data.ncon]
    return bool(
        np.any((pairs == core_geom).any(axis=1) & np.isin(pairs, ground).any(axis=1))
    )


def is_upside_down(data: mj.MjData, core_id: int) -> bool:
    """True if the core's top faces downwards (its z-axis points below horizontal)."""
    return bool(data.xmat[core_id, 8] < 0.0)  # z-component of the body's z-axis


def walk(
    genotype: Genotype,
    model: mj.MjModel,
    config: SimConfig,
    min_progress: float | None = None,
) -> Score:
    """Run one episode and measure it.

    The network is queried every `control_every` physics steps and its output
    is written straight into the hinge servos' target angles (direct control,
    decision D4). The servos move the joints between queries. Posture is
    sampled at every query: does the core touch the ground, is it upside down.

    The episode ends early (decision D16) if:

    - `config.stop_at_target` is on and the core is within `TARGET_RADIUS`
      of the target - it has arrived;
    - `min_progress` is given and, at `config.early_stop_time` seconds, the
      robot has closed less than `min_progress` metres of the distance to the
      target - it is hopeless. It keeps the distance it has at that moment.
    """
    shape: NetworkShape = config.shape
    weights, clock_hz = split(genotype, shape, config.evolve_tempo)
    core_id = model.body(CORE_BODY).id
    core_geom = model.geom(CORE_BODY).id
    ground = ground_geoms(model)
    vision_ground = ground if config.vision else None

    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    start_xy = data.qpos[0:2].copy()
    start_distance = distance_to_target(start_xy, TARGET_XY)

    n_updates = round(config.duration / (model.opt.timestep * config.control_every))
    touching = flipped = updates = 0
    checked_progress = min_progress is None
    for _ in range(n_updates):
        # mj_step1 brings every derived quantity (body positions, orientations,
        # contacts) up to date with the current state; the network reads them,
        # sets the controls, and mj_step2 finishes the step using those controls.
        mj.mj_step1(model, data)

        now_distance = distance_to_target(data.qpos[0:2], TARGET_XY)
        if config.stop_at_target and now_distance < TARGET_RADIUS:
            break
        if not checked_progress and data.time >= config.early_stop_time:
            checked_progress = True
            if start_distance - now_distance < min_progress:
                break

        updates += 1
        touching += core_touches_ground(data, core_geom, ground)
        flipped += is_upside_down(data, core_id)
        inputs = read_inputs(model, data, core_id, TARGET_XY, vision_ground, clock_hz)
        data.ctrl[:] = forward(weights, shape, inputs) * HALF_PI
        mj.mj_step2(model, data)
        mj.mj_step(model, data, nstep=config.control_every - 1)

    final = data.qpos[0:3]
    if not np.all(np.isfinite(final)) or final[2] < FELL_OUT_OF_WORLD_Z:
        return FAILED_SCORE
    return Score(
        distance=distance_to_target(np.asarray(final), TARGET_XY),
        displacement=float(np.linalg.norm(final[:2] - start_xy)),
        ground_contact=touching / max(updates, 1),
        upside_down=flipped / max(updates, 1),
        seconds=float(data.time),
    )


def evaluate(
    genotype: Genotype,
    models: tuple[mj.MjModel, ...],
    config: SimConfig,
    min_progress: float | None = None,
) -> Score:
    """Every part of the score, averaged over the given terrains."""
    scores = [walk(genotype, model, config, min_progress) for model in models]
    return Score(
        **{
            field: float(np.mean([getattr(score, field) for score in scores]))
            for field in Score.__dataclass_fields__
        }
    )


def save_terrains(
    models: list[mj.MjModel], directory: Path, generation: int
) -> tuple[str, ...]:
    """Write a generation's compiled terrains to MuJoCo binary files.

    A compiled model is about 5 MB; sending it with every task would copy it
    once per individual. Instead each terrain is written once (about 1 ms) and
    workers load it by path (about 2 ms), bit-for-bit identical.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, model in enumerate(models):
        path = directory / f"generation{generation}_terrain{index}.mjb"
        mj.mj_saveModel(model, str(path), None)
        paths.append(str(path))
    return tuple(paths)


def run_terrains(
    directory: Path,
    n_terrains: int,
    world_factory: Callable[[], BaseWorld] = RuggedTerrainWorld,
    body: str = DEFAULT_BODY,
) -> tuple[str, ...]:
    """The terrains one seed uses for its whole run, generated on first use.

    Each is a plain `RuggedTerrainWorld()` (a random terrain) compiled once
    and saved as `terrain<i>.mjb` in `directory`. Every later run pointed at
    the same directory - the other migration policies with the same seed -
    reuses the saved files, so all conditions walk exactly the same ground.
    Two runs starting at the same moment must not end up on different
    terrain: each writes its candidate under a private name and publishes it
    with `os.link`, which fails if the file already exists. The loser deletes
    its own candidate and uses the winner's file. Paths are absolute, so a
    run's `config.json` works from any directory.
    """
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(n_terrains):
        path = directory / f"terrain{index}.mjb"
        if not path.exists():
            partial = directory / f"terrain{index}.{os.getpid()}.partial"
            mj.mj_saveModel(build_model(world_factory, body), str(partial), None)
            try:
                os.link(partial, path)
            except FileExistsError:
                pass  # another run published first; use its terrain
            finally:
                partial.unlink()
        paths.append(str(path))
    return tuple(paths)


# Per-worker cache of loaded terrains, keyed by file path.
_LOADED: dict[str, mj.MjModel] = {}
_MAX_LOADED = 8


def load_terrain(path: str) -> mj.MjModel:
    """Load a saved terrain once per worker process."""
    if path not in _LOADED:
        if len(_LOADED) >= _MAX_LOADED:
            _LOADED.clear()
        _LOADED[path] = mj.MjModel.from_binary_path(path)
    return _LOADED[path]


type Task = tuple[list[float], tuple[str, ...], SimConfig, float | None]


def evaluate_task(task: Task) -> Score:
    """`evaluate` with one picklable argument, for `multiprocessing.Pool.map`.

    The task names the terrain files rather than carrying the models, and
    carries this generation's early-stop bar (None when early stopping is off).
    """
    genotype, terrain_paths, config, min_progress = task
    models = tuple(load_terrain(path) for path in terrain_paths)
    return evaluate(
        np.asarray(genotype, dtype=np.float64), models, config, min_progress
    )
