"""One fitness evaluation: let the network walk on the given terrains, measure.

    fitness = distance to the target at the end            (walk there)
            [ + w x the distance to the target averaged over the walk (fast; D20) ]
            + 0.5 x fraction of the run the core touches the ground   (stand)
            + 1.0 x fraction of the run the robot is upside down      (stay upright)
            [ + w x how low the core is carried, 0 at >= carry_height up (D18) ]
            [ + w x how much the least-used leg lags the others    (D18) ]
            [ + w x how much the laziest leg's motors lag the others (D18) ]
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
import csv
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt

# Local libraries (ARIEL)
from ariel.simulation.environments import BaseWorld, RuggedTerrainWorld
from ariel.simulation.tasks.targeted_locomotion import distance_to_target

# Local libraries
from bodies import DEFAULT_BODY, FIRST_BODY, body_info, build_body
from genome import genotype_length, split
from network import Genotype, NetworkShape, forward, parse_hidden
from sensors import CORE_BODY, HALF_PI, n_inputs, ray_set, read_inputs
from terrain import CORE_ABOVE_LOWEST_POINT, ground_geoms, ground_height, spawn_height

SPAWN_XY: tuple[float, float] = (0.0, 0.0)
TEMPLATE_SPAWN_Z: float = 0.1  # the template's SPAWN_POS height
TARGET_XY: npt.NDArray[np.float64] = np.array([2.0, 0.0])

# Given to a controller that blew up (NaN) or fell out of the world. The robot
# starts 2 m from the target, so no genuine run comes anywhere near this.
FAILED_FITNESS: float = 10.0
FELL_OUT_OF_WORLD_Z: float = -1.0
TARGET_RADIUS: float = 0.1  # within this many metres the target counts as reached
# The core counts as carried once its underside is this far above the ground
# under it (decision D18). The default is lenient; spider_8 can hold about 6 cm.
CARRY_HEIGHT: float = 0.02


@dataclass(frozen=True)
class SimConfig:
    """Everything that defines an evaluation. Identical across all conditions."""

    duration: float = 15.0  # seconds of simulated time per episode (D12)
    control_every: int = 10  # physics steps per network update (10 x 2 ms = 50 Hz)
    vision: bool = True  # the terrain-sensing rays as extra inputs
    vision_rays: str = "all"  # which rays: "all" 10 or the 5 "near" ones (D5)
    position: bool = False  # the core's absolute (x, y) as 2 extra inputs (exp. 29)
    hidden_layers: str = "16"  # neurons per hidden layer, e.g. "16" or "8,8" (D6)
    body: str = DEFAULT_BODY  # a John Set body (bodies.BODIES, decision D1)
    evolve_tempo: bool = False  # a tempo gene sets the clock (genome.py, D17)
    ground_contact_weight: float = 0.5  # metres-equivalent for lying down all run
    upside_down_weight: float = 1.0  # metres-equivalent for being flipped all run
    low_body_weight: float = 0.0  # for a core carried below carry_height all run (D18)
    carry_height: float = CARRY_HEIGHT  # metres of lift that count as carried (D18)
    leg_imbalance_weight: float = 0.0  # for one leg never moving (D18)
    work_imbalance_weight: float = 0.0  # for one leg's motors never working (D18)
    speed_weight: float = 0.0  # x the walk's average distance to the target (D20)

    # Ending a walk early (decision D16); all off by default.
    stop_at_target: bool = False  # end the walk once the target is reached
    early_stop_time: float = 5.0  # when a hopeless walk is judged (if enabled)

    def __post_init__(self) -> None:
        """Reject settings that would only fail, or divide by zero, mid-run."""
        ray_set(self.vision_rays)  # raises ValueError for an unknown ray set
        if self.carry_height <= 0:
            msg = f"carry_height must be positive, got {self.carry_height}"
            raise ValueError(msg)

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
            n_inputs(self.vision, self.hinges, self.vision_rays, self.position),
            parse_hidden(self.hidden_layers),
            self.hinges,
        )


def build_model(
    world_factory: Callable[[], BaseWorld] = RuggedTerrainWorld,
    body: str = DEFAULT_BODY,
    yaw: float = 0.0,
    ariel_spawn: bool = False,
) -> mj.MjModel:
    """Compile `body` on a newly generated world (a new random terrain).

    `yaw` turns the robot at spawn by that many degrees about the vertical
    (left for positive), through ARIEL's own `spawn(rotation=...)`; at 0 the
    target lies straight ahead (decision D23).

    `ariel_spawn` places the robot as the template does: ARIEL's own floor
    correction, which puts its lowest point 1 cm above height 0. That is right
    on OlympicArena's flat start; on RuggedTerrainWorld it buries the robot, so
    by default we spawn 2 cm above the real ground instead (decision D2a).
    """
    mj.set_mjcb_control(None)  # MuJoCo's control callback is global; keep it off
    world = world_factory()
    robot = build_body(body)
    if ariel_spawn:
        spawn_z = TEMPLATE_SPAWN_Z  # ARIEL's correction replaces it anyway
    else:
        spawn_z = spawn_height(world.spec, body_info(body).reach, *SPAWN_XY)
    world.spawn(
        robot.spec,
        position=[*SPAWN_XY, spawn_z],
        rotation=(0.0, 0.0, yaw) if yaw else None,
        correct_collision_with_floor=ariel_spawn,
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
    low_body: float = 0.0  # 0 = carried at least carry_height up all run, 1 = lying
    leg_imbalance: float = 0.0  # 0 = every leg moves equally, 1 = one leg never moves
    mean_distance: float = 0.0  # distance to the target averaged over the whole walk
    work_imbalance: float = 0.0  # 0 = every leg's motors work equally, 1 = one idle


FAILED_SCORE = Score(
    distance=FAILED_FITNESS,
    displacement=0.0,
    ground_contact=1.0,
    upside_down=1.0,
    seconds=0.0,
    low_body=1.0,
    leg_imbalance=1.0,
    mean_distance=FAILED_FITNESS,
    work_imbalance=1.0,
)


def saved_sim_config(saved: dict[str, object]) -> SimConfig:
    """A run's SimConfig from the "sim" entry of its config.json.

    A setting a run's config does not record did not exist yet, so the run
    used what is now that setting's default - except the body, whose default
    changed from spider_16 (experiments 1-12, which do not record it) to
    spider_8. Raises ValueError for a setting this version does not know
    (a config from an older, incompatible version).
    """
    unknown = set(saved) - {field.name for field in fields(SimConfig)}
    if unknown:
        msg = f"unknown settings {sorted(unknown)}: an older, incompatible config"
        raise ValueError(msg)
    return SimConfig(**{"body": FIRST_BODY, **saved})


def final_sim_config(run: Path) -> SimConfig:
    """The SimConfig a run's walks ended with.

    That is the "sim" settings of its config.json (`saved_sim_config`), with
    the walk length of its last logged generation: the longer walks only if
    the run got far enough to switch to them (`ea.EAConfig.final_duration`,
    decision D21). A log from before D21 records no walk length.
    """
    saved = json.loads((run / "config.json").read_text())
    config = saved_sim_config(saved["sim"])
    log_file = run / "log.csv"
    if not log_file.exists():
        return config
    with log_file.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if rows and rows[-1].get("duration"):
        return replace(config, duration=float(rows[-1]["duration"]))
    return config


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
        + config.low_body_weight * score.low_body
        + config.leg_imbalance_weight * score.leg_imbalance
        + config.work_imbalance_weight * score.work_imbalance
        + config.speed_weight * score.mean_distance
        - movement_weight * score.displacement
    )


def carry_shortfall(core_height: float, carry_height: float = CARRY_HEIGHT) -> float:
    """How far the core is below "carried": 0 at `carry_height` up or more, 1 lying.

    `core_height` is the core centre's height above the ground under it; the
    core lies on the ground at CORE_ABOVE_LOWEST_POINT.
    """
    lift = core_height - CORE_ABOVE_LOWEST_POINT
    return float(np.clip(1.0 - lift / carry_height, 0.0, 1.0))


def shortfall_of_least(per_leg: npt.NDArray[np.float64]) -> float:
    """1 - (smallest leg's amount) / (mean over legs), in [0, 1].

    0 when every leg has the same amount, 1 when one leg has none. Fewer than
    two legs, or (numerically) nothing at all, scores 0. The ratio ignores
    scale: even a robot that only settles after spawning has some movement
    and work per leg, and is scored on how evenly that is spread.
    """
    if len(per_leg) < 2 or per_leg.mean() < 1e-9:
        return 0.0
    return float(np.clip(1.0 - per_leg.min() / per_leg.mean(), 0.0, 1.0))


def leg_imbalance(
    angles: npt.NDArray[np.float64], legs: tuple[tuple[int, ...], ...]
) -> float:
    """How far the least-used leg falls short of an equal share of the movement.

    `angles` holds the hinge angles over the walk (one row per sample). A leg's
    movement is the summed spread (standard deviation) of its hinges' angles.
    With n legs, each would do 1/n of the total if all moved alike:

        1 - (least-used leg's movement) / (mean leg movement)

    0 when every leg moves equally, 1 when one leg never moves. A body with a
    single limb, or one that does not move at all, scores 0.
    """
    if len(legs) < 2 or len(angles) < 2:
        return 0.0
    spread = angles.std(axis=0)
    return shortfall_of_least(np.array([spread[list(leg)].sum() for leg in legs]))


def work_imbalance(
    hinge_work: npt.NDArray[np.float64], legs: tuple[tuple[int, ...], ...]
) -> float:
    """How far the laziest leg's motors fall short of an equal share of the work.

    `hinge_work` is the positive mechanical work (joules) each hinge's motor
    did over the walk. A leg that only props the body up, or only waves in
    the air, does little work; a leg that drives the gait does a lot.

        1 - (least-working leg's work) / (mean leg work)
    """
    return shortfall_of_least(np.array([hinge_work[list(leg)].sum() for leg in legs]))


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

    legs = body_info(config.body).legs
    n_updates = round(config.duration / (model.opt.timestep * config.control_every))
    touching = flipped = updates = 0
    shortfall = 0.0
    angles = []
    # Positive motor work per hinge, sampled at every network update: power
    # (torque x joint speed, when the motor drives the joint) x update period.
    hinge_work = np.zeros(model.nu)
    update_period = model.opt.timestep * config.control_every
    # The distance to the target summed over every update of the FULL walk
    # (D20): after arriving (stop_at_target) the rest of the walk counts as 0;
    # after a hopeless walk is stopped (D16) it counts as where it stopped.
    distance_sum = 0.0
    rest_distance = 0.0
    checked_progress = min_progress is None
    for _ in range(n_updates):
        # mj_step1 brings every derived quantity (body positions, orientations,
        # contacts) up to date with the current state; the network reads them,
        # sets the controls, and mj_step2 finishes the step using those controls.
        mj.mj_step1(model, data)

        now_distance = distance_to_target(data.qpos[0:2], TARGET_XY)
        if config.stop_at_target and now_distance < TARGET_RADIUS:
            break  # arrived: the rest of the walk counts as distance 0
        if not checked_progress and data.time >= config.early_stop_time:
            checked_progress = True
            if start_distance - now_distance < min_progress:
                rest_distance = now_distance  # hopeless: it would stay about here
                break

        updates += 1
        distance_sum += now_distance
        touching += core_touches_ground(data, core_geom, ground)
        flipped += is_upside_down(data, core_id)
        x, y, z = data.xpos[core_id]
        shortfall += carry_shortfall(
            z - ground_height(model, data, ground, x, y), config.carry_height
        )
        angles.append(data.qpos[7:].copy())
        power = data.actuator_force * data.qvel[6 : 6 + model.nu]
        hinge_work += np.maximum(power, 0.0) * update_period
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
        low_body=shortfall / max(updates, 1),
        leg_imbalance=leg_imbalance(np.asarray(angles), legs),
        mean_distance=(distance_sum + rest_distance * (n_updates - updates))
        / max(n_updates, 1),
        work_imbalance=work_imbalance(hinge_work, legs),
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


def parse_yaws(text: str) -> tuple[float, ...]:
    """Spawn turns in degrees from a comma-separated setting such as "0,30,-30"."""
    return tuple(float(part) for part in text.split(","))


def terrain_yaw(yaws: tuple[float, ...], index: int) -> float:
    """The spawn turn of terrain `index`: the yaws are cycled over the terrains."""
    return yaws[index % len(yaws)]


def terrain_name(index: int, yaw: float, ariel_spawn: bool = False) -> str:
    """`terrain<i>`, plus `_yaw<d>` for a turned start and `_arielspawn` for
    ARIEL's own spawn. The turn is written exactly, so two different starts
    never share a file."""
    name = f"terrain{index}"
    if yaw:
        label = f"{yaw:g}"
        if float(label) != yaw:
            label = repr(yaw)
        name += f"_yaw{label}"
    if ariel_spawn:
        name += "_arielspawn"
    return name


def run_terrains(
    directory: Path,
    n_terrains: int,
    world_factory: Callable[[], BaseWorld] = RuggedTerrainWorld,
    body: str = DEFAULT_BODY,
    yaws: tuple[float, ...] = (0.0,),
    ariel_spawn: bool = False,
) -> tuple[str, ...]:
    """The terrains one seed uses for its whole run, generated on first use.

    Each is a plain `RuggedTerrainWorld()` (a random terrain) compiled once
    and saved as `terrain<i>.mjb` in `directory`, or `terrain<i>_yaw<d>.mjb`
    when the robot starts turned by d degrees on it (`terrain_yaw`, D23), with
    `_arielspawn` added for ARIEL's own spawn (D2a): the saved model includes
    the spawned robot. Every later run pointed at
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
        yaw = terrain_yaw(yaws, index)
        name = terrain_name(index, yaw, ariel_spawn)
        path = directory / f"{name}.mjb"
        if not path.exists():
            partial = directory / f"{name}.{os.getpid()}.partial"
            model = build_model(world_factory, body, yaw, ariel_spawn)
            mj.mj_saveModel(model, str(partial), None)
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
