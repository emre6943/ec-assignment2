"""Watch a run's best network walk.

    uv run --project ../ariel python replay.py RUN                # video
    uv run --project ../ariel python replay.py RUN --viewer       # live window
    uv run --project ../ariel python replay.py RUN --new-terrain  # unseen ground
    uv run --project ../ariel python replay.py RUN --flat         # flat ground
    uv run --project ../ariel python replay.py RUN --duration 30  # walk longer
    uv run --project ../ariel python replay.py RUN --terrain 1    # 2nd arena
    uv run --project ../ariel python replay.py RUN --flat --yaw 30  # turned

RUN is a run folder, e.g. results/final/best/seed20 for the final experiment.

By default the robot walks the terrain it was evolved on (saved with the run).
`--new-terrain` builds its world anew - a random terrain (RuggedTerrainWorld)
or rugged strip (OlympicArena) it never saw, the robustness test - and
`--flat` uses ARIEL's flat world. The video lands in the run's folder as
`replay*.mp4`.

The printed distance and fitness are not read off the video: they come from
`simulate.walk` itself on the same ground, the measurement the EA uses.
With `stop_at_target` (on in the final experiment) that walk ends once the
core is within `TARGET_RADIUS` (0.1 m) of the target, and so does the video
(to within one frame). The live viewer keeps walking past the target.

The video's controller is driven through MuJoCo's control callback (the
viewer and ARIEL's video renderer step the simulation themselves): it queries
the network every `control_every` physics steps and holds the output in
between, as `simulate.walk` does.

MuJoCo's results differ slightly between processor architectures (x86-64 and
ARM), so a run replays exactly only on the kind of machine it ran on. In the
final experiment (99), seeds 10-19 ran on x86-64 and seeds 20-29 on ARM
(Apple).
"""

# Standard library
import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

# Third-party libraries
import mujoco as mj
import numpy as np
from mujoco import viewer

# Local libraries (ARIEL)
from ariel.simulation.environments import RuggedTerrainWorld, SimpleFlatWorld
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Local libraries
from genome import split
from network import Genotype, forward
from run import WORLDS
from sensors import CORE_BODY, HALF_PI, read_inputs
from simulate import (
    TARGET_RADIUS,
    TARGET_XY,
    SimConfig,
    build_model,
    final_sim_config,
    fitness,
    walk,
)
from terrain import ground_geoms


def make_controller(
    genotype: Genotype,
    config: SimConfig,
    model: mj.MjModel,
) -> Callable[[mj.MjModel, mj.MjData], None]:
    """A MuJoCo control callback that runs the network every `control_every` steps."""
    shape = config.shape
    weights, clock_hz = split(genotype, shape, config.evolve_tempo)
    core_id = model.body(CORE_BODY).id
    ground = ground_geoms(model) if config.vision else None

    def control(m: mj.MjModel, d: mj.MjData) -> None:
        step = round(d.time / m.opt.timestep)
        if step % config.control_every == 0:
            inputs = read_inputs(
                m,
                d,
                core_id,
                TARGET_XY,
                ground,
                clock_hz,
                config.vision_rays,
                config.position,
            )
            d.ctrl[:] = forward(weights, shape, inputs) * HALF_PI

    return control


def main() -> None:
    """Replay the best genotype of one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run", type=Path, help="a run folder, e.g. results/final/best/seed20"
    )
    parser.add_argument("--viewer", action="store_true", help="open a live window")
    ground = parser.add_mutually_exclusive_group()
    ground.add_argument("--new-terrain", action="store_true", help="unseen terrain")
    ground.add_argument("--flat", action="store_true", help="walk on flat ground")
    parser.add_argument(
        "--duration", type=float, help="seconds to walk (default: the run's walks)"
    )
    parser.add_argument(
        "--terrain", type=int, default=0, help="which training terrain (default: 0)"
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="degrees the robot starts turned by, with --flat or --new-terrain (D23)",
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if not (np.isfinite(args.yaw) and abs(args.yaw) <= 180):
        parser.error("--yaw must lie between -180 and 180 degrees")

    run_config = json.loads((args.run / "config.json").read_text())
    config = final_sim_config(args.run)
    if args.duration is not None:
        config = replace(config, duration=args.duration)
    genotype = np.load(args.run / "best_genotype.npy")
    training_terrains = run_config.get("terrains", [])
    ariel_spawn = run_config.get("ea", {}).get("ariel_spawn", False)
    if args.flat:
        model = build_model(SimpleFlatWorld, config.body, args.yaw, ariel_spawn)
    elif args.new_terrain or not training_terrains:
        worlds = {factory.__name__: factory for factory in WORLDS.values()}
        world = run_config.get("world", RuggedTerrainWorld.__name__)
        model = build_model(worlds[world], config.body, args.yaw, ariel_spawn)
    elif args.yaw:
        parser.error(
            "--yaw needs --flat or --new-terrain: a training terrain has its own"
        )
    elif not 0 <= args.terrain < len(training_terrains):
        parser.error(f"--terrain must be 0 to {len(training_terrains) - 1}")
    else:
        model = mj.MjModel.from_binary_path(training_terrains[args.terrain])
    data = mj.MjData(model)

    # The walk as the EA measures it; before the callback is set, because
    # `walk` drives the network itself.
    score = walk(genotype, model, config)
    arrived = config.stop_at_target and score.distance < TARGET_RADIUS
    print(
        f"walk: {score.distance:.3f} m from the target, "
        f"fitness {fitness(score, config):.3f}"
    )
    if arrived:
        print(f"reached the target after {score.seconds:.2f} s; the walk ends there")

    mj.set_mjcb_control(make_controller(genotype, config, model))
    try:
        if args.viewer:
            viewer.launch(model=model, data=data)
        else:
            recorder = VideoRecorder(file_name="replay", output_folder=args.run)
            seconds = score.seconds if arrived else config.duration
            video_renderer(model, data, duration=seconds, video_recorder=recorder)
            print(f"video of {data.time:.1f} s saved in {args.run}")
    finally:
        mj.set_mjcb_control(None)


if __name__ == "__main__":
    main()
