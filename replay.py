"""Watch a run's best network walk.

    uv run --project ../ariel python replay.py results/best/seed0                # video
    uv run --project ../ariel python replay.py results/best/seed0 --viewer       # live window
    uv run --project ../ariel python replay.py results/best/seed0 --new-terrain  # unseen ground
    uv run --project ../ariel python replay.py results/best/seed0 --flat         # flat ground

By default the robot walks the terrain it was evolved on (saved with the run).
`--new-terrain` builds its world anew - a random terrain (RuggedTerrainWorld)
or rugged strip (OlympicArena) it never saw, the robustness test - and
`--flat` uses ARIEL's flat world. The video lands in the
run's folder as `replay*.mp4`, and the final distance to the target is printed.

The controller is driven through MuJoCo's control callback here (the viewer
and ARIEL's video renderer step the simulation themselves), but it does the
same thing as `simulate.walk`: query the network every `control_every`
physics steps and hold the output in between.
"""

# Standard library
import argparse
import json
from collections.abc import Callable
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
from simulate import TARGET_XY, SimConfig, build_model, final_sim_config
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
                m, d, core_id, TARGET_XY, ground, clock_hz, config.vision_rays
            )
            d.ctrl[:] = forward(weights, shape, inputs) * HALF_PI

    return control


def main() -> None:
    """Replay the best genotype of one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="a run folder, e.g. results/best/seed0")
    parser.add_argument("--viewer", action="store_true", help="open a live window")
    ground = parser.add_mutually_exclusive_group()
    ground.add_argument("--new-terrain", action="store_true", help="unseen terrain")
    ground.add_argument("--flat", action="store_true", help="walk on flat ground")
    args = parser.parse_args()

    run_config = json.loads((args.run / "config.json").read_text())
    config = final_sim_config(run_config)
    genotype = np.load(args.run / "best_genotype.npy")
    training_terrains = run_config.get("terrains", [])
    if args.flat:
        model = build_model(SimpleFlatWorld, config.body)
    elif args.new_terrain or not training_terrains:
        worlds = {factory.__name__: factory for factory in WORLDS.values()}
        world = run_config.get("world", RuggedTerrainWorld.__name__)
        model = build_model(worlds[world], config.body)
    else:
        model = mj.MjModel.from_binary_path(training_terrains[0])
    data = mj.MjData(model)

    mj.set_mjcb_control(make_controller(genotype, config, model))
    try:
        if args.viewer:
            viewer.launch(model=model, data=data)
        else:
            recorder = VideoRecorder(file_name="replay", output_folder=args.run)
            video_renderer(
                model, data, duration=config.duration, video_recorder=recorder
            )
            final = data.qpos[:2]
            distance = float(np.linalg.norm(final - TARGET_XY))
            print(f"ended at {np.round(final, 2)}, {distance:.2f} m from the target")
            print(f"video saved in {args.run}")
    finally:
        mj.set_mjcb_control(None)


if __name__ == "__main__":
    main()
