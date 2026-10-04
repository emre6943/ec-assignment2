"""The final experiment's flags build the configuration the paper describes.

The code's defaults are the early experiments' settings; the paper's settings
exist only as the flags of experiments/99_final_experiment.sh. This test reads
those flags from the script itself and passes them through run.py's own
parser, so the script, the CLI and the paper cannot drift apart unnoticed.
"""

import re
import shlex
from dataclasses import fields, replace
from pathlib import Path

import pytest

from run import build_parser, configs_from_args
from simulate import Score, fitness

SCRIPT = Path(__file__).resolve().parent.parent / "experiments/99_final_experiment.sh"


def final_flags() -> list[str]:
    """The run.py flags in the script's `run=(...)` array, minus `--workers`."""
    found = re.search(r"^run=\((.*?)\)$", SCRIPT.read_text(), re.MULTILINE | re.DOTALL)
    assert found, f"no run=(...) array in {SCRIPT}"
    words = shlex.split(found.group(1))
    words = words[words.index("run.py") + 1 :]
    workers = words.index("--workers")
    del words[workers : workers + 2]  # "$workers" is filled in by the script
    return words


def test_the_final_experiments_flags_give_the_papers_setup() -> None:
    args = build_parser().parse_args([*final_flags(), "--policy", "best"])
    ea, sim = configs_from_args(args, seed=10)

    # Body, world and walks (D1, D2, D2a, D12, D20).
    assert (args.world, sim.body) == ("olympic", "spider_8")
    assert ea.ariel_spawn
    assert sim.duration == 15.0
    assert sim.stop_at_target

    # The brain: 16 inputs, no vision, 16-8-4-8, 212 weights (D5, D6).
    assert not sim.vision
    assert not sim.position
    assert not sim.evolve_tempo
    assert sim.shape.n_inputs == 16
    assert sim.shape.layer_sizes == (16, 8, 4, 8)
    assert sim.shape.n_weights == 212
    assert sim.genotype_length == 212

    # The EA (D7-D9, D11, D12, D22).
    assert (ea.algorithm, ea.policy) == ("island", "best")
    assert (ea.n_islands, ea.island_size) == (4, 20)
    assert (ea.migration_interval, ea.n_migrants) == (20, 2)
    assert ea.crossover == "neuron"
    assert ea.crossover_probability == 0.9
    assert ea.mutation_sigma == 0.05
    assert ea.mutation_rate == 1.0
    assert ea.tournament_size == 3
    assert ea.n_elites == 2
    assert ea.clock_boost == 1.0
    assert ea.max_evaluations == 12_000

    # None of the earlier experiments' options.
    assert ea.terrain_mode == "per_run"
    assert ea.n_terrains == 1
    assert not (ea.curriculum or ea.early_stop)
    assert ea.stall_generations == 0
    assert ea.final_duration == 0.0
    assert ea.init_from == ""

    # The fitness weights of the paper's Eq. 1: each part alone, at 1.
    zero = Score(
        distance=0.0, displacement=0.0, ground_contact=0.0, upside_down=0.0, seconds=0.0
    )
    weights = {
        part.name: fitness(replace(zero, **{part.name: 1.0}), sim)
        for part in fields(Score)
    }
    assert weights == {
        "distance": 1.0,  # d_T
        "mean_distance": 0.5,  # d-bar
        "ground_contact": 1.0,  # c
        "low_body": 1.0,  # l
        "upside_down": 1.0,  # u
        "work_imbalance": 0.5,  # w
        "leg_imbalance": 0.0,  # leg movement: not in the paper
        "displacement": 0.0,
        "seconds": 0.0,
    }
    assert sim.carry_height == pytest.approx(0.04)  # the 4 cm line of l
