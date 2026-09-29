"""Choosing a different John Set body changes the network and the spawn, nothing else."""

import numpy as np
import pytest
from ariel.simulation.environments import SimpleFlatWorld

from bodies import BODIES, body_info, build_body
from simulate import SimConfig, build_model, walk


def test_every_listed_body_exists_in_the_john_set() -> None:
    for name in BODIES:
        assert build_body(name) is not None


def test_unknown_body_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown body"):
        build_body("octopus")


@pytest.mark.parametrize(("body", "hinges"), [("spider_16", 16), ("snake", 8)])
def test_network_size_follows_the_body(body: str, hinges: int) -> None:
    config = SimConfig(body=body)
    assert body_info(body).hinges == hinges
    assert config.shape.n_outputs == hinges
    assert config.shape.n_inputs == hinges + 2 + 3 + 3 + 10


def test_a_snake_walks_without_errors() -> None:
    config = SimConfig(body="snake", duration=0.5)
    model = build_model(SimpleFlatWorld, "snake")
    assert model.nu == 8
    score = walk(np.zeros(config.shape.n_weights), model, config)
    assert np.isfinite(score.distance)
