"""The genotype layout with the rhythm options (decision D17)."""

import numpy as np
import pytest
from ariel.simulation.environments import SimpleFlatWorld

from genome import (
    DEFAULT_TEMPO_HZ,
    TEMPO_MAX_HZ,
    TEMPO_MIN_HZ,
    crossover,
    genotype_length,
    new_genotype,
    split,
    tempo_hz,
)
from network import NetworkShape, unpack
from simulate import SimConfig, build_model, walk

SHAPE = NetworkShape(n_inputs=6, hidden=(4,), n_outputs=3)
CLOCK_ROWS = (3, 4)


def test_tempo_gene_maps_onto_the_allowed_range() -> None:
    assert tempo_hz(0.0) == pytest.approx(1.0)
    assert TEMPO_MIN_HZ < tempo_hz(-20.0) < TEMPO_MIN_HZ * 1.01
    assert TEMPO_MAX_HZ * 0.99 < tempo_hz(20.0) < TEMPO_MAX_HZ
    assert tempo_hz(-1.0) < tempo_hz(0.0) < tempo_hz(1.0)


def test_split_with_and_without_the_tempo_gene() -> None:
    plain = np.zeros(SHAPE.n_weights)
    weights, hz = split(plain, SHAPE, evolve_tempo=False)
    assert weights.shape == (SHAPE.n_weights,) and hz == DEFAULT_TEMPO_HZ

    with_tempo = np.append(plain, 0.0)
    assert genotype_length(SHAPE, True) == SHAPE.n_weights + 1
    weights, hz = split(with_tempo, SHAPE, evolve_tempo=True)
    assert weights.shape == (SHAPE.n_weights,) and hz == pytest.approx(1.0)

    with pytest.raises(ValueError, match="genes"):
        split(plain, SHAPE, evolve_tempo=True)


def test_clock_boost_scales_only_the_clock_rows() -> None:
    plain = new_genotype(
        SHAPE, np.random.default_rng(0), evolve_tempo=False, clock_rows=CLOCK_ROWS
    )
    boosted = new_genotype(
        SHAPE,
        np.random.default_rng(0),
        evolve_tempo=False,
        clock_rows=CLOCK_ROWS,
        clock_boost=3.0,
    )
    first_plain, first_boosted = unpack(plain, SHAPE)[0], unpack(boosted, SHAPE)[0]
    np.testing.assert_allclose(
        first_boosted[list(CLOCK_ROWS)], 3 * first_plain[list(CLOCK_ROWS)]
    )
    others = [r for r in range(first_plain.shape[0]) if r not in CLOCK_ROWS]
    np.testing.assert_array_equal(first_boosted[others], first_plain[others])


def test_crossover_takes_the_tempo_from_one_parent() -> None:
    parent_a = np.append(np.zeros(SHAPE.n_weights), -1.0)
    parent_b = np.append(np.ones(SHAPE.n_weights), 1.0)
    tempos = {
        crossover(parent_a, parent_b, SHAPE, True, np.random.default_rng(s))[-1]
        for s in range(20)
    }
    assert tempos == {-1.0, 1.0}


def test_a_walk_uses_the_evolved_tempo() -> None:
    config = SimConfig(body="spider_8", duration=1.0, evolve_tempo=True)
    model = build_model(SimpleFlatWorld, "spider_8")
    genotype = new_genotype(
        config.shape,
        np.random.default_rng(0),
        evolve_tempo=True,
        clock_rows=(config.hinges, config.hinges + 1),
        clock_boost=3.0,
    )
    assert genotype.shape == (config.genotype_length,)
    slow, fast = genotype.copy(), genotype.copy()
    slow[-1], fast[-1] = -5.0, 5.0
    assert walk(slow, model, config) != walk(fast, model, config)
