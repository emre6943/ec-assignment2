"""Crossover and tournament selection."""

import numpy as np

from network import NetworkShape, pack, unpack
from operators import neuron_crossover, tournament_select

SHAPE = NetworkShape(n_inputs=4, n_hidden=6, n_outputs=3)


def test_crossover_inherits_whole_neurons() -> None:
    """Every hidden neuron's in- and out-weights come from the same parent."""
    parent_a = np.zeros(SHAPE.n_weights)
    parent_b = np.ones(SHAPE.n_weights)
    for seed in range(20):
        child = neuron_crossover(parent_a, parent_b, SHAPE, np.random.default_rng(seed))
        w1, w2 = unpack(child, SHAPE)
        for j in range(SHAPE.n_hidden):
            neuron_genes = np.concatenate([w1[:, j], w2[j, :]])
            assert np.all(neuron_genes == neuron_genes[0]), f"neuron {j} was split"


def test_crossover_mixes_both_parents() -> None:
    parent_a = np.zeros(SHAPE.n_weights)
    parent_b = np.ones(SHAPE.n_weights)
    child = neuron_crossover(parent_a, parent_b, SHAPE, np.random.default_rng(3))
    assert 0 < child.sum() < SHAPE.n_weights


def test_crossover_keeps_neuron_identity_with_real_weights() -> None:
    """Hidden neuron j of the child equals hidden neuron j of one parent."""
    rng = np.random.default_rng(7)
    parent_a = rng.normal(size=SHAPE.n_weights)
    parent_b = rng.normal(size=SHAPE.n_weights)
    a1, a2 = unpack(parent_a, SHAPE)
    b1, b2 = unpack(parent_b, SHAPE)
    c1, c2 = unpack(neuron_crossover(parent_a, parent_b, SHAPE, rng), SHAPE)
    for j in range(SHAPE.n_hidden):
        from_a = np.array_equal(c1[:, j], a1[:, j]) and np.array_equal(c2[j], a2[j])
        from_b = np.array_equal(c1[:, j], b1[:, j]) and np.array_equal(c2[j], b2[j])
        assert from_a or from_b


def test_crossover_of_identical_parents_is_a_copy() -> None:
    parent = np.random.default_rng(0).normal(size=SHAPE.n_weights)
    child = neuron_crossover(parent, parent.copy(), SHAPE, np.random.default_rng(1))
    np.testing.assert_array_equal(child, parent)
    assert pack(*unpack(child, SHAPE)).shape == parent.shape


def test_tournament_of_everyone_returns_the_best() -> None:
    fitness = np.array([3.0, 0.5, 2.0, 1.0])
    assert tournament_select(fitness, k=4, rng=np.random.default_rng(0)) == 1


def test_bigger_tournaments_pick_fitter_parents() -> None:
    fitness = np.arange(20, dtype=np.float64)
    rng = np.random.default_rng(0)
    mean_k2 = np.mean(
        [fitness[tournament_select(fitness, 2, rng)] for _ in range(2000)]
    )
    mean_k5 = np.mean(
        [fitness[tournament_select(fitness, 5, rng)] for _ in range(2000)]
    )
    assert mean_k5 < mean_k2 < fitness.mean()
