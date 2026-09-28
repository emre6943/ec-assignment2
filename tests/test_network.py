"""The genotype-to-network mapping."""

import numpy as np
import pytest

from network import NetworkShape, forward, pack, random_genotype, unpack

SHAPE = NetworkShape(n_inputs=5, n_hidden=3, n_outputs=4)


def test_weight_count_includes_biases() -> None:
    assert SHAPE.n_weights == (5 + 1) * 3 + (3 + 1) * 4


def test_pack_inverts_unpack() -> None:
    genotype = np.arange(SHAPE.n_weights, dtype=np.float64)
    w1, w2 = unpack(genotype, SHAPE)
    assert w1.shape == (6, 3)
    assert w2.shape == (4, 4)
    np.testing.assert_array_equal(pack(w1, w2), genotype)


def test_unpack_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected"):
        unpack(np.zeros(SHAPE.n_weights + 1), SHAPE)


def test_forward_matches_hand_computation() -> None:
    rng = np.random.default_rng(0)
    genotype = random_genotype(SHAPE, rng)
    inputs = rng.normal(size=5)
    w1, w2 = unpack(genotype, SHAPE)

    hidden = np.tanh(np.append(inputs, 1.0) @ w1)
    expected = np.tanh(np.append(hidden, 1.0) @ w2)

    np.testing.assert_allclose(forward(genotype, SHAPE, inputs), expected)


def test_outputs_stay_in_tanh_range() -> None:
    genotype = random_genotype(SHAPE, np.random.default_rng(1), scale=50.0)
    outputs = forward(genotype, SHAPE, np.full(5, 10.0))
    assert outputs.shape == (4,)
    assert np.all(np.abs(outputs) <= 1.0)
