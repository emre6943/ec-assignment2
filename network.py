"""The controller network and how a flat genotype maps onto it.

An individual's genotype is one flat vector of floats. This module is the single
place that knows how that vector is cut up into the network's weight matrices
(the genotype-to-phenotype mapping). Everything else - the EA, the operators,
the simulator - only sees flat vectors plus a `NetworkShape`.

Architecture (decisions D5 and D6 in docs/decisions.md): a feed-forward network
with any number of hidden layers, all with tanh activations:

    inputs (+ bias) --W1--> hidden 1 (+ bias) --W2--> ... --Wn--> outputs (tanh)

Genotype layout: the weight matrices one after another, in layer order. Each
matrix is (neurons in + 1) x (neurons out); its last row holds the biases of
the layer it feeds. So column j of a matrix is everything flowing INTO neuron j
of the next layer - the grouping the neuron-level crossover in `operators.py`
relies on.
"""

# Standard library
from dataclasses import dataclass

# Third-party libraries
import numpy as np
import numpy.typing as npt

type Genotype = npt.NDArray[np.float64]
type Weights = list[npt.NDArray[np.float64]]


@dataclass(frozen=True)
class NetworkShape:
    """Layer sizes of the controller. Biases are added on top of these."""

    n_inputs: int
    hidden: tuple[int, ...]
    n_outputs: int

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """Neurons per layer, inputs first and outputs last."""
        return (self.n_inputs, *self.hidden, self.n_outputs)

    @property
    def matrix_shapes(self) -> list[tuple[int, int]]:
        """Shape of each weight matrix, bias row included."""
        sizes = self.layer_sizes
        return [(n_in + 1, n_out) for n_in, n_out in zip(sizes, sizes[1:])]

    @property
    def n_weights(self) -> int:
        """Genotype length: every weight and bias in the network."""
        return sum(rows * cols for rows, cols in self.matrix_shapes)


def parse_hidden(text: str) -> tuple[int, ...]:
    """'8' -> (8,), '8,8' -> (8, 8): hidden layer sizes as given on the command line."""
    sizes = tuple(int(part) for part in text.split(",") if part.strip())
    if not sizes or min(sizes) < 1:
        msg = f"hidden layers must be positive sizes like '8' or '8,8', got {text!r}"
        raise ValueError(msg)
    return sizes


def unpack(genotype: Genotype, shape: NetworkShape) -> Weights:
    """Cut a flat genotype into its weight matrices. The results are views."""
    if genotype.shape != (shape.n_weights,):
        msg = f"genotype has shape {genotype.shape}, expected ({shape.n_weights},)"
        raise ValueError(msg)
    matrices = []
    start = 0
    for rows, cols in shape.matrix_shapes:
        matrices.append(genotype[start : start + rows * cols].reshape(rows, cols))
        start += rows * cols
    return matrices


def pack(matrices: Weights) -> Genotype:
    """Inverse of `unpack`: flatten the weight matrices back into one genotype."""
    return np.concatenate([matrix.ravel() for matrix in matrices])


def random_genotype(
    shape: NetworkShape,
    rng: np.random.Generator,
    scale: float = 0.5,
) -> Genotype:
    """Draw every weight from N(0, scale^2) - the initial population."""
    return rng.normal(0.0, scale, size=shape.n_weights)


def forward(
    genotype: Genotype,
    shape: NetworkShape,
    inputs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Run the network once. Returns `n_outputs` values in [-1, 1]."""
    activation = inputs
    for matrix in unpack(genotype, shape):
        activation = np.tanh(activation @ matrix[:-1] + matrix[-1])
    return activation
