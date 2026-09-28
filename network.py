"""The controller network and how a flat genotype maps onto it.

An individual's genotype is one flat vector of floats. This module is the single
place that knows how that vector is cut up into the network's weight matrices
(the genotype-to-phenotype mapping). Everything else - the EA, the operators,
the simulator - only sees flat vectors plus a `NetworkShape`.

Architecture (decisions D5 and D6 in docs/decisions.md):

    inputs (+ bias) --W1--> hidden (tanh) (+ bias) --W2--> outputs (tanh)

Genotype layout, in this order:

    W1 : (n_inputs + 1) x n_hidden   - the last row holds the hidden biases
    W2 : (n_hidden + 1) x n_outputs  - the last row holds the output biases

Column j of W1 (the weights INTO hidden neuron j) together with row j of W2
(the weights OUT OF hidden neuron j) is everything hidden neuron j does. The
neuron-level crossover in `operators.py` relies on that grouping.
"""

# Standard library
from dataclasses import dataclass

# Third-party libraries
import numpy as np
import numpy.typing as npt

type Genotype = npt.NDArray[np.float64]


@dataclass(frozen=True)
class NetworkShape:
    """Layer sizes of the controller. Biases are added on top of these."""

    n_inputs: int
    n_hidden: int
    n_outputs: int

    @property
    def w1_shape(self) -> tuple[int, int]:
        """Shape of the input-to-hidden matrix, bias row included."""
        return (self.n_inputs + 1, self.n_hidden)

    @property
    def w2_shape(self) -> tuple[int, int]:
        """Shape of the hidden-to-output matrix, bias row included."""
        return (self.n_hidden + 1, self.n_outputs)

    @property
    def n_weights(self) -> int:
        """Genotype length: every weight and bias in the network."""
        rows1, cols1 = self.w1_shape
        rows2, cols2 = self.w2_shape
        return rows1 * cols1 + rows2 * cols2


def unpack(
    genotype: Genotype,
    shape: NetworkShape,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cut a flat genotype into (W1, W2). The results are views, not copies."""
    if genotype.shape != (shape.n_weights,):
        msg = f"genotype has shape {genotype.shape}, expected ({shape.n_weights},)"
        raise ValueError(msg)
    split = shape.w1_shape[0] * shape.w1_shape[1]
    w1 = genotype[:split].reshape(shape.w1_shape)
    w2 = genotype[split:].reshape(shape.w2_shape)
    return w1, w2


def pack(w1: npt.NDArray[np.float64], w2: npt.NDArray[np.float64]) -> Genotype:
    """Inverse of `unpack`: flatten (W1, W2) back into one genotype vector."""
    return np.concatenate([w1.ravel(), w2.ravel()])


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
    w1, w2 = unpack(genotype, shape)
    hidden = np.tanh(inputs @ w1[:-1] + w1[-1])
    return np.tanh(hidden @ w2[:-1] + w2[-1])
