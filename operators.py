"""Variation and selection operators on flat genotypes.

Every function takes an explicit `np.random.Generator`, so a run is fully
determined by its seed. The fitness (`simulate.fitness`) is minimised: LOWER IS
BETTER.

- `neuron_crossover`    decision D7
- `weight_crossover`, `blx_crossover`   alternatives tested in experiment 25
- `tournament_select`   decision D9

Mutation is ariel's own `FloatMutator.gaussian` (decision D8); it draws from
ariel.ec's package-level RNG, which `ariel.ec.set_seed` reseeds.
"""

# Third-party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
from network import Genotype, NetworkShape, pack, unpack


def neuron_crossover(
    parent_a: Genotype,
    parent_b: Genotype,
    shape: NetworkShape,
    rng: np.random.Generator,
) -> Genotype:
    """Uniform crossover that keeps every hidden neuron intact.

    For each hidden neuron, a fair coin picks the parent it comes from, and the
    child copies that neuron's weights as a unit: its incoming column (bias
    included) and, for neurons of the LAST hidden layer, also its outgoing row
    into the output layer. Mixing weights of one neuron from two different
    parents usually breaks what the neuron did (the "competing conventions"
    problem), so the neuron is the unit of inheritance.

    With one hidden layer this moves each hidden neuron completely (all its
    in- and out-weights). With more layers, a neuron's outgoing weights into
    the next hidden layer travel with the neuron they feed instead, since each
    weight can only belong to one unit. The output biases belong to no hidden
    neuron; each one is inherited from a random parent.
    """
    matrices_a = unpack(parent_a, shape)
    matrices_b = unpack(parent_b, shape)
    child = [matrix.copy() for matrix in matrices_a]

    last_hidden = len(shape.hidden) - 1
    for layer, n_neurons in enumerate(shape.hidden):
        from_b = rng.random(n_neurons) < 0.5
        child[layer][:, from_b] = matrices_b[layer][:, from_b]
        if layer == last_hidden:
            outgoing = child[layer + 1]
            outgoing[:-1][from_b] = matrices_b[layer + 1][:-1][from_b]

    bias_from_b = rng.random(shape.n_outputs) < 0.5
    child[-1][-1] = np.where(bias_from_b, matrices_b[-1][-1], matrices_a[-1][-1])
    return pack(child)


def weight_crossover(
    parent_a: Genotype, parent_b: Genotype, rng: np.random.Generator
) -> Genotype:
    """Uniform crossover of single weights: each weight from either parent, 50/50.

    Unlike `neuron_crossover`, the weights of one neuron can come from both
    parents (experiment 25 tests whether keeping neurons whole matters).
    """
    return np.where(rng.random(parent_a.shape) < 0.5, parent_b, parent_a)


def blx_crossover(
    parent_a: Genotype,
    parent_b: Genotype,
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> Genotype:
    """BLX-alpha (Eshelman & Schaffer 1993): each weight drawn uniformly from the
    interval between the parents' values, widened by `alpha` times its length on
    both sides. Identical parents give an identical child, so the spread shrinks
    as an island converges.
    """
    low = np.minimum(parent_a, parent_b)
    high = np.maximum(parent_a, parent_b)
    margin = alpha * (high - low)
    return rng.uniform(low - margin, high + margin)


def tournament_select(
    fitness: npt.NDArray[np.float64],
    k: int,
    rng: np.random.Generator,
) -> int:
    """Pick `k` individuals at random (without replacement); return the best index.

    Larger `k` means higher selection pressure: with k=1 this is uniform random
    selection, with k equal to the population size it always returns the best.
    """
    contestants = rng.choice(len(fitness), size=k, replace=False)
    return int(contestants[np.argmin(fitness[contestants])])
