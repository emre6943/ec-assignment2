"""Variation and selection operators on flat genotypes.

Every function takes an explicit `np.random.Generator`, so a run is fully
determined by its seed. Fitness is a distance to the target: LOWER IS BETTER.

- `neuron_crossover`    decision D7
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

    For each hidden neuron, the child copies ALL of that neuron's weights - its
    incoming column in W1 (bias included) and its outgoing row in W2 - from one
    parent, chosen by a fair coin. Mixing weights of one neuron from two
    different parents usually breaks what the neuron did (the "competing
    conventions" problem), so the neuron is the unit of inheritance.

    The output biases (last row of W2) belong to no hidden neuron; each one is
    inherited from a random parent.
    """
    a1, a2 = unpack(parent_a, shape)
    b1, b2 = unpack(parent_b, shape)

    from_b = rng.random(shape.n_hidden) < 0.5
    w1 = np.where(from_b[np.newaxis, :], b1, a1)
    w2 = a2.copy()
    w2[:-1] = np.where(from_b[:, np.newaxis], b2[:-1], a2[:-1])

    bias_from_b = rng.random(shape.n_outputs) < 0.5
    w2[-1] = np.where(bias_from_b, b2[-1], a2[-1])
    return pack(w1, w2)


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
