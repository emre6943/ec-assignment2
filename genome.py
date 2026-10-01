"""The full genotype: the network's weights, plus an optional tempo gene.

    [ w_1 ... w_n | tempo ]      the tempo gene only with `SimConfig.evolve_tempo`

Two "rhythm" options (decision D17) live here, both off by default:

- **Tempo gene** (`SimConfig.evolve_tempo`): one extra gene sets the frequency
  of the clock the network walks to, instead of a fixed 1 Hz. The gene is an
  unbounded number, mapped smoothly onto 0.25-4 Hz (gene 0 = 1 Hz), so
  Gaussian mutation can nudge the tempo like any weight.
- **Rhythmic start** (`clock_boost` in `new_genotype`): at initialisation only,
  the weights from the two clock inputs are multiplied by `clock_boost`.
  Random networks then already command a clear rhythm, each joint with its
  own phase and amplitude (measured: output standard deviation about 0.25 of
  the range at boost 1, 0.48 at boost 3). Every weight stays freely evolvable.
  It applies to every new random genotype, so also to random search.

Neither turns the controller into a CPG: the clock is a fixed input signal;
nothing oscillates inside the network and there are no coupled oscillators.
"""

# Third-party libraries
import numpy as np

# Local libraries
from network import Genotype, NetworkShape, random_genotype, unpack
from operators import blx_crossover, neuron_crossover, weight_crossover

TEMPO_MIN_HZ: float = 0.25
TEMPO_MAX_HZ: float = 4.0
DEFAULT_TEMPO_HZ: float = 1.0  # the clock when the tempo is not evolved


def genotype_length(shape: NetworkShape, evolve_tempo: bool) -> int:
    """Number of genes: every weight, plus the tempo gene if it is evolved."""
    return shape.n_weights + (1 if evolve_tempo else 0)


def tempo_hz(gene: float) -> float:
    """Map an unbounded tempo gene onto [0.25, 4] Hz; gene 0 gives 1 Hz."""
    share = 1.0 / (1.0 + np.exp(-gene))  # logistic: (-inf, inf) -> (0, 1)
    return float(TEMPO_MIN_HZ * (TEMPO_MAX_HZ / TEMPO_MIN_HZ) ** share)


def split(
    genotype: Genotype, shape: NetworkShape, evolve_tempo: bool
) -> tuple[Genotype, float]:
    """(network weights, clock frequency in Hz) encoded by a genotype."""
    if genotype.shape != (genotype_length(shape, evolve_tempo),):
        msg = (
            f"genotype has {genotype.shape[0]} genes, expected "
            f"{genotype_length(shape, evolve_tempo)}"
        )
        raise ValueError(msg)
    if evolve_tempo:
        return genotype[:-1], tempo_hz(genotype[-1])
    return genotype, DEFAULT_TEMPO_HZ


def new_genotype(
    shape: NetworkShape,
    rng: np.random.Generator,
    *,
    evolve_tempo: bool,
    clock_rows: tuple[int, int],
    init_scale: float = 0.5,
    clock_boost: float = 1.0,
) -> Genotype:
    """A random genotype for the initial population (or random search).

    Weights come from N(0, init_scale^2); the rows of the first weight matrix
    that carry the two clock inputs (`clock_rows`) are then multiplied by
    `clock_boost`. The tempo gene, if any, starts from N(0, 1): tempos spread
    around 1 Hz, mostly between about 0.5 and 2 Hz.
    """
    weights = random_genotype(shape, rng, init_scale)
    if clock_boost != 1.0:
        first = unpack(weights, shape)[0]  # a view into `weights`
        first[list(clock_rows)] *= clock_boost
    if not evolve_tempo:
        return weights
    return np.append(weights, rng.normal(0.0, 1.0))


def crossover(
    parent_a: Genotype,
    parent_b: Genotype,
    shape: NetworkShape,
    evolve_tempo: bool,
    rng: np.random.Generator,
    kind: str = "neuron",
) -> Genotype:
    """Crossover of the weights; the tempo gene from either parent.

    `kind` (`ea.EAConfig.crossover`, D7): "neuron" (and "headless", whose second
    parent the EA replaces by a random genotype) keeps neurons whole, "weight"
    mixes single weights, "blx" blends them.
    """
    weights_a, _ = split(parent_a, shape, evolve_tempo)
    weights_b, _ = split(parent_b, shape, evolve_tempo)
    if kind in ("neuron", "headless"):
        child = neuron_crossover(weights_a, weights_b, shape, rng)
    elif kind == "weight":
        child = weight_crossover(weights_a, weights_b, rng)
    elif kind == "blx":
        child = blx_crossover(weights_a, weights_b, rng)
    else:
        msg = f"unknown crossover {kind!r}"
        raise ValueError(msg)
    if not evolve_tempo:
        return child
    tempo = parent_b[-1] if rng.random() < 0.5 else parent_a[-1]
    return np.append(child, tempo)
