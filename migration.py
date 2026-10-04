"""Migration between islands - the part of the EA our research question varies.

Every `interval` generations, each island sends copies of `n_migrants` of its
individuals to the next island in a ring (island i -> island (i + 1) mod n).
On arrival, the immigrants replace the receiving island's worst individuals.

Only the EMIGRANT SELECTION differs between experimental conditions:

    "best"    send the island's fittest individuals
    "worst"   send the island's least fit individuals
    "random"  send individuals drawn uniformly at random
    "none"    no migration at all (the control condition)

Topology, interval, number of migrants and the replacement policy are held
fixed across conditions (decision D11 in docs/decisions.md).

The fitness (`simulate.fitness`) is minimised: LOWER IS BETTER.
"""

# Standard library
from dataclasses import dataclass
from typing import Literal

# Third-party libraries
import numpy as np
import numpy.typing as npt

type Policy = Literal["best", "worst", "random", "none"]
POLICIES: tuple[Policy, ...] = ("best", "worst", "random", "none")


def select_emigrants(
    fitness: npt.NDArray[np.float64],
    n_migrants: int,
    policy: Policy,
    rng: np.random.Generator,
) -> npt.NDArray[np.intp]:
    """Return the indices of the individuals that emigrate under `policy`."""
    if n_migrants > len(fitness):
        msg = f"cannot send {n_migrants} migrants from an island of {len(fitness)}"
        raise ValueError(msg)
    match policy:
        case "best":
            return np.argsort(fitness, kind="stable")[:n_migrants]
        case "worst":
            return np.argsort(fitness, kind="stable")[::-1][:n_migrants]
        case "random":
            return rng.choice(len(fitness), size=n_migrants, replace=False)
        case "none":
            return np.empty(0, dtype=np.intp)
    msg = f"unknown migration policy {policy!r}"
    raise ValueError(msg)


@dataclass(frozen=True)
class Transfer:
    """One island's part of a migration event.

    Copies of `source` island's individuals at `emigrants` go to `target`
    island, where they overwrite the individuals at `replaced`.
    """

    source: int
    target: int
    emigrants: npt.NDArray[np.intp]
    replaced: npt.NDArray[np.intp]


def plan_migration(
    fitness_per_island: list[npt.NDArray[np.float64]],
    n_migrants: int,
    policy: Policy,
    rng: np.random.Generator,
) -> list[Transfer]:
    """Decide who moves where in one ring migration event, without moving anyone.

    Every decision is taken from the islands as they are BEFORE the event, so
    an individual that arrives on an island cannot be sent on again in the
    same event, and immigrants never displace each other. Each island's worst
    `n_migrants` natives are the ones replaced.

    The EA (`ea.Experiment.migrate`) carries the plan out on ariel.ec's
    individuals: emigrants are copied, not moved, so island sizes stay
    constant, and immigrants keep their fitness - fair only because every
    island walks the same terrain in a generation (decision D10).
    """
    if policy == "none" or len(fitness_per_island) < 2:
        return []

    n_islands = len(fitness_per_island)
    return [
        Transfer(
            source=source,
            target=(source + 1) % n_islands,
            emigrants=select_emigrants(fitness, n_migrants, policy, rng),
            replaced=select_emigrants(
                fitness_per_island[(source + 1) % n_islands], n_migrants, "worst", rng
            ),
        )
        for source, fitness in enumerate(fitness_per_island)
    ]
