"""Emigrant selection policies and the ring migration step."""

import numpy as np
import pytest

from migration import Island, migrate, select_emigrants

FITNESS = np.array([0.9, 0.1, 0.5, 0.3, 0.7])


def make_islands(n_islands: int = 3, size: int = 5) -> list[Island]:
    """Island i holds genotypes filled with the value i, fitness i + 0.0 .. 0.4."""
    return [
        Island(
            genotypes=np.full((size, 2), float(i)),
            fitness=i + np.linspace(0.0, 0.4, size),
        )
        for i in range(n_islands)
    ]


def test_best_policy_picks_lowest_distances() -> None:
    chosen = select_emigrants(FITNESS, 2, "best", np.random.default_rng(0))
    assert sorted(chosen.tolist()) == [1, 3]


def test_worst_policy_picks_highest_distances() -> None:
    chosen = select_emigrants(FITNESS, 2, "worst", np.random.default_rng(0))
    assert sorted(chosen.tolist()) == [0, 4]


def test_random_policy_picks_distinct_individuals_and_varies() -> None:
    picks = {
        tuple(sorted(select_emigrants(FITNESS, 2, "random", np.random.default_rng(s))))
        for s in range(30)
    }
    assert all(len(set(p)) == 2 for p in picks)
    assert len(picks) > 3


def test_none_policy_changes_nothing() -> None:
    islands = make_islands()
    before = [(i.genotypes.copy(), i.fitness.copy()) for i in islands]
    migrate(islands, 2, "none", np.random.default_rng(0))
    for island, (genotypes, fitness) in zip(islands, before, strict=True):
        np.testing.assert_array_equal(island.genotypes, genotypes)
        np.testing.assert_array_equal(island.fitness, fitness)


def test_ring_sends_best_to_next_island_replacing_its_worst() -> None:
    islands = make_islands()
    migrate(islands, 2, "best", np.random.default_rng(0))

    # Island 1 received island 0's two best (fitness 0.0 and 0.1) in place of
    # its own two worst (1.3 and 1.4); the ring wraps from island 2 to island 0.
    np.testing.assert_allclose(sorted(islands[1].fitness), [0.0, 0.1, 1.0, 1.1, 1.2])
    assert np.sum(islands[1].genotypes[:, 0] == 0.0) == 2
    assert np.sum(islands[0].genotypes[:, 0] == 2.0) == 2


def test_emigrants_are_copied_not_moved() -> None:
    islands = make_islands()
    migrate(islands, 2, "best", np.random.default_rng(0))
    assert all(len(island.fitness) == 5 for island in islands)
    # Island 0 still has its own best individual (fitness 0.0).
    assert 0.0 in islands[0].fitness


def test_arrivals_are_not_forwarded_in_the_same_event() -> None:
    islands = make_islands()
    migrate(islands, 2, "best", np.random.default_rng(0))
    # Island 2 must only have received island 1's natives, never island 0's.
    assert not np.any(islands[2].genotypes[:, 0] == 0.0)


def test_too_many_migrants_is_an_error() -> None:
    with pytest.raises(ValueError, match="cannot send"):
        select_emigrants(FITNESS, 6, "best", np.random.default_rng(0))


def test_island_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="fitness values"):
        Island(genotypes=np.zeros((3, 2)), fitness=np.zeros(2))
