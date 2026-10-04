"""Emigrant selection policies and the ring migration plan the EA carries out."""

import numpy as np
import numpy.typing as npt
import pytest

from migration import plan_migration, select_emigrants

FITNESS = np.array([0.9, 0.1, 0.5, 0.3, 0.7])


def island_fitness(n_islands: int = 3, size: int = 5) -> list[npt.NDArray]:
    """Island i holds fitness i + 0.0 .. 0.4: index 0 its best, the last its worst."""
    return [i + np.linspace(0.0, 0.4, size) for i in range(n_islands)]


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


def test_none_policy_and_a_single_island_plan_nothing() -> None:
    rng = np.random.default_rng(0)
    assert plan_migration(island_fitness(), 2, "none", rng) == []
    assert plan_migration(island_fitness(n_islands=1), 2, "best", rng) == []


def test_ring_sends_best_to_next_island_replacing_its_worst() -> None:
    plan = plan_migration(island_fitness(), 2, "best", np.random.default_rng(0))

    # Island i sends to island i + 1, and the ring wraps from island 2 to 0.
    assert [(t.source, t.target) for t in plan] == [(0, 1), (1, 2), (2, 0)]
    for transfer in plan:
        # The sender's two best (its first two) replace the receiver's two
        # worst (its last two).
        assert sorted(transfer.emigrants.tolist()) == [0, 1]
        assert sorted(transfer.replaced.tolist()) == [3, 4]


def test_emigrants_are_copied_not_moved() -> None:
    """An island only loses the natives its immigrants replace, never its emigrants."""
    plan = plan_migration(island_fitness(), 2, "best", np.random.default_rng(0))
    replaced_on = {t.target: set(t.replaced.tolist()) for t in plan}
    for transfer in plan:
        assert len(transfer.emigrants) == len(transfer.replaced)  # sizes stay
        assert not set(transfer.emigrants.tolist()) & replaced_on[transfer.source]


def test_arrivals_are_not_forwarded_in_the_same_event() -> None:
    """Island 1 sends its own best natives, not island 0's arrivals.

    Island 0's best are better than all of island 1, so if the plan were made
    after their arrival, island 1 would send them on.
    """
    plan = plan_migration(island_fitness(), 2, "best", np.random.default_rng(0))
    arrival_slots = set(plan[0].replaced.tolist())
    assert plan[1].source == 1
    assert not set(plan[1].emigrants.tolist()) & arrival_slots


def test_too_many_migrants_is_an_error() -> None:
    with pytest.raises(ValueError, match="cannot send"):
        select_emigrants(FITNESS, 6, "best", np.random.default_rng(0))
    with pytest.raises(ValueError, match="cannot send"):
        plan_migration(island_fitness(size=3), 4, "random", np.random.default_rng(0))
