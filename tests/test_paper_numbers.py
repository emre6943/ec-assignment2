"""The helpers behind the paper's loose numbers (paper_numbers.py)."""

# Third-party libraries
import pandas as pd
import pytest

# Local libraries
from paper_numbers import paired_t_power, variance_shares


def test_power_grows_with_seeds_and_matches_a_known_value() -> None:
    # d_z = 0.5 with 34 pairs at alpha 0.05 is the textbook ~80% case
    assert paired_t_power(34, 0.5, 0.05) == pytest.approx(0.80, abs=0.01)
    assert paired_t_power(10, 0.6, 0.05) < paired_t_power(40, 0.6, 0.05)


def test_variance_shares_split_seed_and_condition_effects() -> None:
    # rows are seeds, columns conditions: a pure seed effect, then a pure
    # condition effect, then both
    seed_only = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    assert variance_shares(seed_only) == pytest.approx((1.0, 0.0))
    condition_only = pd.DataFrame({"a": [1.0, 1.0], "b": [3.0, 3.0]})
    assert variance_shares(condition_only) == pytest.approx((0.0, 1.0))
    both = pd.DataFrame({"a": [0.0, 2.0], "b": [4.0, 6.0]})
    seeds, conditions = variance_shares(both)
    assert seeds + conditions == pytest.approx(1.0)
    assert conditions > seeds
