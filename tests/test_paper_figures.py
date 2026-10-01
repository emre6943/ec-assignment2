"""paper_figures.py helpers on hand-made data, where the right answers are known."""

import json
from pathlib import Path

import pandas as pd
import pytest

from paper_figures import (
    common_generations,
    generation_size,
    shared_champion_share,
    weighted_terms,
)
from simulate import Score, SimConfig, fitness

ISLANDS = {"n_islands": 4, "island_size": 20, "n_elites": 2}
STANDARD = {"n_islands": 1, "island_size": 80, "n_elites": 8}


def test_island_and_standard_eas_share_one_generation_size() -> None:
    assert generation_size(ISLANDS) == (80, 72)
    assert generation_size(STANDARD) == (80, 72)


def write_run(folder: Path, ea: dict[str, int], generations: int) -> None:
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"ea": ea}))
    (folder / "log.csv").write_text(
        "generation\n" + "".join(f"{g}\n" for g in range(generations + 1))
    )


def test_common_generations_stop_at_the_shortest_run(tmp_path: Path) -> None:
    write_run(tmp_path / "a" / "seed0", ISLANDS, 3)
    write_run(tmp_path / "b" / "seed0", STANDARD, 2)
    generations, evaluations = common_generations([tmp_path / "a", tmp_path / "b"])
    assert generations.tolist() == [0, 1, 2]
    assert evaluations.tolist() == [80, 152, 224]


def test_unequal_generation_sizes_are_refused(tmp_path: Path) -> None:
    write_run(tmp_path / "a" / "seed0", ISLANDS, 2)
    write_run(tmp_path / "b" / "seed0", {**ISLANDS, "n_elites": 4}, 2)
    with pytest.raises(ValueError, match="generation size"):
        common_generations([tmp_path / "a", tmp_path / "b"])


def test_a_champion_is_shared_only_when_every_island_has_it() -> None:
    log = pd.DataFrame(
        {
            "generation": [0, 0, 0, 1, 1, 1, 1],
            "island": ["0", "1", "all", "0", "1", "all", "all"],
            "best": [1.5, 1.2, 1.2, 1.1, 1.1, 1.1, 1.1],
        }
    )
    assert shared_champion_share(log) == pytest.approx(0.5)


def test_the_weighted_terms_add_up_to_the_fitness() -> None:
    score = Score(
        distance=0.3,
        displacement=1.8,
        ground_contact=0.02,
        upside_down=0.01,
        seconds=15.0,
        low_body=0.1,
        leg_imbalance=0.2,
        mean_distance=1.0,
        work_imbalance=0.15,
    )
    sim = SimConfig(
        ground_contact_weight=1.0,
        low_body_weight=1.0,
        work_imbalance_weight=0.5,
        speed_weight=0.5,
    )
    terms = weighted_terms(score, sim)
    assert sum(terms.values()) == pytest.approx(fitness(score, sim))
    assert terms["mean_distance"] == pytest.approx(0.5)
    assert terms["leg_imbalance"] == 0.0  # weight 0 by default
