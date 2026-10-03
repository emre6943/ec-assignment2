"""The exploratory end-result analysis of experiment 99 (end_results.py, D26)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from end_results import cochran_q, mcnemar_exact, ranking, report, run_measures


def test_cochran_q_with_two_conditions_is_mcnemars_chi_squared() -> None:
    """For k = 2, Q reduces to (b - c)^2 / (b + c) over the discordant seeds."""
    a = pd.Series([1, 1, 1, 1, 1, 0, 0, 1, 0, 0])
    b = pd.Series([0, 0, 0, 1, 1, 1, 0, 1, 0, 0])
    q, _ = cochran_q(pd.DataFrame({"a": a, "b": b}))
    a_only, b_only = 3, 1
    assert q == pytest.approx((a_only - b_only) ** 2 / (a_only + b_only))


def test_cochran_q_without_any_disagreement_is_p_one() -> None:
    blocks = pd.DataFrame({"a": [1, 0, 1], "b": [1, 0, 1], "c": [1, 0, 1]})
    assert cochran_q(blocks) == (0.0, 1.0)


def test_mcnemar_counts_only_the_seeds_that_disagree() -> None:
    a = pd.Series([1, 1, 1, 1, 1, 0, 1, 0])
    b = pd.Series([0, 0, 0, 0, 0, 0, 1, 0])
    assert mcnemar_exact(a, b) == pytest.approx(2 * 0.5**5)  # 5 to 0
    assert mcnemar_exact(a, a) == 1.0


def test_ranking_rewards_reaching_more() -> None:
    table = pd.DataFrame(
        {
            "condition": ["x", "y"] * 3,
            "run": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "reached": [0.3, 0.1, 0.2, 0.0, 0.0, 0.4],
        }
    )
    ranks = ranking(table, "reached", higher_is_better=True)
    assert ranks.loc["x", "best on"] == 2 and ranks.loc["y", "best on"] == 1
    assert ranks.loc["x", "mean rank"] == pytest.approx(4 / 3)


def test_the_report_covers_every_measure() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(20):
        for condition, effect in (("best", 0.0), ("none", 0.1), ("standard", -0.1)):
            reached = float(rng.integers(0, 5)) / 20
            rows.append(
                {
                    "condition": condition,
                    "run": f"seed{seed}",
                    "final_fitness": 1.2 + effect + rng.normal(0, 0.1),
                    "own_reached_15": float(rng.random() < 0.3),
                    "own_reached_30": float(rng.random() < 0.5),
                    "unseen_reached_30": reached,
                    "unseen_missed_30": 1.0 - reached,
                    "unseen_fitness": 1.8 + effect + rng.normal(0, 0.1),
                    "unseen_failed_30": float(rng.integers(0, 2)),
                    "unseen_walks_30": 20.0,
                }
            )
    text = report(pd.DataFrame(rows))
    for heading in (
        "## Per condition",
        "## Ranking per seed",
        "### own target in 15 s",
        "### final fitness",
        "### unseen arenas missed in 30 s",
        "### unseen fitness (15 s)",
    ):
        assert heading in text
    assert "Cochran's Q" in text and "Exploratory" in text


def test_run_measures_reads_a_run_folder(tmp_path: Path) -> None:
    """The files a run of experiment 99 leaves: log, longer walks, unseen tests."""
    run = tmp_path / "seed10"
    run.mkdir()
    (run / "log.csv").write_text(
        "generation,evaluations,island,best_final,best_distance\n"
        "0,80,all,2.0,1.5\n1,152,all,1.4,0.9\n2,224,all,1.5,0.8\n"
    )
    walk = {"distance": 0.5, "seconds": 15.0, "arrived_at": None}
    arrived = {"distance": 0.09, "seconds": 17.0, "arrived_at": 17.0}
    (run / "longer_walks.json").write_text(
        json.dumps({"15": walk, "20": arrived, "30": arrived, "60": arrived})
    )
    unseen = {"fitness_mean": 1.8, "distance": [0.9, 1.1]}
    (run / "unseen.json").write_text(json.dumps({"unseen": unseen}))
    unseen_30 = {"reached": 0.25, "distance": [0.05, 10.0, 0.7, 0.8]}
    (run / "unseen_30s.json").write_text(json.dumps({"unseen": unseen_30}))
    measures = run_measures(run)
    assert measures["final_fitness"] == 1.4  # the best so far, not the last
    assert (measures["own_reached_15"], measures["own_reached_30"]) == (0.0, 1.0)
    assert measures["unseen_reached_30"] == 0.25
    assert measures["unseen_missed_30"] == 0.75
    assert (measures["unseen_failed_30"], measures["unseen_walks_30"]) == (1.0, 4.0)
