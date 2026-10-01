"""analyze.py and unseen.py on hand-made data, where the right answers are known."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyze import (
    friedman,
    holm,
    planned_comparisons,
    run_metrics,
    stats_report,
    summary_markdown,
)
from compute_ledger import EXPERIMENTS, run_tally
from longer_walks import arrival
from probabilities import a12, mean_interval, paired_posterior, pair_rows
from rq_figure import first_below
from simulate import Score, SimConfig
from unseen import load_sim_config, result_file_name, summarise

CURVE = pd.DataFrame(
    {
        "evaluations": [80, 160, 240, 320],
        "fitness": [1.9, 1.8, 1.5, 1.4],
        "distance": [1.9, 1.7, 1.5, 1.3],
    }
)


def test_holm_matches_hand_computation() -> None:
    # sorted: 0.01*3 = 0.03, 0.02*2 = 0.04, 0.04*1 = 0.04 (kept monotone)
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_run_metrics_reads_within_the_budget() -> None:
    grid = np.linspace(0, 240, 4)  # 0, 80, 160, 240: the run went on to 320
    metrics = run_metrics(CURVE, grid, threshold=1.6)
    assert metrics["final_fitness"] == pytest.approx(1.5)  # not 1.4, beyond budget
    assert metrics["final_distance"] == pytest.approx(1.5)
    assert metrics["evals_to_threshold"] == 240
    # before the first log (0) the first value counts: (1.9 + 1.9 + 1.8 + 1.5) / 4
    assert metrics["auc"] == pytest.approx(1.775)


def test_threshold_never_reached_is_nan() -> None:
    metrics = run_metrics(CURVE, np.linspace(0, 320, 5), threshold=1.0)
    assert np.isnan(metrics["evals_to_threshold"])


def make_table(effect: float) -> pd.DataFrame:
    """Five seeds x three conditions; `best` is `effect` lower on every seed."""
    rows = []
    for seed in range(5):
        for condition, shift in (("best", -effect), ("none", 0.0), ("worst", 0.01)):
            value = 1.8 + 0.01 * seed + shift
            rows.append(
                {
                    "condition": condition,
                    "run": f"seed{seed}",
                    "final_fitness": value,
                    "auc": value,
                }
            )
    return pd.DataFrame(rows)


def test_clear_differences_can_be_significant() -> None:
    table = make_table(effect=0.3)
    statistic, p, blocks = friedman(table, "final_fitness")
    assert blocks == 5
    assert p < 0.05
    comparisons = dict(
        (c, p_holm) for c, _, p_holm in planned_comparisons(table, "auc", "none")
    )
    assert comparisons["best"] < 0.05  # the whole point of planned comparisons


def test_friedman_needs_three_conditions() -> None:
    table = make_table(effect=0.3)
    assert friedman(table[table["condition"] != "worst"], "auc") is None


def test_reports_are_written_as_markdown() -> None:
    table = make_table(effect=0.3)
    report = stats_report(table, "none")
    assert "Friedman" in report and "best vs none" in report
    assert "No condition named" in stats_report(table, "missing")
    summary = summary_markdown(table.assign(evals_to_threshold=np.nan), 1.6, 12000)
    assert "not reached (0/5)" in summary


def test_unseen_skips_runs_it_cannot_read(tmp_path: Path) -> None:
    assert load_sim_config(tmp_path) is None  # no config.json at all
    (tmp_path / "config.json").write_text(json.dumps({"sim": {"n_hidden": 8}}))
    assert load_sim_config(tmp_path) is None  # an older, incompatible version
    (tmp_path / "config.json").write_text(json.dumps({"sim": {"duration": 3.0}}))
    # A config that does not record its body is from experiments 1-12: spider_16.
    assert load_sim_config(tmp_path) == SimConfig(duration=3.0, body="spider_16")
    # A log with a header only (a run stopped before its first generation).
    (tmp_path / "log.csv").write_text("generation,evaluations,duration\n")
    assert load_sim_config(tmp_path) == SimConfig(duration=3.0, body="spider_16")


def test_unseen_summary_statistics() -> None:
    config = SimConfig()
    scores = [Score(1.0, 1.0, 0.0, 0.0, 10.0), Score(2.0, 0.0, 1.0, 0.0, 10.0)]
    result = summarise(scores, config)
    assert result["distance_mean"] == pytest.approx(1.5)
    assert result["distance_std"] == pytest.approx(0.5)
    assert result["fitness"] == pytest.approx([1.0, 2.0 + config.ground_contact_weight])
    assert result["reached"] == 0.0
    arrived = Score(0.05, 1.95, 0.0, 0.0, 8.0)  # stop_at_target ended the walk
    assert summarise([*scores, arrived], config)["reached"] == pytest.approx(1 / 3)


def test_a_special_unseen_test_gets_its_own_file() -> None:
    """Only the standard test is `unseen.json`, the file analyze.py reads (D23)."""
    assert result_file_name([0.0], None) == "unseen.json"
    assert result_file_name([0.0, 30.0, -30.0], 20.0) == "unseen_yaws0_30_-30_20s.json"
    assert result_file_name([0.0], 20.0) == "unseen_20s.json"


def test_the_ledger_counts_walks_per_generation(tmp_path: Path) -> None:
    """3 terrains per evaluation; walks that get longer halfway (D21) are counted right."""
    config = {"ea": {"n_terrains": 3}, "sim": {"duration": 15.0}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "log.csv").write_text(
        "generation,evaluations,island,duration,seconds\n"
        "0,80,all,15,10\n0,80,0,15,10\n1,152,all,30,20\n"
    )
    tally = run_tally(tmp_path)
    assert (tally.runs, tally.unfinished, tally.evaluations) == (1, 1, 152)
    assert tally.walks == 152 * 3
    assert tally.simulated_s == 80 * 3 * 15 + 72 * 3 * 30
    assert tally.wall_s == 20


def test_paired_posterior_is_a_student_t() -> None:
    """Probabilities that one condition beats another (probabilities.py)."""
    even = paired_posterior(np.array([-0.2, 0.2, -0.1, 0.1]), rope=0.0)
    assert even["p_a_better_at_all"] == pytest.approx(0.5)
    clear = paired_posterior(np.array([-0.30, -0.25, -0.35, -0.28, -0.32]), rope=0.05)
    assert clear["p_a_better"] > 0.99 and clear["high"] < 0.0
    noisy = paired_posterior(np.array([-0.4, 0.3, -0.2, 0.1, -0.3]), rope=0.05)
    assert 0.3 < noisy["p_a_better"] < 0.8
    total = noisy["p_a_better"] + noisy["p_equal"] + noisy["p_b_better"]
    assert total == pytest.approx(1.0)


def test_a12_counts_wins_and_ties() -> None:
    assert a12(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
    assert a12(np.array([1.0, 3.0]), np.array([2.0, 3.0])) == pytest.approx(0.625)


def test_every_results_folder_belongs_to_one_experiment_only() -> None:
    folders = [folder for _, _, group in EXPERIMENTS for folder in group]
    assert len(folders) == len(set(folders))


def test_an_identical_difference_on_every_seed_is_certain_or_a_tie() -> None:
    tie = paired_posterior(np.zeros(4), rope=0.05)
    assert tie["p_a_better_at_all"] == 0.5 and tie["p_equal"] == 1.0
    ahead = paired_posterior(np.full(4, -0.2), rope=0.05)
    assert ahead["p_a_better_at_all"] == 1.0 and ahead["p_a_better"] == 1.0


def test_pairs_are_matched_by_seed_and_skip_missing_values() -> None:
    table = pd.DataFrame(
        {
            "condition": ["a", "a", "a", "b", "b", "b"],
            "seed": ["seed0", "seed1", "seed2", "seed2", "seed1", "seed0"],
            "final_fitness": [1.0, 2.0, 3.0, 3.5, 2.5, 1.5],
            "unseen_distance": [1.0, 1.0, np.nan, 2.0, 2.0, 2.0],
        }
    )
    final = pair_rows(table, "final_fitness", rope=0.0)
    assert [(row["a"], row["b"]) for row in final] == [("a", "b"), ("b", "a")]
    assert final[0]["mean_difference"] == pytest.approx(-0.5)  # every seed -0.5
    assert final[0]["seeds_a_won"] == 3 and final[0]["p_a_better_at_all"] == 1.0
    unseen = pair_rows(table, "unseen_distance", rope=0.0)
    assert unseen[0]["seeds"] == 2  # seed2 has no unseen test for "a"
    assert len(pair_rows(table.iloc[[0, 3]], "final_fitness", rope=0.0)) == 0


def test_mean_interval_is_a_t_interval() -> None:
    mean, sd, low, high = mean_interval(np.array([1.0, 2.0, 3.0]))
    assert (mean, sd) == (2.0, 1.0)
    assert (low, high) == pytest.approx((2.0 - 2.4841, 2.0 + 2.4841), abs=1e-4)
    single = mean_interval(np.array([1.5]))
    assert single[:2] == (1.5, 0.0) and np.isnan(single[2]) and np.isnan(single[3])


def test_first_below_only_counts_the_common_budget() -> None:
    assert first_below(CURVE, "fitness", 1.6, budget=320) == 240
    assert np.isnan(first_below(CURVE, "fitness", 1.6, budget=160))
    assert np.isnan(first_below(CURVE, "fitness", 1.0, budget=320))


def test_a_walk_arrived_only_inside_the_target_circle() -> None:
    """longer_walks.py: stop_at_target ends the walk on arrival."""
    arrived = Score(0.095, 1.9, 0.0, 0.0, 16.2)
    short = Score(0.30, 1.7, 0.0, 0.0, 30.0)
    assert arrival(arrived) == pytest.approx(16.2)
    assert arrival(short) is None


def test_the_ledger_counts_longer_walks(tmp_path: Path) -> None:
    config = {"ea": {}, "sim": {"duration": 15.0}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "log.csv").write_text(
        "generation,evaluations,island,seconds\n0,80,all,5\n"
    )
    walks = {"15": {"seconds": 15.0}, "30": {"seconds": 17.5}}
    (tmp_path / "longer_walks.json").write_text(json.dumps(walks))
    tally = run_tally(tmp_path)
    assert (tally.test_walks, tally.test_simulated_s) == (2, 32.5)
