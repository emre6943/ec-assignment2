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
from simulate import Score, SimConfig
from unseen import load_sim_config, summarise

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


def test_unseen_summary_statistics() -> None:
    config = SimConfig()
    scores = [Score(1.0, 1.0, 0.0, 0.0, 10.0), Score(2.0, 0.0, 1.0, 0.0, 10.0)]
    result = summarise(scores, config)
    assert result["distance_mean"] == pytest.approx(1.5)
    assert result["distance_std"] == pytest.approx(0.5)
    assert result["fitness"] == pytest.approx([1.0, 2.0 + config.ground_contact_weight])
