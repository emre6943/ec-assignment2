"""analyze.py and unseen.py on hand-made data, where the right answers are known."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import shapiro, ttest_rel

import analyze
from analyze import (
    T_TEST,
    WILCOXON,
    bell_shaped,
    complete_blocks,
    friedman,
    holm,
    normality_checks,
    pair_differences,
    paired_comparisons,
    planned_comparisons,
    rank_biserial,
    run_metrics,
    signed_rank_p,
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


def make_seeds(effects: dict[str, float], seeds: int = 20) -> pd.DataFrame:
    """`seeds` seeds per condition, each condition `effect` from the seed's base.

    The base varies far more between seeds (some arenas are harder) than the
    effects do, plus a little noise per run: only pairing by seed sees them.
    """
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(seeds):
        base = rng.uniform(1.0, 2.0)
        for condition, effect in effects.items():
            value = base + effect + rng.normal(0.0, 0.02)
            rows.append(
                {
                    "condition": condition,
                    "run": f"seed{seed}",
                    "final_fitness": value,
                    "auc": value,
                }
            )
    return pd.DataFrame(rows)


def test_a_clearly_better_condition_is_significant_after_holm() -> None:
    table = make_seeds({"best": -0.1, "none": 0.0, "worst": 0.0})
    tests = {(t.a, t.b): t for t in paired_comparisons(table, "final_fitness")}
    assert list(tests) == [("best", "none"), ("best", "worst"), ("none", "worst")]
    best = tests[("best", "none")]
    assert best.pairs == 20 and best.p_holm < 0.05
    assert best.median_difference == pytest.approx(-0.1, abs=0.02)
    assert best.rank_biserial == -1.0  # lower on every seed
    assert tests[("none", "worst")].p_holm > 0.05  # noise only


def test_identical_conditions_give_p_one() -> None:
    table = make_seeds({"a": 0.0})
    table = pd.concat([table, table.assign(condition="b")])
    (test,) = paired_comparisons(table, "auc")
    assert (test.p, test.p_holm) == (1.0, 1.0)
    assert (test.median_difference, test.rank_biserial) == (0.0, 0.0)


def test_pairing_is_by_seed_not_by_row_order() -> None:
    table = make_seeds({"a": -0.1, "b": 0.0})
    shuffled = table.sample(frac=1.0, random_state=1)
    assert list(shuffled["run"]) != list(table["run"])
    assert paired_comparisons(shuffled, "auc") == paired_comparisons(table, "auc")
    assert paired_comparisons(shuffled, "auc")[0].rank_biserial == -1.0


def test_holm_adjusted_p_values_are_monotone_and_at_most_one() -> None:
    effects = {"best": -0.05, "random": -0.01, "none": 0.0, "standard": -0.08}
    table = make_seeds({**effects, "random_search": 0.5})
    worst = table[table["condition"] == "none"].assign(condition="worst")
    table = pd.concat([table, worst])  # identical to none: p = 1, kept at 1
    tests = sorted(paired_comparisons(table, "final_fitness"), key=lambda t: t.p)
    assert len(tests) == 15  # every pair of six conditions, once
    adjusted = [t.p_holm for t in tests]
    assert adjusted == sorted(adjusted)
    assert all(t.p <= t.p_holm <= 1.0 for t in tests)
    assert adjusted[-1] == 1.0 and adjusted[0] < 0.05


def test_incomplete_blocks_are_dropped() -> None:
    table = make_seeds({"a": -0.1, "b": 0.0, "c": 0.0})
    table = table[~((table["condition"] == "b") & (table["run"] == "seed3"))].copy()
    missing = (table["condition"] == "c") & (table["run"] == "seed7")
    table.loc[missing, "final_fitness"] = np.nan  # e.g. an unfinished test
    assert {t.pairs for t in paired_comparisons(table, "final_fitness")} == {18}
    assert {t.pairs for t in paired_comparisons(table, "auc")} == {19}
    report = stats_report(table, "b")
    assert "final_fitness: 18 seeds" in report and "seed3, seed7" in report


def test_five_seeds_can_never_be_significant() -> None:
    """Experiment 14: all 5 seeds on one side is the most extreme outcome."""
    tests = paired_comparisons(make_table(effect=0.3), "final_fitness")
    assert min(t.p for t in tests) == pytest.approx(0.0625)
    assert all(t.p_holm >= 0.05 for t in tests)
    report = stats_report(make_table(effect=0.3), "none")
    assert "= 0.0625" in report and "no pair can be significant" in report


def test_rank_biserial_splits_a_tie_between_both_sides() -> None:
    assert rank_biserial(np.array([-1.0, -2.0, -3.0])) == -1.0
    # ranks by size: 0 -> 1, 1 -> 2, -2 -> 3; (2 - 3 + 0) / (1 + 2 + 3)
    assert rank_biserial(np.array([1.0, -2.0, 0.0])) == pytest.approx(-1 / 6)


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


def test_signed_rank_p_is_exact_with_and_without_ties() -> None:
    """Without ties it equals scipy's exact test; with ties and zeros, scipy's
    exact permutation test (too slow for 20 seeds, fine for 8)."""
    from scipy.stats import PermutationMethod, wilcoxon

    clean = np.random.default_rng(1).normal(-0.1, 0.1, 12)
    assert signed_rank_p(clean) == pytest.approx(wilcoxon(clean, method="exact").pvalue)
    tied = np.array([-0.3, -0.1, 0.1, -0.2, -0.2, 0.0, -0.4, 0.05])
    exact = PermutationMethod(n_resamples=np.inf)
    assert signed_rank_p(tied) == pytest.approx(
        wilcoxon(tied, zero_method="zsplit", method=exact).pvalue
    )
    # 20 seeds all favouring A, two of them tied: still the extreme 2 / 2^20.
    one_sided = -np.arange(1, 21) / 100
    one_sided[1] = one_sided[0]
    assert signed_rank_p(one_sided) == pytest.approx(2 / 2**20)
    assert signed_rank_p(np.zeros(5)) == 1.0


def test_a_condition_without_the_metric_is_left_out() -> None:
    """No unseen test yet for one condition: the others are still compared."""
    table = make_seeds({"a": -0.1, "b": 0.0, "c": 0.0})
    table["unseen_distance"] = table["final_fitness"]
    table.loc[table["condition"] == "c", "unseen_distance"] = np.nan
    (test,) = paired_comparisons(table, "unseen_distance")
    assert (test.a, test.b, test.pairs) == ("a", "b", 20)
    assert "Left out, no values for this metric: c." in stats_report(table, "b")


def test_bell_shaped_differences_get_the_paired_t_test() -> None:
    """D25: normal noise only, so no pair rejects normality and every pair gets
    the paired t-test, whose numbers are scipy's ttest_rel on A and B."""
    table = make_seeds({"best": -0.1, "none": 0.0, "worst": 0.0})
    checks = normality_checks(pair_differences(table, "final_fitness"))
    assert len(checks) == 3 and bell_shaped(checks)
    blocks = complete_blocks(table, "final_fitness")
    tests = paired_comparisons(table, "final_fitness")
    for test in tests:
        assert test.test == T_TEST
        scipy_test = ttest_rel(blocks[test.a], blocks[test.b])
        interval = scipy_test.confidence_interval(confidence_level=0.95)
        assert test.p == pytest.approx(scipy_test.pvalue, rel=1e-6)
        assert (test.ci_low, test.ci_high) == pytest.approx(
            (interval.low, interval.high), rel=1e-6
        )
        differences = blocks[test.a] - blocks[test.b]
        assert test.mean_difference == pytest.approx(differences.mean())
        assert test.cohen_dz == pytest.approx(differences.mean() / differences.std())
        assert test.p_other_test == pytest.approx(
            signed_rank_p(np.round(differences.to_numpy(), 9))
        )
    best = tests[0]
    assert (best.a, best.b) == ("best", "none") and best.p_holm < 0.05
    assert best.cohen_dz < -2  # 0.1 better against a noise sd of about 0.03


def test_one_outlying_seed_switches_every_pair_to_wilcoxon() -> None:
    """A hard arena can give one seed a far larger difference than the rest:
    the pairs with that seed fail the check, and then all pairs, including the
    one that looks normal, get the Wilcoxon test."""
    table = make_seeds({"a": 0.0, "b": 0.0, "c": 0.0})
    outlier = (table["condition"] == "a") & (table["run"] == "seed0")
    table.loc[outlier, "final_fitness"] += 1.0
    differences = pair_differences(table, "final_fitness")
    checks = {(c.a, c.b): c for c in normality_checks(differences)}
    assert checks[("a", "b")].p_holm < 0.05 and checks[("a", "c")].p_holm < 0.05
    assert checks[("b", "c")].p_holm >= 0.05
    blocks = complete_blocks(table, "final_fitness")
    for test in paired_comparisons(table, "final_fitness"):
        assert test.test == WILCOXON
        assert test.p == signed_rank_p(differences[(test.a, test.b)])
        t_p = ttest_rel(blocks[test.a], blocks[test.b]).pvalue
        assert test.p_other_test == pytest.approx(t_p, rel=1e-6)
    report = stats_report(table, "a")
    assert "normality is rejected for a vs b, a vs c" in report
    assert "| A vs B | median A − B | r | p | p (Holm) |" in report


def test_skewed_differences_get_wilcoxon() -> None:
    """Exponential differences: a few seeds gain a lot, most a little."""
    table = make_seeds({"a": 0.0, "b": 0.0})
    a_rows = table["condition"] == "a"
    table.loc[a_rows, "auc"] += np.random.default_rng(3).exponential(0.2, 20)
    assert not bell_shaped(normality_checks(pair_differences(table, "auc")))
    (test,) = paired_comparisons(table, "auc")
    assert test.test == WILCOXON


def test_a_pair_with_no_spread_is_left_out_of_the_decision() -> None:
    """b is a + 0.1 on every seed and d is a copy of a: their pairs have no
    shape to check. The decision rests on the pairs with c, and the t-test
    still gives the no-spread pairs a sensible result, without warnings."""
    table = make_seeds({"a": -0.1, "c": 0.0})
    a_rows = table[table["condition"] == "a"]
    shifted = a_rows.assign(condition="b", final_fitness=a_rows["final_fitness"] + 0.1)
    table = pd.concat([table, shifted, a_rows.assign(condition="d")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checks = normality_checks(pair_differences(table, "final_fitness"))
        tests = {(t.a, t.b): t for t in paired_comparisons(table, "final_fitness")}
        report = stats_report(table, "c")
    unchecked = {(c.a, c.b) for c in checks if c.unchecked}
    assert unchecked == {("a", "b"), ("a", "d"), ("b", "d")}
    assert all(np.isnan(c.p_holm) for c in checks if c.unchecked)
    assert bell_shaped(checks)  # the three pairs with c are normal noise
    assert {t.test for t in tests.values()} == {T_TEST}
    assert (tests[("a", "d")].p, tests[("a", "d")].cohen_dz) == (1.0, 0.0)
    assert tests[("a", "b")].p == 0.0 and tests[("a", "b")].cohen_dz == -np.inf
    assert tests[("a", "b")].ci_low == pytest.approx(-0.1)
    assert "| a vs b | – | – | the same difference on every seed |" in report
    assert "could not be checked take no part in the decision" in report
    # Fewer than 3 seeds: nothing can be checked, so the Wilcoxon test stays.
    (two_seeds,) = normality_checks({("a", "b"): np.array([0.1, 0.3])})
    assert two_seeds.unchecked == "needs ≥ 3 seeds"
    assert not bell_shaped([two_seeds])
    only_ties = make_seeds({"a": 0.0})
    only_ties = pd.concat([only_ties, only_ties.assign(condition="b")])
    assert "no pair could be checked" in stats_report(only_ties, "a")


def test_the_shapiro_p_values_are_holm_corrected_over_the_checked_pairs() -> None:
    """[0, 0, 0, 2, 5] alone would reject normality (p = 0.033), but not after
    Holm over the 4 pairs that can be checked; the 5th, with no spread, does
    not count in the correction."""
    differences = {
        ("a", "b"): np.array([0.0, 0.0, 0.0, 2.0, 5.0]),
        ("a", "c"): np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ("a", "d"): np.array([-1.0, 0.5, 0.0, 1.0, -0.4]),
        ("b", "c"): np.array([2.0, 2.5, 1.5, 2.2, 1.9]),
        ("b", "d"): np.full(5, 0.3),
    }
    checks = normality_checks(differences)
    raw = [shapiro(d).pvalue for pair, d in differences.items() if pair != ("b", "d")]
    assert [c.p for c in checks[:4]] == pytest.approx(raw)
    assert [c.p_holm for c in checks[:4]] == pytest.approx(holm(raw))
    assert checks[0].p < 0.05 and checks[0].p_holm == pytest.approx(4 * checks[0].p)
    assert checks[4].unchecked and bell_shaped(checks)


def write_runs(folder: Path, finals: list[float]) -> None:
    """Fake finished runs of one condition: two generations each, ending at
    `finals[seed]`, logged as `run.py` logs them."""
    for seed, final in enumerate(finals):
        run = folder / f"seed{seed}"
        run.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps({"ea": {"max_evaluations": 160}}))
        (run / "log.csv").write_text(
            "generation,evaluations,island,best,best_distance\n"
            f"0,80,all,3.0,1.0\n1,160,all,{final},0.5\n"
        )


def test_experiment_14_sized_runs_give_a_test_column_and_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """analyze.py end to end on 6 conditions x 5 seeds, the size of experiment
    14: paired_tests.csv names the test used and keeps the other test's p, and
    there is one Q-Q figure per tested metric (no unseen test here)."""
    effects = {"best": -0.1, "worst": 0.0, "random": -0.05, "none": 0.0}
    table = make_seeds({**effects, "standard": -0.15, "random_search": 1.0}, seeds=5)
    folders = []
    for condition, group in table.groupby("condition", sort=False):
        write_runs(tmp_path / condition, group["final_fitness"].round(4).tolist())
        folders.append(str(tmp_path / condition))
    out = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", ["analyze.py", *folders, "--out", str(out)])
    analyze.main()
    paired = pd.read_csv(out / "paired_tests.csv")
    assert len(paired) == 2 * 15  # final fitness and AUC, 15 pairs each
    assert set(paired["test"]) <= {T_TEST, WILCOXON}
    assert paired.groupby("metric")["test"].nunique().eq(1).all()  # one per metric
    assert paired["p_other_test"].between(0.0, 1.0).all()
    stats = (out / "stats.md").read_text()
    assert stats.count("**Decision:**") == 2 and "5 seeds in every condition" in stats
    for metric in ("final_fitness", "auc"):
        assert (out / f"normality_{metric}.png").stat().st_size > 0
    assert not (out / "normality_unseen_distance.png").exists()
