"""paper_figures.py helpers on hand-made data, where the right answers are known."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from paper_figures import (
    champion,
    common_generations,
    generation_size,
    main,
    missing_inputs,
    seeds_per_condition,
    shared_champion_share,
    weighted_terms,
)
from rq_figure import CONDITIONS
from simulate import Score, SimConfig, fitness

ISLANDS = {"n_islands": 4, "island_size": 20, "n_elites": 2}
STANDARD = {"n_islands": 1, "island_size": 80, "n_elites": 8}
SCORE = Score(
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
SIM = SimConfig(
    ground_contact_weight=1.0,
    low_body_weight=1.0,
    work_imbalance_weight=0.5,
    speed_weight=0.5,
)


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
    terms = weighted_terms(SCORE, SIM)
    assert sum(terms.values()) == pytest.approx(fitness(SCORE, SIM))
    assert terms["mean_distance"] == pytest.approx(0.5)
    assert terms["leg_imbalance"] == 0.0  # weight 0 by default


def write_final_run(run: Path, distance: float, reached: float) -> None:
    """A finished run with every file the figures read, its champion at `distance`."""
    score = replace(SCORE, distance=distance)
    best = fitness(score, SIM)
    run.mkdir(parents=True)
    ea = {**ISLANDS, "max_evaluations": 296}
    (run / "config.json").write_text(json.dumps({"ea": ea, "sim": asdict(SIM)}))
    (run / "log.csv").write_text(
        "generation,evaluations,island,best,best_final,best_distance\n"
        + "".join(
            f"{g},{80 + 72 * g},all,{best + 3 - g},{best + 3 - g},{distance + 3 - g}\n"
            for g in range(4)
        )
    )
    (run / "summary.json").write_text(json.dumps({"best_fitness_seen": best}))
    walk = {"fitness": [best], "scores": [asdict(score)]}
    unseen = {"distance_mean": distance + 0.5, "reached": reached}
    (run / "unseen.json").write_text(json.dumps({"training": walk, "unseen": unseen}))
    (run / "unseen_30s.json").write_text(
        json.dumps({"unseen": {"reached": 2 * reached}})
    )
    arrived = {"arrived_at": 14.0 if distance < 0.2 else None}
    (run / "longer_walks.json").write_text(json.dumps({"15": arrived, "30": arrived}))


def test_the_figures_come_from_the_folders_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    final, out = tmp_path / "final", tmp_path / "figures"
    for i, condition in enumerate(CONDITIONS):
        for seed in range(3):
            run = final / condition / f"seed{seed}"
            write_final_run(run, 0.1 + 0.1 * i + 0.03 * seed, reached=0.05 * seed)
    arguments = ["--final", str(final), "--olympic", str(tmp_path / "olympic")]
    monkeypatch.setattr(
        sys, "argv", ["paper_figures.py", *arguments, "--out", str(out)]
    )
    main()
    drawn = ("convergence", "final_spread", "probabilities", "fitness_terms")
    expected = {
        f"{name}.{kind}" for name in (*drawn, "longer_walks") for kind in ("pdf", "png")
    }
    assert {path.name for path in out.iterdir()} == expected
    # Experiment 26 is not there: only the interval figure is skipped.
    printed = capsys.readouterr().out
    assert "interval.pdf skipped: no seed*/log.csv in" in printed
    assert "WARNING" not in printed  # every run used its whole budget


def test_a_figure_is_skipped_while_its_inputs_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    for seed in range(2):
        write_final_run(tmp_path / "best" / f"seed{seed}", 0.3, reached=0.0)
    (tmp_path / "best" / "seed1" / "unseen.json").unlink()
    assert not missing_inputs("convergence", [tmp_path / "best"])
    assert missing_inputs("final_spread", [tmp_path / "best"], ("unseen.json",))
    assert missing_inputs("interval", [tmp_path / "best", tmp_path / "none"])
    printed = capsys.readouterr().out
    assert "final_spread.pdf skipped: no unseen.json in 1 of 2 run folders" in printed
    assert f"interval.pdf skipped: no seed*/log.csv in {tmp_path / 'none'}" in printed


def test_unequal_seed_counts_are_shown_as_a_range(tmp_path: Path) -> None:
    for seed in range(3):
        write_run(tmp_path / "a" / f"seed{seed}", ISLANDS, 1)
    for seed in range(2):
        write_run(tmp_path / "b" / f"seed{seed}", ISLANDS, 1)
    assert seeds_per_condition([tmp_path / "a"]) == "3"
    assert seeds_per_condition([tmp_path / "a", tmp_path / "b"]) == "2-3"


def test_the_champion_is_its_walk_on_its_own_arena(tmp_path: Path) -> None:
    write_final_run(tmp_path / "seed0", 0.3, reached=0.0)
    best, score = champion(tmp_path / "seed0")
    assert score == replace(SCORE, distance=0.3)
    assert best == pytest.approx(fitness(score, SIM))


def test_a_walk_that_did_not_score_the_runs_best_is_refused(tmp_path: Path) -> None:
    run = tmp_path / "seed0"
    write_final_run(run, 0.3, reached=0.0)
    best = json.loads((run / "summary.json").read_text())["best_fitness_seen"]
    # After a resume the run's best can be the log's, rounded to 4 decimals.
    (run / "summary.json").write_text(json.dumps({"best_fitness_seen": round(best, 4)}))
    champion(run)
    (run / "summary.json").write_text(json.dumps({"best_fitness_seen": best - 0.001}))
    with pytest.raises(ValueError, match="unseen.json"):
        champion(run)
