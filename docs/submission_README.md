# Team 92: Evolutionary Computing, Assignment 2 (neuroevolution)

The report is `92.pdf`. This folder holds our own code, the scripts of every
experiment the report draws on, and the result files its numbers and figures
come from. ARIEL itself is not included; the code uses it unmodified.

**Research question.** How does the emigrant-selection policy of an island-model
EA (migrating the best, the worst or random individuals, or no migration) affect
the convergence speed and the final quality of evolved locomotion controllers?
The robot is ARIEL's `spider_8` on the OlympicArena; the controller is a 16-8-4-8
network whose 212 weights are evolved.

## Setup

The code runs in ARIEL's own environment. Put this folder next to a clone of the
course's ARIEL fork named `ariel`, at the commit the report used:

    some-folder/
    |-- ariel/   <- git clone https://github.com/AndrzejSzczepura/EvolutionaryComputing2026.git ariel
    `-- 92/      <- this folder

    cd ariel && git checkout a435b19 && uv sync && cd ../92

(`uv` is at https://docs.astral.sh/uv/; `uv sync` installs Python 3.12 and
MuJoCo.) Every command below runs from inside `92/`:

    uv run --project ../ariel python -m pytest -q tests     # 206 tests, ~15 s

On a Linux machine without a display, set `MUJOCO_GL=egl` for `filmstrip.py`.

## Reproduce the report's numbers and figures (no evolution, under a minute)

The shipped results are every run's log and summary files (no databases), so all
statistics and figures can be recomputed from them; only `filmstrip.py` walks a
robot again:

    # Table 3, the tests, diversity
    uv run --project ../ariel python analyze.py \
        results/final/{best,worst,random,none,standard,random_search} \
        --reference none --out results/final/analysis
    # the Bayesian probabilities
    uv run --project ../ariel python probabilities.py \
        results/final/{best,worst,random,none,standard,random_search} \
        --out results/final/probabilities.md
    # Section 4.3, Table 4
    uv run --project ../ariel python end_results.py \
        results/final/{best,worst,random,none,standard} \
        --out results/final/end_results.md
    # arrivals within 15, 20, 30 and 60 s
    uv run --project ../ariel python longer_walks.py \
        results/final/{best,worst,random,none,standard,random_search} \
        --out results/final/longer_walks
    uv run --project ../ariel python paper_numbers.py   # the few remaining numbers
    uv run --project ../ariel python paper_figures.py   # Figures 2-4 -> report/figures/
    uv run --project ../ariel python filmstrip.py       # Figure 1 (ARM machines only)

Where each number in the report comes from:

| In the report | File |
|---|---|
| Table 3, Friedman, paired t / Wilcoxon tests, effect sizes, normality checks, diversity (`final_spread`), medians to 1.6 | `results/final/analysis/summary.md`, `stats.md`, `paired_tests.csv` |
| Posterior probabilities ("P(faster than no migration)", 99%, 98%, 76%, ...) | `results/final/probabilities.md` |
| Table 4, Cochran's Q, McNemar, per-seed ranks, falls | `results/final/end_results.md` |
| 19 / 39 / 40 arrivals in 15 / 20 / 60 s | `results/final/longer_walks.md` |
| Gait terms' sum (0.18-0.20) | printed by `paper_figures.py` (column `gait`) |
| 4.3% shared champion, runs still improving, the standard EA's early lag (14 of 20 seeds, 50 of 80 islands, generation 40), 5 of 2,000 unseen arrivals in 15 s, power, 37% / 14% of variance, 0.94 m | printed by `paper_numbers.py` |
| Preliminary study (Figure 2; 0.957; 42-58%, 93-99%, 96-100%, 81%) | `results/olympic/analysis*/`, `results/olympic/probabilities*.md` |
| Table 2 (tuning) | `results/tuning/analysis*/summary.md` |
| Compute paragraph | `docs/compute.md` (written by `compute_ledger.py` from the full `results/` of the repository; run in this folder it would count only the shipped runs) |
| Section 2's pilot measurements (0.84 m, 1.6 cm, 22 J, 5 N m, 25°, ...) | the design log `docs/decisions.md` in the public repository (the pilots' raw results are not shipped) |

## Re-run the evolution

One run of the final experiment (migrating the best, seed 20; about 7-8 minutes
on a laptop) with exactly the flags of `experiments/99_final_experiment.sh`, on
the seed's saved arena and next to the shipped results rather than over them:

    mkdir -p results/rerun && cp -r results/final/terrains results/rerun/
    uv run --project ../ariel python run.py --policy best --seeds 20 \
        --world olympic --body spider_8 --duration 15 --no-evolve-tempo \
        --no-vision --clock-boost 1 --hidden-layers 8,4 \
        --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04 \
        --work-imbalance-weight 0.5 --leg-imbalance-weight 0 \
        --speed-weight 0.5 --stop-at-target \
        --crossover-probability 0.9 --ariel-spawn --max-evaluations 12000 \
        --migration-interval 20 --out results/rerun/best
    diff <(cut -d, -f1-19 results/rerun/best/seed20/log.csv) \
         <(cut -d, -f1-19 results/final/best/seed20/log.csv) && echo identical

On an ARM machine the new `log.csv` equals the shipped one in every column but
the wall-clock time. For another condition, use `--policy worst`, `random` or
`none`, `--standard`, or `--algorithm random_search`.

The whole experiment, all six conditions of every seed followed by the unseen
tests and the analysis, is `OUT=results/rerun bash experiments/99_final_experiment.sh`
(seeds as arguments run a subset; the analysis needs at least two). It took 8.4
hours split over two machines. Do not run it into `results/final` itself: the
finished runs are skipped, but the unseen tests are walked again and would
overwrite the shipped ones.

**Determinism.** Every walk is deterministic on a given kind of processor, but
MuJoCo's floating-point results differ slightly between x86-64 and ARM, and the
differences grow over a walk. Seeds 10-19 ran on an Intel i7-12700H (x86-64) and
seeds 20-29 on an Apple M3 Pro (ARM), so each half repeats exactly only on its
own kind of machine. ARIEL draws the OlympicArena's bumps without a seed, so
every arena is saved and shipped: `results/final/terrains/` (training, one per
seed), `results/terrains/olympic/test/spider_8/` (the 20 unseen test arenas),
and the preliminary study's and tuning's arenas.

## What is here

    92.pdf                 the report
    run.py                 command line for one run; `--help` groups the options
                           the final experiment uses apart from earlier ones
    ea.py                  the island-model EA, built on ariel.ec
    migration.py           emigrant selection (best / worst / random / none), ring
    operators.py           tournament selection, neuron-level crossover
    genome.py network.py   the 212-weight genome and the 16-8-4-8 tanh network
    sensors.py             the 16 inputs (joint angles, clock, target, tilt)
    simulate.py            one walk and the fitness (the report's Equation 1)
    terrain.py bodies.py   saved arenas; the body's legs
    unseen.py              walks each run's best controller on the 20 unseen arenas
    longer_walks.py        walks it on its own arena for 15, 20, 30 and 60 s
    analyze.py probabilities.py end_results.py paper_numbers.py   statistics
    paper_figures.py filmstrip.py rq_figure.py                    figures
    replay.py              watch or record a run's best controller
    plot.py                a run's curves (called by the pilot scripts)
    compute_ledger.py      the compute accounting behind docs/compute.md
    tests/                 unit tests (pytest)
    experiments/           one script per experiment, see below
    results/               the shipped result files and arenas
    docs/compute.md        compute per phase of the study

The code's defaults are the settings of our earliest experiments; the report's
settings are the flags in `experiments/99_final_experiment.sh`
(`tests/test_final_config.py` checks them against the report). Comments refer to
design decisions as D1-D26; the full design log is `docs/decisions.md` in
https://github.com/emre6943/ec-assignment2, which also holds the experiments
the report does not use.

## Experiments

| Script | What the report uses it for |
|---|---|
| `99_final_experiment.sh` | the final experiment: six conditions x 20 seeds (Sections 4.2-4.3, Tables 3-4, Figures 1, 3, 4) |
| `14_main_olympic.sh` | the preliminary study's six conditions on seeds 0-4, migrating every 10 generations |
| `26_migration_interval.sh` | the preliminary study's migration intervals (Figure 2) |
| `27_statistics.sh` | the preliminary study's probabilities |
| `19_tuning.sh`, `19b_tuning_more.sh` | Table 2: mutation step, tournament size, population, elites, sparse mutation, crossover probability |
| `22_stagnation_ablation.sh` | Table 2: the mutation-step rule (dropped) |
| `23_vision_ablation.sh` | Table 2: 5, 3, 1 or no distance rays |
| `24_brain_shape.sh` | Table 2: six network shapes |
| `25_crossover_operators.sh` | Table 2: four crossover operators |
| `07_pilot.sh`, `11_body_pilot.sh`, `13_walking_pilot.sh`, `15_olympic_long_run.sh`, `16_gait_pilot.sh`, `17_gait_pilot_2.sh`, `18_inputs_pilot.sh`, `21_robustness_pilot.sh` | the pilots behind the choice of body, world, mutation step, inputs, fitness terms, walk length and budget (Section 2); their raw results are not shipped |
