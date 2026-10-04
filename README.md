# Evolutionary Computing 2026 - Assignment 2

Brain evolution (neuroevolution) for X_400111 at the VU. Assignment 1 evolved a body;
this one takes a fixed body and evolves the weights of its neural-network controller, so
that it walks towards a target. Our research question compares **island-model migration
policies**: how does the emigrant-selection policy (migrating the best, the worst or
random individuals, or not migrating) affect the convergence speed and the final quality
of evolution?

**The paper**, [`report/main.pdf`](report/main.pdf), reports the final experiment,
[`experiments/99_final_experiment.sh`](experiments/99_final_experiment.sh): `spider_8`
on `OlympicArena`, six conditions × 20 seeds (10-29), migrating every 20 generations,
results in `results/final/`. Its preliminary study is experiment 14 (the same six
conditions on 5 seeds, migrating every 10 generations), 26 (how often to migrate) and 27
(their probabilities); the settings were tuned in 19, 19b and 22-25, and the pilots were
07, 11, 13, 15-18 and 21. [`experiments/README.md`](experiments/README.md) lists every
experiment.

**The answer in one paragraph:** every EA clearly beats random search (final fitness
1.13-1.24 against 2.49, lower is better; Holm-corrected paired t-tests, p < 10⁻⁹).
Between the policies the differences are small: migrating the best or random individuals
probably converges faster than no migration (posterior probabilities 99% and 98% on the
area under the convergence curve), but after correcting for the 15 pairwise comparisons
no difference between two EAs is significant in speed, final fitness or distance on 20
unseen arenas. Migrating the best made the islands as alike as a single population, at
little cost within 166 generations; migrating the worst behaved like no migration.
Random migration ended with the best mean fitness and was never the worst EA on any seed.

**Read in this order:**

1. This README: setup, how to run things, where everything is.
2. [`report/main.pdf`](report/main.pdf): the paper.
3. [`docs/decisions.md`](docs/decisions.md): every design choice, the alternatives and
   why we chose what we did.
4. [`experiments/README.md`](experiments/README.md): every experiment we ran, what it
   showed, and the script that reproduces it.

## Quick start

**1. Install `uv`** (the Python tool every command uses), once:

    curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
    # Windows (PowerShell): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

The commands below are for a macOS or Linux terminal. On Windows, use WSL or Git Bash:
the experiment scripts are bash scripts.

**2. Get the code.** This repo has no Python environment of its own. Every command runs
against the course's ARIEL fork at commit `a435b19`, which must sit **next to** this
folder and be named `ariel`:

    mkdir EvolutionaryComputing && cd EvolutionaryComputing
    git clone https://github.com/AndrzejSzczepura/EvolutionaryComputing2026 ariel
    git -C ariel checkout a435b19
    git clone https://github.com/emre6943/ec-assignment2 assignment2
    cd assignment2
    uv sync --project ../ariel        # ARIEL's dependencies: Python 3.12+, MuJoCo, ...

    EvolutionaryComputing/
    ├── ariel/         # the course's fork, unmodified
    └── assignment2/   # this repo

Then run everything from inside `assignment2/` as `uv run --project ../ariel python ...`.

**3. Check it works** (under a minute):

    uv run --project ../ariel python -m pytest -q tests

**4. A tiny evolution run** (about 15 seconds; small population, short walks):

    uv run --project ../ariel python run.py --policy best --seeds 0 \
        --max-evaluations 150 --island-size 6 --n-elites 1 --n-migrants 1 \
        --migration-interval 2 --duration 3 --out results/smoke

**5. Look at it:**

    uv run --project ../ariel python plot.py results/smoke/seed0     # -> results/smoke/seed0/run.png
    uv run --project ../ariel python replay.py results/smoke/seed0   # -> a video in that folder
    uv run --project ../ariel python replay.py results/smoke/seed0 --viewer   # live 3D window

## How to reproduce the paper

The code's defaults are the **early experiments'** settings, not the paper's. The paper's
settings are the flags in `experiments/99_final_experiment.sh`;
`tests/test_final_config.py` checks that they build the configuration the paper
describes, and `run.py --help` notes the paper's value next to each flag that differs.
Run everything from inside `assignment2/`. Results go to `results/`, which is not in git;
the paper's arenas (`results/final/terrains/`, the preliminary study's
`results/olympic/terrains/` and the 20 unseen arenas in
`results/terrains/olympic/test/spider_8/`) are submitted with the report.

**The final experiment** (Sections 4.2-4.3): 120 runs of 12,032 evaluations, about 7-8
minutes each, so about 15 hours on one machine (the paper's runs were split by seed over
two machines and took 8.4 hours). It runs as many walks in parallel as the machine has
CPUs (`WORKERS` sets fewer), then tests every run's best controller on the 20 unseen
arenas at 15 and 30 s and runs `analyze.py`, `probabilities.py`, `longer_walks.py` and
`end_results.py`. It skips finished runs and resumes a cut-off one from its last saved
generation, so it can simply be started again:

    bash experiments/99_final_experiment.sh                   # 6 conditions x 20 seeds (10-29)
    WORKERS=8 bash experiments/99_final_experiment.sh 10 11   # 8 parallel walks, seeds 10 and 11 only

For the paper's unseen arenas, put them in `results/terrains/olympic/test/spider_8/`
first; otherwise `unseen.py` builds 20 new ones there.

**One run of the final experiment** (about 7-8 minutes): migrate the best, seed 25, with
exactly the script's flags, into a folder of its own. For another condition, replace
`--policy best` with `--policy worst`, `--policy random`, `--policy none`, `--standard`
or `--algorithm random_search`.

    mkdir -p results/check/terrains/olympic/spider_8
    cp -R results/final/terrains/olympic/spider_8/seed25 results/check/terrains/olympic/spider_8/
    uv run --project ../ariel python run.py --policy best --seeds 25 \
        --world olympic --body spider_8 --duration 15 --no-evolve-tempo \
        --no-vision --clock-boost 1 --hidden-layers 8,4 \
        --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04 \
        --work-imbalance-weight 0.5 --leg-imbalance-weight 0 \
        --speed-weight 0.5 --stop-at-target \
        --crossover-probability 0.9 --ariel-spawn --max-evaluations 12000 \
        --migration-interval 20 --out results/check/best
    diff <(cut -d, -f1-19 results/check/best/seed25/log.csv) \
         <(cut -d, -f1-19 results/final/best/seed25/log.csv) && echo identical

To repeat a run exactly you need its seed's saved arena: a run builds its arena on first
use in `<out>/../terrains/`, and OlympicArena places its bumps at random on every build,
so without the `cp` the run evolves on different ground. Even on the same arena, MuJoCo's
results differ slightly between processor architectures: **seeds 10-19 ran on x86-64
(an Intel i7) and repeat exactly only there; seeds 20-29 ran on ARM (an Apple M3 Pro) and
repeat exactly only there.** The `diff` compares every column of `log.csv` but the
wall-clock time.

**The preliminary study** (Section 4.1, results in `results/olympic/`): experiment 14,
then 26 (the best policy every 5, 20 and 50 generations, on the same seeds and arenas),
then 27 (no new runs: their probabilities). 14 takes about 4 hours, 26 about 2.5:

    bash experiments/14_main_olympic.sh          # 6 conditions x 5 seeds (0-4), every 10 generations
    bash experiments/26_migration_interval.sh    # -> results/olympic/analysis_interval/
    bash experiments/27_statistics.sh            # -> results/olympic/probabilities*.md

The tuning and pilot scripts are listed in `experiments/README.md`.

**The tables, numbers and figures**, from the saved runs (each takes under a minute;
script 99 already ran the first three):

    uv run --project ../ariel python analyze.py results/final/{best,worst,random,none,standard,random_search} \
        --reference none --out results/final/analysis        # Table 3, the tests, diversity
    uv run --project ../ariel python probabilities.py results/final/{best,worst,random,none,standard,random_search} \
        --out results/final/probabilities.md                  # the posterior probabilities (Table 3's last column)
    uv run --project ../ariel python end_results.py results/final/{best,worst,random,none,standard} \
        --out results/final/end_results.md                    # Table 4, Section 4.3
    uv run --project ../ariel python paper_numbers.py         # the few numbers no other script prints
    uv run --project ../ariel python paper_figures.py         # Figures 2-4 -> report/figures/*.pdf
    uv run --project ../ariel python filmstrip.py             # Figure 1 -> report/figures/walk_filmstrip.png
    uv run --project ../ariel python compute_ledger.py        # the compute totals -> docs/compute.md

- `analyze.py`'s `summary.md` has Table 3's columns: final fitness, closest distance,
  evaluations to a fitness of 1.6 (mean, runs that reached it, and the median over all
  runs), AUC, `final_spread` (the genotype diversity at the end) and the unseen distance.
  `stats.md` and `paired_tests.csv` hold the Friedman and the pairwise tests.
- `longer_walks.py` (run by script 99, into `results/final/longer_walks.md`) gives the
  arrivals within 15, 20, 30 and 60 s.
- `paper_figures.py` draws Figure 2 (`interval.pdf`) from experiment 26 in
  `results/olympic/`, and Figures 3 and 4 (`convergence.pdf`, `final_spread.pdf`) from
  `results/final/`. It also draws `fitness_terms.pdf`, which is not in the paper, and
  prints the fitness terms' means that the Limitations quote.
- `filmstrip.py` walks the standard EA's seed-25 controller on its own arena. As that
  seed ran on ARM, it reproduces the published figure only on an ARM machine; it checks
  its walk against the recorded one and warns if they differ.

**Build the paper** (`report/`, the course's GECCO'19 LaTeX template): install
[Tectonic](https://tectonic-typesetting.github.io) (`brew install tectonic` on a Mac), then

    cd report && tectonic -X compile main.tex

**Build the hand-in** (git-ignored): `bash make_submission.sh` copies the compiled
paper as `92.pdf`, the code, the experiment scripts the paper uses, the arenas and the
result files its numbers come from into `submission/92/`, with
`docs/submission_README.md` as its README. Zip it with
`cd submission && zip -r -X 92.zip 92`.

## What the code does

**The robot:** `spider_8` from ARIEL's John Set (`--body`): a core with 4 legs × 2 hinges,
8 position-controlled motors. It spawns in an ARIEL world (`--world`, default `olympic` =
`OlympicArena`: a flat start, then from x = 0.5 m a strip of centimetre-high bumps on which
the target lies at (2, 0) m) and has 15 seconds to walk to the target. A walk ends early
once the core is within 0.1 m of it.

**The brain:** a feed-forward neural network, 16 → 8 → 4 → 8, `tanh`, queried at 50 Hz.

- **16 inputs:** the 8 hinge angles; a clock (sin and cos of a 1 Hz beat); where the target
  is relative to the robot's heading (distance, and sin and cos of its bearing); how the
  body is tilted (3 values).
- **8 outputs:** one target angle per hinge.
- **The genotype** is simply all 212 weights (biases included) in one list.

The code's defaults differ: 10 "vision" rays to the ground as extra inputs and one hidden
layer of 16, i.e. 26 inputs and 568 weights for spider_8. The paper switches them off
with `--no-vision --hidden-layers 8,4`: the rays sensed almost nothing on OlympicArena,
and 16-8-4-8 learned best (decisions D5, D6). The robot's absolute (x, y) is not an input:
adding it (`--position`, experiment 29) did not help.

**The fitness** (lower is better), Equation 1 of the paper:

    f = d_T + 0.5 d̄ + 1.0 c + 1.0 ℓ + 1.0 u + 0.5 w

    d_T  distance to the target at the end
    d̄    the distance averaged over the whole walk, 0 after arriving   (be fast)
    c    share of the walk with the core on the ground                (stand)
    ℓ    how far the core is carried below 4 cm, on average           (carry the body)
    u    share of the walk upside down                                (stay upright)
    w    how far the laziest leg's motor work falls short of the mean (use all legs)

A robot that holds still scores 5.0; one that arrives at once, standing tall and using
all legs alike, scores close to 0. The code's defaults are the early fitness,
d_T + 0.5 c + 1.0 u (decisions D15, D18, D20).

**The EA:** 4 islands of 20 networks each, in a ring. Every generation, on each island:

1. The 2 best networks survive as they are (elitism).
2. The other 18 are children: two parents are picked by tournament (size 3). With
   probability 0.9 (code default 0.5; D22) they are combined by neuron-level crossover,
   which copies each hidden neuron whole from one parent. The child is then mutated:
   every weight gets Gaussian noise, σ = 0.05.
3. Every 20 generations (code default 10; D11), each island copies 2 individuals to the
   next island, where they replace its 2 worst. **Which 2 are sent is the research
   question:** the `best`, the `worst`, `random` ones, or `none` (no migration, the
   control).

A run stops at the first generation that completes 12,000 evaluations: 80 + 166 × 72 =
12,032. `random_search` is the baseline: the same loop, but children are random networks.
`run.py --standard` runs the same EA as **one population** (1 × 80, 8 elites, no
migration), a standard EA to compare the island model against.

**The terrain:** OlympicArena's bumps are random on every build. So each seed's arena is
built once, on first use, and saved next to the condition folders:
`<results>/terrains/<world>/<body>/seed<S>/` (spider_16 keeps
`<results>/terrains/<world>/seed<S>/`), e.g.
`results/final/terrains/olympic/spider_8/seed10/terrain0_arielspawn.mjb`. Every condition
with that seed walks exactly the same ground, so the conditions are paired by seed.

## Code map

| File | What it holds |
|---|---|
| `run.py` | **Start here.** Command line: one condition, one or more seeds (`--standard` for no islands) |
| `bodies.py` | The John Set bodies we can use (`--body`), and their hinge count and leg reach |
| `genome.py` | The genotype: the network's weights plus the optional tempo gene; the rhythm options |
| `ea.py` | The island-model EA as `ariel.ec` operations: reproduce, evaluate, migrate, log |
| `migration.py` | Emigrant selection (best / worst / random / none) and the ring's migration plan (`plan_migration`) |
| `operators.py` | Neuron-level crossover (and experiment 25's alternatives) and tournament selection (mutation is ARIEL's own) |
| `network.py` | The neural network, and how a flat genotype maps onto its weights |
| `sensors.py` | The network inputs (and the core's body name), including the vision rays and the absolute (x, y) of the earlier experiments |
| `simulate.py` | One evaluation: walk the arena, measure, compute the fitness |
| `terrain.py` | What counts as ground in each world, and a spawn height that clears it |
| `plot.py` | Curves of one run: fitness, per-island best, posture, genetic diversity |
| `replay.py` | Watch a run's best network walk, as a video or in a live viewer |
| `compare.py` | Overlay a few runs on the plain distance, by evaluations and by wall-clock time |
| `unseen.py` | Test each run's best network on 20 arenas it never saw |
| `analyze.py` | **The paper's main numbers:** convergence curves, summary table, Friedman and pairwise tests |
| `probabilities.py` | For every pair of conditions, the probability that one is better (Bayesian paired t-test), A12 and seeds won |
| `longer_walks.py` | Walk the best brains for longer than they trained (15/20/30/60 s) |
| `end_results.py` | The exploratory end results (D26, Table 4): arrivals, unseen reach and fitness, falls, per-seed ranks |
| `paper_numbers.py` | The few numbers in the paper that no other script prints |
| `paper_figures.py` | Figures 2-4 of the paper, as PDFs, into `report/figures/` |
| `filmstrip.py` | Figure 1: a controller walking to the target, seen from above |
| `rq_figure.py` | Experiment 14's answer in one four-panel figure (not in the paper) |
| `compute_ledger.py` | Counts the compute every experiment used and writes `docs/compute.md` |
| `report/` | The paper: `main.tex`, `references.bib`, the course's ACM template, `figures/` and the built `main.pdf` |
| `experiments/` | One script per experiment we ran, plus the log of what each one showed |
| `docs/decisions.md` | Every design decision, with its reasoning |
| `docs/compute.md` | How many runs, evaluations and hours every experiment used |
| `tests/` | Unit tests for all of the above, plus tiny end-to-end runs |
| `A2_template_2026.py` | The course template, for reference; our code does not use it (see Rules) |
| `Assignment2.pdf` | The assignment |

## Running experiments

    uv run --project ../ariel python run.py --policy best --seeds 0 1 2 3 4 \
        --out results/mine/best
    uv run --project ../ariel python run.py --algorithm random_search --seeds 0 1 2 3 4 \
        --out results/mine/random_search
    uv run --project ../ariel python run.py --help      # every setting is a flag

The defaults are the paper's body, world and walk length (spider_8, OlympicArena, 15 s),
but the early experiments' brain, fitness and EA settings. For the paper's setup, copy the
flags of `experiments/99_final_experiment.sh` (see "One run of the final experiment"
above). Always pass `--out`: without it a run writes to `results/<condition>/`, where
experiment 8's results live. A new results folder builds new arenas.

- **Output:** each run writes to `<out>/seed<S>/`:
  - `config.json`: every setting of the run, and the path of its arena;
  - `log.csv`: per generation, per island and overall;
  - `database.db`: ARIEL's record of every individual;
  - `best_genotype.npy`: the best network, saved whenever it improves;
  - `summary.json`.
- **Cost:** a run of the paper's setup takes about 7 minutes on an Apple M3 Pro and 8.4
  minutes on an Intel i7-12700H. `run.py` uses 10 parallel walks; on a smaller machine pass
  `--workers 4` (or your core count). Run experiments one after another: two at once
  just makes each slower.
- **Re-running:** `--skip-done` skips seeds already finished with exactly the same
  settings, and `--resume` continues a cut-off run from its last saved generation.
- **Useful flags:**
  - `--body spider_16`: another John Set body;
  - `--world rugged|flat|olympic|amphitheatre|crater`: another ARIEL world;
  - `--standard`: one population of 80 instead of 4 islands of 20;
  - `--duration 10`: shorter walks (seconds; 15 by default).
- **Options of the earlier experiments** stay available; `run.py --help` lists those no
  final setting uses in a group of their own, with the experiment that used each:
  `--curriculum` and `--early-stop` (D16), `--terrain-mode per_generation` (D10),
  `--n-terrains` and `--spawn-yaws` (D23), among others. Options the final experiment
  sets to a non-default value, such as `--clock-boost` (1 there, 3 in early pilots) and
  `--evolve-tempo` (off there; rhythm options, D17), note the paper's value instead. See `experiments/README.md` for what they did.

## Looking at results

A run of seed 20-29 replays exactly on an ARM machine, a run of seed 10-19 on x86-64 (see
above); on the other kind the walk differs slightly.

    uv run --project ../ariel python plot.py results/final/standard/seed25       # its curves -> run.png
    uv run --project ../ariel python replay.py results/final/standard/seed25     # its best walk -> replay*.mp4
    uv run --project ../ariel python replay.py results/final/standard/seed25 --viewer        # live 3D window
    uv run --project ../ariel python replay.py results/final/standard/seed25 --duration 30   # walk longer
    uv run --project ../ariel python replay.py results/final/standard/seed25 --new-terrain   # a new arena
    uv run --project ../ariel python compare.py results/final/best/seed25 results/final/none/seed25
    uv run --project ../ariel python unseen.py results/check/best/seed25         # the 20 unseen arenas -> unseen.json
    uv run --project ../ariel python rq_figure.py                                # experiment 14 in one figure

`plot.py`, `replay.py` and `unseen.py` write into the run's folder, `compare.py` next to
the first run; `replay.py` also prints how far from the target the robot ended.
`analyze.py` writes to `--out` (default `results/analysis/`):

- `convergence.png`: mean ± std across seeds of the best fitness so far, the plot the
  assignment asks for;
- `summary.csv` / `summary.md`: per run and per condition, the best fitness at the
  budget, the closest distance, evaluations to a fitness of 1.6 (mean over the runs that
  reached it, and median over all runs), the area under the curve, `final_spread` (the
  genotypes' mean distance to their centroid at the end) and, after `unseen.py`, the
  distance on the unseen arenas;
- `stats.md`: a Friedman test across all conditions (blocked by seed); unpaired
  Mann-Whitney U tests against `--reference`, kept from the 5-seed experiments; and
  paired tests of every pair of conditions, also in `paired_tests.csv`: per metric, the
  paired t-test if no pair's differences reject normality (Shapiro-Wilk, Holm),
  otherwise the Wilcoxon signed-rank test, Holm-corrected across the pairs (D3, D25);
- `normality_<metric>.png`: Q-Q plots of every pair's differences, the picture behind
  that normality check.

## Rules we follow

- **ARIEL is used unmodified** (`../ariel/src/ariel`, commit `a435b19`), with its classes
  as shipped. The final experiment passes no ARIEL setting beyond what the template passes
  (`--ariel-spawn`: the template's spawn). Without that flag (the pilots and tuning runs),
  our own spawn passes `correct_collision_with_floor=False` to ARIEL's `spawn()`, because
  the template's spawn buries the robot in rugged terrain (D2a).
- **The course template is not edited beyond one line.** `A2_template_2026.py` is a copy
  of ARIEL's `assignments/assignment_2/A2_template_2026.py` that differs only in line 71
  (`MODE = "simple"`, headless, instead of `"launcher"`). Our code lives in new files and
  does not import it.
- **No black-box optimisers** (nevergrad, CMA-ES libraries). Selection, crossover and
  migration are our own code; mutation and the population/database machinery come from
  `ariel.ec`, which the course allows.
- **Every design decision is written down** in `docs/decisions.md`, with its reasoning;
  every experiment has a script in `experiments/`.

## How this meets the assignment

- **Body and world:** free choices from ARIEL's John Set and environments; we chose
  `spider_8` on `OlympicArena` after pilots with `spider_16` and `RuggedTerrainWorld`
  (D1, D2, D17).
- **Controller:** a neural network whose weights are evolved; a fixed clock input
  supplies the rhythm, so there is no CPG.
- **Fitness:** the assignment's is the distance from the core to the target; ours adds
  speed and gait terms, defined in the paper's Equation 1 (D15, D18, D20).
- **Experiments:** 20 independent runs per configuration (at least 5 required), a
  random-search baseline at the same budget, the mean ± std line plot (Figure 3a), and a
  seed set on the command line (`--seeds`). We use a fixed budget rather than stopping on
  a plateau (D12).
