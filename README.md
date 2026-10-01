# Evolutionary Computing 2026 - Assignment 2

Brain evolution (neuroevolution) for X_400111 at the VU. Team of 4, worth 10 points.
**Deadline Tuesday 13 October 2026, 09:00 CEST.** Each day late costs 0,5 points.

Assignment 1 evolved a body. This one takes a fixed body and evolves the weights of its
neural-network controller, so that it walks towards a target. Our research question
compares **island-model migration policies**.

**Status (2026-10-01): every experiment is done, and the paper draft is in
[`report/`](report/).** The final setup is `spider_8` on `OlympicArena` (the code's
defaults). The research-question experiment is `experiments/14_main_olympic.sh`; its
follow-ups are 26 (how often to migrate), 27 (statistics), 28 (longer test walks) and 29
(the absolute position as two extra inputs: no gain, not adopted).

**The answer in one paragraph:** migrating (any policy) makes the islands converge faster
than no migration (89-96% probability), and migrating the best is the fastest. But after
12,000 evaluations the policies end at about the same fitness and do equally well on
unseen arenas, and one standard population (no islands) ended best, although that lead
did not survive adding two inputs to the brain (experiment 29). How *often* the
islands migrate matters more than *whom* they send: every 20 generations was best. Only 3
of 30 best brains reach the target in the 15 s they trained for; 11 do given 20 s.

**New here? Read in this order:**

1. This README: setup, how to run things, where everything is.
2. [`report/main.pdf`](report/main.pdf): the paper draft, the short version of everything.
3. [`docs/decisions.md`](docs/decisions.md): every design choice, the alternatives and why
   we chose what we did. This is the raw material for the report's Methods section.
4. [`experiments/README.md`](experiments/README.md): every experiment we ran, what we
   learned, and the script to reproduce it.

Experiments 1-12 used `spider_16` on `RuggedTerrainWorld`: spider_16 is too weak to lift
its own body, and rugged terrain stopped every body we tried (decisions D1, D2, D17).
Their scripts pass their old settings explicitly, so they still reproduce.

## Quick start

**1. Install `uv`** (the Python tool every command uses), once:

    curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
    # Windows (PowerShell): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

The commands below are for a macOS or Linux terminal. On Windows, use WSL or Git Bash:
the experiment scripts are bash scripts.

**2. Get the code.** This repo has no Python environment of its own. Every command runs
against the course's ARIEL fork, which must sit **next to** this folder and be named
`ariel`:

    mkdir EvolutionaryComputing && cd EvolutionaryComputing
    git clone https://github.com/AndrzejSzczepura/EvolutionaryComputing2026 ariel
    git clone https://github.com/emre6943/ec-assignment2 assignment2
    cd assignment2

    EvolutionaryComputing/
    ├── ariel/         # the course's fork; never edit it
    └── assignment2/   # this repo

Then run everything from inside `assignment2/` as `uv run --project ../ariel python ...`.
The first command installs ARIEL's dependencies (Python 3.12+, MuJoCo), which takes a
minute or two.

**3. Check it works** (about 30 seconds):

    uv run --project ../ariel python -m pytest tests -q

**4. A tiny evolution run** (about 15 seconds; small population, short walks):

    uv run --project ../ariel python run.py --policy best --seeds 0 \
        --max-evaluations 150 --island-size 6 --n-elites 1 --n-migrants 1 \
        --migration-interval 2 --duration 3 --out results/smoke

**5. Look at it:**

    uv run --project ../ariel python plot.py results/smoke/seed0     # -> results/smoke/seed0/run.png
    uv run --project ../ariel python replay.py results/smoke/seed0   # -> a video in that folder
    uv run --project ../ariel python replay.py results/smoke/seed0 --viewer   # live 3D window

## Recipes: the final experiment, videos, figures and the paper

Copy and paste these from inside `assignment2/`. Everything writes under `results/`,
which is not in git (see "Using someone else's results" below).

**Run one condition of the final experiment for one seed** (about 10 minutes). This is
experiment 14's exact setup; change `--policy` to `worst`, `random` or `none`, or replace
`--policy best` with `--standard` or `--algorithm random_search`:

    uv run --project ../ariel python run.py --skip-done --seeds 0 \
        --world olympic --body spider_8 --duration 15 --no-evolve-tempo \
        --no-vision --clock-boost 1 --hidden-layers 8,4 \
        --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04 \
        --work-imbalance-weight 0.5 --leg-imbalance-weight 0 \
        --speed-weight 0.5 --stop-at-target \
        --crossover-probability 0.9 --ariel-spawn --max-evaluations 12000 \
        --policy best --out results/olympic/best

**Run the whole final experiment and its follow-ups** (one after another; about 4 + 2.5
+ 3.5 hours of evolution on a 10-core Mac, plus the tests). Each script skips runs that are
already finished, so an interrupted one can simply be started again:

    bash experiments/14_main_olympic.sh          # the research question: 6 conditions x 5 seeds
    bash experiments/14_main_olympic.sh 0 1      # ...or only seeds 0 and 1
    bash experiments/26_migration_interval.sh    # migrate every 5 / 20 / 50 generations
    bash experiments/27_statistics.sh            # probabilities: results/olympic/probabilities*.md
    bash experiments/28_longer_walks.sh          # 15 / 20 / 30 / 60 s test walks
    bash experiments/29_absolute_position.sh     # 14 again, brains also told their (x, y)

**Watch a brain walk.** `replay.py` saves an `.mp4` in the run's folder and prints how far
from the target the robot ended:

    uv run --project ../ariel python replay.py results/olympic/standard/seed4               # the best brain of all
    uv run --project ../ariel python replay.py results/olympic/standard/seed4 --duration 30 # walk longer
    uv run --project ../ariel python replay.py results/olympic/standard/seed4 --new-terrain # an arena it never saw
    uv run --project ../ariel python replay.py results/olympic/standard/seed4 --viewer      # live 3D window

The best run of each condition (lowest final fitness) is `standard/seed4`, `best/seed0`,
`best_int20/seed0`, `best_int5/seed0`, `random/seed4`, `best_int50/seed0`, `worst/seed0`,
`none/seed0` and `random_search/seed3`, from best to worst.

**Make the figures and tables:**

    uv run --project ../ariel python paper_figures.py     # the paper's figures -> report/figures/
    uv run --project ../ariel python rq_figure.py         # one-page answer -> results/olympic/rq_figure.png
    uv run --project ../ariel python rq_figure.py --metric distance

More analysis commands are under "Looking at results" below.

**Build the paper** (`report/`, the course's GECCO'19 LaTeX template):

- **Overleaf** (easiest): zip `main.tex`, `references.bib`, `acmart.cls`,
  `ACM-Reference-Format.bst` and the `figures/` folder from `report/`, then in Overleaf
  choose *New Project → Upload Project* and pick the zip.
- **Locally:** install [Tectonic](https://tectonic-typesetting.github.io) (`brew install
  tectonic` on a Mac), then `cd report && tectonic -X compile main.tex`.

The limit is 6 pages without the cover page and the bibliography. `\todo{...}` marks what
still has to be filled in.

**Using someone else's results.** `results/` is not in git, and each seed's arena is
random per machine, so you cannot regenerate the exact same results elsewhere. To work
with the team's numbers, copy the whole `results/olympic/` folder from whoever ran it
(Emre) into your `results/`, terrains included. Without the `database.db` files it is
about 50 MB, and everything still works except the fitness-terms figure of
`paper_figures.py`.

## What the code does

**The robot:** a body from ARIEL's John Set (`--body`): `spider_8` (4 legs × 2 hinges =
8 motors, the default) or e.g. `spider_16` (4 legs × 4 hinges). It spawns in an ARIEL world
(`--world`, default `olympic` = `OlympicArena`: a flat start, then a gently rugged strip
where the target lies) and has 15 seconds to walk to a target 2 m away.

**The brain** (final setup): a feed-forward neural network, 16 → 8 → 4 → 8 with `tanh`.

- **16 inputs:** the 8 joint angles; a clock (sin and cos of a 1 Hz beat); where the
  target is relative to the robot's heading (distance, sin and cos of its direction); how
  the body is tilted (3 values). The robot's absolute (x, y) is not an input: the target
  direction already carries it, and adding it (`--position`, experiment 29) did not help.
- **8 outputs:** one target angle per hinge.
- The genotype is simply all 212 weights (biases included) in one list.

The code's defaults are older: 10 "vision" rays to the ground and one hidden layer of 16
(34 inputs and 832 weights for spider_16). The final setup switches them off with
`--no-vision --hidden-layers 8,4`, because the rays sensed almost nothing on OlympicArena
and the 8-4 bottleneck learned best (decisions D5, D6).

**Rhythm options** (decision D17, off by default): `--clock-boost 3` makes new random
networks respond 3× more strongly to the clock, so they start with a clear rhythm; it was
used until experiment 18, and the final setup keeps the default of 1. `--evolve-tempo`
adds a gene that sets the clock's frequency (0.25-4 Hz); it is **not** used, because an
evolved non-weight rhythm parameter comes too close to the spec's "no CPG" rule (D17).

**The fitness** of the final setup (lower is better):

    distance to the target at the end
      + 0.5 × the distance averaged over the whole walk     (be fast; 0 after arriving)
      + 1.0 × share of the walk with the body on the ground (stand)
      + 1.0 × share of the walk upside down                 (stay upright)
      + 1.0 × how far the body sits below 4 cm, on average  (carry the body)
      + 0.5 × how far the laziest leg's motor work falls short of the average (use all legs)

Holding still scores 5.0; arriving at once, standing tall, scores 0. The code's defaults
only have the first, third and fourth terms (with 0.5 for the ground); the others are
flags (decisions D15, D18, D20). Walks end as soon as the robot is within 0.1 m of the
target.

**The EA:** 4 islands of 20 networks each, in a ring. Every generation, on each island:

1. The 2 best networks survive as they are (elitism).
2. The rest are children: two parents are picked by tournament (size 3). With probability
   0.9 (the code's default is 0.5; tuning chose 0.9, D22) they are combined by neuron-level crossover, which
   copies each hidden neuron whole from one parent. The child is then mutated: every
   weight gets Gaussian noise, σ = 0.05.
3. Every 10 generations, each island copies 2 individuals to the next island, where they
   replace its 2 worst. **Which 2 are sent is the research question:** the `best`, the
   `worst`, `random` ones, or `none` (no migration, the control).

`random_search` is the baseline: the same loop, but children are random networks.
`run.py --standard` runs the same EA as **one population** (1 × 80, 8 elites, no
migration), a standard EA to compare the island model against (experiment 9).

**The terrain:** OlympicArena's rugged strip and RuggedTerrainWorld are random on every
build. So each seed's world is built once, on first use, and saved next to the condition
folders: `<results>/terrains/<world>/<body>/seed<S>/` (spider_16 keeps
`<results>/terrains/<world>/seed<S>/`), e.g. `results/olympic/terrains/...` for experiment
14. Every condition with that seed walks exactly the same ground, which makes the
comparison between policies fair.

## Code map

| File | What it holds |
|---|---|
| `run.py` | **Start here.** Command line: one condition, one or more seeds (`--standard` for no islands) |
| `bodies.py` | The John Set bodies we can use (`--body`), and their hinge count and leg reach |
| `genome.py` | The genotype: the network's weights plus the optional tempo gene; the rhythm options |
| `ea.py` | The island-model EA as `ariel.ec` operations: reproduce, evaluate, migrate, log |
| `migration.py` | Emigrant selection (best / worst / random / none) and the ring migration |
| `operators.py` | Neuron-level crossover (and experiment 25's alternatives) and tournament selection (mutation is ARIEL's own) |
| `network.py` | The neural network, and how a flat genotype maps onto its weights |
| `sensors.py` | The network inputs, including the 10 vision rays and the optional absolute (x, y) |
| `simulate.py` | One evaluation: walk the terrain, measure, compute the fitness |
| `terrain.py` | What counts as ground in each world, and a spawn height that clears it (the template's spawn buries the robot) |
| `plot.py` | Curves of one run: fitness, per-island best, posture, genetic diversity |
| `replay.py` | Watch a run's best network walk, as a video or in a live viewer |
| `compare.py` | Overlay a few runs on the plain distance, by evaluations and by wall-clock time |
| `unseen.py` | Test each run's best network on 20 terrains it never saw (robustness) |
| `analyze.py` | **The report's numbers:** mean ± std curves per condition, summary table, statistical tests |
| `probabilities.py` | How sure we are: for every pair of conditions, the probability that one is better (Bayesian paired t-test), A12 and seeds won |
| `rq_figure.py` | The research question's answer in one four-panel figure (experiment 14) |
| `longer_walks.py` | Walk the best brains for longer than they trained (15/20/30/60 s; step 28) |
| `paper_figures.py` | The paper's figures, as PDFs, into `report/figures/` |
| `compute_ledger.py` | Counts the compute every experiment used and writes `docs/compute.md` |
| `report/` | The paper: `main.tex`, `references.bib`, the course's ACM template, `figures/` and the built `main.pdf` |
| `experiments/` | One script per experiment we ran, plus the log of what each one showed |
| `docs/decisions.md` | Every design decision, with its reasoning |
| `docs/compute.md` | How many evaluations and hours every experiment used |
| `tests/` | Unit tests for all of the above, plus tiny end-to-end runs |
| `A2_template_2026.py` | The course template, **unchanged** (except line 71: headless mode). Not used by our code |
| `Assignment2.pdf` | The spec |

## Running experiments

    uv run --project ../ariel python run.py --policy best --seeds 0 1 2 3 4 \
        --out results/mine/best
    uv run --project ../ariel python run.py --algorithm random_search --seeds 0 1 2 3 4 \
        --out results/mine/random_search
    uv run --project ../ariel python run.py --help      # every setting is a flag

The defaults are the final body, world and episode length (spider_8, OlympicArena, 15 s).
The rest of the final setup (inputs, brain shape, fitness terms, crossover probability,
spawn) is a set of flags: copy them from `experiments/14_main_olympic.sh`. Always pass
`--out`: without it a run writes to `results/<condition>/`, where experiment 8's results
live.

- **Output:** each run writes to `<out>/seed<S>/`:
  - `config.json`: every setting of the run;
  - `log.csv`: per generation, per island and overall;
  - `database.db`: ARIEL's record of every individual;
  - `best_genotype.npy`: the best network, saved whenever it improves;
  - `summary.json`.
- **Cost:** a run of the final setup (12,000 evaluations) takes 8-10 minutes on a 10-core
  Mac. `run.py` uses 10 worker processes; on a smaller machine pass `--workers 4` (or
  your core count). Run experiments **one after another**: two at once just makes each
  slower.
- **The full experiment** for the report is `experiments/14_main_olympic.sh`: 6
  conditions (4 migration policies, the standard EA, random search) × 5 seeds, about 4
  hours, written to `results/olympic/`. It passes `--skip-done`, which skips any run
  already finished with exactly the same settings, so an interrupted run can be resumed.
  (`08_main_experiment.sh` is the same experiment on the first setup, spider_16 on rugged.)
- **Useful flags:**
  - `--body spider_16`: another John Set body;
  - `--world rugged|flat|olympic|amphitheatre|crater`: another ARIEL world;
  - `--clock-boost 3`: the rhythmic start used before experiment 18 (D17);
    `--evolve-tempo` exists but is not used (D17);
  - `--standard`: one population of 80 instead of 4 islands of 20;
  - `--skip-done`: skip seeds that are already finished with the same settings;
  - `--duration 10`: shorter walks (seconds; 15 by default).
- **Terrain files are per machine.** `results/` is not in git, so the same seed gives a
  *different* terrain on your laptop than on someone else's. For the final results, one
  person runs everything on one machine, or shares their `results/terrains/` folder.
- **Options we tried and rejected** stay available for experimenting:
  - `--curriculum` and `--early-stop` (decision D16); `--stop-at-target` is used again,
    with the speed term (D20);
  - `--terrain-mode per_generation` (D10).

  See `experiments/README.md` for what they did.

## Looking at results

    uv run --project ../ariel python plot.py results/olympic/best/seed0          # one run's curves
    uv run --project ../ariel python replay.py results/olympic/best/seed0        # its best walk (video)
    uv run --project ../ariel python replay.py results/olympic/best/seed0 --viewer  # live 3D window
    uv run --project ../ariel python compare.py results/olympic/best/seed0 results/olympic/none/seed0
    uv run --project ../ariel python unseen.py results/olympic/best/seed*  # robustness: 20 fresh arenas
    uv run --project ../ariel python analyze.py results/olympic/{best,worst,random,none,standard,random_search} \
        --reference none --out results/olympic/analysis                         # report figure + stats
    uv run --project ../ariel python probabilities.py results/olympic/{best,worst,random,none,standard,random_search} \
        --out results/olympic/probabilities.md                                  # P(A better than B)
    uv run --project ../ariel python rq_figure.py                               # the research-question figure
    uv run --project ../ariel python longer_walks.py results/olympic/{best,worst,random,none,standard,random_search} \
        --out results/olympic/longer_walks                                      # 15-60 s test walks
    uv run --project ../ariel python compute_ledger.py                          # docs/compute.md

`analyze.py` writes to `--out` (default `results/analysis/`):

- `convergence.png`: mean ± std across seeds, the plot the spec asks for;
- `summary.csv` / `summary.md`: per run and per condition, the best fitness at the
  budget, evaluations to reach a threshold, the area under the curve and (after
  `unseen.py`) the distance on unseen terrain;
- `stats.md`: a Friedman test across all conditions (blocked by seed), and Mann-Whitney
  U tests of each condition against `--reference` (default `none`), Holm-corrected
  (decision D3 explains why). Pass `--out` to write somewhere else than `results/analysis/`.

## Rules we follow

- **Never change, re-implement or reconfigure ARIEL** (`../ariel/src/ariel`). The course
  treats changing it as fraud. We use its classes as shipped, with default settings. The
  one exception is a spawn argument; see decision D2a.
- **Never edit `A2_template_2026.py`.** Our code lives in new files.
- **No black-box optimisers** (nevergrad, CMA-ES libraries). Selection, crossover and
  migration are our own code; mutation and the population/database machinery come from
  `ariel.ec`, which the course allows.
- **Every design decision gets written down** in `docs/decisions.md`, with its reasoning;
  every experiment goes in `experiments/`.
- **Before committing:** the tests pass (`uv run --project ../ariel python -m pytest tests -q`)
  and formatting is clean (`uvx black --check --force-exclude A2_template_2026.py .`).
  Don't commit `results/`; it is gitignored.

## The assignment in short

**Task:** build an EA on top of `ariel.ec` that evolves the weights of a neural-network
controller, so that a fixed John Set body walks from its spawn point to a fixed target.
Investigate **one aspect of the EA** as a research question.

**Fixed for the whole assignment:**

- **Body:** a free choice from `ariel...prebuilt_robots.john_set`. We chose `spider_16`,
  then switched to `spider_8` (D1, D17).
- **World:** a free choice from `ariel.simulation.environments`, except
  `SimpleTiltedWorld`. We chose `RuggedTerrainWorld`, then switched to `OlympicArena`
  (D2, D17).
- **Controller:** a neural network whose weights are evolved. **No CPGs.**

**The spec's fitness** is the flat-ground distance from the core to the target, lower is
better. We may argue for our own metric if we define it clearly; ours adds two posture
terms (D15).

**Experimental requirements:**

- at least **5 independent runs** per configuration, reported as mean and spread;
- a **baseline** at the same evaluation budget (ours: random search);
- a line plot of mean/std fitness across runs;
- a seed that can be set on the command line.

The spec's tips also suggest stopping on a plateau; we explain our fixed budget in D12.

**Hard rules:**

- no changes to `ariel/src/ariel`;
- the search must be our own design;
- no off-the-shelf optimisers.

**Hand-in:**

- One `groupnumber.zip` containing a folder of the same name, with the report
  (`groupnumber.pdf`) and the code.
- The report: max 6 pages in the **GECCO19 template**, not counting the cover page and
  bibliography.
  - Sections: Introduction (with a clear research question), Methods, Results and
    discussion, Conclusions, Literature.
  - The cover page names the course, the task number and name, the team, every member's
    name and student ID, and the date.
- Grading is on the report; the code is checked for correctness and coherence with it.
  Assignment 1 and Assignment 2 each count for 50% of the practical grade.
