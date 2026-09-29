# Evolutionary Computing 2026 - Assignment 2

Brain evolution (neuroevolution) for X_400111 at the VU. Team of 4, worth 10 points.
**Deadline Tuesday 13 October 2026, 09:00 CEST.** Each day late costs 0,5 points.

Assignment 1 evolved a body. This one takes a fixed body and evolves the weights of its
neural-network controller, so that it walks towards a target. Our research question
compares **island-model migration policies**.

**The final setup (2026-09-29): `spider_8` on `OlympicArena`**, and those are now the
code's defaults. Experiments 1-12 used `spider_16` on `RuggedTerrainWorld`: spider_16 is
too weak to lift its own body, and rugged terrain stopped every body we tried (decisions
D1, D2, D17). Their scripts pass their old settings explicitly, so they still reproduce.
The research-question experiment on the final setup is `experiments/14_main_olympic.sh`.

**New here? Read in this order:**

1. This README: setup, how to run things, where everything is.
2. [`docs/decisions.md`](docs/decisions.md): every design choice, the alternatives and why
   we chose what we did. This is the raw material for the report's Methods section.
3. [`experiments/README.md`](experiments/README.md): every experiment we ran, what we
   learned, and the script to reproduce it.

## Quick start

**1. Setup.** This repo has no Python environment of its own. Every command runs against
the course's ARIEL fork, which must sit **next to** this folder and be named `ariel`:

    EvolutionaryComputing/
    ├── ariel/         # git clone https://github.com/AndrzejSzczepura/EvolutionaryComputing2026 ariel
    └── assignment2/   # this repo

Then run everything from inside `assignment2/` as `uv run --project ../ariel python ...`.

**2. Check it works** (about 30 seconds):

    uv run --project ../ariel python -m pytest tests -q

**3. A tiny evolution run** (about 15 seconds; small population, short walks):

    uv run --project ../ariel python run.py --policy best --seeds 0 \
        --max-evaluations 150 --island-size 6 --n-elites 1 --n-migrants 1 \
        --migration-interval 2 --duration 3 --out results/smoke

**4. Look at it:**

    uv run --project ../ariel python plot.py results/smoke/seed0     # -> results/smoke/seed0/run.png
    uv run --project ../ariel python replay.py results/smoke/seed0   # -> a video in that folder
    uv run --project ../ariel python replay.py results/smoke/seed0 --viewer   # live 3D window

## What the code does

**The robot:** a body from ARIEL's John Set (`--body`): `spider_8` (4 legs × 2 hinges =
8 motors, the default) or e.g. `spider_16` (4 legs × 4 hinges). It spawns in an ARIEL world
(`--world`, default `olympic` = `OlympicArena`: a flat start, then a gently rugged strip
where the target lies) and has 15 seconds to walk to a target 2 m away.

**The brain:** a feed-forward neural network. Numbers below are for spider_16, then
spider_8:

- **34 / 26 inputs:**
  - one joint angle per hinge (16 / 8);
  - a clock (sin/cos at 1 Hz, or at an evolved tempo with `--evolve-tempo`);
  - where the target is relative to the robot's heading (3 values);
  - how the body is tilted (3 values);
  - 10 "vision" rays that measure the distance to the ground around it.
- **One hidden layer of 16 neurons.**
- **16 / 8 outputs:** one target angle per hinge.
- The genotype is simply all the weights in one list: 832 / 568 of them, plus one tempo
  gene with `--evolve-tempo`.

**Rhythm options** (decision D17, off by default): `--clock-boost 3` makes new random
networks respond 3× more strongly to the clock, so they start with a clear rhythm; the
final experiments use it. `--evolve-tempo` adds a gene that sets the clock's frequency
(0.25-4 Hz); it is **not** used, because an evolved non-weight rhythm parameter comes too
close to the spec's "no CPG" rule (D17).

**The fitness** (lower is better):

    distance to the target at the end
      + 0.5 × share of the walk with the body on the ground
      + 1.0 × share of the walk upside down

Standing still scores 2.0.

**The EA:** 4 islands of 20 networks each, in a ring. Every generation, on each island:

1. The 2 best networks survive as they are (elitism).
2. The rest are children: two parents are picked by tournament (size 3). Half of the time
   they are combined by neuron-level crossover, which copies each hidden neuron whole from
   one parent. The child is then mutated: every weight gets Gaussian noise, σ = 0.05.
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
| `operators.py` | Neuron-level crossover and tournament selection (mutation is ARIEL's own) |
| `network.py` | The neural network, and how a flat genotype maps onto its weights |
| `sensors.py` | The network inputs, including the 10 vision rays |
| `simulate.py` | One evaluation: walk the terrain, measure, compute the fitness |
| `terrain.py` | What counts as ground in each world, and a spawn height that clears it (the template's spawn buries the robot) |
| `plot.py` | Curves of one run: fitness, per-island best, posture, genetic diversity |
| `replay.py` | Watch a run's best network walk, as a video or in a live viewer |
| `compare.py` | Overlay a few runs on the plain distance, by evaluations and by wall-clock time |
| `unseen.py` | Test each run's best network on 20 terrains it never saw (robustness) |
| `analyze.py` | **The report's numbers:** mean ± std curves per condition, summary table, statistical tests |
| `experiments/` | One script per experiment we ran, plus the log of what each one showed |
| `docs/decisions.md` | Every design decision, with its reasoning |
| `tests/` | Unit tests for all of the above, plus tiny end-to-end runs |
| `A2_template_2026.py` | The course template, **unchanged** (except line 71: headless mode). Not used by our code |
| `Assignment2.pdf` | The spec |

## Running experiments

    uv run --project ../ariel python run.py --policy best --seeds 0 1 2 3 4 \
        --clock-boost 3 --out results/mine/best
    uv run --project ../ariel python run.py --algorithm random_search --seeds 0 1 2 3 4 \
        --clock-boost 3 --out results/mine/random_search
    uv run --project ../ariel python run.py --help      # every setting is a flag

The defaults are the final body, world and episode length (spider_8, OlympicArena, 15 s);
`--clock-boost 3` completes the final setup. Always pass `--out`: without it a run writes
to `results/<condition>/`, where experiment 8's results live.

- **Output:** each run writes to `<out>/seed<S>/`:
  - `config.json`: every setting of the run;
  - `log.csv`: per generation, per island and overall;
  - `database.db`: ARIEL's record of every individual;
  - `best_genotype.npy`: the best network, saved whenever it improves;
  - `summary.json`.
- **Cost:** a default run (12,000 evaluations) takes about 20 minutes on a 10-core Mac.
  `run.py` uses 10 worker processes; on a smaller machine pass `--workers 4` (or your
  core count). Run experiments **one after another**: two at once just makes each slower.
- **The full experiment** for the report is `experiments/14_main_olympic.sh`: 6
  conditions (4 migration policies, the standard EA, random search) × 5 seeds, about 10
  hours, written to `results/olympic/`. It passes `--skip-done`, which skips any run
  already finished with exactly the same settings, so an interrupted run can be resumed.
  (`08_main_experiment.sh` is the same experiment on the first setup, spider_16 on rugged.)
- **Useful flags:**
  - `--body spider_16`: another John Set body;
  - `--world rugged|flat|olympic|amphitheatre|crater`: another ARIEL world;
  - `--clock-boost 3`: the rhythmic start used in the final setup (D17);
    `--evolve-tempo` exists but is not used (D17);
  - `--standard`: one population of 80 instead of 4 islands of 20;
  - `--skip-done`: skip seeds that are already finished with the same settings;
  - `--duration 10`: shorter walks (seconds; 15 by default).
- **Terrain files are per machine.** `results/` is not in git, so the same seed gives a
  *different* terrain on your laptop than on someone else's. For the final results, one
  person runs everything on one machine, or shares their `results/terrains/` folder.
- **Options we tried and rejected** stay available for experimenting:
  - `--curriculum`, `--early-stop` and `--stop-at-target` (decision D16);
  - `--terrain-mode per_generation` (D10).

  See `experiments/README.md` for what they did.

## Looking at results

    uv run --project ../ariel python plot.py results/olympic/best/seed0          # one run's curves
    uv run --project ../ariel python replay.py results/olympic/best/seed0        # its best walk (video)
    uv run --project ../ariel python replay.py results/olympic/best/seed0 --viewer  # live 3D window
    uv run --project ../ariel python compare.py results/olympic/best/seed0 results/olympic/none/seed0
    uv run --project ../ariel python unseen.py results/best/seed*   # robustness test (rugged runs only)
    uv run --project ../ariel python analyze.py results/olympic/{best,worst,random,none,standard,random_search} \
        --threshold 0.8 --reference none --out results/olympic/analysis         # report figure + stats

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
