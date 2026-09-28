# Evolutionary Computing 2026 - Assignment 2

Brain evolution (neuroevolution) for X_400111 at the VU. Team of 4, worth 10 points.
**Deadline Tuesday 13 October 2026, 09:00 CEST.** Each day late costs 0,5 points.

Assignment 1 evolved a body. This one takes a fixed body and evolves its controller.

## The task

Build an EA on top of `ariel.ec` that evolves the **weights of a neural-network
controller** so a fixed body from the 'John Set' walks from its spawn position to a
fixed target position.

Fitness is the Euclidean distance in the flat ground plane between the core body's
final position and the target. Lower is better. On top of that we have to
investigate **one aspect of the EA** - mutation, crossover, parent selection,
survivor selection, population size or genotype representation - as an explicit
research question, the same shape as Assignment 1.

Fixed choices, each fixed for the whole assignment:

- **Body**: free choice from `ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set`.
  Available: `baby_a`, `baby_b`, `gecko`, `linkin_modified`, `snake`, `turtle`,
  `iguana`, `spider_8`, `spider_12`, `spider_16`, `centipede_3`, `centipede_4`,
  `centipede_5`.
- **World**: free choice from `ariel.simulation.environments`, **except
  `SimpleTiltedWorld`**, which is not supported - John Set bodies slide off the
  platform under a neutral gait. `SimpleFlatWorld`, `RuggedTerrainWorld`,
  `CraterTerrainWorld`, `AmphitheatreTerrainWorld` and `OlympicArena` are all valid.
- **Controller**: a neural network whose weights we evolve. **CPG-based controllers
  are not supported for this task.**

`ariel.simulation.tasks.targeted_locomotion` holds ready-made fitness variants if we
want to build on one: plain remaining distance, distance reduced from the starting
position, distance plus a control-effort penalty, one that penalises falling over,
one that penalises a wandering path, and one that rewards reaching the target
quickly. We may argue for our own metric as long as we define it clearly.

## Experimental requirements

- At least **5 independent runs** of the final experiments, reported as mean and
  spread.
- A **baseline** at the same evaluation budget - random search, or a fixed
  non-evolved controller.
- A line plot across generations showing avg/std of fitness over the independent
  runs.
- Stopping criterion should be the fitness curve plateauing, not a fixed generation
  count.
- Seed parametrised (CLI argument or env var) so the >=5-seed repeat is one script
  invocation per configuration.

## Hard rules

- **No changes to anything in `ariel/src/ariel`.** The course calls that fraud and
  it fails the assignment.
- `ariel.ec` utilities are allowed as building blocks, but the search itself must be
  our own design and implementation. **No off-the-shelf black-box optimizers**
  (nevergrad, CMA-ES libraries).

## Hand-in

A single `groupnumber.zip` containing a folder of the same name, with the report as
`groupnumber.pdf` and the code. Report: max 6 pages excluding cover page and
bibliography, **GECCO19 template**. Sections: Introduction (with a clear research
question), Methods, Results and discussion, Conclusions, Literature list. Cover page
carries the course name, the task number and name, the team number/name, all members'
names and student IDs, and the date.

Grading is on the report; the code is checked for correctness and coherence with the
report, but pure performance is not graded. Assignment 1 and Assignment 2 are 50%
each of the practical grade.

## Files

    A2_template_2026.py    course template: spawns a robot, drives it with a
                           RANDOM-weight network, reports the distance to target.
                           There is deliberately no evolution in it.
    Assignment2.pdf        the spec, as published on Canvas 2026-09-21

The template comes from the ARIEL fork at
github.com/AndrzejSzczepura/EvolutionaryComputing2026. Our own work goes in new files
alongside it. The only edit made to it so far is line 71, `MODE = "simple"` instead of
`"launcher"`, so it runs headless instead of opening a viewer window.

⚠️ The template's `build_robot()` imports `gecko` from
`ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko`, **not** from `john_set`.
The spec requires a John Set body, so that import has to change before any real run -
`john_set` has its own `gecko()`, among twelve others.

## Setup

Same as Assignment 1. This repo holds only our own code and has no environment of its
own - every command runs against the ARIEL clone, which must sit **next to** this
directory and must be named `ariel`:

    EvolutionaryComputing/
    ├── ariel/         # the course fork, cloned as "ariel"
    └── assignment2/   # this repo

Then, from inside this directory:

    uv run --project ../ariel python A2_template_2026.py

## Code

Every design choice, with its alternatives and rationale, is in
[`docs/decisions.md`](docs/decisions.md). Read that first.

| File | What it holds |
|---|---|
| `network.py` | The controller network and how a flat genotype maps onto its weights |
| `sensors.py` | The 34 network inputs: hinge angles, clock, target direction, tilt, 10 vision rays |
| `terrain.py` | A spawn height that clears the rugged ground (the template's spawn buries the robot) |
| `simulate.py` | One fitness evaluation: build the world, walk, measure the distance |
| `operators.py` | Neuron-level crossover and tournament selection |
| `migration.py` | Emigrant selection (best / worst / random / none) and the ring migration |
| `ea.py` | The island-model EA as `ariel.ec` operations |
| `run.py` | Command line: one condition, one or more seeds |
| `plot.py` | Fitness and diversity curves of one run (`run.png` in the run folder) |
| `replay.py` | Watch a run's best network walk: video, or a live viewer window |
| `compare.py` | Overlay several runs on the plain distance, by evaluations and by wall-clock time |
| `tests/` | Unit tests for all of the above, plus a tiny end-to-end run |

Run one condition for several seeds (each 3,000-evaluation run takes ~4.5 min on 10 cores):

    uv run --project ../ariel python run.py --policy best --seeds 0 1 2 3 4
    uv run --project ../ariel python run.py --algorithm random_search --seeds 0 1 2 3 4

Then look at what happened:

    uv run --project ../ariel python plot.py results/best/seed0     # -> results/best/seed0/run.png
    uv run --project ../ariel python replay.py results/best/seed0   # -> a video in the same folder
    uv run --project ../ariel python replay.py results/best/seed0 --viewer   # live window

Every setting is a flag (`python run.py --help`). `--world flat` runs on ARIEL's flat
world; it is for debugging only, since the experiment world is rugged. `--curriculum`,
`--early-stop` and `--stop-at-target` switch on the options of decision D16. A 10-second smoke test:

    uv run --project ../ariel python run.py --policy best --seeds 0 \
        --max-evaluations 150 --island-size 6 --n-elites 1 --n-migrants 1 \
        --migration-interval 2 --n-terrains 1 --duration 3 --out results/smoke

Tests:

    uv run --project ../ariel python -m pytest tests

## Measured baseline

A single evaluation with random weights, gecko body, `SimpleFlatWorld`,
`SIM_DURATION = 15.0`, headless:

    controller inputs (len(data.qpos)) : 15
    controller outputs (model.nu)      : 8
    genotype length (total weights)    : 138
    fitness                            : 1.9911   (target is 2.0 away, so it moved ~1cm)

Wall clock was ~1.5s for the whole invocation including imports and two model
compiles, so one headless evaluation is well under a second on this machine. That
still means a 50x100 run is a few thousand evaluations - budget the full experiment
grid before committing to it, as the spec's tips warn.
