# Experiment log

Every experiment we ran, in order, with the script that reproduces it and what we
learned. Each finding changed a decision in [`../docs/decisions.md`](../docs/decisions.md);
the decision numbers (Dx) point there. This log is the story behind the Methods section:
*why* the final setup looks the way it does.

Run a script from anywhere, e.g. `experiments/07_pilot.sh`. Results land in `results/`,
which is gitignored: each person's results stay on their own machine.

| # | Script | Question | Answer | Led to |
|---|---|---|---|---|
| 1 | `01_terrain_per_generation.sh` | Does the EA learn with a fresh rugged terrain every generation? | **No.** The terrain's difficulty moved every island's score up and down together; the best robot wiggled in place. | D10 |
| 2 | `02_flat_world_debug.sh` | Is the EA broken, or is it the terrain? | **The terrain.** On ARIEL's flat world the same EA went from 1.84 m to 1.02 m and walked about 1 m. | D2b |
| 3 | `03_posture_fitness.sh` | Does rewarding a raised, upright body help? | **Not on its own:** the terrain noise still dominated. | D15 |
| 4 | `04_fixed_terrain.sh` | One fixed terrain per seed: does it learn now? | **Yes, slowly:** 1.86 → 1.75 in 41 generations; robots learned to stand. | D10 |
| 5 | `05_long_run.sh` | More population and generations? | **Better:** 1.84 → 1.39 over 164 generations; flat after about generation 115. The robot shuffles rather than walks. | D12 |
| 6 | `06_curriculum_vs_plain.sh` | Curriculum + early stopping? | **No:** early lead, then stalled at about 1.55 m while the plain run reached 1.46 m; also slower. Stopped at generation 86. | D16 |
| 7 | `07_pilot.sh` | Which brain shape and mutation σ? | **One layer of 16, σ = 0.05.** σ = 0.2 was worst everywhere; a second layer did not help. | D6, D8 |
| 8 | `08_main_experiment.sh` | **The research question:** best vs worst vs random vs no migration, plus standard EA and random search | **No measurable effect of the migration policy.** All five EA variants end at 1.59–1.62 (± ~0.12); random search is clearly worse (1.73). Friedman across all six: AUC p = 0.011, final p = 0.053, driven by random search; no planned comparison is significant. Unseen terrain: ~1.97 for all. See *Details*. | D3 |
| 9 | `09_standard_vs_island.sh` | Are islands any better than one standard population of the same size? | **Not significantly** (3 seeds, Friedman p = 0.37): final fitness 1.54 for both; islands with migration far more consistent (std 0.03 vs 0.13) and ahead mid-budget; no migration slightly worse (1.58). Seeds 3–4 come with experiment 8. | D3, D11 |
| 10 | `10_overnight_best.sh` | How good a walker can the best setup get with a big budget (4 × 50, 120,000 evaluations, 15 s)? | **1.82 → 1.37** (about 0.8 m walked in 15 s), almost all of it in the first 20,000 evaluations; no improvement at all after about generation 440 of 624. Genetic diversity collapsed early. 2 h 47 min. | — |
| 11 | `11_body_pilot.sh` | Can another John Set body move better than spider_16 (whose motors cannot lift it)? | **No.** spider_16 1.50 and spider_8 1.52 lead; gecko 1.60, turtle 1.68, snake 1.72, linkin_modified 1.78 (distance, 2 seeds, 4,000 evaluations). | D1 |
| 12 | `12_crossover_or_not.sh` | Does neuron-level crossover help, or would mutation alone do as well? | **Keep it, no proof it matters:** 1.607 with crossover vs 1.645 mutation only (5 seeds); better on 3 seeds, tied on 1, worse on 1; not significant (p = 0.31). | D7 |
| 13 | `13_walking_pilot.sh` | Can spider_8 (which can lift itself) learn to *walk*, with the rhythm options, on flat / OlympicArena / rugged ground? Judged by video. | *running* - first result: on flat ground, without the rhythm options, spider_8 reached the target (fitness **0.02**) in 5 minutes. | D17 |

Numbers are *fitness* (distance to the target plus posture penalties, D15; a robot that
does not move scores 2.0) unless they say "m". Runs 1–6 used 15-second episodes and a
hidden layer of 8 with σ = 0.1. Runs 7–8 use 10-second episodes, so their numbers are not
directly comparable with 1–6.

## Details

### 1–3: why the terrain may not change every generation

With a new random terrain every generation, the same controller scores very differently
from one generation to the next (±0.14 m, measured over 10 terrains). That is more than
weak controllers move, so selection mostly rewarded terrain luck. Neither a posture-aware
fitness (3) nor more of the same helped. The flat-world check (2) proved the EA itself
learns fine without that noise.

*Reproduction note:* 1 and 3 ran before a few later changes (an earlier 5-ray vision
layout; the seeding fix from the code review). The scripts are the closest reproduction
with the current code, so their numbers will differ slightly.

### 4–6: learning on a fixed terrain

Fixing the terrain per seed made the fitness deterministic, and the curves became clean
staircases. Migration events are visible: the islands' best values snap together.

- **The long run (5)** shows where the budget matters: steady progress until about
  18,000 evaluations with 4 × 40 individuals, then flat.
- **The curriculum (6)** was the team's idea to speed this up: reward any movement early,
  then only movement toward the target. It helped only for the first ~20 generations.

### 7: the pilot

One seed and one terrain for all nine runs, 4 × 20 individuals, 4,000 evaluations, 10 s
episodes. Best fitness at the end (lower is better):

| Brain | σ 0.05 | σ 0.1 | σ 0.2 |
|---|---|---|---|
| 1 layer × 8 | 1.58 | 1.56 | 1.60 |
| 1 layer × 16 | **1.51** | 1.70 | 1.67 |
| 2 layers × 8 | 1.57 | 1.57 | 1.72 |

- **σ = 0.2** was the worst for every shape and the slowest to improve: steps that big
  destroy gaits that already work.
- **σ 0.05 and 0.1** are close.
- **A second hidden layer** brought no gain, and it makes the neuron-level crossover less
  clean (D7).
- **8 vs 16 neurons** cannot be separated on one seed: 16 had the best end result, 8
  improved fastest early on.
- **Chosen:** one layer of 16, σ = 0.05. A bigger layer costs nothing measurable, because
  simulation time is dominated by the physics.

**Robustness check** (`unseen.py`, 20 terrains no run has seen): the pilot winner reaches
1.48 m on its own training terrain, but **1.92 ± 0.11 m** on unseen terrain; the
runner-up (1 × 8, σ 0.1) gets 1.53 m and 1.99 ± 0.12 m. So the evolved brains
**specialise to the terrain they were evolved on** and barely transfer. That is the price
of fixing the terrain per seed (D10), and it belongs in the Discussion.

### 8: the main experiment (2026-09-28/29)

Six conditions × five seeds, default settings (16 hidden, σ 0.05, 4 × 20 or 1 × 80
individuals, 12,000 evaluations, 10 s episodes). From `results/analysis/`:

| condition | final fitness | AUC | reached 1.6 | unseen distance |
|---|---|---|---|---|
| best | 1.607 ± 0.133 | 1.675 ± 0.111 | 4/5 | 1.983 ± 0.022 |
| worst | 1.600 ± 0.113 | 1.654 ± 0.108 | 3/5 | 1.979 ± 0.023 |
| random | 1.600 ± 0.134 | 1.650 ± 0.119 | 4/5 | 1.982 ± 0.045 |
| none | 1.623 ± 0.124 | 1.666 ± 0.114 | 3/5 | 1.976 ± 0.031 |
| standard | 1.593 ± 0.125 | 1.677 ± 0.074 | 3/5 | 1.964 ± 0.042 |
| random search | 1.729 ± 0.072 | 1.757 ± 0.081 | 0/5 | 1.980 ± 0.049 |

- **Friedman test across all six** (blocked by seed): AUC p = 0.011, final fitness
  p = 0.053. That is a difference somewhere, and the figure shows it comes from random
  search being worse.
- **Planned comparisons against `none`:** none significant; every Holm-corrected p is at
  least 0.75. Among the EA variants the spread *between seeds* (±0.12, i.e. between
  terrains) is several times larger than the differences *between policies* (≤0.03).
- **Unseen terrain:** every condition's best network ends about 1.97 m from the target,
  hardly better than standing still. Controllers specialise to their training terrain.
- **Reading:** on this problem, *which* individuals migrate, and whether the population is
  split into islands at all, makes no measurable difference within 12,000 evaluations.
  The EA does clearly beat random search.

### 10: the overnight run (2026-09-29)

The best setup with 10× the budget: islands migrating the best, 4 × 50 individuals,
120,000 evaluations (624 generations), 15 s episodes.

- **Best fitness 1.82 → 1.37.** The robot ends 1.34 m from the target, about 0.8 m
  walked in 15 s. Nearly all of the progress came in the first 20,000 evaluations. The
  last improvement was around generation 440, then nothing for 180 generations.
- **Genetic diversity collapsed early.** The islands' genotype spread fell from about 12
  to about 3 by 15,000 evaluations (the bottom panel of `run.png`), and never recovered.
  Migrating the *best* copies each island's winner everywhere, and with a small σ the
  population ends up in one basin. That is the premature-convergence effect Cantú-Paz
  (2001) describes, and it belongs in the Discussion.
- **Conclusion:** with this setup, more compute does not buy a walker that reaches the
  target. The limit is the search (lost diversity) and the terrain, not the budget.

### 11: the body pilot (2026-09-29)

**Why:** every John Set motor is capped at 0.66 N·m. No static pose lifts spider_16's
core above its resting height; its hip motors saturate without moving. So spider_16 can
only shuffle, and the idea was that a snake, which never lifts itself, might do better.

**Setup:** six bodies, 2 seeds each, 4,000 evaluations. The "core on the ground" penalty
was off for all of them: it would punish a snake for moving like a snake.

| body | final distance | reached 1.6 |
|---|---|---|
| spider_16 | **1.496 ± 0.009** | 2/2 |
| spider_8 | 1.521 ± 0.092 | 2/2 |
| gecko | 1.597 ± 0.043 | 1/2 |
| turtle | 1.675 ± 0.158 | 1/2 |
| snake | 1.716 ± 0.167 | 0/2 |
| linkin_modified | 1.784 ± 0.120 | 0/2 |

- **No body does clearly better than spider_16.** Its long legs paddle furthest, even
  without lifting the body.
- **The snake was near the bottom.** Undulation needs a strong rhythm at the right tempo,
  which our randomly initialised brains lack at first.
- **Caveat:** each body's terrains differ, because a saved terrain includes the robot
  (they are in `results/bodies/terrains/`, separate from the main experiment's). So
  the "blocked by seed" Friedman test in `results/bodies/analysis/` is not meaningful
  here; read the table only as a rough ranking.
- **Decision at the time:** keep spider_16. Superseded the same day (D17): every body
  here only wiggled on video, and spider_16 physically cannot lift itself, so the team
  switched to spider_8, the body that can (experiment 13).

### 12: crossover or not (2026-09-29)

**Why:** before changing the EA further, check that its one recombination operator
earns its place. Same setup as the `best` condition of experiment 8 (spider_16, rugged,
islands migrating the best, 5 seeds, 12,000 evaluations); the only change is crossover
probability 0 instead of 0.5. `best` is reused from experiment 8.

| seed | with crossover | mutation only |
|---|---|---|
| 0 | **1.545** | 1.562 |
| 1 | **1.512** | 1.739 |
| 2 | **1.561** | 1.613 |
| 3 | 1.576 | 1.575 |
| 4 | 1.841 | **1.737** |
| mean ± sd | **1.607 ± 0.133** | 1.645 ± 0.087 |

- Mann-Whitney p = 0.31 (final fitness), 0.69 (AUC); `results/analysis_crossover/`.
- **Decision: keep crossover** (D7): it never clearly hurt, it was ahead on average, and
  it is the operator through which immigrants' neurons mix with the natives'. With 5
  seeds, the report can only claim that it does not hurt and may help a little.
