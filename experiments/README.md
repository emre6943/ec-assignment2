# Experiment log

Every experiment we ran, in order, with the script that reproduces it and what we
learned. Each finding changed a decision in [`../docs/decisions.md`](../docs/decisions.md);
the decision numbers (Dx) point there. This log is the story behind the Methods section:
*why* the final setup looks the way it does.

The compute every experiment used (runs, evaluations, walks, hours) is in
[`../docs/compute.md`](../docs/compute.md); `compute_ledger.py` rewrites it from `results/`.

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
| 8 | `08_main_experiment.sh` | **The research question, first setup** (spider_16, rugged): best vs worst vs random vs no migration, plus standard EA and random search | **No measurable effect of the migration policy.** All five EA variants end at 1.59–1.62 (± ~0.12); random search is clearly worse (1.73). Friedman across all six: AUC p = 0.011, final p = 0.053, driven by random search; no planned comparison is significant. Unseen terrain: ~1.97 for all. See *Details*. | D3 |
| 9 | `09_standard_vs_island.sh` | Are islands any better than one standard population of the same size? | **Not significantly** (3 seeds, Friedman p = 0.37): final fitness 1.54 for both; islands with migration far more consistent (std 0.03 vs 0.13) and ahead mid-budget; no migration slightly worse (1.58). Seeds 3–4 come with experiment 8. | D3, D11 |
| 10 | `10_overnight_best.sh` | How good a walker can the best setup get with a big budget (4 × 50, 120,000 evaluations, 15 s)? | **1.82 → 1.37** (about 0.8 m walked in 15 s), almost all of it in the first 20,000 evaluations; no improvement at all after about generation 440 of 624. Genetic diversity collapsed early. 2 h 47 min. | — |
| 11 | `11_body_pilot.sh` | Can another John Set body move better than spider_16 (whose motors cannot lift it)? | **No.** spider_16 1.50 and spider_8 1.52 lead; gecko 1.60, turtle 1.68, snake 1.72, linkin_modified 1.78 (distance, 2 seeds, 4,000 evaluations). | D1 |
| 12 | `12_crossover_or_not.sh` | Does neuron-level crossover help, or would mutation alone do as well? | **Keep it, no proof it matters:** 1.607 with crossover vs 1.645 mutation only (5 seeds); better on 3 seeds, tied on 1, worse on 1; not significant (p = 0.31). | D7 |
| 13 | `13_walking_pilot.sh` | Can spider_8 (which can lift itself) learn to *walk*, with the rhythm options, on flat / OlympicArena / rugged ground? Judged by video. | **On flat ground, yes:** it reaches the target with or without the rhythm options (fitness 0.020 / 0.002). **OlympicArena:** 0.55 m from the target, still improving. **Rugged:** 1.61 m, as stuck as spider_16. The terrain, not only the body, was the blocker. | D17 |
| 14 | `14_main_olympic.sh` | **The research question on the final setup** (the script's header lists it: spider_8, OlympicArena, ARIEL's spawn, 15 s, no vision (16 inputs), brain 16-8-4-8, gait and speed terms, crossover 0.9, no stagnation rule): best vs worst vs random vs no migration, plus standard EA and random search; 5 seeds, 12,000 evaluations, then the unseen test | **Done 2026-10-01.** Final fitness (mean ± sd): standard 0.957 ± 0.25, best 1.130 ± 0.32, random 1.137 ± 0.22, worst 1.172 ± 0.28, none 1.245 ± 0.21, random search 2.603 ± 0.11. Every EA beats random search (Holm p = 0.04); no policy differs significantly. Paired probabilities (AUC, `results/olympic/probabilities.md`): migrating beats not migrating with 89-96%; best is fastest (88% vs worst, 68% vs random); the standard EA is as fast or faster (65-80%). 3 brains reached their own target in 15 s. Figure: `results/olympic/rq_figure.png` (`rq_figure.py`) | D3 |
| 15 | `15_olympic_long_run.sh` | How good does the final setup get on one seed with a big budget (80,000 evaluations)? Does spider_8 reach the target on OlympicArena? | **Stuck at 0.84 m** (fitness 0.8755): 50% of the gain by 1,200 evaluations, 95% by 5,000, the last improvement at 8,800, then nothing until it was stopped at 48,600. Diversity collapsed early. Walking 45 s instead of 15 only gets it to 0.63 m. On video it crouches and keeps one leg tucked. | — |
| 16 | `16_gait_pilot.sh` | Do the gait and speed terms (D18, D20) and the stagnation rule (D19) give a better walker than experiment 15, on the same seed and arena? | **Better posture, less distance.** `gait` (12,000): 1.11 m left (vs 0.84), core on the ground 0.5% (vs 8%), all legs moving; but it barely stands (2.5 cm up) and one leg only props. `gait_stall` stopped at 9,400: σ maxed out on every island, no gain. | D18-D20 |
| 17 | `17_gait_pilot_2.sh` | Revised gait fitness: motor-work share per leg instead of movement, carry line 4 cm with weight 1.0, plus the stagnation rule. Better walker? | **Best so far:** 0.53 m left (vs 0.84 old fitness, 1.11 round 1); body 3.1 cm up; all legs' motors work alike (27-32 J). But the front and back legs are folded: their outputs sit at ±90° 50-61% of the walk, so only the side legs row. Walking 30 s: stuck in a small trough at x = 1.49 from 15 s on. | D18, D19 |
| 18 | `18_inputs_pilot.sh` | Fewer vision rays (5 instead of 10), and clock boost 1 instead of 3 against the saturated outputs: better legs and body height? | **5 rays + boost 1 is the best walker yet:** 0.53 m left (as good as 17), fitness 1.377 (best), body 4.7 cm up (vs 3.1) and under 2 cm only 4% of the time, fewer outputs pinned (26%); but the back leg's outer joint now sits folded (+86°, 92%). 5 rays with boost 3: 1.05 m. Walking 30 s: 0.36 m closest, stalls at x = 1.7. On 20 fresh arenas: 1.24 ± 0.30 m. | D5, D17 |
| 19 | `19_tuning.sh` | Are σ, the population size and the tournament size right for the final setup? One factor at a time, 3 seeds (100-102), 6,000 evaluations each | Nothing passes the rule (beat base on all 3 seeds and the mean): all values kept. Tournament 5 (1.61) and 4 × 40 (1.69) beat base's mean (1.80) but not on every seed | D22 |
| 19b | `19b_tuning_more.sh` | Part 2: the elite count (survivor selection), sparse vs dense mutation, and the crossover probability, same seeds and budget | **Crossover 0.9 wins on all 3 seeds** (1.37 ± 0.42 against 1.80 ± 0.07; 0.50 m left against 0.81 m). Crossover 0 is the worst setting. Elites and sparse mutation: no change | D22 |
| 20 | `20_long_walk.sh` | One long single-seed run: 15 s walks until 6,000 evaluations, then 30 s. Does spider_8 reach the target? With crossover 0.9 (D22) | **No:** 0.82 m left after 30 s (fitness 1.538), 1.11 ± 0.25 m unseen. Stuck at 1.10 m for the whole 15 s phase, no gain in the last 15,000 evaluations; experiment 18's brain is better on the same arena | D21 |
| 21 | `21_robustness_pilot.sh` | Can training make brains reach the target reliably? 2 × 2: 3 arenas with turned starts (0°, ±30°) vs 1 arena, 20 s vs 15 s walks; 3 seeds, 6,000 evaluations; tested on 100 unseen walks (5 turns, each on its own 20 arenas, 20 s) | **No.** Nothing reaches the target on unseen arenas (3 of 1,200 walks, all base's seed-101 brain). Mean distance left: base 1.02 m, walk20 1.35, arenas3 1.11, both 1.22. Nothing adopted | D23 |
| 22 | `22_stagnation_ablation.sh` | Does the stagnation rule (D19) earn its place? The final EA with the rule on (19b's xover_0.9) vs off; 3 seeds, 6,000 evaluations | **No:** on 1.763 / 0.921 / 1.421, off 1.531 / 1.091 / 1.421 (mean 1.369 vs 1.348). Helped 1 seed, hurt 1, no effect on 1. Dropped | D19 |
| 23 | `23_vision_ablation.sh` | Fewer vision rays: 5 (final) vs 3, 1 or none? Decided on the 20 unseen arenas; 3 seeds, 6,000 evaluations | **Keep 5.** Unseen mean: 5 rays 0.92 m, 3 rays 1.05, 1 ray 1.04, none 0.97; no smaller set beats 5 on all seeds. No vision ties 5 rays (training 1.339 vs 1.348) and learns a bit faster: vision adds nothing measurable, and is not what causes overfitting. The rays barely sense the few-cm bumps (they mostly measure the robot's own bobbing), so experiment 14 drops vision (Emre) | D5 |
| 24 | `24_brain_shape.sh` | Which brain shape learns to walk best? 16-16-8 (final) vs 16-8-8, 16-32-8, 16-8-8-8, 16-16-16-8, 16-8-4-8; 3 seeds, 6,000 evaluations | **16-8-4-8 (Emre's bottleneck idea), 212 weights:** the only shape better on all 3 seeds (1.203 vs 1.339 mean), also best unseen (0.85 vs 0.97 m) and fastest. Wider/deeper-wide shapes were clearly worse at this budget | D6 |
| 25 | `25_crossover_operators.sh` | Is our neuron crossover the right one? On 16-8-4-8: vs weight-level uniform, BLX-0.5 and headless chicken; 3 seeds, 6,000 evaluations | **Keep ours.** Headless chicken far worse (2.21 vs 1.20): crossover helps by combining real parents, not as a big mutation. Weight-level lost every seed: keeping neurons whole matters. BLX-0.5 won 2 of 3 (mean 1.17) and was best unseen (0.72 m) but fails the rule; it would also dilute immigrants | D7 |
| 26 | `26_migration_interval.sh` | How often should the islands migrate? The best policy every 5, 20 or 50 generations (14 gives 10 and never); 5 seeds, experiment 14's setup and arenas | **A sweet spot at 10-20.** Final fitness: every 20 1.093, 5 1.120, 10 1.130, 50 1.255, never 1.245. 5/10/20 learn equally fast and beat 50/never (93-100%). Every 20 is best on unseen arenas (0.76 m) and keeps islands apart (one shared champion in 6% of generations vs 28% at 10, 35% at 5) | D11 |
| 27 | `27_statistics.sh` | No new runs: how sure are we? For every pair of conditions in 14 and 26: mean difference ± 95% interval, the probability that one is truly better (Bayesian paired t-test), A12, seeds won | **Done:** `results/olympic/probabilities.md` (14) and `probabilities_interval.md` (26) | D3 |
| 28 | `28_longer_walks.sh` | No new evolution: is it the 15 s limit? The best brains of 14 and 26 walk their own arena for 15/20/30/60 s and the 20 unseen arenas for 30 s | **Partly.** Own arena: 3 of 30 reach the target in 15 s, 11 in 20 s, 11 in 30 s, 12 in 60 s: 15 s binds for a third, the rest stall. Unseen: 0-4% in 15 s, 5-13% in 30 s. Figure `results/olympic/longer_walks.png` | D12 |
| 29 | `29_absolute_position.sh` | Should the brain also get its absolute (x, y)? Experiment 14's five EA conditions rerun with 2 extra inputs (18 inputs, 228 weights), same seeds and arenas, each paired with its experiment-14 twin | **No: D5 stands** (2026-10-01). Over the 25 pairs: training fitness the same (1.128 → 1.153; P(position better) 37%), slower (AUC 14%), slightly worse on unseen arenas (0.869 → 0.940 m; 16%, the rule needed 90%). No condition gains consistently; the standard EA drops from first to last (0.957 → 1.288). Details in D5 | D5 |
| X | `x_best_walk.sh` | *For fun, not in the paper:* how good a walker can we make? A standard EA whose first population holds the 45 best brains of 14 and 26 (`--init-from`; 35 random), evolved for 5,000 generations (360,080 walks, 5.9 h; resumed once after the laptop's battery ran out) on a fresh arena, then the same blind test on the 20 unseen arenas | **The best brain yet** (2026-10-02). Fitness 0.617 (best earlier 0.682); it reaches its own target in 11.0 s. Blind test: 0.47 ± 0.22 m from the target on the 20 unseen arenas (best earlier brain 0.56 m, mean of the 45 starting brains 0.94 m), reaching it on 1 of 20 in 15 s and 8 of 20 in 30 s (best earlier 5 of 20). Generation 0 started 0.48 m away on the new arena | — |
| 99 | `99_final_experiment.sh` | **THE FINAL EXPERIMENT** (the paper reports this, not 14): experiment 14's six conditions with migration every 20 generations (D11, after 26) on 20 fresh seeds (10-29) for firmer statistics (D3); then the unseen test at 15 and 30 s, analysis, probabilities, longer walks. To run on emre-server, about 18 h; the launch command is in the script header | *Prepared and smoke-tested on emre-server 2026-10-02; not run yet* | D3, D11 |

Numbers are *fitness* (distance to the target plus posture penalties, D15; a robot that
does not move scores 2.0) unless they say "m". Runs 1–6 used 15-second episodes and a
hidden layer of 8 with σ = 0.1. Runs 7–9, 11 and 12 use 10-second episodes, so their
numbers are not directly comparable with 1–6.

Experiments 1–12 used spider_16 on RuggedTerrainWorld; 13 compared worlds with spider_8;
14 is the final setup (spider_8 on OlympicArena, D1, D2). Every script passes the body,
world and episode length it ran with, so it still reproduces now that the defaults are the
final setup.

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

### 13: can spider_8 walk? (2026-09-29)

**Why:** spider_16 cannot lift itself (D17), and every body in experiment 11 only
wiggled. spider_8 can hold its core 6 cm up. Does it learn to *walk*, and on which ground?

**Setup:** spider_8, seed 0, islands migrating the best, 15 s walks, 8,000 evaluations.
"Rhythm" = `--clock-boost 3 --evolve-tempo` (D17).

| run | ground | rhythm | best fitness | ended from the target | evolved tempo | wall-clock |
|---|---|---|---|---|---|---|
| flat_plain | SimpleFlatWorld | off | 0.020 | 0.02 m | (1 Hz fixed) | 5 min |
| flat_rhythm | SimpleFlatWorld | on | **0.002** | **0.00 m** | 0.88 Hz | 7 min |
| olympic | OlympicArena | on | 0.670 | 0.55 m | 0.67 Hz | 14 min |
| rugged | RuggedTerrainWorld | on | 1.637 | 1.61 m | 2.7 Hz | 8 min |

- **Flat ground is solved.** Both flat runs walk onto the target and stay there, with the
  core off the ground. With the rhythm options it got there sooner (fitness below 0.1
  after about 1,950 evaluations vs 2,170; below 0.5 after 1,230 vs 1,810). One seed, so
  a hint only.
- **OlympicArena is hard but learnable.** The robot crosses the flat start and gets about
  1 m into the rugged strip. The best fitness fell steadily (1.05 → 0.84 → 0.76 → 0.67 at
  1k / 3k / 5k / 7k evaluations) and had not levelled off at 8,000.
- **Rugged terrain stays out of reach** even for spider_8 (1.70 → 1.64). Its evolved
  tempo of 2.7 Hz suggests jittering rather than stepping. So the 25° slopes were a
  blocker on their own, not just spider_16's weak motors.
- **Code note:** the olympic run is the first with the fixed ground detection (D17,
  "worlds made of several pieces"); no OlympicArena result from before the fix is used.
