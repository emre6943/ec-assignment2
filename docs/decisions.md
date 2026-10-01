# Design decisions — Assignment 2

Every choice that shapes the experiment, why we made it, and what we gave up. This is the
raw material for the **Methods** section of the report: if a reader could ask "why this
and not that?", the answer belongs here.

Status legend:

- ✅ **Decided** — agreed by the team.
- 🟡 **Proposed** — a recommendation that needs the team's yes/no.
- ⏳ **Waiting on a measurement** — the recommendation depends on numbers we are still collecting.

---

## Summary

| # | Decision | Status | Current choice |
|---|---|---|---|
| D1 | Body | ✅ | `john_set.spider_8` since 2026-09-29 (was `spider_16`, which cannot lift itself; D17) |
| D2 | World | ✅ | `OlympicArena` since 2026-09-29, ARIEL defaults (was `RuggedTerrainWorld`: too steep to walk on; D17) |
| D2a | Spawn height | ✅ | Experiment 14 uses ARIEL's own spawn, as in the template (`--ariel-spawn`); earlier runs spawned 2 cm above the real ground, which rests in the same pose on OlympicArena |
| D2b | Terrain bump height | ✅ | ARIEL's default; nothing in ARIEL changed or re-implemented |
| D3 | Research question | ✅ (wording 🟡) | Effect of the emigrant-selection policy on convergence speed |
| D4 | Controller outputs | 🟡 | One output per hinge (8 for spider_8; 16 for spider_16), direct position control |
| D5 | Controller inputs | ✅ | 16 for spider_8: 8 joint angles + clock (2) + target vector (3) + tilt (3); no vision since experiment 23 (the rays sensed almost nothing) |
| D6 | Network shape | ✅ (experiment 24) | Fixed MLP 16-8-4-8 (two hidden layers, a bottleneck of 4); evolve weights only (212 weights) |
| D7 | Crossover | ✅ (experiments 12, 19b, 25) | Neuron-level uniform crossover, p = 0.9 since tuning (was 0.5); kept against weight-level, BLX-0.5 and headless chicken (experiment 25) |
| D8 | Mutation | ✅ (pilot) | Gaussian perturbation of every weight, σ = 0.05 |
| D9 | Selection | 🟡 | Tournament (parents) + generational with elitism (survivors) |
| D10 | Terrain and noisy fitness | ✅ | One fixed terrain per seed, shared by all conditions with that seed |
| D11 | Island model settings | 🟡 | 4 islands, ring, migrate every 10 generations, replace worst; every 5-20 learned equally fast in experiment 26, so 10 stays |
| D12 | Budget and stopping | ✅ rule / 🟡 size | Fixed budget of 12,000 evaluations per run; 15 s training walks on OlympicArena; the final brains are also tested with 20-60 s walks (step 28) |
| D13 | Baselines and controls | 🟡 | Random search + no-migration islands |
| D14 | Final evaluation | 🟡 | Best controllers re-tested on unseen terrains; for OlympicArena after experiment 14 |
| D15 | Fitness function | ✅ (weights 🟡) | Distance + penalties for the core touching the ground and for being upside down |
| D16 | Curriculum and early stopping | ❌ rejected | Lost to the plain setup on one seed; kept behind flags, off by default |
| D17 | Rhythm options and body/world rethink | ✅ | spider_8 on OlympicArena; clock boost 1 since experiment 18 (was 3); tempo gene dropped |
| D18 | Gait terms in the fitness | ✅ (exp. 16-18) | Contact penalty 1.0, carry term (4 cm line, weight 1.0), motor-work balance across legs (0.5) |
| D19 | Stagnation rule for the mutation step | ❌ dropped (exp. 22) | Tested alone: helped 1 seed, hurt 1, no effect on 1; same mean. Off in experiment 14 |
| D20 | Rewarding speed | ✅ (exp. 16-18) | + 0.5 × the walk-averaged distance to the target; walks end on arrival; episodes stay 15 s |
| D21 | Longer walks late in a run | ❌ (experiment 20) | Showcase only: 15 s then 30 s walks; did not reach the target |
| D22 | Parameter tuning | ✅ (experiments 19, 19b) | Crossover probability 0.9; σ, population, tournament, elites and dense mutation kept |
| D23 | Train on several situations | ❌ (experiment 21) | 3 arenas, turned starts and 20 s walks tried; nothing reached unseen targets, nothing adopted |

---

## D1. Body — ✅ `spider_8` (was `spider_16` until 2026-09-29)

**Now spider_8.** On 2026-09-29 the team switched to spider_8, because spider_16 turned
out to be physically unable to lift its own body: its motors are too weak for its weight
(measured in D17). Every experiment up to experiment 12 used spider_16; the
research-question experiment is re-run on spider_8.

**Why spider_16 was chosen first.** 16 hinges: 4 legs, 4 hinges each. It is symmetric, so
it can walk in any direction, which suits a target-reaching task.

Pilot on flat ground (200 evaluations of a simple sine-gait hill-climber, 5 seeds each,
remaining distance to a target 2 m away; lower is better):

| body | mean | std | cost per evaluation |
|---|---|---|---|
| spider_8 | 1.39 | 0.18 | 1× |
| **spider_16** | **1.38** | **0.12** | ~2× |
| spider_12 | 1.64 | 0.10 | ~1.7× |

spider_16 was as good as spider_8 and more consistent, at twice the cost per evaluation.
That pilot only tested a sine gait on flat ground, so it could not show that spider_16
never lifts its body; the video of the evolved controllers did.

## D2. World — ✅ `OlympicArena` (was `RuggedTerrainWorld` until 2026-09-29)

**Now OlympicArena** (team decision, 2026-09-29, after experiment 13). The spec lists it as
a valid world: "SimpleFlatWorld, RuggedTerrainWorld, CraterTerrainWorld,
AmphitheatreTerrainWorld, and OlympicArena are all valid choices", fixed for the whole
assignment. Along our path it is a flat start (spawn at x = 0), then from x = 0.5 a
rugged strip with bumps of a few centimetres, where the target (x = 2) lies. An uphill
ramp begins beyond the target. The arena is 2 m wide with drops at the sides.

- **Why not rugged:** its 25° slopes stopped every body and every setup we tried, even
  spider_8 (experiment 13: 1.61 m left after 8,000 evaluations).
- **Why not flat:** spider_8 solves it within about 2,000 evaluations, so every condition
  of the research question would reach the target and only differ in speed.
- **Why OlympicArena:** learnable but not trivial (experiment 13: 0.55 m left and still
  improving at 8,000 evaluations), so the conditions can differ in speed *and* in the
  final result.
- **Its rugged strip is random too.** ARIEL draws it with unseeded Perlin noise on every
  build, like RuggedTerrainWorld, only a few centimetres high. So everything in D10
  applies: each seed gets one saved arena, shared by every condition with that seed, and
  controllers can be re-tested on fresh arenas (D14). Evolved brains do specialise to
  their strip: experiment 13's OlympicArena winner ends 0.55 m from the target on its own
  arena, and 0.76-1.37 m away on 5 fresh ones.
- *Correction, for the record:* we first wrote here that OlympicArena is identical on
  every build, because its class has a `load_precompiled` option. ARIEL ships no
  pre-built copy, so the option does nothing; the code review caught it the same day.

**What follows is the history: why we first chose RuggedTerrainWorld** (experiments 1-12).

`RuggedTerrainWorld` draws a **new random Perlin-noise terrain every time it is
constructed**. The terrain generator is called without a seed in
`ariel/src/ariel/simulation/environments/heightmap_functions.py`. We use the class exactly
as shipped, and do not re-implement or seed it.

- **Our first plan** was to give every generation new terrain, so evolution would produce a
  controller that copes with ground it has not seen. In the literature this is
  **evaluating on randomised environments**. It is the same idea as adding noise to
  simulations so that controllers transfer to reality (Jakobi 1997; "domain
  randomisation" in robot learning).
- **In practice that made the fitness too noisy to learn** (experiment 1): the same
  controller scores very differently on different terrains.
- **What we do now:** each seed gets one random terrain for its whole run (D10), and
  robustness is tested afterwards on unseen terrains (D14).

Measured on 10 terrains each: final distance to the target, starting from 2.0, with the
spawn fix from D2a applied.

| Controller | Flat ground | Rugged: mean ± sd across terrains |
|---|---|---|
| Standing still (zero control) | 2.00 | 1.98 ± 0.05 |
| Random network with a clock input | 2.08 | 2.01 ± 0.19 |
| Sine gait tuned on flat ground (walks 1.04 m there) | 0.96 | 1.96 ± 0.14 |

The terrain scatters a controller's score by about ±0.12–0.14 m. That is **larger than
what weak controllers achieve**. Averaging over *k* terrains shrinks the scatter by √*k*:
about 0.08 m at *k* = 3 and 0.05 m at *k* = 10.

## D2a. Spawn height — ✅ ARIEL's own spawn for experiment 14 (ours on rugged terrain)

**The template's spawn buries the robot in rugged terrain.**

- `correct_collision_with_floor=True` lifts the robot so that its lowest point is just
  above z = 0. It ignores the terrain.
- The ground under the spawn point is typically 0.1–0.5 m higher. In 9 of 10 fresh
  terrains we checked, the core started **below the ground surface**.
- MuJoCo then pushes the robot out violently. In 26% of runs the robot ended upside down,
  and in one it fell through the world, before the controller had done anything.

Fix (`terrain.py`, our code only): sample the terrain under the leg span, spawn the robot
2 cm above the highest point, and let it drop. With the fix, no run ended upside down.

**This is the one place we pass a non-default ARIEL argument:** `world.spawn(...,
correct_collision_with_floor=False)`. ARIEL's default is `True`, and the template passes
`True` explicitly, but that default is what buries the robot. No ARIEL code is changed; it
is an argument of ARIEL's public `spawn()` call, like the spawn position the template
leaves to us.

- It conflicts with the team rule in D2b ("no settings away from ARIEL's defaults").

**Resolution (Emre, 2026-09-30): experiment 14 uses ARIEL's own spawn** (`--ariel-spawn`:
the template's `position=[0, 0, 0.1]`, `correct_collision_with_floor=True`).

- The problem above only exists on rugged ground. OlympicArena's start is flat at height
  0, which is exactly what ARIEL's correction assumes.
- Measured on OlympicArena: ARIEL's spawn starts the core at 8.5 cm, ours at 9.5 cm, and
  after 1 s both rest at 7.5 cm (`test_ariel_spawn_settles_like_ours_on_olympic_arena`).
  Tuning and the pilots (experiments 15-21) used ours; the robot ends in the same pose,
  so their conclusions carry over.
- So the final experiment passes no setting away from ARIEL's defaults beyond what the
  template itself passes. Experiments 1-13 on rugged terrain keep our spawn.
- An arena built with ARIEL's spawn includes the robot, so it is saved under its own name
  (`terrain0_arielspawn.mjb`), and the unseen test arenas likewise.

## D2b. Bump height — ✅ ARIEL's default, untouched

**Team rule: nothing in ARIEL is changed, configured away from its defaults, or
re-implemented.** The world is `RuggedTerrainWorld()` exactly as shipped, with bumps up to
0.5 m and a median slope of about 33°.

This matters for the Discussion, because the terrain is hard for this body: spider_16's
core cube is 15 cm on a side. A gait that walks 1.04 m on flat ground manages only about
0.18 m here. For context only (we do not use these settings), the same gait on lower bumps:

| tallest bump | final distance | distance walked |
|---|---|---|
| **0.50 m (ARIEL default, used)** | 1.96 ± 0.14 | 0.18 m |
| 0.20 m | 1.80 ± 0.20 | 0.25 m |
| 0.10 m | 1.51 ± 0.21 | 0.58 m |
| 0.05 m | 1.27 ± 0.14 | 0.76 m |

Expect small absolute improvements; the comparison between migration policies is what
counts.

**Evidence from the first full run (2026-09-28).** With plain distance as the fitness, one
terrain per generation and 3,000 evaluations:

- **Rugged terrain:** the EA learned nothing visible. All four islands' best scores rose
  and fell *together* from generation to generation, so the terrain's difficulty decided
  the score, not evolution. The final best network wiggled in place: its joint commands
  swung only ±0.13 on a scale of ±1.
- **The same EA on ARIEL's flat world** (a debug check, `run.py --world flat`): steady
  improvement from 1.84 m to 1.02 m. The evolved network swings its joints ±0.39 and walks
  about 1 m. Placed on rugged terrain, the same network moves only about 0.1 m.

Conclusion: the EA works; the default rugged terrain is hard for this body, and the
per-generation terrain noise hides small improvements. The team kept rugged terrain and
changed the fitness instead (D15).

## D3. Research question — ✅ (final wording 🟡)

> How does the emigrant-selection policy of an island-model EA (migrating the **best**,
> the **worst**, or **random** individuals) affect convergence speed and final fitness when
> evolving a neural-network controller for `spider_8` on OlympicArena?

(The first version, experiment 8, used `spider_16` on rugged terrain; D1, D2 and D17
explain the switch. The final experiment is experiment 14.)

**Hypothesis (team):** migrating the best individuals converges fastest.

**Suggested refinement:** state what we expect for final quality too. The classic result is
that sending the best migrants raises **selection pressure**: good genes spread across all
islands quickly, so the islands become similar sooner. That speeds convergence but can lock
every island onto the same mediocre solution (**premature convergence**). A two-part
hypothesis makes the discussion richer whatever the outcome:

> Migrating the best converges fastest, but ends at an equal or worse plateau than
> random migration. Migrating the worst behaves close to having no migration at all.

Grounding: Cantú-Paz (2001), *Migration policies, selection pressure, and parallel
evolutionary algorithms*, Journal of Heuristics 7(4). This is the key citation.

**Making "speed" measurable.** The true optimum is unknown, so "time to reach the optimum"
cannot be measured directly. Instead we report:

1. **Evaluations to reach a threshold**: the first evaluation count at which the best
   fitness drops below a fixed threshold. Experiment 14 uses 1.6 (`analyze.py`'s
   default): in tuning (D22), 2 of 3 runs of the tuned setup crossed it within 6,000
   evaluations and none of the untuned base runs (crossover 0.5) did, so it separates
   fast runs from slow ones. Experiment 8 used 1.6 too, which the rugged pilot's better
   settings reached within about 1,400 evaluations.
2. **Area under the best-fitness curve**, which rewards being good early.
3. **Final fitness** at the budget cap, on the training terrain; robustness is reported
   separately as the distance reached on unseen terrains (D14).

**Statistics** (`analyze.py`), on the final fitness and the AUC:

- **A Friedman test across all six conditions, blocked by seed.** Conditions with the
  same seed walk the same terrain (D10), so each seed is a matched block. This asks: do
  the conditions differ at all?
- **Planned comparisons of each condition against `none`,** the no-migration control:
  two-sided Mann-Whitney U tests with Holm's correction.
- **Why not all pairs?** With 5 seeds per condition, the smallest possible Mann-Whitney
  p-value is 0.008. Correcting for all 15 pairs of the six conditions would multiply it
  past 0.05, so nothing could ever come out significant. The five planned comparisons
  can (5 × 0.008 = 0.04).
- A paired Wilcoxon test is no use either: its smallest possible p with 5 pairs is 0.06.
- **More seeds** would allow more comparisons, if compute allows.

**Only the emigrant selection changes between conditions.** Island count, island size,
migration interval, number of migrants, topology and replacement policy stay fixed (D11).
Varying any of them as well would be a second research question.

**How sure are we? (added 2026-09-30, `probabilities.py`, step 27).** Besides the
significance tests above, the report gives, for every pair of conditions paired by seed:
the mean difference with its 95% interval; the probability that one condition is truly
better, from a Bayesian paired t-test with a flat prior (Benavoli et al. 2017), split
into better / practically equal (within ±0.05) / worse; the A12 effect size (Vargha &
Delaney 2000); and the seeds won. This answers "how likely is policy A better than B?"
directly, which a p-value does not. The probabilities are per pair and not corrected for
the number of pairs. Tried on experiment 24: 93% that 16-8-4-8 beats 16-16-8.

## D4. Controller outputs — 🟡 one per hinge, direct position control

*Numbers in D4-D6 are for spider_16, the body of experiments 1-12, unless they say
otherwise. The final body, spider_8 (D1), has 8 hinges, so 8 outputs; its final brain has
16 inputs (D5) and 212 weights (16-8-4-8, D6).*

**What is actually controlled?** The robot is built from modules: a core, bricks and
hinges. Only the **hinges** have motors, and `spider_16` has 16 of them (`model.nu == 16`).
Each motor is a **position servo**. We give it a target angle in [−π/2, π/2] and MuJoCo's
servo drives the joint toward that angle. So "one output per node" becomes **one output per
hinge**: 16 outputs, each a target angle. The body fixes this number; it is not a free
choice.

**Direct or delta control** (both appear in the template):

- **Direct**: `ctrl = output × π/2`. The network states the angle it wants. Simple and
  common.
- **Delta**: `ctrl += output × α`, then clip. The network states a change. It is smoother,
  but it accumulates, so the network also needs to know the current angle.

Recommendation: **direct**, with the network queried every **10 physics steps (50 Hz)**
instead of every step (500 Hz). The servos move the joints in between.

- **Speed:** measured on rugged terrain, it cuts about 16% of the run time (0.73 s down to
  0.61 s per 15 s episode), because the physics itself dominates.
- **Rhythm:** a 1 Hz gait still gets 50 updates per stride.

## D5. Controller inputs — ✅ 16 inputs for spider_8, no vision since experiment 23 (`sensors.py`)

The network can only react to what it is told. The spec warns about this directly: *a
controller with no signal telling it where the target is, or nothing to drive rhythmic
movement with, has very little to work with.*

| Input | Size | Why | Status |
|---|---|---|---|
| Hinge angles (proprioception) | 16 | Where each leg is now; needed to coordinate legs | implemented |
| Clock `sin(2πft)`, `cos(2πft)` | 2 | A beat to walk to. A feed-forward net has no memory, so without a clock it can only produce rhythm through feedback from the body | implemented |
| Target in the robot's own frame: distance, `sin(bearing)`, `cos(bearing)` | 3 | "Where is the goal relative to where I am facing". This is our answer to "we must know our position" (see below) | implemented |
| Body tilt: the world's up-direction seen from the core | 3 | Rugged terrain tips the body by 25–40° routinely; this lets the network react before falling | implemented |
| Bias | — | Every neuron has its own bias weight in the genotype | implemented |
| Vision | 10 | Distance to the terrain along 10 rays: 1 down, 1 up, and 2 per face (see below) | implemented |

Implemented: 16 + 2 + 3 + 3 + 10 = **34 inputs**. Vision is not needed for the
research question, but `run.py --no-vision` gives the 24-input version for an
optional comparison.

**Why not the absolute position (x, y)?** Knowing "I am at (0.7, 0.2)" only helps if the
network also works out where the target is from there, and it would have to learn that
separately for every location. The target vector **relative to the robot's heading** gives
exactly the useful part in a form that means the same everywhere: "the goal is 1.3 m away,
30° to my left". Absolute position is therefore replaced, not dropped.

**Vision (team decision: included, 10 rays).** Rays start at a point 0.25 m above the
core's centre. That point is fixed to the body, so the rays tilt and turn with the robot.
Each ray reports how far away the terrain is along it.

| Ray | Direction | What it tells the robot | Reading on flat ground, upright |
|---|---|---|---|
| Down | straight down | height above the ground | 0.34 m |
| Up | straight up | nothing while upright; sees the ground once the robot has flipped | no hit (3 m) |
| Ahead × 4 | along each face, 15° below horizontal | the ground about 1.3 m away: bumps and drops coming up | 1.33 m |
| Steep × 4 | along each face, 45° below horizontal | the ground just past the body, about 0.35 m away | 0.49 m |

The four faces are the core's ±x and ±y sides, which are the directions the four legs
stick out.

**How the rays are cast:**

- We intersect each ray with the ground geoms only (`terrain.ray_to_ground`: MuJoCo's
  `mj_rayHfield` for heightfields, `mj_rayMesh` / `mju_rayGeom` for the other pieces of
  OlympicArena). In our own code, the rays are computed 50 times a second, when the
  network is queried.
- **Why terrain-only:** MuJoCo's built-in rangefinder sensors also hit the robot itself.
  Because each leg lies along its face, the steep rays hit the leg instead of the ground.
  This happened all the time when the legs lay flat, and 23–48% of the time while
  walking.
- **Nothing is added to the robot:** no sensor, no marker. Nothing in ARIEL is touched.
- **Cost:** no measurable slowdown (about 0.6–0.7 s per walk with or without vision).

**Special cases:**

- A ray that hits nothing reads the maximum range, 3 m. The up ray always reads that while
  the robot is upright.
- If the robot is upside down and pressed into the ground, the ray origin ends up below
  the terrain. Every ray then reads 0, an unmistakable "I have flipped" signal.
- Readings are scaled to [0, 1] by dividing by 3 m.

These readings are checked by `tests/test_sensors.py` on ARIEL's flat world, where the
correct distances are known exactly.

**Fewer rays? (experiment 18, 2026-09-29).** Over experiment 17's best walk on
OlympicArena, the up ray always read its maximum (the robot never flipped), and the four
15° rays swung between about 0.8 and 3 m, mostly measuring the 2 m wide arena's edges and
the body's turning. The down ray (body height, ±1 cm) and the four 45° rays (the ground
just ahead, 2-7 cm variation over the bumps) carried the terrain information. The 10
rays cost 160 of the 568 weights. `--vision-rays near` keeps only the down ray and the
four 45° rays: 21 inputs and 488 weights for spider_8. The default stays `all`, so
earlier runs reproduce.

Result (one seed): with 5 rays and clock boost 3 the robot ended 1.05 m from the target,
against 0.53 m with 10 rays; with 5 rays and boost 1 it ended 0.53 m away and stood
clearly taller (see D17). A 0.5 m gap from the ray count alone mostly shows how noisy a
single-seed pilot is; the posture differences between the boost settings are large and
consistent, the distance differences are not.

**Even fewer rays? (experiment 23, 2026-09-30).** Stuck robots mostly do not touch the
ground: their legs keep stepping, but the brain switches to a step pattern that does not
push (tested on 4 stuck brains; only one scraped a bump). The rays let a brain react to its
own arena's exact bumps, which may cause both that switching and the overfitting to one
arena (D23). Fewer inputs also mean fewer weights: 488 (5 rays), 456 (3), 424 (1), 408
(none), so at most 16% fewer. Tested against the final EA (experiment 22's runs without
the stagnation rule) on seeds 100-102:

- `near3`: straight down plus the 45° rays ahead and behind along the core's +x axis, the
  axis the target bearing is measured from;
- `down`: straight down only, the body's height above the ground;
- none: `--no-vision`.

Rule fixed before the results: a smaller set replaces the 5 rays only if its brains end
closer to the target on the 20 unseen arenas (15 s) than the 5-ray brains on all 3 seeds
and on the mean; if several pass, the one with the lowest mean.

| Rays (weights) | Unseen distance, seeds 100 / 101 / 102 | Unseen mean | Training fitness at 6,000 | AUC |
|---|---|---|---|---|
| 5 `near` (488), the setup then | 1.100 / 0.653 / 1.013 | **0.922** | 1.348 ± 0.229 | 1.913 |
| 3 `near3` (456) | 1.132 / 1.096 / **0.912** | 1.047 | 1.628 ± 0.228 | 1.979 |
| 1 `down` (424) | **0.961** / 0.860 / 1.300 | 1.041 | 1.513 ± 0.306 | 1.906 |
| none (408) | **0.935** / 1.086 / **0.890** | 0.970 | **1.339 ± 0.184** | **1.768** |

Unseen distance in metres left after 15 s (lower is better); bold where a set beats the 5
rays. No unseen walk reached the target.

**First decision, by the rule: the 5 near rays stay.** No smaller set beats them on all 3
seeds.

- **No vision did as well as 5 rays, not better.** It trained to the same fitness, learned
  a little faster (lower AUC; below 1.6 sooner) and was as good on unseen arenas (0.97 vs
  0.92 m). The rule fixed for this test keeps the current set on a tie; experiment 22's
  rule preferred the simpler EA on a tie. This test's own rule applies.
- **For the report:** at this scale the rays add nothing measurable. The bumps are a few
  centimetres, and the down ray, the tilt input and the joint angles seem to carry what
  matters.
- **Vision is not what makes brains overfit or get stuck:** blind brains lose as much on
  unseen arenas (0.45 m on their own arena, 0.97 m on new ones) as brains with rays. The
  arena-specific behaviour must come through the other inputs and the walk's dynamics.

**Final decision (Emre, 2026-09-30): no vision in experiment 14 (`--no-vision`).** Why the
rays sense almost nothing on OlympicArena, measured over a 15 s walk of the best 5-ray
brain (seed 101), on its own arena and on flat ground:

| Input (scaled to 0-1) | Spread while walking, own arena | Same, flat ground |
|---|---|---|
| The 5 rays | 0.005-0.013 (1.4-4.0 cm) | 0.003-0.012 (0.9-3.7 cm) |
| Joint angles | 0.21 | 0.21 |

- **The signal is tiny.** A ray reads up to 3 m and the bumps are about ±2 cm, so a bump
  moves a ray input by less than 0.01, 20-40 times less than the joint angles move.
- **The rays mostly measure the robot itself.** On flat ground, with no bumps at all,
  they vary almost as much as on the arena: what they pick up is the body bobbing and
  tilting as it walks, which the tilt input and the joint angles already report.
- **They were never aimed at the target.** They are terrain sensors fixed to the body:
  one straight down and four 45° down around it. The target has its own inputs (distance
  and direction), and the robots often walk sideways, so no fixed ray would point at the
  target anyway.
- **Why vision was there:** it was chosen for RuggedTerrainWorld, whose hills are up to
  0.5 m high and worth seeing. OlympicArena's bumps are a few centimetres.
- **Why this overrides the test's own tie rule:** the rule kept the current set on a tie,
  but this measurement shows the tie is no accident: the rays carry almost no information
  about the terrain. Experiment 22 already let the simpler EA win a tie. Removing them
  leaves 16 inputs and 408 weights, and changes nothing else.
- Tuning (D22) and experiment 22 ran with the 5 rays. Experiment 23's blind runs used the
  same settings and trained as well (1.339 vs 1.348), so those settings carry over.

**The 16 inputs of the final brain** (all scaled to about [-1, 1]):

| Inputs | Count | What they tell the brain |
|---|---|---|
| Joint angles | 8 | Where each of the 8 hinges is now |
| Clock | 2 | sin and cos of a 1 Hz beat, for a rhythm |
| Target | 3 | Distance to the target / 2 m, and sin and cos of its direction seen from the robot |
| Tilt | 3 | Which way is up, seen from the core: how the body leans |

## D6. Network shape — ✅ 16-8-4-8 since experiment 24 (was one hidden layer of 16)

**Weights only, or topology too?** There are two families of neuroevolution:

- **Fixed topology, evolve weights**: the network's shape is chosen by us and the genotype
  is simply the list of all weights. This is what the assignment describes.
- **Topology and weights** (e.g. NEAT, Stanley & Miikkulainen 2002): evolution also adds
  neurons and connections. It is powerful but much more complex, and a changing genome
  length would interfere with our migration research question.

We use a **fixed topology** with `tanh` activations. The code supports any number of
hidden layers (`--hidden-layers 16`, or `8,8` for two):

```
34 inputs ──► hidden (tanh) ──► 16 outputs (tanh × π/2)
```

Genotype length, with a bias on every neuron:

| Hidden layers | Weights |
|---|---|
| 8 | 35·8 + 9·16 = 424 |
| **16 (used)** | 35·16 + 17·16 = **832** |
| 8, 8 | 35·8 + 9·8 + 9·16 = 496 |

**Trade-off:** more neurons or layers can express more complex gaits, but every weight is
one more dimension evolution has to search. One hidden layer can already represent any
smooth mapping in principle (the universal approximation theorem). A second layer also
makes the neuron-level crossover less clean: only the last hidden layer's neurons can be
kept fully intact (D7).

**How we chose: a pilot** (`experiments/07_pilot.sh`, 2026-09-28). There is no way to know
the best brain in advance, so we compared shapes and mutation step sizes (D8) under
identical conditions:

- the same seed and terrain for all nine runs;
- 4 × 20 individuals, 4,000 evaluations, 10 s episodes.

Best fitness at the end (lower is better; standing still scores 2.0):

| Brain | σ 0.05 | σ 0.1 | σ 0.2 |
|---|---|---|---|
| 1 layer × 8 | 1.58 | 1.56 | 1.60 |
| 1 layer × 16 | **1.51** | 1.70 | 1.67 |
| 2 layers × 8 | 1.57 | 1.57 | 1.72 |

- **A second layer did not help.**
- **16 vs 8 neurons** cannot be separated on one seed: 16 had the best end result, 8
  improved fastest early on.
- We took **one layer of 16**, the best end result. It costs nothing measurable, because
  the physics dominates the simulation time.
- **Limitation to state in the report:** this is a single-seed pilot, meant to pick
  reasonable settings, not to prove one shape is better. It is a development choice, not
  part of the research question.

**Shape test on the final setup (experiment 24, 2026-09-30).** The one hidden layer of 16
was chosen on spider_16, rugged terrain, the old fitness and one seed (experiment 7);
since then the body, world, inputs (16, D5) and fitness have all changed. Five shapes
against 16-16-8 (408 weights), seeds 100-102, 6,000 evaluations:

| Shape | Weights | Tests |
|---|---|---|
| 16-8-8 | 208 | narrower |
| 16-32-8 | 808 | wider |
| 16-8-8-8 | 280 | deeper, small |
| 16-16-16-8 | 680 | deeper at the current width |
| 16-8-4-8 | 212 | a bottleneck of 4 (Emre's idea) |

How a shape is usually chosen in neuroevolution, and why this way:

- **Compare candidates on several seeds** with a rule fixed in advance: the tuning
  approach (Eiben & Smit 2011), used here.
- **Evolve the topology** (NEAT, Stanley & Miikkulainen 2002): start without hidden nodes
  and add nodes and links by mutation. Principled, but a different EA: genomes change
  length, so crossover and migration would need redesigning, which changes the research
  question. Future work.
- **Budget per weight:** 12,000 evaluations give 408 weights about 29 evaluations each;
  808 weights get half that. At a fixed budget, bigger brains are harder to search.

Rule fixed before the results (D22's rule): a shape replaces 16-16-8 only if its best
fitness at 6,000 evaluations beats 16-16-8 on all 3 seeds and on the mean; if several do,
the one with the lowest mean. Every brain also gets the standard unseen test. Known bias:
σ = 0.05 was tuned for 16-16-8, and a bigger network gets more total mutation noise per
child, which slightly favours smaller networks.

| Shape | Weights | Best fitness, seeds 100 / 101 / 102 | Mean | Beats 16-16-8 on | Unseen mean | AUC |
|---|---|---|---|---|---|---|
| 16-16-8 (was final) | 408 | 1.196 / 1.546 / 1.276 | 1.339 | — | 0.970 m | 1.768 |
| 16-8-8 | 208 | 1.384 / 1.808 / 1.250 | 1.481 | 1 of 3 | 1.019 m | 1.971 |
| 16-32-8 | 808 | 1.825 / 1.708 / 1.881 | 1.805 | 0 of 3 | 1.194 m | 2.235 |
| 16-8-8-8 | 280 | 0.927 / 1.450 / 1.649 | 1.342 | 2 of 3 | 1.003 m | 1.885 |
| 16-16-16-8 | 680 | 1.766 / 1.696 / 1.442 | 1.635 | 0 of 3 | 1.024 m | 2.075 |
| **16-8-4-8** | **212** | **1.176 / 1.367 / 1.065** | **1.203** | **3 of 3** | **0.845 m** | **1.699** |

**Decision: 16-8-4-8, the bottleneck (`--hidden-layers 8,4`), 212 weights.** It is the only
shape that passes the rule, and it is also the best on the unseen arenas (better on 2 of 3
seeds), learns fastest (lowest AUC) and crossed fitness 1.6 in all 3 runs.

- **Why a bottleneck can help:** the 8 motors are driven by only 4 shared signals, so the
  legs are pushed to move in coordinated patterns rather than 8 independent ones, and there
  are half as many weights to search. Coordinated patterns of this kind are how legged
  gaits are usually described; the result fits that, though it does not prove it.
- **The big networks were still improving at 6,000 evaluations** (16-32-8's last gains came
  at 5,600-6,000), so this measures learning speed at a fixed budget, which is what the
  research question is about. At 12,000 they might close some of the gap.
- **Margins:** seed 100 is close (1.176 vs 1.196); the rule asks for consistency, not size.
  The σ bias above favours small networks.
- **Crossover:** with two hidden layers, our neuron crossover moves the first layer's
  neurons without their outgoing weights. Experiment 25 tests the crossover on this shape.

## D7. Crossover — ✅ neuron-level uniform crossover, kept after experiments 12 and 25; p = 0.9 since tuning (D22)

Crossover builds a child from two parents. Three candidates:

**(a) Arithmetic / averaging** (your first idea): `child = α·p1 + (1−α)·p2`, usually
α = 0.5.

- Simple, and smooth in weight space.
- ✗ It pulls every child toward the middle of its parents, so the population's diversity
  shrinks every generation. That matters for us: an immigrant's distinct weights get
  averaged away within a few generations, which weakens exactly the effect we want to
  measure.
- ✗ **The competing-conventions problem** (Schaffer, Whitley & Eshelman 1992). Two networks
  can solve the task with their hidden neurons in a different order. Hidden neuron 3 might
  drive the front-left leg in parent A and the back-right leg in parent B. Averaging them
  gives a neuron that does neither.

**(b) Uniform crossover at the weight level** ("share half of the weights"): each weight is
copied from one parent or the other, chosen at random.

- Keeps real weight values, with no averaging.
- ✗ It still mixes the weights of one neuron from two parents, so the conventions problem
  remains.

**(c) Neuron-level uniform crossover** (recommended): for each hidden neuron, copy **all**
of its incoming and outgoing weights from one parent, chosen at random.

- Neurons stay intact as functional units: the child inherits whole "leg drivers" from each
  parent.
- It does not fully solve the conventions problem, but it avoids the worst damage.
- It is easy to implement, about 20 lines.
- It also gives a point for the discussion: within an island, individuals share ancestors,
  so their conventions mostly agree. Immigrants come from another lineage, so crossing
  them with natives is where the conventions problem would show.

**How much comes from each parent?** For each of the 16 hidden neurons, a fair coin
decides which parent it comes from. That neuron's 35 incoming weights (34 inputs + bias)
and 16 outgoing weights are copied together. On average, half of the neurons come from
each parent. The 16 output biases belong to no hidden neuron, so each is taken from a
random parent.

On the final brain (16-8-4-8, D6), each of the 8 first-layer neurons moves with its 17
incoming weights (16 inputs + bias), and each of the 4 second-layer neurons with its 9
incoming weights (8 + bias) and its 8 outgoing ones; the 8 output biases come from a
random parent each. A weight belongs to exactly one unit, so the first layer's outgoing
weights travel with the second-layer neuron they feed.

Crossover probability: 0.5. The other half of the children are a copy of one parent. Every
child is then mutated (D8). These values are identical across all research-question
conditions.

With the tempo gene (D17), the weights are crossed over as above and the child takes the
tempo gene from one parent, chosen at random.

**Does crossover help? Tested in experiment 12 (2026-09-29).** Same EA (islands migrating
the best, spider_16, rugged, 5 seeds, 12,000 evaluations), once with crossover (p = 0.5)
and once with mutation only (p = 0). Final fitness, lower is better:

| seed | with crossover | mutation only |
|---|---|---|
| 0 | **1.545** | 1.562 |
| 1 | **1.512** | 1.739 |
| 2 | **1.561** | 1.613 |
| 3 | 1.576 | 1.575 |
| 4 | 1.841 | **1.737** |
| mean ± sd | **1.607 ± 0.133** | 1.645 ± 0.087 |

Two-sided Mann-Whitney U: p = 0.31 on the final fitness, p = 0.69 on the area under the
curve. Not significant.

**Decision: keep crossover (p = 0.5).**

- It was better on 3 seeds, tied on 1 and worse on 1, and better on the mean. There is no
  sign that it hurts.
- Recombination is what makes this a genetic algorithm rather than a population of
  parallel hill-climbers (Eiben & Smith 2015). It lets good neurons found in different
  individuals be combined.
- The research question is about migration. With crossover, an immigrant's neurons can be
  combined with the natives' neurons; without it, an immigrant can only compete with them.
  Removing crossover would take away one of the two ways migration can act.
- **Limitation to state in the report:** with 5 seeds, a difference this small (about
  0.04) cannot be shown to be real. The honest claim is "crossover does not hurt and may
  help a little", not "crossover helps".

**Update (D22):** on the final setup, tuning raised the crossover probability from 0.5 to
0.9, and switching crossover off was the worst of the tuned settings. Both point the same
way as experiment 12, now more clearly.

**Why does it work if most children are worse? (2026-09-30)** Measured on the final EA
(experiment 23's no-vision run, seed 100): real parent pairs rebuilt from saved generations
with the EA's own tournament, 60 children per operator, each child compared with its
better parent.

| Child made by | Beats its better parent (generation 10 / 40 / 83) |
|---|---|
| Copy + mutation (no crossover) | 7% / 13% / 3% |
| Neuron crossover + mutation (ours) | 2% / 10% / 7% |
| Neuron crossover, no mutation | 3% / 10% / 8% |
| Each weight from either parent + mutation | 3% / 7% / 2% |
| Average of the parents + mutation | 7% / 15% / 7% |

- Most children are worse than their better parent for **every** operator, mutation
  included (85-98%). That is normal: variation mostly breaks things, selection keeps the
  rare improvement, and the elites protect the best. A bad child costs one evaluation.
- Our crossover is not much worse than mutation alone. What it adds is the kind of rare
  improvement: whole neurons from two lineages combined, and bigger structured steps.
- Mixing single weights was the most damaging, as the conventions argument predicts.
- One run and 60 children per cell: differences of a few percent are noise.

**Literature (research summary, citations verified):**

- The operator is Montana & Davis's (1989) neuron-as-unit idea with SANE/ESP's unit
  definition (Moriarty & Miikkulainen 1996; Gomez & Miikkulainen 1997): a hidden neuron
  moves with its in- and out-weights. Montana & Davis found little difference between
  weight-level, neuron-level and matched crossover.
- The competing-conventions problem is milder in practice than its reputation (Hancock
  1992). It needs parents from different lineages. Within an island everyone descends from
  a few elites, so neuron j means the same in both parents and the child lands close to
  both: crossover acts as a small, structured mutation scaled by the parents' distance.
  Where it can bite is a native crossed with a fresh immigrant: part of the Discussion.
- Crossover helps between aligned (related) parents (NEAT's ablation, Stanley &
  Miikkulainen 2002) and hurts between unrelated networks (Bodnar et al. 2020, MuJoCo Ant).
  Uriot & Izzo (2020) first align the parents' neurons by how well they correlate, then
  cross them; on MNIST and CIFAR-10 the children kept far more of their parents' skill
  than with naive crossover.
- "Headless chicken" crossover (Jones 1995), crossing a parent with a random genotype,
  separates a real recombination benefit from a large-mutation effect.

**Experiment 25:** three operators against ours, chosen by Emre, 3 seeds, 6,000
evaluations, the final setup with experiment 24's brain:

| Operator | Definition | Tests |
|---|---|---|
| Weight-level uniform | each weight from either parent, 50/50 | whether keeping neurons whole matters |
| BLX-0.5 | each weight drawn uniformly from [lo − 0.5d, hi + 0.5d], d = hi − lo of the parents | blending; its spread shrinks as an island converges (Eshelman & Schaffer 1993) |
| Headless chicken | our neuron crossover with a fresh random genotype instead of a second parent | whether our crossover is just a big mutation (Jones 1995) |

Rule fixed before the results: an operator replaces ours only if it beats it on all 3
seeds and on the mean best fitness at 6,000 evaluations. Mutation afterwards is unchanged.

**Results (experiment 25, on 16-8-4-8, D6):**

| Operator | Best fitness, seeds 100 / 101 / 102 | Mean | Beats ours on | Unseen mean (seeds) |
|---|---|---|---|---|
| **Neuron (ours)** | 1.176 / 1.367 / 1.065 | 1.203 | — | 0.845 m (0.72 / 0.90 / 0.92) |
| Weight-level uniform | 1.252 / 1.762 / 1.235 | 1.416 | 0 of 3 | 0.906 m |
| BLX-0.5 | **1.004 / 1.178** / 1.322 | **1.168** | 2 of 3 | **0.716 m (0.61 / 0.70 / 0.83)** |
| Headless chicken | 2.356 / 2.129 / 2.136 | 2.207 | 0 of 3 | 1.331 m |

**Decision: our neuron crossover stays.** No operator beats it on all 3 seeds.

- **Headless chicken is by far the worst** (2.21 vs 1.20; it never crossed 1.6). Crossing
  with a random brain instead of a real partner ruins the search, so our crossover is
  not just a big mutation: combining two good parents is what helps. This answers the
  doubt that started the test.
- **Weight-level uniform lost on every seed,** as the competing-conventions argument
  predicts: keeping neurons whole matters.
- **BLX-0.5 is the interesting one.** Better on 2 of 3 seeds and on the mean, and better on
  unseen arenas on all 3 seeds; its seed-100 brain is the first to reach the target on
  unseen arenas within 15 s (1 of 20). It fails the rule on seed 102 (1.322 vs 1.065), so
  it is not adopted. There is also a reason tied to the research question: a blend pulls
  an immigrant toward the natives' average and dilutes it within a few generations (see
  above), which would blur the very effect of migration we measure. For the report:
  promising for robustness, future work.

## D8. Mutation — ✅ Gaussian perturbation, σ = 0.05 (chosen by the pilot)

Every weight gets a small random nudge: `w ← w + N(0, σ²)`. Weights start as N(0, 0.5²).

- If σ is too large, children bear no resemblance to their parents and the search becomes
  random search.
- If σ is too small, progress is glacial.

**How we chose σ:** the same pilot as D6 compared σ = 0.05, 0.1 and 0.2.

- **σ = 0.2 was the worst for every network shape**, and the slowest to reach even a modest
  fitness: steps that big destroy gaits that already work.
- **0.05 and 0.1 were close.** 0.05 gave the best result overall, so it is the default.

A self-adapting σ is a well-known improvement, but it is another moving part and not our
research question.

Implemented with ariel's own `FloatMutator.gaussian`, which does exactly this. That keeps
the EA "built on ariel.ec" as the template asks. Its random stream is kept separate from
our operators' stream (see `ea.operator_rng`): seeding both with the same number made the
first mutations exact copies of initial weights, which the code review caught.

## D9. Selection — 🟡 tournament for parents, generational with elitism for survivors

**Is tournament selection computationally heavy?** No. Almost all of the cost is the
physics simulation. Every individual has to walk once per generation to get a fitness,
whatever selection we use. Selection only compares the resulting numbers, which takes
microseconds.

**"Just let them walk and the ones closest to the goal win."** That is exactly what
happens: walking *is* the fitness evaluation. Picking the top *k* closest is called
**truncation selection**. The difference from a tournament:

- **Truncation** always picks the top *k*. It gives high, fixed selection pressure, so
  diversity is lost fast.
- **Tournament of size *k***: draw *k* individuals at random and the best of them becomes a
  parent. Repeat for every parent. Weaker individuals occasionally win, which keeps
  diversity. *k* tunes the pressure: *k* = 2 is gentle, *k* = 5 is strong. It also matches
  what we used in Assignment 1.

Recommendation:

- **Parent selection**: tournament, *k* = 3, within each island.
- **Survivor selection**: generational replacement (children replace parents) with
  **elitism of 2**: the island's 2 best survive unchanged. Elitism stops the best solution
  from being lost. Keeping it small preserves diversity, and the research question already
  varies selection pressure through migration.

## D10. Terrain and noisy fitness — ✅ one fixed terrain per seed

Because every `RuggedTerrainWorld()` is a different random terrain (D2), *which* terrain a
controller walks on changes its score. Three options were considered:

| Option | How it works | Pros | Cons |
|---|---|---|---|
| New terrain per evaluation (template default) | Every individual gets its own terrain | No extra work | Unfair: a good controller on hard ground loses to a bad one on easy ground |
| One terrain per generation (tried first) | Everyone in a generation walks the same terrain; the next generation gets a new one | Controllers must generalise | **Measured: no learning.** The terrain's difficulty moved all islands' scores up and down together, and that noise buried the progress |
| **One terrain per seed** (used) | Each run keeps one terrain for all generations | Deterministic fitness: progress is visible, and elites keep their fitness | The controller may specialise to its terrain; tested on unseen terrain afterwards (D14) |

**How it works (`--terrain-mode per_run`, the default):**

- The first run with seed *s* builds a plain world (`RuggedTerrainWorld()` in
  experiments 1-12, `OlympicArena()` in experiment 14) and saves it to
  `<results>/terrains/<world>/[<body>/]seed<s>/terrain0.mjb`, where `<results>` is the
  folder above the condition folders (`results/` for experiment 8, `results/olympic/` for
  experiment 14).
- Every other condition with seed *s* (best, worst, random, none, standard, random search) loads
  **the same file**. Conditions are compared on identical ground, which makes the
  comparison paired, and different seeds cover different terrains.
- Elites keep their fitness instead of being re-evaluated, since the same controller on
  the same terrain always scores the same. That saves 10% of evaluations.
- **Reproducibility:** with the terrain file kept, a run can be repeated exactly. The EA's
  randomness is seeded, and the terrain is loaded rather than regenerated.
- The `terrains` entry in each run's `config.json` records which file it used.

`--terrain-mode per_generation` keeps the old behaviour, which could make a "robustness"
side experiment. `--n-terrains k` gives each seed *k* terrains, averaged per individual.

## D11. Island model settings — 🟡

An **island model** splits the population into sub-populations that evolve separately and
occasionally exchange individuals (**migration**). Isolation lets islands explore different
solutions; migration spreads good genes. Our research question varies *which* individuals
migrate.

| Setting | Proposed | Reason |
|---|---|---|
| Islands | 4 | Enough to have diversity between islands; each island still reasonably large |
| Island size | 20 (80 in total) | ⏳ depends on the budget (D12) |
| Topology | Ring (island *i* sends to island *i*+1) | The standard, simplest choice (Cantú-Paz 2001) |
| Migration interval | Every 10 generations | Islands get time to diverge between exchanges |
| Migrants per event | 2 (10% of an island) | The usual range is 5–10% |
| Emigrant selection | **best / worst / random** | ← **the research question** |
| Replacement on arrival | Immigrants replace the island's worst | Fixed for all conditions |
| Copy or move | Copy (the emigrant also stays home) | Standard; keeps island sizes constant |

**Follow-up: how often to migrate (experiment 26, queued after experiment 14,
2026-09-30).** Migrating rarely keeps the islands apart, so each explores its own
solutions; migrating often spreads good ones fast but can pull every island into the
same local optimum. With the best policy every 10 generations, all 4 islands held the
same champion within about 2,000 evaluations (experiment 20). The interval was
deliberately not tuned (D22), since it sets the context of the research question, so
this is a separate question on experiment 14's setup, seeds and arenas: the best policy
migrating every 5, 20 and 50 generations; experiment 14 supplies every 10 (`best`) and
never (`none`). Exploratory: no setting changes because of it. Measured: convergence and
final fitness across the five intervals, and how different the islands stay (the
genotype spread per island, and how soon every island holds the same champion).
Background: Cantú-Paz (2001); Skolicki & De Jong (2005).

**Results (experiment 26, 2026-10-01; probabilities from step 27, paired by seed):**

| Migrate the best every | Final fitness | AUC | Unseen distance | All islands share one champion | Within-island spread at the end |
|---|---|---|---|---|---|
| 5 generations | 1.120 ± 0.219 | 1.503 | 0.931 m | 35% of generations, from about 4,000 evaluations | 0.99 |
| 10 (experiment 14) | 1.130 ± 0.317 | 1.491 | 0.857 m | 28%, from about 6,800 | 1.11 |
| **20** | **1.093 ± 0.241** | 1.497 | **0.764 m** | 6% (2 of 5 runs ever) | 1.24 |
| 50 | 1.255 ± 0.276 | 1.602 | 0.939 m | never | 1.53 |
| never (experiment 14's `none`) | 1.245 ± 0.211 | 1.649 | 0.911 m | never | 1.31 |

- **There is a sweet spot, as Emre expected.** Every 5, 10 or 20 generations learn equally
  fast (AUC probabilities 42-58% between them), and all three beat migrating every 50
  (93-99%) or never (96-100%). Every 50 is about as bad as no migration (final fitness
  47%): 3 migrations in 166 generations are too few to spread anything.
- **Every 20 is the best compromise:** the best final fitness (100% likely better than
  never, 85% than every 50; 58-59% against 5 and 10, a tie) and the best on unseen arenas
  (77% vs every 10, 92% vs every 5, 99% vs never).
- **Diversity, measured:** the more often the best migrate, the sooner every island holds
  the same champion (from about 4,000 evaluations at 5, 6,800 at 10; rarely at 20; never
  at 50) and the less diverse each island stays. Every 20 keeps the islands apart while
  still sharing good solutions often enough to speed up the search.
- One policy and 5 seeds: exploratory. Experiment 14 keeps its literature value of 10.

## D12. Budget and stopping — ✅ fixed budget; size chosen from the curves

**Rule (team decision, 2026-09-28): every run stops after a fixed number of evaluations**,
the same for all conditions and the baseline, as a fair comparison needs.

- A plateau-detection rule existed first. It never fired: the budget always ran out
  earlier, and the long run showed flat stretches of up to about 25 generations *followed
  by further improvement*, so any short window would have stopped runs too early.
- The spec's tips say to stop on a plateau, not a fixed count. We meet the intent by
  **choosing the budget from where the curves flattened** in the pilot and in the long
  single-seed run (improvement until about generation 115 with 4 × 40 individuals), and
  say so in Methods.

An "evaluation" is one individual walking the run's terrain(s). Random search is budgeted
the same way.

**Episode length: 15 s on OlympicArena** (2026-09-29; the template's value). Experiments
7-12 used 10 s: no spider_16 came close to the target, and a shorter episode cut the cost
by a third. spider_8 on OlympicArena can reach the target, and experiment 13 ran at 15 s,
so the final setup keeps the template's 15 s.

**Final budget (experiment 14): 12,000 evaluations per run**, 8-10 minutes on
OlympicArena, about 4 hours for 6 conditions × 5 seeds (measured; `docs/compute.md`). The
OlympicArena pilot was still improving at 8,000 evaluations.

**Measured cost** on this Mac (M3 Pro, 10 worker processes, vision on):

| Setup | Speed | One run |
|---|---|---|
| 15 s episodes, 4 × 20, 3,000 evaluations | ~10 evaluations/s | ~5 min |
| 15 s episodes, 4 × 40, 25,000 evaluations | ~5.4 evaluations/s (walkers get costlier) | 77 min |
| 10 s episodes, 4 × 20, 4,000 evaluations (pilot) | ~7.5 evaluations/s (while other work ran) | ~9 min |

**Budget of the first main experiment (experiment 8): 12,000 evaluations per run** (4 × 20
individuals, about 166 generations, about 17 minutes per run with 10 s episodes, 5
conditions × 5 seeds in about 7 h; the standard EA was added later as a sixth).

- The long run (experiment 5) kept improving until about 18,000 evaluations, with twice
  the population and 15 s episodes.
- On OlympicArena, experiment 15 made 95% of its progress by 5,000 evaluations and its
  last improvement at 8,800, then nothing until 48,600. So 12,000 covers the learning
  phase plus a stretch of plateau, and stays the budget for experiment 14.

**Longer test walks (step 28, `longer_walks.py`, 2026-10-01).** Training stays at 15 s.
After experiment 14 we asked whether the brains miss the target because of the 15 s limit
or because of how they walk: every run's best brain walked its own arena for 15, 20, 30
and 60 s (ending on arrival), and the 20 unseen arenas for 30 s.

| Condition | Reached in 15 s | 20 s | 30 s | 60 s | Unseen arenas, 15 s | Unseen, 30 s |
|---|---|---|---|---|---|---|
| migrate best | 1 of 5 | 2 | 2 | 3 | 0% | 6% |
| migrate worst | 1 of 5 | 3 | 3 | 3 | 0% | 13% |
| migrate random | 0 of 5 | 1 | 1 | 1 | 0% | 5% |
| no migration | 0 of 5 | 2 | 2 | 2 | 0% | 7% |
| standard EA | 1 of 5 | 3 | 3 | 3 | 4% | 7% |
| random search | 0 of 5 | 0 | 0 | 0 | 0% | 0% |
| **all** | **3 of 30** | **11** | **11** | **12** | | |

(`results/olympic/longer_walks.md` and `.png`; experiment 26's intervals in
`longer_walks_interval.*`.)

- **15 s is binding for about a third of the brains.** 8 brains that miss in 15 s arrive
  between 15.2 and 19.6 s: they head for the target and run out of time.
- **The rest stall, and more time does not help:** from 20 to 60 s only one more brain
  arrives. They get stuck 0.2-0.7 m away (legs moving, body still: D23), and one fell over
  during the 60 s walk.
- **On unseen arenas more time helps a little:** at most 4% of the walks arrive in 15 s,
  5-13% in 30 s. Generalising to new arenas stays the main limit (D23).
- **The policy hardly matters here** (1-3 of 5 per condition at 30 s); random search
  never arrives.
- **Why we keep training at 15 s:** it is the research question's fixed budget, and
  training with 20 s walks did not help in experiment 21 (old setup). Brains that learn to
  get as close as possible in 15 s are mostly heading the right way, and a longer test
  shows what they have learnt. Report both: the 15 s results for the research question,
  and the 30 s test for what the brains can do.
- "Reached" here is about each run's best-fitness brain. During evolution, other brains
  of 5 of the 30 runs entered the target circle within 15 s (`rq_figure.py --metric
  distance`, panel C), because the closest brain is not always the fittest one.

## D13. Baselines and controls — 🟡

- **A standard EA** (`run.py --standard`, experiment 9): the same EA as one population of
  80 with 8 elites and no migration, so only the population structure differs. It answers
  a question the research question depends on: does the island model matter here at all?
  If it doesn't, the migration policies can hardly differ either, and another research
  question may be more interesting (mutation σ already showed a clear effect in the pilot,
  D8).

- **Random search** with the same evaluation budget: required by the spec, and nearly free
  to implement. Implemented as the same loop with every child drawn at random instead of
  bred (`run.py --algorithm random_search`). It keeps the same elites, so the best random
  networks found so far are carried along, just like the EA's.
- **No migration** (4 isolated islands): the control condition for the research question.
  Without it we could not tell whether migration matters at all.

## D14. Final evaluation — 🟡 re-test on unseen terrains

**For OlympicArena (experiment 14):** its rugged strip is random on every build (D2), so
the same test applies. `unseen.py` builds 20 fresh copies of each run's own world
(`results/terrains/<world>/test/<body>/`) and skips only SimpleFlatWorld, which never
changes. First result: experiment 18's best brain ends 0.53 m from the target on its own
arena and 1.24 ± 0.30 m on 20 fresh ones - it specialises to its strip, as the rugged
brains did. What follows describes the rugged runs (experiments 7-12).

The fitness logged during a run comes from the seed's training terrain. For the headline
numbers, also take each run's final best controller and evaluate it on **20 fresh terrains
it never saw** (a "test set"). Report the mean and spread per condition. This also answers
"did we get a brain that adapts to changing terrain?".

`unseen.py` does this; the 20 test terrains are generated once
(`results/terrains/rugged/test/`) and shared by every run. **First result (pilot winner):**
1.48 m on its training terrain against 1.92 ± 0.11 m on unseen terrain. The brains
specialise to their own terrain (see `experiments/README.md`, experiment 7).

---

## D15. Fitness function — ✅ (weights 🟡)

```
fitness = distance to the target at the end                     (walk there)
        + 0.5 × share of the run the core touches the ground     (stand)
        + 1.0 × share of the run the robot is upside down        (stay upright)
```

Lower is better. With several terrains (`--n-terrains`), it is averaged over them.
This is the base fitness of experiments 1-15. From experiment 16 on, gait terms (D18) and
a speed term (D20) are added, each behind its own flag.

**Why:** real spiders carry their body on their legs; they don't drag it. At rest,
spider_16's core lies on the ground (3 contact points, measured). So avoiding the ground
penalty requires the robot to push itself up. That rewards standing and walking
mechanics, not just any wriggle that happens to move the body.

- **Distance** is ARIEL's own `distance_to_target` (`ariel.simulation.tasks.targeted_locomotion`).
- **Ground contact:** at every network update (50 Hz) we check MuJoCo's contact list for
  a contact between the core and the terrain. The penalty is the share of updates where
  there is one.
- **Upside down:** the share of updates where the core's up-axis points below the
  horizon.

**Why not ARIEL's `fitness_survival_and_locomotion`?** It treats a core lower than
z = 0.05 m as fallen. That is an absolute height: on rugged terrain the ground itself
lies 0–0.5 m high, so the check is meaningless there.

**The weights set the exchange rate between posture and progress:**

| Behaviour over the whole run | Fitness |
|---|---|
| Lies still | 2.0 + 0.5 = **2.5** |
| Stands still | **2.0** |
| Drags its body 0.5 m forward | 1.5 + 0.5 = **2.0**: dragging 0.5 m is worth the same as standing still |
| Walks 0.5 m standing up | **1.5** |
| Flipped for the whole run | at least **3.0** |

**Reporting:** the headline result stays the plain **distance**, as the spec defines it.
We log distance, ground contact and upside-down separately (`log.csv`, and tags on every
individual in the database), so the report can show both.

## D16. Curriculum and early stopping — ❌ tried and rejected (flags, off by default)

Two ideas from the team to reach good walkers faster and more cheaply. Both are
implemented behind flags so they can be compared with the plain setup on the same seed
and terrain before we decide.

**Curriculum** (`--curriculum`), also called incremental evolution or fitness shaping
(e.g. Gomez & Miikkulainen 1997, *Incremental evolution of complex general behavior*):

```
fitness(g) = D15 fitness − w(g) × displacement
w(g) = 0.5 × (1 − g / 50)   for the first 50 generations, then 0
```

- *Displacement* is how far the robot moved from its spawn point, in any direction.
- Early generations are rewarded for **moving at all**, so they learn to take steps.
- By generation 50 only **moving toward the target** counts.
- Every generation, everyone alive (elites and immigrants too) is re-scored from their
  stored measurements with that generation's weight, so old and new fitness values are
  never mixed.
- The run's "best" is always judged by the final fitness, with no movement reward.

**Early stopping** (`--early-stop`): at `early_stop_time` (5 s), a walk that has closed
less than *p(g)* metres of the distance to the target is cut. Its measurements up to that
moment are what it gets.

- *p(g)* rises from 2 cm to 10 cm over the same 50 generations: lenient while robots are
  weak, stricter once they walk.
- This saves simulation time. It does not reduce the number of evaluations, so runs are
  compared both by evaluations and by wall-clock time (`compare.py`).

**Stopping at the target** (`--stop-at-target`): a walk ends once the core is within
10 cm of the target.

**Result** (`experiments/06_curriculum_vs_plain.sh`, one seed, same terrain and budget as
the plain long run):

- The curriculum run led early: 1.58 against 1.75 m at generation 20.
- It then stalled around 1.55 m from generation 40, while the plain run kept improving to
  1.46 m by generation 86.
- It was also *slower* per generation: the movement reward bred more active robots, which
  are costlier to simulate, and early stopping cut almost nothing.
- The run was stopped at generation 86.
- **Decision:** the plain setup is used; the options stay in the code behind flags as a
  documented negative result.

**Caveats for the report:**

- Shaping is not guaranteed to help. It helps most when the real objective gives no
  signal at first, and ours already rewards any progress.
- A badly chosen schedule can mislead evolution, for example robots that learn to move
  fast in circles.
- Early stopping can kill "late starters" (robots that first stand up, then walk).
- For the research question, the migration policies must be compared on the plain
  distance, which is logged in every run regardless of the flags.

## D17. Can it walk at all? Body physics, world difficulty and rhythm — ✅ spider_8, OlympicArena

**Why spider_16 only shuffles.** Every John Set motor is capped by ARIEL at **0.66 N·m**.

- spider_16 weighs 4 kg. With a foot 0.5 m out from the hip, holding the body up needs
  about 5 N·m per leg, 7–8× more than a motor gives.
- Measured: no static pose raises its core above resting height. The hip motors
  saturate at 0.66 N·m and the joints do not move (commanded −90°, actual 0°).
- Maximum *held* lift, found by hill-climbing over static poses:

| body | lift |
|---|---|
| centipede_5 | 9.1 cm |
| spider_8 | 6.1 cm |
| baby_a | 4.9 cm |
| gecko | 4.0 cm |
| spider_16 | 0.6 cm |

**Why rugged terrain makes it worse.** Along the path to the target:

| world | mean slope | height range |
|---|---|---|
| RuggedTerrainWorld | **25°** | 0.38 m |
| CraterTerrainWorld | 9° | 0.33 m |
| AmphitheatreTerrainWorld | 7° | 0.20 m |
| OlympicArena | 5° | 0.08 m |
| SimpleFlatWorld | 0° | 0 m |

**Why random brains barely move.** With default initialisation, the network's outputs
swing with a standard deviation of only about 0.25 of their range (about ±22° commanded;
measured on spider_8 and spider_16, 10 random networks each, flat ground). Evolution has
to discover rhythm before it can improve a gait.

**Rhythm options** (`genome.py`, both flags, off by default):

- `--clock-boost 3`: at initialisation only, the weights from the two clock inputs are
  multiplied by 3. The commanded swing roughly doubles (0.25 → 0.48 of the range, about
  ±22° → ±43°), while only 25% of outputs sit at their limits. We chose 3 from a sweep of
  1–6. Honest caveat: the joints themselves move about the same either way (standard
  deviation 16–17° of the actual angle), because the weak servos cannot follow the larger
  commands. The boost changes *what the network asks for* more than what the legs do.
  It also applies to every random-search sample, so random search keeps the same starting
  distribution as the EA.
- `--evolve-tempo`: one extra gene sets the clock's frequency, mapped onto 0.25–4 Hz
  (gene 0 = 1 Hz), and is mutated like any weight. Crossover takes it from either parent.
- **Not a CPG:** the clock is a fixed input signal; nothing oscillates inside the network,
  and there are no coupled oscillators. The spec itself says a controller needs
  "something to drive rhythmic movement with".
- **A grey area to argue in Methods:** evolving the tempo means evolving one parameter of
  an open-loop oscillator, which is also something a CPG does. The difference is that a
  CPG has one oscillator per joint, with evolved amplitudes, phases and couplings that
  produce the movement themselves. Here a single shared beat is only an input; the
  network decides every joint's movement. If the TA disagrees, `--evolve-tempo` is simply
  left off (the clock is then a fixed 1 Hz).

**Clock boost 3 or 1? (experiment 18, 2026-09-29).** With the gait terms on, the
boost-3 brains held their front and back legs folded: those outputs sat pinned at ±90°
50-61% of the walk, a third of all outputs overall, so only the side legs rowed. With
boost 1 (and 5 rays) the pinned share fell to 26%, the body rode 4.7 cm up instead of
3.1 cm (under 2 cm only 4% of the time, against 22%), and the robot got as far (0.53 m
from the target). One joint, the back leg's outer hinge, still sat folded (+86°, 92% of
the walk). Boost 3 mostly added saturation: it doubles what the network asks of the
joints, which the weak servos cannot follow anyway (see above).

**Worlds made of several pieces.** Most ARIEL worlds have one ground geom, named `floor`.
OlympicArena is built from a flat start box (named `floor`), an unnamed rugged heightfield
strip where the target lies, an incline and a finish. Our code first assumed the ground is
the `floor` geom only, so on OlympicArena the vision rays did not see the rugged strip and
lying on the strip was not penalised (found in code review, 2026-09-29, before any
OlympicArena result was used). `terrain.ground_geoms()` now takes every geom fixed to the
world that can collide. On the other worlds that is exactly the `floor` geom, and scores
there are bit-for-bit identical to before the change.

**Team decisions (2026-09-29): switch the body to spider_8**, the body that can lift
itself, **and the world to OlympicArena** (D2), after the walking pilot (experiment 13):
spider_8 reached the target on flat ground, got 0.55 m from it on OlympicArena, and stayed
stuck on rugged terrain. Both are free choices in the spec, but must stay fixed for the
whole assignment, so the research-question experiment is re-run on them (experiment 14).
**Team decision (2026-09-29): keep `--clock-boost 3`, drop `--evolve-tempo`.** The spec
asks for "a neural network whose weights you evolve" and rules out CPG-based controllers.
With the clock boost, every evolved number is still a weight of the network (only their
starting values change), so it is clearly within the rules. The tempo gene is an evolved
number that is not a weight, and evolving an oscillator's frequency is one ingredient of a
CPG; nothing forbids it, but it is not worth the risk:

- the flat run *without* any rhythm option reached the target too (experiment 13);
- the OlympicArena pilot had both options on, so there is no evidence the tempo gene
  helped there.

The clock therefore stays a fixed 1 Hz input. The tempo code stays behind its flag, off.

## D18. Gait terms in the fitness — ✅ in the final setup (experiments 16-18)

**Why.** The best brain of experiment 15 reaches 0.84 m from the target, but on video it
does not walk like a spider (measured over its 15 s walk):

- its core touches the ground 8% of the time, which the 0.5 contact weight prices at only
  0.04;
- the core rides on average 1.6 cm above lying height, although spider_8 can hold it
  about 6 cm up (D17): it crouches;
- one leg is tucked away: its two hinges sit at −85° and +85°, near their limits, and
  the network commands only 5–9° of movement there, against 17–77° for the other legs.

**The terms** (flags, all off by default; `simulate.py`):

- `--ground-contact-weight 1.0` (was 0.5): lying on the ground costs twice as much.
- `--low-body-weight 0.5`, the **carry term**: at every network update, how far the core
  is below "carried", `clip(1 − lift / 2 cm, 0, 1)`, averaged over the walk. `lift` is the
  core's height above lying height over the ground under it. 0 once the core is at least
  2 cm up, 1 lying down. Emre asked for it to be lenient: anything above 1–2 cm counts as
  carried, so the term only punishes crouching, not a low but working gait. Unlike the
  contact penalty, it keeps rewarding every millimetre up to the line, which gives
  evolution a gradient to follow.
- `--leg-imbalance-weight 0.5`, the **leg-balance term**: each leg's movement is the
  summed spread (standard deviation) of its hinges' angles over the walk. With four legs,
  each should do about a quarter. The term is `1 − least-used leg / mean leg`: 0 when all
  legs move alike, 1 when one never moves. The legs are found from the body tree
  (`bodies.limbs`): every branch hanging off the core is one limb.

Experiment 15's best brain would score 0.30 on the carry term and 0.34 on leg balance:
fitness 1.23 instead of 0.88. The distance still dominates; the terms decide between
walkers that get equally far.

**Round 1 result (experiment 16, `gait`).** The terms worked on posture: core on the
ground 8% → 0.5% of the time, carry term 0.30 → 0.13, leg imbalance 0.34 → 0.09. But the
robot ended further away (1.11 m against 0.84 m) and, on video, still barely stood, with
one leg doing little. Measured per leg (our own analysis script, contact forces and
actuator power from MuJoCo):

| leg | weight carried | net push towards the target | positive motor work |
|---|---|---|---|
| 0 | 30% | +0.4 | 40 J |
| 1 | 27% | −10.6 (brakes) | 39 J |
| 2 | 24% | +0.3 | 22 J |
| 3 | 19% | +13.3 | 32 J |

- Every leg carries 19-30% of the weight, so a load-sharing term (Emre's first idea)
  would not catch the lazy leg: it is a *prop* that holds the body up but does not drive.
- The joint-movement measure does not catch it either (0.09): the leg moves, but its
  motors do little.
- Motor work does: 22 J against up to 40 J here, and 8 J against about 27 J for the
  tucked leg of experiment 15.
- Pushing is lopsided (one leg drives, one brakes), but forcing every leg to push
  equally would prescribe an unnatural gait: in real four-legged gaits the front legs
  partly brake. The speed term already charges for legs fighting each other.
- The core rode 2.5 cm up on average (34% of the time under 2 cm, never above 4.7 cm).
  A heavier carry weight alone cannot make it stand taller, because the term stops
  rewarding at its line.

**Round 2 (experiment 17), the current proposal:**

- **leg term → motor-work share** (`--work-imbalance-weight 0.5`, movement term off):
  each leg's positive motor work (torque × joint speed when the motor drives the joint,
  sampled at every network update), and `1 − least-working leg / mean leg`. 0 when every
  leg works alike, 1 when one leg's motors are idle.
- **carry line 2 cm → 4 cm, weight 0.5 → 1.0** (`--carry-height 0.04 --low-body-weight
  1.0`): every millimetre of lift up to 4 cm now pays. spider_8 can hold 6.1 cm (D17), so
  4 cm is demanding but reachable.

**Caveats for the report.** This is behavioural shaping: we prescribe part of *how* to
walk, not only where to go. The spec allows our own metric if it is clearly defined.
A robot could game the leg term by jiggling a useless leg, so we check the videos. The
last time we added posture terms (experiment 3) they did not help, but that was on noisy
rugged terrain. Every condition of the research question uses the same fitness.

## D19. Stagnation rule for the mutation step — ❌ dropped after experiment 22 (used in 16-21)

**Why.** Experiment 15 stopped improving after about 9,000 evaluations and never moved
again (225 generations), with the genotype spread collapsed from 14 to about 1.5. The
data of experiment 8 show that the migration policy does not prevent this *inside* an
island: migrating the worst keeps the islands different from each other (spread 13.9 vs
2.8 when migrating the best), but within every island the spread ends at 2.4–3.0 whatever
the policy, and the final fitness is the same.

**Why not self-adaptive σ** (each individual carries its own σ, the textbook ES method,
Eiben & Smith ch. 4): a σ only spreads when a child *beats* its parents. On a plateau no
child ever does, so the σ values drift and die with their children; self-adaptation
cannot get an island off a plateau.

**First result (experiment 16, `gait_stall`, stopped at 9,400 evaluations):** by then
every island had doubled its σ up to the 0.4 maximum without improving, and the best
fitness (1.986) was no better than without the rule. One seed, and the fitness was being
revised at the time; experiment 17 tests it again with the revised fitness.

**The rule** (`--stall-generations 15`, off at 0; `ea.next_sigma`): per island, when its
best has not improved for 15 generations, its σ doubles (0.05 → 0.1 → 0.2 → 0.4 at most,
`--max-sigma`); it drops back to 0.05 as soon as the island improves. This is a
deterministic, feedback-based parameter control (Eiben & Smith ch. 8). The island's
elites are always kept, so a wide step cannot lose the best gait found so far: it only
lets the children search further away. The same rule runs in every condition; migration
can reset an island's σ by bringing in a better individual, which is part of what
migration does in this EA and belongs in the Discussion.

**Test on its own (experiment 22, 2026-09-30):** the rule had never been tested alone, and
it mixes with the research question (an immigrant that improves an island resets that
island's σ). Experiment 19b's crossover-0.9 runs (rule on) against the same runs with the
rule off, seeds 100-102, 6,000 evaluations. Rule fixed before the results: the stagnation
rule stays only if the runs with it beat the runs without it on all 3 seeds and on the mean.

| Seed | Rule on | Rule off | Island-generations with a widened σ (rule on) |
|---|---|---|---|
| 100 | 1.763 | **1.531** | 72 of 336, up to 0.2 |
| 101 | **0.921** | 1.091 | 32 of 336, up to 0.2 |
| 102 | 1.421 | 1.421 | 47 of 336, up to 0.4 |
| mean ± sd | 1.369 ± 0.423 | **1.348 ± 0.229** | |

Best fitness at 6,000 evaluations, lower is better. Runs below 1.6: 2 of 3 with the rule,
3 of 3 without. Mann-Whitney p = 1.0.

**Decision: the rule is dropped (`--stall-generations 0`, the default).**

- It helped on one seed (101: the run whose brain reaches the target), hurt on another
  (100), and did nothing on the third: on seed 102 σ widened to 0.4 on some islands, yet
  the run ended exactly where the run without the rule did, because no wider step found
  anything better.
- The means are the same within noise, so the rule fails the test fixed in advance, and
  the simpler EA wins the tie.
- Dropping it also takes it out of the research question: migration no longer acts
  through resetting σ, so a difference between policies is migration's alone.
- The code stays (`ea.next_sigma`, off at 0), so experiments 16-21 remain reproducible.

## D20. Rewarding speed — ✅ in the final setup (experiments 16-18)

**Why.** The fitness so far only looks at where the robot ends. A robot that closes 1 m
in 10 s and one that needs all 15 s score the same, and so do a robot that arrives after
8 s and one that arrives after 14 s. Emre wants the faster one to win in both cases.

**The term** (`--speed-weight 0.5 --stop-at-target`; `simulate.walk`): the distance to
the target averaged over every network update of the full 15 s walk.

- Closing distance early lowers the average: from 2 m, reaching 1 m at 10 s and holding
  averages 1.33 m; reaching 1 m only at 15 s averages 1.50 m.
- With `--stop-at-target` the walk ends when the core is within 0.1 m of the target, and
  the rest of the walk counts as distance 0: arriving at 8 s beats arriving at 14 s. This
  also saves simulation time once robots start arriving.
- Moving away from the target raises the average, so only speed in the right direction
  counts.

**Why not ARIEL's `fitness_speed_to_target`** (arrival time ÷ duration if the robot
arrives, else 1 + its closest distance): it rewards speed only after arrival, so the two
robots that each closed 1 m would still tie, and no robot reaches the OlympicArena target
yet. Among robots that do arrive, our term orders them the same way: earlier is better.

**Rejected: episode length growing with progress** (Emre's idea: 5, 10, 15, 20 s blocks
as the population gets better, to save budget early and favour fast brains). The research
question compares fitness-versus-evaluations curves; a longer episode makes every robot
score better at once, and with progress-triggered switches each condition would change
episode length at a different moment, so the curves could no longer be compared. It
would also force re-evaluating every elite at each switch, and very short walks favour
lunging over gaits (the same trap as the curriculum of D16). Episodes stay a fixed 15 s.

## D21. Longer walks late in the showcase run — ❌ tried once (experiment 20), did not reach the target

**Why.** Every brain so far stalls where its 15 s training walk ends: walked for 30 s,
experiment 18's best gets to 0.36 m from the target and stops at x = 1.7. The network has
no memory, so it reacts only to what it senses now, and it never trained on the ground
past where 15 s gets it. Emre wants one long run that reaches the target.

**The rule** (`--final-duration 30 --final-duration-from 6000`; `ea.Experiment.lengthen_walks`):
the run starts with 15 s walks, and from the first generation after 6,000 evaluations
every walk lasts 30 s. Starting at 15 s lets the run first learn to walk at half the cost
per evaluation (95% of experiment 15's progress came by 5,000 evaluations); the 30 s
phase then trains on the rest of the way.

At the switch, fitness values from the short walks stop being comparable with those from
the long ones, so everything that compares fitness across generations starts afresh:

- the elites walk again instead of keeping their scores;
- each island's stagnation record (D19) restarts, which also resets its σ;
- the run's best restarts. The best short-walk network is kept as
  `best_genotype_short.npy`, and `best_genotype.npy` becomes the best long-walk network.

The log's `duration` column records each generation's walk length, `plot.py` marks the
switch, and `replay.py` and `unseen.py` walk the brain for the length of the run's last
logged generation: the long walks only if the run got as far as the switch.

**Why this does not contradict D20**, which rejected walk lengths that grow with progress:

- D20 is about the research question. Experiment 14 keeps a fixed 15 s for every
  condition, and its curves stay comparable. This rule is used in one single-seed
  showcase run only.
- The switch happens at a fixed evaluation count, not when the population reaches some
  level, so it would happen at the same moment in every run anyway.
- The re-evaluation cost D20 names is paid once: the 8 elites walk again.
- The first phase is the proven 15 s walk, not a very short one that favours lunging.

**Result (experiment 20, seed 0, crossover 0.9 from D22):** 0.82 m from the target after
30 s, fitness 1.538; 1.11 ± 0.25 m on 20 fresh arenas. Worse than experiment 18's brain on
the same arena (0.53 m after 15 s, 0.36 m closest in a 30 s replay).

- The 15 s phase stalled at 1.10 m from about 2,500 evaluations to the switch; experiment
  18 was at 0.59 m by 6,000 on the same arena. The switch itself worked: the same brains
  scored 1.63 instead of 2.08 once they had 30 s.
- After about 8,700 evaluations the best never improved again (15,000 evaluations). σ sat
  at the 0.4 cap for the last 130 generations while the population mean got worse: once
  a run is stuck, the stagnation rule (D19) only adds noise.
- From about 2,000 evaluations all four islands had the same best: the best-emigrant
  policy copies the champion around the ring until every island holds it. That is the
  high selection pressure Cantú-Paz (2001) describes, and one reason the RQ compares
  policies.
- One run, so this does not show that crossover 0.9 or the switch is worse: the tuning
  runs differ by up to 0.9 between seeds with identical settings.

## D22. Parameter tuning — ✅ crossover probability 0.9, everything else kept (experiments 19 and 19b)

**What was tuned so far, and where:**

| Parameter | Value | How it was chosen |
|---|---|---|
| Mutation σ | 0.05 | Pilot (experiment 7): spider_16, rugged, old fitness, 1 seed |
| Network shape | 1 hidden layer of 16 | The same pilot |
| Crossover | neuron-level, p = 0.5 | Tested in experiment 12 (5 seeds) |
| Budget | 12,000 evaluations | From where the curves flatten (D12) |
| Population | 4 × 20 | Literature value, never tuned (D11) |
| Tournament size | 3 | Literature value, never tuned (D9) |
| Elites | 2 per island | Literature value, never tuned (D9) |
| Migration | 2 migrants every 10 generations | Literature value (D11) |

Since the pilot, the body, the world, the inputs and the fitness have all changed, so
even σ may no longer be right.

**Design** (`experiments/19_tuning.sh`): one factor at a time around the current values,
3 seeds each, 6,000 evaluations per run:

| Factor | Values tried | Current |
|---|---|---|
| Mutation σ | 0.02, 0.1 | 0.05 |
| Population | 4 × 10, 4 × 40 (elites and migrants scaled to keep their 10% share) | 4 × 20 |
| Tournament size | 2, 5 | 3 |
| Elites per island (part 2) | 1, 5 | 2 |
| Mutation spread (part 2) | 10% of the weights with σ = 0.16 (the same expected step) | every weight, σ = 0.05 |
| Crossover probability (part 2) | 0, 0.9 | 0.5 |

Part 2 (`experiments/19b_tuning_more.sh`) runs right after part 1 with the same seeds and
arenas, and is compared against part 1's `base`. The elite count is the survivor
selection: every generation each island keeps its best few unchanged and replaces the
rest with children. 5 elites of 20 comes close to a (μ + λ) scheme, where parents compete
with their children.

- **Seeds 100-102, not experiment 14's seeds 0-4.** The settings are then not tuned on
  the same arenas that answer the research question, the usual split between tuning and
  test instances.
- **The random emigrant policy** favours none of the policies the research question
  compares.
- **The migration interval and count are not tuned.** They are the context of the
  research question (which individuals migrate). They stay at the literature values
  (Cantú-Paz 2001) and are the same for every condition.
- **6,000 evaluations:** 95% of experiment 15's progress came by 5,000, and 6,000 is where
  the showcase run (D21) switches to long walks.

**Decision rule, fixed before the results:** a value replaces the current one only if its
best fitness at 6,000 evaluations beats `base` on all three seeds (paired: the same arena)
and on the mean. Otherwise the current value stays. With 3 seeds no test can reach
significance (the smallest possible Mann-Whitney p is 0.1), so the rule asks for a
consistent effect instead.

**Limits:** one factor at a time misses interactions between factors (a larger population
may want a different σ). A proper tuner (irace, SPOT, REVAC; Eiben & Smit 2011) would find
those, but needs far more runs than we can afford. It would also measure each setting on
noisy single runs just the same.

**Results** (best fitness at 6,000 evaluations, lower is better; `results/tuning/analysis/`):

| Setting | Seed 100 | Seed 101 | Seed 102 | Mean ± sd | Beats base on |
|---|---|---|---|---|---|
| base | 1.854 | 1.724 | 1.822 | 1.800 ± 0.068 | — |
| σ 0.02 | 2.063 | 1.905 | 1.696 | 1.888 ± 0.184 | 1 of 3 |
| σ 0.1 | 1.922 | 2.271 | 1.922 | 2.038 ± 0.201 | 0 of 3 |
| 4 × 10 | 2.429 | 1.352 | 2.278 | 2.020 ± 0.583 | 1 of 3 |
| 4 × 40 | 1.281 | 1.909 | 1.889 | 1.693 ± 0.357 | 1 of 3 |
| tournament 2 | 2.045 | 1.466 | 2.119 | 1.877 ± 0.358 | 1 of 3 |
| tournament 5 | 2.289 | 1.342 | 1.204 | 1.612 ± 0.591 | 2 of 3 |
| 1 elite | 1.920 | 1.728 | 1.725 | 1.791 ± 0.112 | 1 of 3 |
| 5 elites | 1.856 | 2.004 | 1.572 | 1.811 ± 0.220 | 1 of 3 |
| sparse mutation | 2.039 | 1.993 | 1.961 | 1.997 ± 0.039 | 0 of 3 |
| crossover 0 | 2.243 | 1.941 | 2.004 | 2.063 ± 0.160 | 0 of 3 |
| **crossover 0.9** | **1.763** | **0.921** | **1.421** | **1.369 ± 0.423** | **3 of 3** |

**Decision: crossover probability 0.9; every other setting keeps its value.**

- Crossover 0.9 is the only setting that passes the rule. It also got closest to the
  target on every seed (0.50 ± 0.33 m left against base's 0.81 ± 0.13 m), and 2 of its 3
  runs reached fitness 1.6, against none of base's.
- Crossover is also the factor with the clearest trend: the mean fitness improves from
  probability 0 (2.063) to 0.5 (1.800) to 0.9 (1.369). That fits D7: combining neurons from
  different parents helps on this task.
- Tournament 5 and 4 × 40 have better means than base but lose on at least one seed by a
  lot. Their spread (sd 0.59 and 0.36) is what the rule is there to catch.
- Nothing is significant (Friedman p = 0.24; every Mann-Whitney p ≥ 0.1 before Holm, 1.0
  after), as expected with 3 seeds. The claim for the report is "the only consistent
  improvement", not "a proven one".
- Only one factor changed, so the interactions that one-factor-at-a-time tuning misses
  cannot come from combining two winners.
- Experiment 20 uses it. Experiment 14 must use it too, so that the research question is
  answered with the tuned EA.

## D23. Train on several situations — ❌ tried in experiment 21, nothing adopted

**The problem.** After tuning, only 1 brain of those tested reaches the target (19b's
crossover-0.9 run on seed 101, at 16 s). Tests on the best brains (2026-09-30):

| Finding | Evidence |
|---|---|
| Robots get stuck; they do not run out of time | Legs keep moving as hard as while walking, but the body stays put: experiment 18's brain moved 5 mm in 8 s |
| Each brain only works on its own arena | The seed-0 brains walk about 0.7 m in 30 s on flat ground, against 1.2-1.7 m on their own arena; 1.1-1.2 m left on unseen arenas |
| Steering is unreliable | Started turned 30° or 60° on flat ground, only the best brain still gets within 0.15 m; others drift 0.2-0.9 m sideways and stop near x = 2 |
| 15 s is short | The one brain that arrives needs 16 s; 2 m in 15 s is faster than most gaits |

The common cause: every brain trains on exactly one situation, one arena and one start
pose, and evolution finds a trick for exactly that walk. Jakobi (1997) describes this
for the gap between simulation and reality; the remedy is to vary what the controller
may not rely on during evaluation.

**Options:**

| Option | Against | Cost per evaluation |
|---|---|---|
| (a) 3 arenas per brain, fitness averaged (`--n-terrains 3`) | memorising one arena, getting stuck | 3× |
| (b) Turned starts: one arena each at 0°, +30°, −30° (`--spawn-yaws 0,30,-30`) | not steering | nothing extra with (a) |
| (c) 20 s walks (`--duration 20`) | the time limit | 1.33× |
| (d) Restart islands that stop improving | runs that stall early | none, but it changes the island dynamics the research question studies |

**Chosen for the pilot: (a), (b) and (c)** (Emre, 2026-09-30). Not (d).

- **The 3 arenas are fixed for the whole run.** They are not new every generation as in
  experiment 1 (D10), so a brain's score means the same in every generation, elites keep
  their scores, and the stagnation rule (D19) still works.
- **Turned starts use ARIEL's own `spawn(rotation=...)` argument**, the public API, like
  the spawn position we already pass (Emre agreed, 2026-09-30). No ARIEL code changes. The
  turn is saved in the arena's file (`terrain1_yaw30.mjb`), so a turned arena never mixes
  with an unturned one.
- Arena 0 is unturned and is the same arena the other runs of that seed use.

**Design** (`experiments/21_robustness_pilot.sh`): 2 × 2 on seeds 100-102 and their
tuning arenas, 6,000 evaluations each; base is 19b's crossover-0.9 runs (1 arena, 15 s).

**Test:** every brain walks 20 s from each of 5 turns (0°, ±30°, and the untrained ±60°),
each turn on its own 20 unseen arenas: 100 walks, the same 100 for every brain.
`unseen.py --yaws ... --duration 20` writes them to `unseen_yaws0_30_-30_60_-60_20s.json`,
so they never replace the standard test (`unseen.json`) that `analyze.py` reads. All brains get the same 20 s at test time, so a 15 s
brain is not penalised for its shorter training walk. Each turn has different arenas, so a
per-turn comparison (e.g. ±60° against 0°) mixes the turn with the arenas' difficulty; the
decision uses the pooled share only.

**Decision rule, fixed before the results:** a change is adopted if it raises the share of
the 100 unseen walks that reach the target on all 3 seeds and on the mean. The unseen
distance is the tie-breaker.

**Cost and limits:**

- With 3 arenas one evaluation is 3 walks, so at the same number of evaluations these runs
  simulate 3× as much. If they win, some of that may be the extra simulation; the
  question here is whether training can make brains reliable at all.
- Experiment 14 would cost about 3× (4× with 20 s). At 6,000 instead of 12,000
  evaluations that is about 16-22 h instead of 11 h.

**Results** (`results/robustness/unseen.log`; 100 unseen walks per brain, 20 s each):

| Setting | Reached the target (seeds 100 / 101 / 102) | Mean distance left | Within 0.5 m |
|---|---|---|---|
| base (1 arena, 15 s) | 0% / 3% / 0% | 1.02 m | 13% |
| walk20 (1 arena, 20 s) | 0% / 0% / 0% | 1.35 m | 0% |
| arenas3 (3 turned arenas, 15 s) | 0% / 0% / 0% | 1.11 m | 4% |
| arenas3_walk20 (both) | 0% / 0% / 0% | 1.22 m | 0% |

**Decision: nothing is adopted; the setup stays 1 arena, 15 s (D22).** No setting beat base
on any seed, let alone all three.

- On unseen arenas no brain reaches the target reliably: 3 of 1,200 walks arrived, all
  from base's seed-101 brain. That brain is also the best generaliser (0.59 m median
  unseen, 37% of walks within 0.5 m) and the only one that reaches its own arena (0.10 m).
- Training on 3 turned arenas did make one brain nearly insensitive to the turn (arenas3
  seed 100: 1.04 / 1.01 / 1.06 m from 0°, ±30°, ±60°), but it walked worse overall. At the
  same number of evaluations, the harder task leaves brains less far along.
- 20 s training walks made nothing better, with 1 arena or with 3.
- arenas3_walk20's seed-101 brain failed 9 of its 100 unseen walks (the failed score:
  the simulation went unstable or the robot left the world).
- **For the report:** the gap between the training arena and unseen arenas is the main
  limitation of this controller (a reactive network without a CPG, D17) and budget.
  Remedies that help in the literature (Jakobi 1997) did not help at 6,000 evaluations;
  they may need a far larger budget than this assignment allows.

## References (to verify when writing the report)

- Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change: a
  tutorial for comparing multiple classifiers through Bayesian analysis. *Journal of
  Machine Learning Research*, 18(77), 1–36.
- Bodnar, C., Day, B., & Lió, P. (2020). Proximal distilled evolutionary reinforcement
  learning. *AAAI*, 34(04), 3283–3290.
- Cantú-Paz, E. (2001). Migration policies, selection pressure, and parallel evolutionary
  algorithms. *Journal of Heuristics*, 7(4), 311–334.
- Eiben, A. E., & Smit, S. K. (2011). Parameter tuning for configuring and analyzing
  evolutionary algorithms. *Swarm and Evolutionary Computation*, 1(1), 19–31.
- Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2nd ed.).
  Springer. (Chapters on island models, selection pressure and recombination.)
- Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and
  interval-schemata. *FOGA 2*, 187–202.
- Gomez, F., & Miikkulainen, R. (1997). Incremental evolution of complex general behavior.
  *Adaptive Behavior*, 5(3–4), 317–342.
- Hancock, P. J. B. (1992). Genetic algorithms and permutation problems: a comparison of
  recombination operators for neural net structure specification. *COGANN-92*, 108–122.
- Jakobi, N. (1997). Evolutionary robotics and the radical envelope-of-noise hypothesis.
  *Adaptive Behavior*, 6(2), 325–368.
- Jones, T. (1995). Crossover, macromutation, and population-based search. *ICGA-95*, 73–80.
- Montana, D. J., & Davis, L. (1989). Training feedforward neural networks using genetic
  algorithms. *IJCAI-89*, 762–767.
- Moriarty, D. E., & Miikkulainen, R. (1996). Efficient reinforcement learning through
  symbiotic evolution. *Machine Learning*, 22, 11–32.
- Schaffer, J. D., Whitley, D., & Eshelman, L. J. (1992). Combinations of genetic
  algorithms and neural networks: A survey of the state of the art. *COGANN-92*.
- Skolicki, Z., & De Jong, K. (2005). The influence of migration sizes and intervals on
  island models. *GECCO '05*, 1295–1302.
- Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting
  topologies. *Evolutionary Computation*, 10(2), 99–127.
- Uriot, T., & Izzo, D. (2020). Safe crossover of neural networks through neuron alignment.
  *GECCO '20*. doi:10.1145/3377930.3390197
- Vargha, A., & Delaney, H. D. (2000). A critique and improvement of the CL common
  language effect size statistics of McGraw and Wong. *Journal of Educational and
  Behavioral Statistics*, 25(2), 101–132.
