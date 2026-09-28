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
| D1 | Body | ✅ | `john_set.spider_16` |
| D2 | World | ✅ | `RuggedTerrainWorld`, ARIEL defaults; one random terrain per seed (D10) |
| D2a | Spawn height on rugged terrain | 🟡 **awaiting OK** | Spawn above the ground, `correct_collision_with_floor=False` |
| D2b | Terrain bump height | ✅ | ARIEL's default; nothing in ARIEL changed or re-implemented |
| D3 | Research question | ✅ (wording 🟡) | Effect of the emigrant-selection policy on convergence speed |
| D4 | Controller outputs | 🟡 | One output per hinge (16), direct position control |
| D5 | Controller inputs | ✅ | 34: joint angles + clock + target vector + tilt + 10 vision rays |
| D6 | Network shape | ✅ (pilot) | Fixed MLP, one hidden layer of 16; evolve weights only (832 weights) |
| D7 | Crossover | 🟡 | Neuron-level uniform crossover |
| D8 | Mutation | ✅ (pilot) | Gaussian perturbation of every weight, σ = 0.05 |
| D9 | Selection | 🟡 | Tournament (parents) + generational with elitism (survivors) |
| D10 | Terrain and noisy fitness | ✅ | One fixed terrain per seed, shared by all conditions with that seed |
| D11 | Island model settings | 🟡 | 4 islands, ring, migrate every 10 generations, replace worst |
| D12 | Budget and stopping | ✅ rule / 🟡 size | Fixed budget of 12,000 evaluations per run (proposed); 10 s episodes |
| D13 | Baselines and controls | 🟡 | Random search + no-migration islands |
| D14 | Final evaluation | 🟡 | Best controllers re-tested on unseen terrains |
| D15 | Fitness function | ✅ (weights 🟡) | Distance + penalties for the core touching the ground and for being upside down |
| D16 | Curriculum and early stopping | ❌ rejected | Lost to the plain setup on one seed; kept behind flags, off by default |

---

## D1. Body — ✅ `spider_16`

16 hinges: 4 legs, 4 hinges each. It is symmetric, so it can walk in any direction, which
suits a target-reaching task.

Pilot on flat ground (200 evaluations of a simple sine-gait hill-climber, 5 seeds each,
remaining distance to a target 2 m away; lower is better):

| body | mean | std | cost per evaluation |
|---|---|---|---|
| spider_8 | 1.39 | 0.18 | 1× |
| **spider_16** | **1.38** | **0.12** | ~2× |
| spider_12 | 1.64 | 0.10 | ~1.7× |

spider_16 was as good as spider_8 and more consistent, at twice the cost per evaluation.

## D2. World — ✅ `RuggedTerrainWorld`, ARIEL defaults

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

## D2a. Spawn height on rugged terrain — 🟡 (a bug fix in our code; awaiting the team's OK)

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
- **Recommendation: keep it**, and state it in Methods with the evidence above. Awaiting the team's OK.

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
> evolving a neural-network controller for `spider_16` on rugged terrain?

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
   fitness drops below a fixed threshold (set from the pilot; `analyze.py` uses 1.6 by
   default, which the pilot's better settings reached within about 1,400 evaluations).
2. **Area under the best-fitness curve**, which rewards being good early.
3. **Final fitness** at the budget cap, on the training terrain; robustness is reported
   separately as the distance reached on unseen terrains (D14).

**Statistics** (`analyze.py`), on the final fitness and the AUC:

- **A Friedman test across all five conditions, blocked by seed.** Conditions with the
  same seed walk the same terrain (D10), so each seed is a matched block. This asks: do
  the conditions differ at all?
- **Planned comparisons of each condition against `none`,** the no-migration control:
  two-sided Mann-Whitney U tests with Holm's correction.
- **Why not all pairs?** With 5 seeds per condition, the smallest possible Mann-Whitney
  p-value is 0.008. Correcting for all 10 pairs would multiply it past 0.05, so nothing
  could ever come out significant. Four planned comparisons can.
- A paired Wilcoxon test is no use either: its smallest possible p with 5 pairs is 0.06.
- **More seeds** would allow more comparisons, if compute allows.

**Only the emigrant selection changes between conditions.** Island count, island size,
migration interval, number of migrants, topology and replacement policy stay fixed (D11).
Varying any of them as well would be a second research question.

## D4. Controller outputs — 🟡 one per hinge, direct position control

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

## D5. Controller inputs — ✅ 34 inputs including vision (`sensors.py`)

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

- We use MuJoCo's `mj_rayHfield`, which intersects a ray with the terrain only. In our
  own code, the rays are computed 50 times a second, when the network is queried.
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

## D6. Network shape — ✅ one hidden layer of 16, chosen by a pilot

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

## D7. Crossover — 🟡 neuron-level uniform crossover

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

Crossover probability: 0.5. The other half of the children are a copy of one parent. Every
child is then mutated (D8). These values are identical across all research-question
conditions.

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

- The first run with seed *s* generates a plain `RuggedTerrainWorld()` terrain and saves
  it to `results/terrains/rugged/seed<s>/terrain0.mjb`.
- Every other condition with seed *s* (best, worst, random, none, random search) loads
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

**Episode length: 10 s** (team decision, 2026-09-28; the template uses 15 s). No robot
comes close to reaching the target in 15 s, and a shorter episode cuts the simulation cost
by a third. The fitness is the distance at the end, so robots simply cover less ground;
that is the same for every condition.

**Measured cost** on this Mac (M3 Pro, 10 worker processes, vision on):

| Setup | Speed | One run |
|---|---|---|
| 15 s episodes, 4 × 20, 3,000 evaluations | ~10 evaluations/s | ~5 min |
| 15 s episodes, 4 × 40, 25,000 evaluations | ~5.4 evaluations/s (walkers get costlier) | 77 min |
| 10 s episodes, 4 × 20, 4,000 evaluations (pilot) | ~7.5 evaluations/s (while other work ran) | ~9 min |

**Proposed budget for the main experiment: 12,000 evaluations per run** (4 × 20
individuals, about 166 generations, about 17 minutes per run).

- The long run (experiment 5) kept improving until about 18,000 evaluations, with twice
  the population and 15 s episodes.
- 12,000 is the compromise that fits 5 conditions × 5 seeds into one night (about 7 h).
- If the curves are still clearly falling at 12,000, say so in the Discussion, or re-run
  with a larger budget if time allows.

*Conditions* = best, worst, random, none, and random search.

## D13. Baselines and controls — 🟡

- **Random search** with the same evaluation budget: required by the spec, and nearly free
  to implement. Implemented as the same loop with every child drawn at random instead of
  bred (`run.py --algorithm random_search`). It keeps the same elites, so the best random
  networks found so far are carried along, just like the EA's.
- **No migration** (4 isolated islands): the control condition for the research question.
  Without it we could not tell whether migration matters at all.

## D14. Final evaluation — 🟡 re-test on unseen terrains

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

## References (to verify when writing the report)

- Cantú-Paz, E. (2001). Migration policies, selection pressure, and parallel evolutionary
  algorithms. *Journal of Heuristics*, 7(4), 311–334.
- Gomez, F., & Miikkulainen, R. (1997). Incremental evolution of complex general behavior.
  *Adaptive Behavior*, 5(3–4), 317–342.
- Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2nd ed.).
  Springer. (Chapters on island models, selection pressure and recombination.)
- Jakobi, N. (1997). Evolutionary robotics and the radical envelope-of-noise hypothesis.
  *Adaptive Behavior*, 6(2), 325–368.
- Schaffer, J. D., Whitley, D., & Eshelman, L. J. (1992). Combinations of genetic
  algorithms and neural networks: A survey of the state of the art. *COGANN-92*.
- Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting
  topologies. *Evolutionary Computation*, 10(2), 99–127.
