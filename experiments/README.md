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
| 8 | `08_main_experiment.sh` | **The research question:** best vs worst vs random vs no migration, plus random search | *to run (overnight, ~7 h)* | D3 |

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
