#!/usr/bin/env bash
# Experiment 29 (2026-10-01): should the brain also know where it is?
#
# Why (D5): the brain is told where the target is relative to itself
# (distance and bearing in its own frame), never its absolute position. The
# target never moves from (2, 0), so that vector already fixes the robot's
# position; absolute (x, y) adds only the world heading, which walking to the
# target does not need. But (x, y) does let a brain tie actions to places on
# its one training arena ("a bump at x = 0.8: lift the legs here").
# Hypothesis, fixed before the results: with (x, y) the training fitness is
# about the same or better, and the brains do worse on unseen arenas (they
# memorise their own arena; the same worry as the vision rays, D23).
# Design: experiment 14 again with `--position`: the core's world (x, y),
#         divided by 2, as 2 extra inputs (18 inputs, 228 weights instead of
#         212). Every EA condition of 14 (best, worst, random, none,
#         standard; not random search, the baseline), the same 5 seeds and
#         arenas, so each run pairs with its experiment-14 twin that differs
#         only in the 2 inputs. 25 runs, about 3.5 hours.
# Decision rule, fixed before the results: D5 gains the position inputs only
#         if, over the 25 pairs, the position brains end closer to the target
#         on the unseen arenas with P >= 0.9 (Bayesian paired t-test,
#         `probabilities.py`) and are not worse on training fitness.
#         Experiment 14 is not rerun either way (deadline): a gain would be
#         reported as a finding, not folded into the final EA.
#
# Pass seeds as arguments to run a subset, e.g. `29_absolute_position.sh 0 1`.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then seeds=("$@"); else seeds=(0 1 2 3 4); fi
out=results/olympic  # experiment 14's folder: its arenas and twins
run=(uv run --project ../ariel python run.py --skip-done --seeds "${seeds[@]}"
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo
    --no-vision --clock-boost 1 --hidden-layers 8,4
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target
    --crossover-probability 0.9 --ariel-spawn --max-evaluations 12000
    --position)

for policy in best worst random none; do
    "${run[@]}" --policy "$policy" --out "$out/pos_$policy" > "$out/pos_$policy.log" 2>&1
done
"${run[@]}" --standard --out "$out/pos_standard" > "$out/pos_standard.log" 2>&1

conditions=(best worst random none standard)
runs=()
for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do runs+=("$out/pos_$condition/seed$seed"); done
done
uv run --project ../ariel python unseen.py "${runs[@]}" > "$out/position_unseen.log" 2>&1

# Each condition against its experiment-14 twin, paired by seed.
for condition in "${conditions[@]}"; do
    uv run --project ../ariel python probabilities.py \
        "$out/$condition" "$out/pos_$condition" \
        --out "$out/probabilities_position_$condition.md"
done
# The position conditions among themselves, as experiment 14 did.
uv run --project ../ariel python analyze.py "${conditions[@]/#/$out/pos_}" \
    --reference pos_none --out "$out/analysis_position"
uv run --project ../ariel python compute_ledger.py
