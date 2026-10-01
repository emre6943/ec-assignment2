#!/usr/bin/env bash
# Experiment 24 (2026-09-30): which brain shape learns to walk best?
#
# Why (D6): one hidden layer of 16 was chosen by the experiment 7 pilot, on
# spider_16, rugged terrain, the old fitness and one seed. Since then the
# body, world, inputs (16, no vision: D5) and fitness have all changed.
# Design: the final EA (experiment 23's rays0 runs: 16 inputs -> 16 -> 8
#         outputs, 408 weights) against five other shapes:
#           h8       16-8-8        208 weights   narrower
#           h32      16-32-8       808 weights   wider
#           h8_8     16-8-8-8      280 weights   deeper, small
#           h16_16   16-16-16-8    680 weights   deeper, current width
#           h8_4     16-8-4-8      212 weights   a bottleneck of 4
#         Seeds 100-102 and their arenas, the random policy, 6,000
#         evaluations. About 2 hours, then the standard unseen test (20
#         arenas, 15 s) for all 18 brains.
# Decision rule, fixed before the results (D22's rule): a shape replaces
# 16-16-8 only if its best fitness at 6,000 evaluations beats 16-16-8 on all
# 3 seeds and on the mean; if several do, the one with the lowest mean.
# Known bias: sigma 0.05 was tuned for 16-16-8, and a bigger network gets more
# total mutation noise per child, which slightly favours smaller networks.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/tuning
run=(uv run --project ../ariel python run.py --policy random --seeds 100 101 102
    --workers 10 --skip-done
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo
    --no-vision --clock-boost 1
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target
    --crossover-probability 0.9 --max-evaluations 6000)

conditions=(
    "h8 --hidden-layers 8"
    "h32 --hidden-layers 32"
    "h8_8 --hidden-layers 8,8"
    "h16_16 --hidden-layers 16,16"
    "h8_4 --hidden-layers 8,4"
)
for condition in "${conditions[@]}"; do
    read -r name flags <<< "$condition"
    # shellcheck disable=SC2086  # the flags are meant to split into words
    "${run[@]}" $flags --out "$out/$name" > "$out/$name.log" 2>&1
done

shapes=(rays0 h8 h32 h8_8 h16_16 h8_4)
brains=()
for shape in "${shapes[@]}"; do brains+=("$out/$shape"/seed10{0,1,2}); done
uv run --project ../ariel python unseen.py "${brains[@]}" > "$out/shape_unseen.log" 2>&1
uv run --project ../ariel python analyze.py "${shapes[@]/#/$out/}" \
    --reference rays0 --out "$out/analysis_shape"
