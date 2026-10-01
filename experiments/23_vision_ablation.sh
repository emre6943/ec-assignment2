#!/usr/bin/env bash
# Experiment 23 (2026-09-30): fewer vision rays - 5, 3, 1 or none?
#
# Why (D5): stuck robots mostly do not touch the ground; their legs keep
# stepping, but the brain switches to a step pattern that does not push. The
# rays let a brain react to its own arena's exact bumps, which may cause both
# that switching and the overfitting to one arena (D23). Fewer inputs also
# mean fewer weights to search: 488 (5 rays), 456 (3), 424 (1), 408 (none).
# Design: the final EA (experiment 22's xover_0.9_nostall runs: 5 near rays,
#         crossover 0.9, no stagnation rule) against the same with
#           rays3   --vision-rays near3   down + the 45 degree rays ahead and
#                                         behind along the core's +x axis
#           rays1   --vision-rays down    down only: height above the ground
#           rays0   --no-vision           no rays at all
#         Seeds 100-102 and their arenas, the random policy, 6,000
#         evaluations. About 50 minutes, then the standard unseen test (20
#         arenas, 15 s) for all 12 brains.
# Decision rule, fixed before the results: a smaller set replaces the 5 rays
# only if its brains end closer to the target on the unseen arenas than the
# 5-ray brains on all 3 seeds and on the mean; if several pass, the one with
# the lowest mean. The training fitness at 6,000 evaluations is reported too.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/tuning
run=(uv run --project ../ariel python run.py --policy random --seeds 100 101 102
    --workers 10 --skip-done
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo
    --clock-boost 1
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target
    --crossover-probability 0.9 --max-evaluations 6000)

conditions=(
    "rays3 --vision-rays near3"
    "rays1 --vision-rays down"
    "rays0 --no-vision"
)
for condition in "${conditions[@]}"; do
    read -r name flags <<< "$condition"
    # shellcheck disable=SC2086  # the flags are meant to split into words
    "${run[@]}" $flags --out "$out/$name" > "$out/$name.log" 2>&1
done

brains=("$out"/{xover_0.9_nostall,rays3,rays1,rays0}/seed10{0,1,2})
uv run --project ../ariel python unseen.py "${brains[@]}" > "$out/vision_unseen.log" 2>&1
uv run --project ../ariel python analyze.py "$out"/{xover_0.9_nostall,rays3,rays1,rays0} \
    --reference xover_0.9_nostall --out "$out/analysis_vision"
