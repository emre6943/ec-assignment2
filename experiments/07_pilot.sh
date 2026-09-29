#!/usr/bin/env bash
# Experiment 7 (2026-09-28): tuning pilot for the brain and the mutation step
# size (decisions D6 and D8).
#
# Question: which network shape and mutation sigma learn best?
# Setup:    3 shapes x 3 sigmas, one seed (same terrain for all), 4 x 20
#           individuals, 4,000 evaluations, 10 s episodes. ~50 minutes.
# Result:   sigma 0.2 was worst for every shape; 0.05 and 0.1 were close; a
#           second hidden layer did not help. Best: one layer of 16, sigma 0.05
#           (fitness 1.51). These became the defaults. One seed only, so small
#           differences are within noise.
set -euo pipefail
cd "$(dirname "$0")/.."

for hidden in 8 16 8,8; do
    for sigma in 0.05 0.1 0.2; do
        name="h${hidden/,/x}_s${sigma}"
        echo "=== $name"
        uv run --project ../ariel python run.py --policy best --seeds 0 \
            --world rugged --body spider_16 --duration 10 \
            --hidden-layers "$hidden" --mutation-sigma "$sigma" \
            --max-evaluations 4000 --out "results/pilot/$name"
    done
done

# One table ranking all nine. No figure: the chart only has colours for the
# five standard conditions plus three others, and never cycles colours.
uv run --project ../ariel python analyze.py results/pilot/h* \
    --threshold 1.7 --out results/pilot/analysis
