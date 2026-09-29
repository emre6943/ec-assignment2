#!/usr/bin/env bash
# Experiment 6 (2026-09-28): curriculum + early stopping (decision D16) against
# the plain long run (experiment 5), same seed, terrain and budget.
#
# Question: does a fading "move at all" reward plus cutting hopeless walks
#           reach good walkers faster or more cheaply?
# Result:   no. It led early (1.58 vs 1.75 m at generation 20) but stalled
#           around 1.55 m from generation 40, while the plain run reached
#           1.46 m by generation 86. It was also slower per generation, and
#           early stopping cut almost nothing. Stopped at generation 86; the
#           options stay behind flags, off by default.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --project ../ariel python run.py --policy best --seeds 0 \
    --world rugged --body spider_16 --duration 15 --hidden-layers 8 --mutation-sigma 0.1 \
    --island-size 40 --n-migrants 4 --max-evaluations 25000 \
    --curriculum --early-stop \
    --out results/exp06_curriculum

# Overlay both on the plain distance, by evaluations and by wall-clock time:
uv run --project ../ariel python compare.py \
    results/exp05_long_run/seed0 results/exp06_curriculum/seed0 \
    --out results/exp06_comparison.png
