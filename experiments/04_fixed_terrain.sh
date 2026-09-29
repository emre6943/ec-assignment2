#!/usr/bin/env bash
# Experiment 4 (2026-09-28): one fixed terrain per seed (decision D10).
#
# Question: with the terrain noise gone, does the EA learn on rugged ground?
# Result:   yes, but slowly: best fitness 1.86 -> 1.75 in 41 generations, and
#           the robots learned to stand (core on the ground 35% -> 5%).
#           Migration visibly synchronised the islands.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --project ../ariel python run.py --policy best --seeds 0 \
    --world rugged --body spider_16 --duration 15 --hidden-layers 8 --mutation-sigma 0.1 \
    --max-evaluations 3000 \
    --out results/exp04_fixed_terrain
