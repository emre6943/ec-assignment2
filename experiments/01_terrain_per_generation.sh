#!/usr/bin/env bash
# Experiment 1 (2026-09-28): a fresh rugged terrain every generation.
#
# Question: does the EA learn when every generation walks new random terrain?
# Result:   no. All islands' scores rose and fell together with the terrain's
#           difficulty; the best robot wiggled in place. Led to decision D10.
#
# Closest reproduction with the current code. The original run used the plain
# distance as fitness (the posture weights below switch those terms off) and an
# earlier 5-ray vision layout, so the numbers will not match exactly.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --project ../ariel python run.py --policy best --seeds 0 \
    --world rugged --body spider_16 --terrain-mode per_generation --duration 15 \
    --hidden-layers 8 --mutation-sigma 0.1 --max-evaluations 3000 \
    --ground-contact-weight 0 --upside-down-weight 0 \
    --out results/exp01_terrain_per_generation
