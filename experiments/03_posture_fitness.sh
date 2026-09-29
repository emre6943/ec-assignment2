#!/usr/bin/env bash
# Experiment 3 (2026-09-28): posture-aware fitness (D15), terrain still changing
# every generation.
#
# Question: does rewarding "core off the ground, not upside down" help?
# Result:   not on its own - the terrain noise still dominated (fitness flat
#           around 1.8). The rugged bumps already hold the body up most of the
#           time, so the ground penalty barely bit.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --project ../ariel python run.py --policy best --seeds 0 \
    --world rugged --body spider_16 --terrain-mode per_generation --duration 15 \
    --hidden-layers 8 --mutation-sigma 0.1 --max-evaluations 3000 \
    --out results/exp03_posture_fitness
