#!/usr/bin/env bash
# Experiment 2 (2026-09-28): the same EA on ARIEL's flat world - a debug check.
#
# Question: is the EA itself broken, or is the rugged terrain the problem?
# Result:   the EA works. On flat ground it improved steadily from 1.84 m to
#           1.02 m and the robot walked about 1 m; the same network moved only
#           about 0.1 m on rugged terrain. See decisions D2b.
#
# The flat world is for debugging only; the experiment world is rugged.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --project ../ariel python run.py --world flat --policy best --seeds 0 \
    --terrain-mode per_generation --duration 15 \
    --hidden-layers 8 --mutation-sigma 0.1 --max-evaluations 3000 \
    --ground-contact-weight 0 --upside-down-weight 0 \
    --out results/exp02_flat_world_debug
