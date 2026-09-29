#!/usr/bin/env bash
# Experiment 5 (2026-09-28): a long single-seed run - bigger population, more
# generations.
#
# Question: does it keep improving with more budget?
# Result:   yes, up to a point: best fitness 1.84 -> 1.39 (closest 1.28 m from
#           the target) over 164 generations; improvement stopped around
#           generation 115 (about 18,000 evaluations). 77 minutes. The robot
#           shuffles rather than walks, about 0.7 m in 15 s.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --project ../ariel python run.py --policy best --seeds 0 \
    --world rugged --body spider_16 --duration 15 --hidden-layers 8 --mutation-sigma 0.1 \
    --island-size 40 --n-migrants 4 --max-evaluations 25000 \
    --out results/exp05_long_run
