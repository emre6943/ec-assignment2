#!/usr/bin/env bash
# Experiment 20 (2026-09-29): one long run that should reach the target.
#
# Why: every brain so far stalls where its 15 s training walk ends (D21).
# Setup: experiment 18's best setup (5 rays, clock boost 1, gait and speed
#        terms, stagnation rule) on seed 0 and experiment 15's arena. 15 s
#        walks until 6,000 evaluations, then 30 s walks (D21), 24,000
#        evaluations in all. About 1.5 hours.
#        Tuning (experiments 19 and 19b, D22) changed one setting: crossover
#        probability 0.9 instead of 0.5. Everything else kept its value.
set -euo pipefail
cd "$(dirname "$0")/.."

# A run in results/long_walk takes seed 0's arena from results/terrains/:
# experiment 15's.
out=results/long_walk
mkdir -p "$out"

uv run --project ../ariel python run.py --policy best --seeds 0 --workers 10 \
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo \
    --vision-rays near --clock-boost 1 \
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04 \
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0 \
    --speed-weight 0.5 --stop-at-target --stall-generations 15 \
    --crossover-probability 0.9 \
    --final-duration 30 --final-duration-from 6000 \
    --max-evaluations 24000 --out "$out" > "$out/run.log" 2>&1

uv run --project ../ariel python plot.py "$out/seed0"
uv run --project ../ariel python replay.py "$out/seed0"
uv run --project ../ariel python unseen.py "$out/seed0"
