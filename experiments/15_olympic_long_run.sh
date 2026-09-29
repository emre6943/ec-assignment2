#!/usr/bin/env bash
# Experiment 15 (2026-09-29): how good does the final setup get with a big
# budget? Does spider_8 reach the target on OlympicArena?
#
# Why: the experiment 13 pilot got 0.55 m from the target in 8,000 evaluations
# and was still improving. This is one seed of the final setup (the same
# settings as experiment 14) with about 7x the budget, to see how far it goes
# and whether 12,000 evaluations cut the curves off early.
# Setup: seed 0, spider_8, OlympicArena, 15 s walks, --clock-boost 3, no tempo
#        gene, islands migrating the best, 80,000 evaluations. About 2.5-3 h.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/olympic_long
uv run --project ../ariel python run.py --policy best --seeds 0 \
    --world olympic --body spider_8 --duration 15 --clock-boost 3 --no-evolve-tempo \
    --max-evaluations 80000 --out "$out"

uv run --project ../ariel python plot.py "$out/seed0"
uv run --project ../ariel python replay.py "$out/seed0"
