#!/usr/bin/env bash
# Experiment 10 (2026-09-28): one long run with the best setup, for the best
# walker we can get - not part of the research-question comparison.
#
# Setup: islands migrating the best (as good as the standard EA on average and
#        the most consistent, experiment 9), 4 x 50 = 200 individuals, 120,000
#        evaluations (~600 generations), 15 s episodes so reaching the target
#        is physically possible, walks end early once the target is reached,
#        16 hidden neurons, sigma 0.05 (the pilot winner). Seed 0, the same
#        terrain as the other seed-0 runs. About 6 hours.
# The best network is saved on every improvement, so the run can be stopped
# at any time without losing it.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --project ../ariel python run.py --policy best --seeds 0 \
    --world rugged --body spider_16 \
    --island-size 50 --n-migrants 5 --max-evaluations 120000 \
    --duration 15 --stop-at-target \
    --out results/overnight
