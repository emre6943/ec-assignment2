#!/usr/bin/env bash
# Experiment 12 (2026-09-29): does crossover help at all?
#
# Question: every child so far had a 50% chance of neuron-level crossover
#           (decision D7). Would mutation alone do as well or better?
# Setup:    exactly the main experiment's `best` condition (spider_16, islands
#           migrating the best, 16 hidden, sigma 0.05, 12,000 evaluations, 10 s)
#           but with crossover probability 0, seeds 0-4. Same seeds means same
#           terrains, so the comparison with results/best/ (crossover 0.5) is
#           paired. About 75 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --project ../ariel python run.py --policy best --seeds 0 1 2 3 4 \
    --crossover-probability 0 --out results/no_crossover

uv run --project ../ariel python analyze.py results/best results/no_crossover \
    --reference best --out results/analysis_crossover
