#!/usr/bin/env bash
# Step 27 (2026-09-30): how sure are we? No new runs: statistics for the
# research question (experiment 14) and the migration-interval follow-up (26).
#
# For every pair of conditions, paired by seed: the mean difference with its
# 95% interval, the probability that one is truly better (Bayesian paired
# t-test, flat prior; Benavoli et al. 2017), split into better / practically
# equal (within 0.05) / worse, the A12 effect size (Vargha & Delaney 2000) and
# the seeds won. On the best fitness at the budget, the AUC and the unseen
# distance. Complements analyze.py's significance tests (D3).
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/olympic
uv run --project ../ariel python probabilities.py \
    "$out"/{best,worst,random,none,standard,random_search} \
    --out "$out/probabilities.md"
uv run --project ../ariel python probabilities.py \
    "$out"/{best_int5,best,best_int20,best_int50,none} \
    --out "$out/probabilities_interval.md"
