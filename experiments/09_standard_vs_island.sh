#!/usr/bin/env bash
# Experiment 9 (2026-09-28): is the island model any better than a standard EA?
#
# Question: before committing the research question to migration policies, does
#           splitting the population into islands matter at all here?
# Setup:    the then-default settings (spider_16, rugged, 16 hidden, sigma 0.05,
#           12,000 evaluations, 10 s episodes) for three set-ups, seeds 0-2, same terrain per seed:
#             standard  one population of 80, 8 elites, no migration
#             best      4 islands x 20, migrating the best (the default)
#             none      4 islands x 20, never migrating
# If islands make no difference, migration policies will barely differ either,
# and another research question may be the better choice.
#
# The `best` and `none` runs double as seeds 0-2 of the main experiment
# (08_main_experiment.sh), so they are not wasted. ~2.5 hours; ordered by seed
# so a first comparison is available after the first three runs (~50 minutes).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then seeds=("$@"); else seeds=(0 1 2); fi

for seed in "${seeds[@]}"; do
    for condition in --standard "--policy best" "--policy none"; do
        # shellcheck disable=SC2086  # $condition is one or two words
        uv run --project ../ariel python run.py $condition --seeds "$seed" \
            --world rugged --body spider_16 --duration 10
    done
done

uv run --project ../ariel python analyze.py results/standard results/best results/none \
    --reference standard --out results/analysis_standard_vs_island
