#!/usr/bin/env bash
# Experiment 8: THE research-question experiment (decisions D3, D11-D14).
#
# Five conditions x five seeds, all with the default settings (one hidden layer
# of 16, sigma 0.05, 4 islands x 20, 12,000 evaluations, 10 s episodes):
#
#   best / worst / random   the three emigrant-selection policies
#   none                    isolated islands (the control)
#   random_search           the baseline at the same budget
#
# Every condition with the same seed walks the same terrain
# (results/terrains/rugged/seed<S>/). About 17 minutes per run, ~7 hours in
# total on a 10-core Mac: run it overnight. Runs one after another on purpose -
# running them in parallel only makes each slower.
#
# Pass seeds as arguments to run a subset, e.g. `08_main_experiment.sh 0 1`.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then seeds=("$@"); else seeds=(0 1 2 3 4); fi

for policy in best worst random none; do
    uv run --project ../ariel python run.py --policy "$policy" --seeds "${seeds[@]}"
done
uv run --project ../ariel python run.py --algorithm random_search --seeds "${seeds[@]}"

# Robustness: every best network on the same 20 unseen terrains (D14).
uv run --project ../ariel python unseen.py \
    results/{best,worst,random,none,random_search}/seed*

# Figures, tables and statistics for the report.
uv run --project ../ariel python analyze.py \
    results/best results/worst results/random results/none results/random_search
