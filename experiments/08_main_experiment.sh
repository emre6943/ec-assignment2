#!/usr/bin/env bash
# Experiment 8: THE research-question experiment (decisions D3, D11-D14).
#
# Six conditions x five seeds, all with the then-default settings (spider_16 on
# rugged terrain, one hidden layer of 16, sigma 0.05, 12,000 evaluations, 10 s
# episodes, 80 individuals). The final setup is experiment 14.
#
#   best / worst / random   the three emigrant-selection policies (4 islands x 20)
#   none                    isolated islands (the control)
#   standard                one population of 80, no islands (experiment 9)
#   random_search           the baseline at the same budget
#
# Every condition with the same seed walks the same terrain
# (results/terrains/rugged/seed<S>/). About 14-17 minutes per run, ~7-8 hours
# in total on a 10-core Mac: run it overnight. Runs one after another on
# purpose - running them in parallel only makes each slower. Runs that already
# finished with exactly these settings (e.g. from experiment 9) are skipped.
#
# Pass seeds as arguments to run a subset, e.g. `08_main_experiment.sh 0 1`.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then seeds=("$@"); else seeds=(0 1 2 3 4); fi
# The settings this experiment ran with (the defaults have moved on since).
run=(uv run --project ../ariel python run.py --skip-done --seeds "${seeds[@]}"
    --world rugged --body spider_16 --duration 10)

for policy in best worst random none; do
    "${run[@]}" --policy "$policy"
done
"${run[@]}" --standard
"${run[@]}" --algorithm random_search

conditions=(results/best results/worst results/random results/none results/standard
    results/random_search)

# Robustness: every best network on the same 20 unseen terrains (D14).
runs=()
for condition in "${conditions[@]}"; do runs+=("$condition"/seed*); done
uv run --project ../ariel python unseen.py "${runs[@]}"

# Figures, tables and statistics for the report: the migration policies against
# the no-migration control, and all set-ups against the standard EA.
uv run --project ../ariel python analyze.py "${conditions[@]}" \
    --reference none --out results/analysis
uv run --project ../ariel python analyze.py "${conditions[@]}" \
    --reference standard --out results/analysis_vs_standard
