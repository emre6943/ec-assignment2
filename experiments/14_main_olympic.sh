#!/usr/bin/env bash
# Experiment 14: THE research-question experiment on the final setup (D1, D2, D3).
#
# spider_8 on OlympicArena (decisions D1, D2, D17), 15 s walks, a rhythmic
# start (--clock-boost 3; the tempo gene is left out, see D17), one hidden
# layer of 16, sigma 0.05, 12,000 evaluations, 80 individuals.
# Six conditions x five seeds:
#
#   best / worst / random   the three emigrant-selection policies (4 islands x 20)
#   none                    isolated islands (the control)
#   standard                one population of 80, no islands
#   random_search           the baseline at the same budget
#
# OlympicArena's rugged strip is random on every build, so each seed gets one
# arena (results/olympic/terrains/olympic/spider_8/seed<S>/), shared by all six
# conditions. Seed 0's arena is the one experiment 15 used (copied in before
# this ran), so 15 is a long continuation of this experiment's seed 0.
# About 20 minutes per run, ~10 hours in total on a 10-core Mac. Runs that
# already finished with exactly these settings are skipped, so an interrupted
# run can simply be started again.
#
# Pass seeds as arguments to run a subset, e.g. `14_main_olympic.sh 0 1`.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then seeds=("$@"); else seeds=(0 1 2 3 4); fi
out=results/olympic
run=(uv run --project ../ariel python run.py --skip-done --seeds "${seeds[@]}"
    --world olympic --body spider_8 --duration 15 --clock-boost 3 --no-evolve-tempo)

for policy in best worst random none; do
    "${run[@]}" --policy "$policy" --out "$out/$policy"
done
"${run[@]}" --standard --out "$out/standard"
"${run[@]}" --algorithm random_search --out "$out/random_search"

conditions=("$out/best" "$out/worst" "$out/random" "$out/none" "$out/standard"
    "$out/random_search")

# Figures, tables and statistics for the report: the migration policies against
# the no-migration control, and all set-ups against the standard EA. The speed
# threshold is 0.8: the experiment 13 pilot crossed it mid-budget (0.84 after
# 3,000 evaluations, 0.76 after 5,000), so it separates fast runs from slow ones.
# (That pilot had the tempo gene on and its own random strip: a rough guide.)
uv run --project ../ariel python analyze.py "${conditions[@]}" \
    --threshold 0.8 --reference none --out "$out/analysis"
uv run --project ../ariel python analyze.py "${conditions[@]}" \
    --threshold 0.8 --reference standard \
    --out "$out/analysis_vs_standard"
