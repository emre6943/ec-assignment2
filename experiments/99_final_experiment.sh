#!/usr/bin/env bash
# Experiment 99: THE FINAL EXPERIMENT (prepared 2026-10-02, to run on
# emre-server). The research question (D3), final version; the paper reports
# this, not experiment 14. Numbered 99 so it always sorts last. Experiment 14
# again, with two changes:
#
#   migrate every 20 generations, not 10 (D11). In experiment 26, every 20 had
#       the best final fitness (1.093 vs 1.130) and the best unseen distance
#       (0.76 vs 0.86 m) and learned as fast. It also keeps the islands apart:
#       one champion shared by all 4 islands in 6% of the generations, against
#       28% at every 10. Islands that hold the same individuals cannot show
#       whether it matters which ones migrate.
#   20 seeds, not 5, on fresh arenas (seeds 10-29). The TAs asked for firmer
#       statistics. With 5 seeds a paired Wilcoxon test can never reach 0.05;
#       with 20 every pair of conditions can be compared with a paired test.
#       The interval was chosen on experiment 14's seeds 0-4, so those seeds
#       and arenas are not reused: the same tuning/test split as D22.
#
# Everything else is experiment 14 (its header lists the setup): 6 conditions
# (best / worst / random / none / standard / random_search) x 20 seeds = 120
# runs of 12,000 evaluations. The runs go seed by seed, so a run stopped early
# still leaves complete paired blocks of all six conditions. About 9 minutes
# a run, 17 hours in all, on emre-server (measured 2026-10-02: 14 and 20
# workers are equally fast there, and two runs side by side gain nothing).
# WORKERS sets the pool size (default: every CPU).
#
# On emre-server, as a user service that survives logging out (from the
# assignment2 folder; `systemctl --user stop ec-final` stops it, starting it
# again resumes):
#   systemd-run --user --unit=ec-final --working-directory="$PWD" \
#       --setenv=PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
#       bash -c 'bash experiments/99_final_experiment.sh > results/final_driver.log 2>&1'
#
# Unseen test (D14): unseen.py reuses results/terrains/olympic/test/spider_8/
# when it exists. On another machine, copy that folder from the Mac first, so
# these brains face the same 20 unseen arenas as every earlier brain.
#
# Re-running skips finished runs and resumes a cut-off one from its last saved
# generation (--resume). Pass seeds as arguments to run a subset, e.g.
# `99_final_experiment.sh 10 11`; the analysis then covers those seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then seeds=("$@"); else seeds=($(seq 10 29)); fi
workers=${WORKERS:-$(python3 -c "import os; print(os.cpu_count())")}
out=results/final
mkdir -p "$out"
run=(uv run --project ../ariel python run.py --skip-done --resume --workers "$workers"
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo
    --no-vision --clock-boost 1 --hidden-layers 8,4
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target
    --crossover-probability 0.9 --ariel-spawn --max-evaluations 12000
    --migration-interval 20)

for seed in "${seeds[@]}"; do
    for policy in best worst random none; do
        "${run[@]}" --seeds "$seed" --policy "$policy" --out "$out/$policy" \
            >> "$out/$policy.log" 2>&1
    done
    "${run[@]}" --seeds "$seed" --standard --out "$out/standard" \
        >> "$out/standard.log" 2>&1
    "${run[@]}" --seeds "$seed" --algorithm random_search --out "$out/random_search" \
        >> "$out/random_search.log" 2>&1
done

conditions=("$out/best" "$out/worst" "$out/random" "$out/none" "$out/standard"
    "$out/random_search")
runs=()
for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do runs+=("$condition/seed$seed"); done
done

# Every run's best brain on the 20 unseen arenas, at 15 s (D14) and 30 s (as 28).
py=(uv run --project ../ariel python)
"${py[@]}" unseen.py "${runs[@]}" --workers "$workers" > "$out/unseen.log" 2>&1
"${py[@]}" unseen.py "${runs[@]}" --workers "$workers" --duration 30 \
    > "$out/unseen_30s.log" 2>&1

# Figures, tables and statistics, as for experiment 14 (and 27, 28).
"${py[@]}" analyze.py "${conditions[@]}" --reference none --out "$out/analysis"
"${py[@]}" analyze.py "${conditions[@]}" --reference standard \
    --out "$out/analysis_vs_standard"
"${py[@]}" probabilities.py "${conditions[@]}" --out "$out/probabilities.md"
"${py[@]}" longer_walks.py "${conditions[@]}" --workers "$workers" \
    --out "$out/longer_walks"
"${py[@]}" compute_ledger.py
