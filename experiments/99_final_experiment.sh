#!/usr/bin/env bash
# Experiment 99: THE FINAL EXPERIMENT, the one the paper reports: the
# research question (D3), final version. Numbered 99 so it always sorts last.
# It is experiment 14 again, with two changes:
#
#   migrate every 20 generations, not 10 (D11). In experiment 26, every 20 had
#       the best final fitness (1.093 vs 1.130) and the best unseen distance
#       (0.76 vs 0.86 m) and learned as fast. It also keeps the islands apart:
#       one champion shared by all 4 islands in 6% of the generations, against
#       28% at every 10. Islands that hold the same individuals cannot show
#       whether it matters which ones migrate.
#   20 seeds, not 5, on fresh arenas (seeds 10-29), for firmer statistics.
#       With 5 seeds a paired Wilcoxon test can never reach 0.05;
#       with 20 every pair of conditions can be compared with a paired test.
#       The interval was chosen on experiment 14's seeds 0-4, so those seeds
#       and arenas are not reused: the same tuning/test split as D22.
#
# Everything else is experiment 14 (its header lists the setup): 6 conditions
# (best / worst / random / none / standard / random_search) x 20 seeds = 120
# runs of 12,000 evaluations. The paper's settings are the flags in `run=`
# below. Many differ from the code's defaults, which are the early
# experiments' values (see ea.EAConfig and simulate.SimConfig).
#
# How to run, from the assignment2 folder:
#   bash experiments/99_final_experiment.sh          # all 20 seeds (10-29)
# WORKERS sets the number of parallel walks (default: every CPU). Pass seeds
# as arguments to run a subset, e.g. `99_final_experiment.sh 10 11`.
#
# The paper's runs took 8.4 hours on two machines in parallel, split by seed:
# seeds 20-29 on an Apple M3 Pro (ARM), seeds 10-19 on an Intel i7-12700H
# (x86-64), about 7-8 minutes a run. MuJoCo's results differ slightly between
# CPU architectures, so a run repeats exactly only on the kind of machine it
# ran on. The analysis steps need every seed in one results/final/ folder.
#
# Unseen test (D14): unseen.py reuses results/terrains/olympic/test/spider_8/
# when it exists. On another machine, copy that folder over first, so these
# brains face the same 20 unseen arenas as every earlier brain.
#
# Re-running skips finished runs and resumes a cut-off one from its last saved
# generation (--resume). The runs go seed by seed, all six conditions of a
# seed before the next, so a stop leaves at most one seed incomplete. With
# seeds as arguments the unseen test covers only those seeds, but the analysis
# steps read every seed*/ under results/final/. analyze.py and end_results.py
# keep only the seeds complete in all the conditions they compare;
# probabilities.py pairs two conditions on the seeds both have.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then seeds=("$@"); else seeds=($(seq 10 29)); fi
workers=${WORKERS:-$(python3 -c "import os; print(os.cpu_count())")}
# OUT=results/rerun writes a fresh copy elsewhere (copy results/final/terrains
# there first to walk the same arenas).
out=${OUT:-results/final}
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
# The exploratory end-result analysis (D26), added after the results were known.
"${py[@]}" end_results.py "$out"/{best,worst,random,none,standard} \
    --out "$out/end_results.md"
