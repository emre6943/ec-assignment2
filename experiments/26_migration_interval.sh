#!/usr/bin/env bash
# Experiment 26 (2026-09-30): how often should the islands migrate?
#
# Why (D11): migrating rarely keeps the islands apart, so each can explore its
# own solutions; migrating often spreads good ones fast but can pull every
# island into the same local optimum. With the best policy every 10
# generations, all 4 islands held the same champion within about 2,000
# evaluations (experiment 20). The interval was deliberately not tuned (D22):
# it sets the context of the research question. This is a follow-up
# question, run after experiment 14 on the same setup, seeds and arenas.
# Design: the best-emigrant policy (it spreads champions fastest, so it is the
#         most sensitive to the interval), migrating every 5, 20 and 50
#         generations. Experiment 14 supplies every 10 (`best`) and never
#         (`none`). 12,000 evaluations are about 166 generations, so 50 means
#         only 3 migrations. 15 runs, about 2.5 hours.
# Analysis (exploratory, no setting is changed by it): convergence and final
# fitness across the five intervals, and how different the islands stay.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then seeds=("$@"); else seeds=(0 1 2 3 4); fi
out=results/olympic
run=(uv run --project ../ariel python run.py --skip-done --seeds "${seeds[@]}"
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo
    --no-vision --clock-boost 1 --hidden-layers 8,4
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target
    --crossover-probability 0.9 --ariel-spawn --max-evaluations 12000
    --policy best)

for interval in 5 20 50; do
    "${run[@]}" --migration-interval "$interval" --out "$out/best_int$interval" \
        > "$out/best_int$interval.log" 2>&1
done

runs=()
for interval in 5 20 50; do
    for seed in "${seeds[@]}"; do runs+=("$out/best_int$interval/seed$seed"); done
done
uv run --project ../ariel python unseen.py "${runs[@]}" > "$out/interval_unseen.log" 2>&1
uv run --project ../ariel python analyze.py "$out"/{best_int5,best,best_int20,best_int50,none} \
    --reference best --out "$out/analysis_interval"
uv run --project ../ariel python compute_ledger.py
