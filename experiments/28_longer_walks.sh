#!/usr/bin/env bash
# Step 28 (2026-10-01): trained on 15 s walks, tested with more time.
#
# Why (D12): only 3 of experiment 14's 25 EA brains reach the target within the
# 15 s they were trained on, but a first check found 11 arriving within 20 s.
# No new evolution: every run's best brain (experiments 14 and 26) walks its
# own arena for 15, 20, 30 and 60 s, ending on arrival, and the 20 unseen
# arenas for 30 s. The training length stays 15 s; this is a test of what the
# evolved brains can do with more time.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/olympic
conditions=("$out"/{best,worst,random,none,standard,random_search})
interval=("$out"/{best_int5,best,best_int20,best_int50,none})

runs=()
for condition in "${conditions[@]}" "$out"/best_int{5,20,50}; do
    runs+=("$condition"/seed*)
done
uv run --project ../ariel python unseen.py "${runs[@]}" --duration 30 \
    > "$out/unseen_30s.log" 2>&1
uv run --project ../ariel python longer_walks.py "${conditions[@]}" \
    --out "$out/longer_walks"
uv run --project ../ariel python longer_walks.py "${interval[@]}" \
    --out "$out/longer_walks_interval"
uv run --project ../ariel python compute_ledger.py
