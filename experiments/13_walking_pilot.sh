#!/usr/bin/env bash
# Experiment 13 (2026-09-29): can spider_8 learn to WALK - judged by video?
#
# Why: spider_16 cannot lift its body (motors capped at 0.66 N·m, experiment
#      11 notes); spider_8 can hold its core 6.1 cm up. Random networks barely
#      oscillate, so the "rhythm" options (decision D17) start every network
#      with a clear rhythm (clock inputs x3 at init) and let evolution set the
#      clock's tempo (a tempo gene, 0.25-4 Hz).
# Setup: spider_8, seed 0, 15 s walks (enough time to reach the target 2 m
#        away), islands migrating the best, 8,000 evaluations each:
#          flat_plain    flat world, no rhythm options (the reference)
#          flat_rhythm   flat world, rhythm options on
#          olympic       ARIEL's OlympicArena (gentle, 5 degree slopes), rhythm on
#          rugged        RuggedTerrainWorld (25 degree slopes), rhythm on
#        About 75 minutes. Then look at the videos, not just the numbers.
set -euo pipefail
cd "$(dirname "$0")/.."

common=(--policy best --seeds 0 --body spider_8 --duration 15 --max-evaluations 8000)
rhythm=(--clock-boost 3 --evolve-tempo)
run=(uv run --project ../ariel python run.py)

"${run[@]}" "${common[@]}" --clock-boost 1 --no-evolve-tempo --world flat \
    --out results/walking/flat_plain
"${run[@]}" "${common[@]}" "${rhythm[@]}" --world flat --out results/walking/flat_rhythm
"${run[@]}" "${common[@]}" "${rhythm[@]}" --world olympic --out results/walking/olympic
"${run[@]}" "${common[@]}" "${rhythm[@]}" --world rugged --out results/walking/rugged

for run_dir in results/walking/*/seed0; do
    uv run --project ../ariel python plot.py "$run_dir"
    uv run --project ../ariel python replay.py "$run_dir"
done
