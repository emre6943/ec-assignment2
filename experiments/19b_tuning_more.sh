#!/usr/bin/env bash
# Experiment 19b (2026-09-29): tuning, part 2 - survivor selection and variation.
#
# Why: part 1 (19_tuning.sh) covers sigma, the population size and the
# tournament size. This part covers the rest of the loop (D22):
#   elites_1     --n-elites 1   only the island's best survives (5%)
#   elites_5     --n-elites 5   the best quarter survives (25%): closer to a
#                               (mu + lambda) scheme where parents compete
#                               with their children
#   sparse_mut   --mutation-rate 0.1 --mutation-sigma 0.16: a tenth of the
#                weights change, by more; sqrt(0.1) x 0.16 = 0.05, so the
#                expected step is the same size as base's
#   xover_0      --crossover-probability 0   (mutation only, as in exp 12)
#   xover_0.9    --crossover-probability 0.9
# Everything else as part 1: seeds 100-102 and their arenas, the random
# policy, 6,000 evaluations, compared against part 1's base. 15 runs, about
# 2 hours.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/tuning
run=(uv run --project ../ariel python run.py --policy random --seeds 100 101 102
    --workers 10 --skip-done
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo
    --vision-rays near --clock-boost 1
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target --stall-generations 15
    --max-evaluations 6000)

conditions=(
    "elites_1 --n-elites 1"
    "elites_5 --n-elites 5"
    "sparse_mut --mutation-rate 0.1 --mutation-sigma 0.16"
    "xover_0 --crossover-probability 0"
    "xover_0.9 --crossover-probability 0.9"
)
mkdir -p "$out"
for condition in "${conditions[@]}"; do
    read -r name flags <<< "$condition"
    # shellcheck disable=SC2086  # the flags are meant to split into words
    "${run[@]}" $flags --out "$out/$name" > "$out/$name.log" 2>&1
done

uv run --project ../ariel python analyze.py \
    "$out"/base "$out"/sigma_0.02 "$out"/sigma_0.1 "$out"/pop_4x10 \
    "$out"/pop_4x40 "$out"/tour_2 "$out"/tour_5 "$out"/elites_1 \
    "$out"/elites_5 "$out"/sparse_mut "$out"/xover_0 "$out"/xover_0.9 \
    --reference base --out "$out/analysis"
