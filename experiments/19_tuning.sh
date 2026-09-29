#!/usr/bin/env bash
# Experiment 19 (2026-09-29): parameter tuning on the final setup.
#
# Why: the mutation step (D8) and network shape (D6) were tuned by a pilot on
# spider_16 and rugged terrain with the old fitness; the population size,
# tournament size and elite count were never tuned (literature values, D9 and
# D11). The body, world, inputs and fitness have all changed since.
# Design: one factor at a time around the current values (sigma 0.05, 4 x 20,
#         tournament 3), 3 seeds each:
#           base         the current values
#           sigma_0.02   --mutation-sigma 0.02
#           sigma_0.1    --mutation-sigma 0.1
#           pop_4x10     --island-size 10 (1 elite, 1 migrant: the same shares)
#           pop_4x40     --island-size 40 (4 elites, 4 migrants)
#           tour_2       --tournament-size 2
#           tour_5       --tournament-size 5
#         Seeds 100-102, not experiment 14's seeds 0-4, so the settings are
#         not tuned on the runs that answer the research question. The random
#         emigrant policy favours none of the research question's policies.
#         The migration interval and count stay at their literature values:
#         they set the context of the research question, so they are fixed,
#         not tuned. 6,000 evaluations (95% of the progress came by 5,000 in
#         experiment 15). 21 runs, about 4.5 hours.
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
    "base"
    "sigma_0.02 --mutation-sigma 0.02"
    "sigma_0.1 --mutation-sigma 0.1"
    "pop_4x10 --island-size 10 --n-elites 1 --n-migrants 1"
    "pop_4x40 --island-size 40 --n-elites 4 --n-migrants 4"
    "tour_2 --tournament-size 2"
    "tour_5 --tournament-size 5"
)
mkdir -p "$out"
for condition in "${conditions[@]}"; do
    read -r name flags <<< "$condition"
    # shellcheck disable=SC2086  # the flags are meant to split into words
    "${run[@]}" $flags --out "$out/$name" > "$out/$name.log" 2>&1
done

uv run --project ../ariel python analyze.py \
    "$out"/base "$out"/sigma_0.02 "$out"/sigma_0.1 "$out"/pop_4x10 \
    "$out"/pop_4x40 "$out"/tour_2 "$out"/tour_5 \
    --reference base --out "$out/analysis"
