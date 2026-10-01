#!/usr/bin/env bash
# Experiment 25 (2026-09-30): is our neuron-level crossover the right one?
#
# Why (D7): most children are worse than their better parent for every
# operator, mutation included; the question is which operator makes the rare
# better child most often. Three alternatives, chosen by Emre after a
# literature check:
#   h8_4_weight     --crossover weight     each weight from either parent: does
#                                          keeping neurons whole matter?
#   h8_4_blx        --crossover blx        BLX-0.5, a blend whose spread shrinks
#                                          as an island converges
#   h8_4_headless   --crossover headless   our crossover with a random genotype
#                                          as the second parent (Jones 1995): is
#                                          crossover only a big mutation?
# Design: on the brain chosen by experiment 24 (16-8-4-8, D6), against its
#         runs there (neuron crossover); seeds 100-102 and their arenas, the
#         random policy, crossover probability 0.9, 6,000 evaluations. About
#         45 minutes, then the standard unseen test for all 12 brains.
# Decision rule, fixed before the results (D7): an operator replaces ours only
# if it beats it on all 3 seeds and on the mean best fitness at 6,000
# evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/tuning
run=(uv run --project ../ariel python run.py --policy random --seeds 100 101 102
    --workers 10 --skip-done
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo
    --no-vision --clock-boost 1 --hidden-layers 8,4
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target
    --crossover-probability 0.9 --max-evaluations 6000)

for kind in weight blx headless; do
    "${run[@]}" --crossover "$kind" --out "$out/h8_4_$kind" \
        > "$out/h8_4_$kind.log" 2>&1
done

operators=(h8_4 h8_4_weight h8_4_blx h8_4_headless)
brains=()
for name in "${operators[@]}"; do brains+=("$out/$name"/seed10{0,1,2}); done
uv run --project ../ariel python unseen.py "${brains[@]}" > "$out/crossover_unseen.log" 2>&1
uv run --project ../ariel python analyze.py "${operators[@]/#/$out/}" \
    --reference h8_4 --out "$out/analysis_crossover_ops"
uv run --project ../ariel python compute_ledger.py
