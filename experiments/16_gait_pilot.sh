#!/usr/bin/env bash
# Experiment 16 (2026-09-29): do the gait terms and the stagnation rule help?
#
# Why: the best brain of experiment 15 crouched (core within 2 cm of the
# ground about 30% of the time), kept one leg tucked against its joint limits,
# and got stuck at 0.84 m from the target from evaluation ~9,000 on, with the
# population's diversity collapsed.
# Setup: seed 0 on the same arena as experiment 15 (its terrain is copied in),
#        the final setup (spider_8, OlympicArena, 15 s, --clock-boost 3), and:
#          gait        gait fitness (D18): contact penalty 1.0, carry-height
#                      term 0.5, leg-balance term 0.5; plus the speed term
#                      (D20): 0.5 x the walk-averaged distance to the target,
#                      and the walk ends on arrival; 12,000 evaluations
#          gait_stall  the same plus the stagnation rule (D19): an island's
#                      sigma doubles after 15 generations without progress;
#                      25,000 evaluations, to see whether it breaks a plateau
#        Compared with experiment 15 (old fitness, same seed and arena) on the
#        distance, the posture measurements and the videos. About 1 hour.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/gait_pilot
arena=results/terrains/olympic/spider_8/seed0/terrain0.mjb  # experiment 15's
mkdir -p "$out/terrains/olympic/spider_8/seed0"
cp -n "$arena" "$out/terrains/olympic/spider_8/seed0/terrain0.mjb"

run=(uv run --project ../ariel python run.py --policy best --seeds 0
    --world olympic --body spider_8 --duration 15 --clock-boost 3 --no-evolve-tempo
    --ground-contact-weight 1.0 --low-body-weight 0.5 --leg-imbalance-weight 0.5
    --speed-weight 0.5 --stop-at-target)
"${run[@]}" --max-evaluations 12000 --out "$out/gait"
"${run[@]}" --max-evaluations 25000 --stall-generations 15 --out "$out/gait_stall"

for name in gait gait_stall; do
    uv run --project ../ariel python plot.py "$out/$name/seed0"
    uv run --project ../ariel python replay.py "$out/$name/seed0"
done
