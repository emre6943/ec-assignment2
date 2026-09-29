#!/usr/bin/env bash
# Experiment 18 (2026-09-29): fewer vision rays, and a gentler rhythmic start.
#
# Why: experiment 17's best brain walked with its front and back legs folded:
# their outputs sat pinned at +-90 degrees 50-61% of the walk (a third of all
# outputs overall), so only the side legs rowed and the body stayed low. The
# clock boost doubles what the network asks of the joints but barely changes
# what they do (D17), so it may mostly add this saturation. And of the 10
# vision rays, the up ray never fired and the four shallow ones mostly
# measured the arena's edges; only the down ray and the four steep ones saw
# the ground under and just ahead of the robot.
# Setup: experiment 17's setup (seed 0, experiment 15's arena, gait and speed
#        terms, stagnation rule, 12,000 evaluations) with
#          rays5         --vision-rays near (5 rays, 488 weights), clock boost 3
#          rays5_boost1  the same with --clock-boost 1
#        Both run at once with 5 workers each. About 50 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/inputs_pilot
arena=results/terrains/olympic/spider_8/seed0/terrain0.mjb  # experiment 15's
mkdir -p "$out/terrains/olympic/spider_8/seed0"
cp -n "$arena" "$out/terrains/olympic/spider_8/seed0/terrain0.mjb"

run=(uv run --project ../ariel python run.py --policy best --seeds 0 --workers 5
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target --stall-generations 15
    --max-evaluations 12000 --vision-rays near)
"${run[@]}" --clock-boost 3 --out "$out/rays5" > "$out/rays5.log" 2>&1 &
"${run[@]}" --clock-boost 1 --out "$out/rays5_boost1" > "$out/rays5_boost1.log" 2>&1 &
wait

for name in rays5 rays5_boost1; do
    uv run --project ../ariel python plot.py "$out/$name/seed0"
    uv run --project ../ariel python replay.py "$out/$name/seed0"
done
