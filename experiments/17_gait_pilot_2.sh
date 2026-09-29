#!/usr/bin/env bash
# Experiment 17 (2026-09-29): the revised gait fitness, with the stagnation rule.
#
# Why: in experiment 16 the gait terms made spider_8 carry its body (contact
# 8% -> 0.5%) and move every leg, but it stood only 2.5 cm up on average and
# one leg acted as a prop: it carried weight (every leg 19-30% of the load) but
# its motors did the least work (22 J against up to 40 J) and it pushed
# nothing. The joint-movement term cannot see that. Changes (D18):
#   - leg term: the share of the motor work each leg does (replaces movement)
#   - carry line: 4 cm instead of 2 cm (spider_8 can hold 6.1 cm), weight 1.0
#   - the stagnation rule (D19) on, as it will be in the final setup
# Setup: seed 0, experiment 15's arena, spider_8, OlympicArena, 15 s,
#        --clock-boost 3, speed term 0.5 with stop at target (D20), 12,000
#        evaluations. About 15 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/gait_pilot_2
arena=results/terrains/olympic/spider_8/seed0/terrain0.mjb  # experiment 15's
mkdir -p "$out/terrains/olympic/spider_8/seed0"
cp -n "$arena" "$out/terrains/olympic/spider_8/seed0/terrain0.mjb"

uv run --project ../ariel python run.py --policy best --seeds 0 \
    --world olympic --body spider_8 --duration 15 --clock-boost 3 --no-evolve-tempo \
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04 \
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0 \
    --speed-weight 0.5 --stop-at-target --stall-generations 15 \
    --max-evaluations 12000 --out "$out/gait"

uv run --project ../ariel python plot.py "$out/gait/seed0"
uv run --project ../ariel python replay.py "$out/gait/seed0"
