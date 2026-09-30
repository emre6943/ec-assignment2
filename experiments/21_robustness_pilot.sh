#!/usr/bin/env bash
# Experiment 21 (2026-09-30): can training make brains reach the target reliably?
#
# Why (D23): only 1 of the brains so far reaches the target. The rest get
# stuck (legs moving, body still), walk only on their own arena (the seed-0
# brains manage 0.7 m in 30 s on flat ground), steer unreliably from a turned
# start, and 15 s is short: the one that arrives needs 16 s. Every brain
# trains on one situation - one arena, one start pose - and evolution finds
# a trick for exactly that walk (Jakobi 1997).
# Design: 2 x 2 around the tuned setup (D22: crossover 0.9), 3 seeds each:
#           base             1 arena, 15 s: experiment 19b's xover_0.9 runs
#           walk20           1 arena, 20 s
#           arenas3          3 arenas, the robot turned 0, +30, -30 degrees
#                            on them, fitness averaged, 15 s
#           arenas3_walk20   both
#         Seeds 100-102 and experiment 19's arenas: a run here takes its
#         arenas from results/robustness/terrains, a link to
#         results/tuning/terrains, so arena 0 of each seed is base's arena.
#         6,000 evaluations each, as in tuning; with 3 arenas an evaluation
#         is 3 walks. About 2 hours.
# Test: every brain walks 20 s on 20 unseen arenas from each of 5 turns (0,
#       +-30 and the untrained +-60): 100 walks. The share that reaches the
#       target decides (D23).
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/robustness
mkdir -p "$out"
[ -e "$out/terrains" ] || ln -s ../tuning/terrains "$out/terrains"

run=(uv run --project ../ariel python run.py --policy random --seeds 100 101 102
    --workers 10 --skip-done
    --world olympic --body spider_8 --no-evolve-tempo
    --vision-rays near --clock-boost 1
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target --stall-generations 15
    --crossover-probability 0.9 --max-evaluations 6000)

conditions=(
    "walk20 --duration 20"
    "arenas3 --duration 15 --n-terrains 3 --spawn-yaws 0,30,-30"
    "arenas3_walk20 --duration 20 --n-terrains 3 --spawn-yaws 0,30,-30"
)
for condition in "${conditions[@]}"; do
    read -r name flags <<< "$condition"
    # shellcheck disable=SC2086  # the flags are meant to split into words
    "${run[@]}" $flags --out "$out/$name" > "$out/$name.log" 2>&1
done

brains=(results/tuning/xover_0.9/seed10{0,1,2} "$out"/{walk20,arenas3,arenas3_walk20}/seed10{0,1,2})
uv run --project ../ariel python unseen.py "${brains[@]}" \
    --yaws 0 30 -30 60 -60 --duration 20 > "$out/unseen.log" 2>&1
