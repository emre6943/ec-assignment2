#!/usr/bin/env bash
# Experiment 11 (2026-09-29): which John Set body can actually learn to move
# on rugged terrain?
#
# Why: spider_16 cannot lift its own body - every John Set motor is capped at
# 0.66 N·m, and no static pose raises its core above resting height (see
# bodies.py) - so it can only shuffle. A snake never needs to lift itself, and
# short-legged bodies might manage to.
# Setup: our default EA (islands, migrate best, 16 hidden, sigma 0.05, 10 s),
#        4,000 evaluations, seeds 0 and 1, for six bodies. The "core on the
#        ground" penalty is switched off for every body: lying on the ground
#        is how a snake moves, and spider_16 could not avoid it anyway. The
#        fitness is then the distance to the target plus the upside-down
#        penalty. About an hour.
# Note: a saved terrain includes the robot, so each body has its own terrains
#       (results/bodies/terrains/rugged/<body>/seed<S>/, and for spider_16
#       results/bodies/terrains/rugged/seed<S>/ - not the main experiment's
#       terrains); two seeds per body average out some of that terrain luck.
set -euo pipefail
cd "$(dirname "$0")/.."

for body in snake gecko spider_8 turtle linkin_modified spider_16; do
    uv run --project ../ariel python run.py --policy best --seeds 0 1 \
        --body "$body" --ground-contact-weight 0 --max-evaluations 4000 \
        --out "results/bodies/$body"
done

uv run --project ../ariel python analyze.py results/bodies/* \
    --reference spider_16 --out results/bodies/analysis
