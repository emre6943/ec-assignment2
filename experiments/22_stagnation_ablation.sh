#!/usr/bin/env bash
# Experiment 22 (2026-09-30): does the stagnation rule (D19) earn its place?
#
# Why: the rule was never tested on its own. Its one clean trial (experiment
# 16, one seed) showed no gain; experiment 17 changed it together with the
# fitness; in experiment 20 it held sigma at the 0.4 cap for 130 generations
# while the population got worse. It also mixes with the research question:
# an immigrant that improves an island resets that island's sigma, so part of
# what migration does would come from the rule.
# Design: experiment 19b's xover_0.9 runs (the final EA, rule on) against the
#         same runs with the rule off (--stall-generations 0): seeds 100-102,
#         the same arenas and spawn, the random policy, 6,000 evaluations.
#         About 20 minutes.
# Decision rule, fixed before the results (D22's rule, with the simpler EA as
# the default): the rule stays only if the runs with it beat the runs without
# it on all 3 seeds and on the mean best fitness at 6,000 evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/tuning
uv run --project ../ariel python run.py --policy random --seeds 100 101 102 \
    --workers 10 --skip-done \
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo \
    --vision-rays near --clock-boost 1 \
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04 \
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0 \
    --speed-weight 0.5 --stop-at-target --stall-generations 0 \
    --crossover-probability 0.9 --max-evaluations 6000 \
    --out "$out/xover_0.9_nostall" > "$out/xover_0.9_nostall.log" 2>&1

uv run --project ../ariel python analyze.py "$out/xover_0.9" "$out/xover_0.9_nostall" \
    --reference xover_0.9 --out "$out/analysis_stagnation"
