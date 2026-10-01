#!/usr/bin/env bash
# Experiment 14: THE research-question experiment on the final setup (D3).
#
# The final EA, and the decision or experiment behind each part:
#   body and world   spider_8 on OlympicArena, target 2 m ahead (D1, D2); the
#                    robot placed by ARIEL's own spawn, as in the template (D2a)
#   walk             15 s, ended early on arrival (D12, D20)
#   brain            16 inputs (8 joint angles, a 1 Hz clock as sin/cos, the
#                    target's distance and direction, body tilt) -> 8 tanh ->
#                    4 tanh -> 8 motor targets: 212 weights (D5, D6,
#                    experiment 24); no vision rays (experiment 23); clock
#                    boost 1 (D17, experiment 18)
#   fitness          distance left + 0.5 x average distance during the walk
#                    + 1.0 x core on the ground + 1.0 x upside down + 1.0 x
#                    core below 4 cm + 0.5 x uneven work between legs
#                    (D15, D18-D20, experiments 16-17)
#   EA               4 islands x 20; per island and generation 2 elites and
#                    18 children; tournament of 3; neuron-level crossover with
#                    probability 0.9 (D7, D22, experiment 25); Gaussian
#                    mutation sigma 0.05 on every weight (D8), kept fixed: the
#                    stagnation rule was dropped after experiment 22 (D19);
#                    2 migrants every 10 generations around a ring (D11);
#                    12,000 evaluations (D12)
#
# Six conditions x five seeds:
#
#   best / worst / random   the three emigrant-selection policies (4 islands x 20)
#   none                    isolated islands (the control)
#   standard                one population of 80 with 8 elites, no islands
#   random_search           the baseline at the same budget
#
# OlympicArena's rugged strip is random on every build, so each seed gets one
# arena (results/olympic/terrains/olympic/spider_8/seed<S>/
# terrain0_arielspawn.mjb), shared by all six conditions (D10).
# 8-10 minutes per run, about 4 hours in total on a 10-core Mac. Runs that
# already finished with exactly these settings are skipped, so an interrupted
# run can simply be started again.
#
# Pass seeds as arguments to run a subset, e.g. `14_main_olympic.sh 0 1`.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then seeds=("$@"); else seeds=(0 1 2 3 4); fi
out=results/olympic
run=(uv run --project ../ariel python run.py --skip-done --seeds "${seeds[@]}"
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo
    --no-vision --clock-boost 1 --hidden-layers 8,4
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0
    --speed-weight 0.5 --stop-at-target
    --crossover-probability 0.9 --ariel-spawn --max-evaluations 12000)

for policy in best worst random none; do
    "${run[@]}" --policy "$policy" --out "$out/$policy"
done
"${run[@]}" --standard --out "$out/standard"
"${run[@]}" --algorithm random_search --out "$out/random_search"

conditions=("$out/best" "$out/worst" "$out/random" "$out/none" "$out/standard"
    "$out/random_search")

# Every run's best brain on 20 unseen arenas (D14); analyze.py reads the result.
runs=()
for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do runs+=("$condition/seed$seed"); done
done
uv run --project ../ariel python unseen.py "${runs[@]}" > "$out/unseen.log" 2>&1

# Figures, tables and statistics for the report: the migration policies against
# the no-migration control, and all set-ups against the standard EA. "Fast
# enough" is best fitness below 1.6 (analyze.py's default): in tuning (D22),
# 2 of 3 runs of the tuned setup crossed it within 6,000 evaluations and none
# of the untuned base runs (crossover 0.5) did, so it separates fast runs from
# slow ones.
uv run --project ../ariel python analyze.py "${conditions[@]}" \
    --reference none --out "$out/analysis"
uv run --project ../ariel python analyze.py "${conditions[@]}" \
    --reference standard --out "$out/analysis_vs_standard"

# The compute ledger for the report, now including this experiment.
uv run --project ../ariel python compute_ledger.py
