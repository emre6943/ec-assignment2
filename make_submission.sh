#!/usr/bin/env bash
# Builds the hand-in folder submission/92/ (git-ignored): the report as 92.pdf,
# our code, the experiment scripts the report draws on, and the result files
# and arenas its numbers come from. ARIEL is not included. Zip it with
#   cd submission && zip -r -X 92.zip 92
# Compile report/main.pdf first (cd report && tectonic -X compile main.tex).
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")"
if [ report/main.pdf -ot report/main.tex ]; then
    echo "report/main.pdf is older than main.tex: compile it first" >&2
    exit 1
fi
D=submission/92
rm -rf "$D"
mkdir -p "$D/tests" "$D/experiments" "$D/docs"

cp run.py ea.py genome.py network.py operators.py migration.py sensors.py \
    simulate.py terrain.py bodies.py unseen.py longer_walks.py analyze.py \
    probabilities.py end_results.py paper_numbers.py paper_figures.py \
    rq_figure.py filmstrip.py replay.py plot.py compute_ledger.py "$D/"
cp tests/*.py "$D/tests/"
for script in 99_final_experiment 14_main_olympic 26_migration_interval \
    27_statistics 19_tuning 19b_tuning_more 22_stagnation_ablation \
    23_vision_ablation 24_brain_shape 25_crossover_operators 07_pilot \
    11_body_pilot 13_walking_pilot 15_olympic_long_run 16_gait_pilot \
    17_gait_pilot_2 18_inputs_pilot 21_robustness_pilot; do
    cp "experiments/$script.sh" "$D/experiments/"
done
cp report/main.pdf "$D/92.pdf"
cp docs/compute.md "$D/docs/"
cp docs/submission_README.md "$D/README.md"

# copy_runs <experiment> "<files>" <condition>...: those files of every seed,
# where a run has them (only some tuning runs had the unseen test).
copy_runs() {
    local experiment=$1 files=$2
    shift 2
    for condition in "$@"; do
        for run in results/"$experiment"/"$condition"/seed*; do
            mkdir -p "$D/$run"
            for file in $files; do
                if [ -f "$run/$file" ]; then cp "$run/$file" "$D/$run/"; fi
            done
        done
    done
}
copy_runs final \
    "config.json log.csv summary.json best_genotype.npy unseen.json unseen_30s.json longer_walks.json" \
    best worst random none standard random_search
copy_runs olympic "config.json log.csv summary.json best_genotype.npy unseen.json" \
    best worst random none standard random_search best_int5 best_int20 best_int50
copy_runs tuning "config.json log.csv summary.json best_genotype.npy unseen.json" \
    base sigma_0.02 sigma_0.1 \
    pop_4x10 pop_4x40 tour_2 tour_5 elites_1 elites_5 sparse_mut xover_0 \
    xover_0.9 xover_0.9_nostall rays3 rays1 rays0 h8 h32 h8_8 h16_16 h8_4 \
    h8_4_weight h8_4_blx h8_4_headless

final_files=$(find "$D/results/final" -path '*/seed*/*' -type f | wc -l | tr -d ' ')
if [ "$final_files" -ne 840 ]; then  # 6 conditions x 20 seeds x 7 files
    echo "expected 840 files of the final experiment's runs, found $final_files"
    exit 1
fi

# The saved arenas: ARIEL builds the OlympicArena without a seed (D10).
copy_file() { mkdir -p "$D/$(dirname "$1")" && cp "$1" "$D/$1"; }
for seed in $(seq 10 29); do
    copy_file "results/final/terrains/olympic/spider_8/seed$seed/terrain0_arielspawn.mjb"
done
for seed in 0 1 2 3 4; do
    copy_file "results/olympic/terrains/olympic/spider_8/seed$seed/terrain0_arielspawn.mjb"
done
for seed in 100 101 102; do
    copy_file "results/tuning/terrains/olympic/spider_8/seed$seed/terrain0.mjb"
done
for arena in $(seq 0 19); do
    copy_file "results/terrains/olympic/test/spider_8/terrain${arena}_arielspawn.mjb"
done
copy_file results/terrains/olympic/spider_8/seed0/terrain0.mjb # pilots 15-18

# The outputs the report cites.
for file in summary.md summary.csv stats.md paired_tests.csv; do
    copy_file "results/final/analysis/$file"
done
for file in probabilities.md probabilities.pairs.csv probabilities.runs.csv \
    end_results.md longer_walks.md; do
    copy_file "results/final/$file"
done
for folder in results/olympic/analysis results/olympic/analysis_interval \
    results/tuning/analysis*; do
    for file in summary.md summary.csv stats.md paired_tests.csv; do
        if [ -f "$folder/$file" ]; then copy_file "$folder/$file"; fi
    done
done
for file in results/olympic/probabilities{,_interval}.{md,pairs.csv,runs.csv}; do
    copy_file "$file"
done

# The runs' JSON files name their arenas by absolute paths on the machine that
# ran them; make them relative to this folder.
find "$D/results" -name '*.json' -print0 | xargs -0 sed -i.bak -E \
    's#"/[^"]*/assignment2/#"#g'
find "$D/results" -name '*.json.bak' -delete
if grep -rlq '/Users/\|/home/' "$D/results"; then
    echo "absolute paths left in:" && grep -rl '/Users/\|/home/' "$D/results"
    exit 1
fi
echo "built $D: $(find "$D" -type f | wc -l | tr -d ' ') files, $(du -sh "$D" | cut -f1)"
