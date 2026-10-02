#!/usr/bin/env bash
# Experiment X (2026-10-02): the best walk we can make. For fun, not in the paper.
#
# What: one standard EA (one population of 80, no islands; experiment 14's
#       final settings otherwise) that does not start from scratch. Its first
#       population holds the best brain of every final-setup run so far,
#       experiments 14 and 26: 45 brains with the same 212-weight network
#       (experiment 29's brains have 2 more inputs and cannot join). The
#       other 35 start random.
# How long: 5,000 generations instead of 166. Generation 0 walks all 80,
#       every later one walks its 72 children (the 8 elites keep their
#       score), so the budget is 80 + 5,000 x 72 = 360,080 walks: about 4.5
#       hours on a 10-core Mac.
# Where: a fresh OlympicArena that none of the 45 brains has seen
#       (results/x/terrains/). With one arena, the brain may learn this one
#       strip by heart (D10), so the blind test below is the real verdict.
# Blind test: the same 20 unseen arenas as every other brain (unseen.py), at
#       the training length and at 30 s, so the numbers compare directly with
#       results/olympic/*/seed*/unseen*.json; then longer walks on its own
#       arena and a video.
#
# Re-running skips the evolution if it already finished with these settings,
# and continues it from its last saved generation if it was cut off (a crash
# or a power cut: `--resume`). Never delete results/x/standard/seed0 to retry.
set -euo pipefail
cd "$(dirname "$0")/.."

out=results/x
mkdir -p "$out"

# The starting brains: every run's best of experiments 14 and 26.
uv run --project ../ariel python - "$out" <<'EOF'
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
conditions = ("best", "worst", "random", "none", "standard", "random_search",
              "best_int5", "best_int20", "best_int50")
files = [Path("results/olympic") / c / f"seed{s}" / "best_genotype.npy"
         for c in conditions for s in range(5)]
np.save(out / "winners.npy", np.array([np.load(f) for f in files]))
(out / "winners.txt").write_text("".join(f"{f}\n" for f in files))
print(f"{len(files)} starting brains -> {out / 'winners.npy'}")
EOF

uv run --project ../ariel python run.py --skip-done --seeds 0 \
    --world olympic --body spider_8 --duration 15 --no-evolve-tempo \
    --no-vision --clock-boost 1 --hidden-layers 8,4 \
    --ground-contact-weight 1.0 --low-body-weight 1.0 --carry-height 0.04 \
    --work-imbalance-weight 0.5 --leg-imbalance-weight 0 \
    --speed-weight 0.5 --stop-at-target \
    --crossover-probability 0.9 --ariel-spawn --max-evaluations 360080 \
    --standard --init-from "$out/winners.npy" --out "$out/standard" --resume \
    >> "$out/standard.log" 2>&1

run="$out/standard/seed0"
uv run --project ../ariel python unseen.py "$run" > "$out/unseen.log" 2>&1
uv run --project ../ariel python unseen.py "$run" --duration 30 > "$out/unseen_30s.log" 2>&1
uv run --project ../ariel python longer_walks.py "$out/standard" > "$out/longer_walks.log" 2>&1
uv run --project ../ariel python plot.py "$run"
uv run --project ../ariel python replay.py "$run"
