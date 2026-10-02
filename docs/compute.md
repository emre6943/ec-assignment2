# Compute used

Written by `compute_ledger.py` from the run folders in `results/`; rerun it
to update. Lower bounds: runs deleted or overwritten by a rerun, smoke tests
and scratch diagnostics are not counted. An evaluation walks every terrain of
its run once (3 in experiment 21's multi-arena runs, else 1). Simulated hours
are nominal (walks that stopped at the target were shorter) and include the
unseen-terrain tests; wall-clock hours add up the runs, some of which ran in
parallel or were paused.

## By phase

| Phase | Runs | Evaluations | Walks | Simulated h | Wall-clock h | Unseen-test walks |
|---|---|---|---|---|---|---|
| development | 69 (3 unfinished) | 797,352 | 870,024 | 3,239.4 | 20.7 | 1,593 |
| tuning | 111 | 833,004 | 833,004 | 3,343.9 | 14.5 | 1,626 |
| final | 45 | 541,440 | 541,440 | 2,269.1 | 6.4 | 2,070 |
| showcase | 1 | 360,080 | 360,080 | 1,500.6 | 5.9 | 46 |
| **total** | 226 (3 unfinished) | 2,531,876 | 2,604,548 | 10,353.0 | 47.4 | 5,335 |

- **development**: building and debugging the problem (world, body, fitness,
  inputs, walk length): experiments 1-6, 8-11, 13, 15-18, 20, 21.
- **tuning**: choosing the EA's and the brain's settings by controlled
  comparisons: experiments 7, 12, 19, 19b, 22-25, 29.
- **final**: the research-question experiments: 14, 26's follow-up on the
  migration interval, and 99, the final experiment.
- **showcase**: experiment X, one long seeded run for the best walk; not part
  of the research.

## By experiment

| Experiment | Runs | Evaluations | Walks | Simulated h | Wall-clock h | Unseen-test walks |
|---|---|---|---|---|---|---|
| 1 (development) | 1 | 3,040 | 3,040 | 12.7 | 0.1 | 0 |
| 2 (development) | 1 | 3,040 | 3,040 | 12.7 | 0.0 | 0 |
| 3 (development) | 1 | 3,040 | 3,040 | 12.7 | 0.1 | 0 |
| 4 (development) | 1 | 3,032 | 3,032 | 12.6 | 0.1 | 0 |
| 5 (development) | 1 | 25,088 | 25,088 | 104.5 | 1.3 | 0 |
| 6 (development) | 1 (1 unfinished) | 13,992 | 13,992 | 58.3 | 0.6 | 0 |
| 7 (tuning) | 9 | 36,360 | 36,360 | 101.1 | 0.8 | 42 |
| 8-9 (development) | 30 | 360,960 | 360,960 | 1,004.4 | 7.2 | 630 |
| 10 (development) | 1 | 120,008 | 120,008 | 500.0 | 2.8 | 0 |
| 11 (development) | 12 | 48,480 | 48,480 | 134.7 | 0.9 | 0 |
| 12 (tuning) | 5 | 60,160 | 60,160 | 167.1 | 1.4 | 0 |
| 13 (development) | 4 | 32,000 | 32,000 | 133.3 | 0.6 | 0 |
| 14 (final) | 30 | 360,960 | 360,960 | 1,512.7 | 4.2 | 1,380 |
| 15 (development) | 1 (1 unfinished) | 48,608 | 48,608 | 202.5 | 1.2 | 0 |
| 16 (development) | 2 (1 unfinished) | 21,400 | 21,400 | 89.2 | 0.4 | 0 |
| 17 (development) | 1 | 12,032 | 12,032 | 50.1 | 0.4 | 0 |
| 18 (development) | 2 | 24,064 | 24,064 | 100.4 | 1.2 | 21 |
| 19 (tuning) | 21 | 127,080 | 127,080 | 529.5 | 4.0 | 0 |
| 19b (tuning) | 15 | 90,588 | 90,588 | 379.1 | 1.2 | 303 |
| 20 (development) | 1 | 24,064 | 24,064 | 175.4 | 0.6 | 21 |
| 21 (development) | 9 | 54,504 | 127,176 | 636.0 | 3.2 | 921 |
| 22 (tuning) | 3 | 18,168 | 18,168 | 76.0 | 0.4 | 63 |
| 23 (tuning) | 9 | 54,504 | 54,504 | 227.9 | 0.9 | 189 |
| 24 (tuning) | 15 | 90,840 | 90,840 | 379.8 | 1.5 | 315 |
| 25 (tuning) | 9 | 54,504 | 54,504 | 227.9 | 0.6 | 189 |
| 26 (final) | 15 | 180,480 | 180,480 | 756.4 | 2.2 | 690 |
| 29 (tuning) | 25 | 300,800 | 300,800 | 1,255.5 | 3.5 | 525 |
| X (showcase) | 1 | 360,080 | 360,080 | 1,500.6 | 5.9 | 46 |
