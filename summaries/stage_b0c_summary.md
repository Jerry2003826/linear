# Stage B0C Summary

## Result

- Curriculum sampler audit: PASS
- B0C.1 N_VALUES=8 curriculum reproducibility: FAIL
- B0C.2 N_VALUES=16 formal curriculum pilot: SKIPPED
- B0C.3 LR fallback, if run: not_run
- Cost update: complete

## Gate

- B0C.1:
  FAIL

- B0C.2:
  SKIPPED

- Whether full Stage B is allowed:
  NO

- Recommended Stage B recipe:
  do not proceed

## Key metrics

| substage | recipe | n_values | lr | seed | step | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=16 acc | N=32 acc | N=64 acc | CE_N8 | CE_N16 | random_acc | random_CE | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0C.1 | curriculum N_VALUES=8 | 8 | 0.001 | 0 | 30000 | 1.0000 | 1.0000 | 0.6071 | 0.3938 |  |  |  | 1.3916 |  | 0.1250 | 2.0794 | completed |
| B0C.1 | curriculum N_VALUES=8 | 8 | 0.001 | 1 | 21000 | 1.0000 | 1.0000 | 0.9989 | 0.9375 |  |  |  | 0.1990 |  | 0.1250 | 2.0794 | pass_early_stop |
| B0C.1 | curriculum N_VALUES=8 | 8 | 0.001 | 2 | 30000 | 1.0000 | 0.9988 | 0.4047 | 0.3088 |  |  |  | 1.5943 |  | 0.1250 | 2.0794 | completed |

## Interpretation

1. Did curriculum stabilize N=8 across seeds? NO.
2. Did the model learn a flexible recall rule across N=1/2/4/8, or only fixed N=8? NO.
3. Is formal N_VALUES=16 training viable? NO.
4. Should full Stage B use curriculum mixed-load? NO.
5. Should the same curriculum be imposed on Mamba for fairness? YES, if Stage B proceeds, Mamba should use the same curriculum for fairness.
6. Is full Stage B allowed under cost estimate? NO.

## Recommended next step

- Do not enter Stage B; revise task/training recipe.
