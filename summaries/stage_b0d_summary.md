# Stage B0D Summary

## Result

- Dense multi-query sampler audit: PASS
- B0D.1 N_VALUES=8 dense reproducibility: FAIL
- B0D.2 N_VALUES=16 formal dense pilot: SKIPPED
- B0D.3 LR fallback, if run: not_run
- Cost update: complete

## Gate

- B0D.1:
  FAIL

- B0D.2:
  SKIPPED

- Whether full Stage B is allowed:
  NO

- Recommended formal task:
  do not proceed

- Recommended Stage B recipe:
  do not proceed

## Key metrics

| substage | eval_mode | n_values | q_count_mode | q_cap | lr | seed | step | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=16 acc | N=32 acc | N=64 acc | CE_N8 | CE_N16 | all_correct_N8 | all_correct_N16 | random_acc | random_CE | status |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0D.1 | dense_capped | 8 | capped | 8 | 0.001 | 0 | 30000 | 1.0000 | 0.9956 | 0.8187 | 0.5307 |  |  |  | 0.9975 |  | 0.0012 |  | 0.1250 | 2.0794 | completed |
| B0D.1 | single_query | 8 | single | 1 | 0.001 | 0 | 30000 | 1.0000 | 0.9906 | 0.7095 | 0.4011 |  |  |  | 1.4269 |  | 0.4011 |  | 0.1250 | 2.0794 | completed |
| B0D.1 | dense_capped | 8 | capped | 8 | 0.001 | 1 | 30000 | 1.0000 | 0.9999 | 0.9991 | 0.9209 |  |  |  | 0.2224 |  | 0.4460 |  | 0.1250 | 2.0794 | completed |
| B0D.1 | single_query | 8 | single | 1 | 0.001 | 1 | 30000 | 1.0000 | 0.9998 | 0.9980 | 0.9927 |  |  |  | 0.0345 |  | 0.9927 |  | 0.1250 | 2.0794 | completed |
| B0D.1 | dense_capped | 8 | capped | 8 | 0.001 | 2 | 30000 | 1.0000 | 0.8855 | 0.6093 | 0.4777 |  |  |  | 1.0754 |  | 0.0005 |  | 0.1250 | 2.0794 | completed |
| B0D.1 | single_query | 8 | single | 1 | 0.001 | 2 | 30000 | 1.0000 | 0.7743 | 0.4174 | 0.3162 |  |  |  | 1.5901 |  | 0.3162 |  | 0.1250 | 2.0794 | completed |

## Interpretation

1. Did dense multi-query supervision stabilize N=8 across seeds? NO.
2. Did the model learn a flexible recall rule across N=1/2/4/8? NO.
3. Does the dense-trained model transfer to single-query eval? NO.
4. Is formal N_VALUES=16 training viable? NO.
5. Should the formal task be redefined as dense capped multi-query recall? NO.
6. Should the same recipe be imposed on Mamba for fairness? NO, because Stage B is still blocked.
7. Is full Stage B allowed under cost estimate? NO.

## Recommended next step

- Do not enter Stage B; revise architecture/training recipe.
