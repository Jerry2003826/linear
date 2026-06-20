# Stage B0E Summary

## Result

- Sampler audit: PASS
- B0E.1 N_VALUES=8 induction-aligned reproducibility: FAIL
- B0E.2 N_VALUES=16 formal pilot: SKIPPED
- Cost update: complete

## Gate

- B0E.1:
  FAIL

- B0E.2:
  SKIPPED

- Whether full Stage B is allowed:
  NO

- Recommended formal task:
  do not proceed

- Recommended Stage B recipe:
  do not proceed

## Key metrics

| substage | eval_mode | task_format | n_values | q_count_mode | q_cap | lr | seed | step | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=16 acc | N=32 acc | N=64 acc | CE_N8 | CE_N16 | all_correct_N8 | all_correct_N16 | random_acc | random_CE | status |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0E.1 | induction_capped | key_next_value | 8 | capped | 8 | 0.001 | 0 | 30000 | 1.0000 | 1.0000 | 0.9989 | 0.7126 |  |  |  | 0.5589 |  | 0.0189 |  | 0.1250 | 2.0794 | completed |
| B0E.1 | single_query | key_next_value | 8 | single | 1 | 0.001 | 0 | 30000 | 1.0000 | 1.0000 | 0.9998 | 0.5995 |  |  |  | 0.8407 |  | 0.5995 |  | 0.1250 | 2.0794 | completed |
| B0E.1 | induction_capped | key_next_value | 8 | capped | 8 | 0.001 | 1 | 12000 | 1.0000 | 1.0000 | 0.9999 | 0.9997 |  |  |  | 0.0012 |  | 0.9977 |  | 0.1250 | 2.0794 | pass_early_stop |
| B0E.1 | single_query | key_next_value | 8 | single | 1 | 0.001 | 1 | 12000 | 1.0000 | 1.0000 | 0.9999 | 0.9999 |  |  |  | 0.0006 |  | 0.9999 |  | 0.1250 | 2.0794 | pass_early_stop |
| B0E.1 | induction_capped | key_next_value | 8 | capped | 8 | 0.001 | 2 | 30000 | 1.0000 | 1.0000 | 0.9998 | 0.6142 |  |  |  | 0.8656 |  | 0.0020 |  | 0.1250 | 2.0794 | completed |
| B0E.1 | single_query | key_next_value | 8 | single | 1 | 0.001 | 2 | 30000 | 1.0000 | 1.0000 | 1.0000 | 0.6077 |  |  |  | 0.9021 |  | 0.6077 |  | 0.1250 | 2.0794 | completed |

## Interpretation

1. Did induction multi-query supervision stabilize N=8 across seeds? NO.
2. Did the model learn a flexible recall rule across N=1/2/4/8? NO.
3. Does the induction-trained model transfer to single-query eval? NO.
4. Is formal N_VALUES=16 training viable? NO.
5. Should the formal task be redefined as key_next_value recall? NO.
6. Should the same recipe be imposed on Mamba for fairness? NO, because Stage B is still blocked.
7. Is full Stage B allowed under cost estimate? NO.

## Recommended next step

- increase Transformer control size
