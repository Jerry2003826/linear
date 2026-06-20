# Stage B0 Summary

## Result

- B0.0 config audit: complete.
- B0.1 N=8 reproducibility: FAIL.
- B0.2 uniform mixed-load pilot: not_run.
- B0.2 curriculum mixed-load pilot: not_run.
- B0.3 cost update: complete.

## Gate

- B0.1: FAIL
- B0.2: FAIL
- Recommended Stage B recipe:
  - do not proceed

## Key metrics

| substage | recipe | lr | seed | step | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=16 acc | N=32 acc | N=64 acc | CE_N8 | CE_N16 | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0.1 | single-condition N=8 | 0.001 | 0 | 20000 |  |  |  | 0.3215 |  |  |  | 1.5702 |  | completed |
| B0.1 | single-condition N=8 | 0.001 | 1 | 20000 |  |  |  | 0.3119 |  |  |  | 1.5668 |  | completed |
| B0.1 | single-condition N=8 | 0.001 | 2 | 20000 |  |  |  | 0.3315 |  |  |  | 1.5577 |  | completed |

## Interpretation

1. A1 and A2 differed because A2 used a restarted run with checkpointing and a custom code path; core model/task hyperparameters matched, but A1 had no checkpoint evidence. B0 also identified RNG coupling: evaluation consumed the global train RNG, so eval interval changed the later training samples.
2. N=8 reproducibly learnable: no.
3. Formal n_values=16 mixed-load training viable: not tested because B0.1 failed before B0.2.
4. Stage B recipe: do not proceed.
5. Full Stage B allowed under budget estimate: no.


## Recommended next step

- Do not enter Stage B; extend B0.
