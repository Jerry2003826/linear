# Stage A2 Summary

## Result

- A2.0 diagnosis: complete; see `summaries/stage_a2_pre_diagnosis.md`.
- A2.1 N=8 fixed-batch overfit: pass_overfit.
- A2.2 continuation to 100k: pass_strong.
- A2.3 alternate LR: not_run.
- A2.3 curriculum: not_run.
- Budget cap: 4.00 GPU-hour; estimated worst-case: 2.744 GPU-hour; budget pause: NO.

## Gate

- Best N=8 accuracy: 0.9733
- Best N=8 CE: 0.0762
- Category: PASS_STRONG
- Whether Stage B is allowed: YES

## Key metrics

| substage | config | lr | seed | step | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=8 CE | random acc | random CE | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A2.1 | N=8 fixed batch overfit | 0.001 | 0 | 500 |  |  |  | 1.0000 | 0.0000 | 0.1250 | 2.0794 | pass_overfit |
| A2.2 | N=8 best LR to 100k | 0.001 | 0 | 6000 |  |  |  | 0.9733 | 0.0762 | 0.1250 | 2.0794 | pass_strong |

## Interpretation

1. N=8 training-budget issue: not strictly proven; A1 checkpoint was missing, so A2.2 restarted the same config and showed N=8 is learnable.
2. N=8 LR issue: not clearly; the original best LR reached PASS_STRONG on restart.
3. Need curriculum: not required by observed pass condition.
4. Transformer as Stage B/C control: yes for the N=8 targeted gate.
5. Stage B plan modification: no strategy change required, but checkpointing and seed bookkeeping should be treated as mandatory.


## Recommended next step

- Enter Stage B with original plan.
