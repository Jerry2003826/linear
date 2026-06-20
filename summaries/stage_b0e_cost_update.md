# Stage B0E Cost Update

- Measured sec/step for B0E.1: 0.03923
- Measured sec/step for B0E.2: not run
- Actual Stage B0E elapsed GPU-hours: 0.780
- Estimated full Stage B cost under key_next_value format: 3.705 GPU-hours; one model covers all eval_n_pairs.
- Estimated full Stage C cost under same recipe: unavailable until Mamba induction sequence throughput probe.
- Mamba throughput probe should be refreshed: YES, if Stage B becomes allowed, because the token format changed.
- Sequence length reduction versus old <V>/<ANS> format: old N=64 q_cap=8 x length 288; key_next_value x length 216.
- Current B0E cap likely sufficient: YES
- B0E.1 gate: FAIL
- B0E.2 gate: SKIPPED
