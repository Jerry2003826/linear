# Stage B0D Cost Update

- Measured sec/step for B0D.1: 0.03907
- Measured sec/step for B0D.2: not run
- Measured sec/step for B0D.3 fallback: not run
- Actual Stage B0D elapsed GPU-hours: 0.977
- Estimated full Stage B cost under dense capped multi-query curriculum: 3.690 GPU-hours; one model covers all eval_n_pairs.
- Estimated full Stage C cost under same recipe: unavailable until Mamba dense sequence throughput probe.
- Mamba throughput probe should be refreshed: YES, because q_cap=8 increases sequence length and answer positions per sample.
- q_cap=8 N=64 x-sequence length: 288; acceptable for 24GB 4090 with conservative batch sizing, but Mamba/Transformer throughput should be remeasured before Stage B/C.
- Current B0D cap likely sufficient: YES
- B0D.1 gate: FAIL
- B0D.2 gate: SKIPPED
