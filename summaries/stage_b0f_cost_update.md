# Stage B0F Cost Update

- Measured sec/step for resumed 4L/256: 0.03335
- Measured sec/step for 6L/256: 0.04033
- Measured sec/step for 4L/384: 0.03146
- Measured sec/step for B0F.3: not run
- Actual Stage B0F elapsed GPU-hours: 2.953
- Estimated full Stage B cost under selected control (do not proceed): 3.150 GPU-hours.
- Estimated full Stage C cost if Mamba uses same task format/curriculum: 3.150 GPU-hours before Mamba throughput refresh.
- Mamba throughput probe should be refreshed: YES, if Stage B becomes allowed.
- Expected extra Stage B cost versus original 4L/256 plan: 0.000 GPU-hours.
- Current B0F cap likely sufficient: YES
- B0F.1 gate: FAIL_PLATEAU
- B0F.2 gate: FAIL
- B0F.3 gate: SKIPPED
