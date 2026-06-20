# Stage B0C Cost Update

- Measured sec/step for B0C.1: 0.03623
- Measured sec/step for B0C.2: not run
- Measured sec/step for B0C.3 fallback: not run
- Actual Stage B0C elapsed GPU-hours: 0.807
- Estimated cost for full curriculum Stage B: 3.422 GPU-hours; this does not multiply by n_pairs because one model covers the full curve.
- Estimated cost for full curriculum Stage C: unavailable until Mamba curriculum smoke refresh
- Should Mamba cost estimate be refreshed before Stage C: YES; Mamba throughput should be re-measured with the same curriculum, dtype decision, and checkpoint cadence before Stage C.
- Current B0C cap likely sufficient: YES
- B0C.1 gate: FAIL
- B0C.2 gate: SKIPPED
