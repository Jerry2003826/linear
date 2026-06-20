# Stage B0F B0E Audit

- B0E summary exists: YES
- B0E results CSV exists: YES
- B0E nohup log exists: YES
- Seed0/seed2 latest checkpoints resumable: YES

## Seed Audit

### seed0

- Source run: `stage_b0e_b01_induction_v8_seed0`
- Latest checkpoint exists: YES
- Latest checkpoint valid: YES
- Latest step: 30000
- Checkpoint includes optimizer/scheduler/RNG state: YES
- Best capped N=8 acc: 0.7126 at step 30000
- Best capped N=8 CE: 0.5589 at step 30000
- Curve at 30k: improving (last-10k acc delta=0.0379, CE drop=0.1063)
- N=1/2/4 already solved: YES
- Valid to resume to 100k: YES

### seed2

- Source run: `stage_b0e_b01_induction_v8_seed2`
- Latest checkpoint exists: YES
- Latest checkpoint valid: YES
- Latest step: 30000
- Checkpoint includes optimizer/scheduler/RNG state: YES
- Best capped N=8 acc: 0.6142 at step 30000
- Best capped N=8 CE: 0.8656 at step 30000
- Curve at 30k: improving (last-10k acc delta=0.0346, CE drop=0.0750)
- N=1/2/4 already solved: YES
- Valid to resume to 100k: YES

## Conclusion

- PASS
