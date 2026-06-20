# Reproduce Notes

This repository contains the cleaned source code and key text artifacts from the AutoDL RTX 4090 run.

## Environment

- Platform: AutoDL
- GPU: single RTX 4090 24GB
- Base image: state-spaces/mamba community image
- Expected packages: PyTorch, mamba-ssm, causal-conv1d, numpy, pandas, matplotlib
- Do not reinstall mamba-ssm/causal-conv1d or change CUDA/PyTorch versions.

## Quick Checks

```bash
cd src
python test_core.py
```

The core tests cover:

- answer-label alignment at `<ANS>` positions
- causal mask invariance
- no accidental final-position-only classifier path

## Main Artifacts

- `configs/training_strategy.md`: mixed-load Stage B/C strategy override.
- `results/stage_a1_summary.csv`: early Transformer ladder summary.
- `results/stage_a2_results.csv`: N=8 targeted diagnosis result.
- `results/stage_b0_results.csv`: B0 reproducibility result.
- `results/stage_b0r_results.csv`: B0R reliability audit result.
- `results/run_manifest.csv`: run manifest.
- `summaries/stage_b0r_summary.md`: final gate summary.
- `summaries/stage_b0r_*`: checkpoint re-eval, RNG, data fingerprint, and leak-probe audits.

## Current Conclusion

B0R concluded `HIGH_VARIANCE_UNSTABLE`: the A2 checkpoint itself still evaluates high, but fresh controlled retraining did not reproduce the quick N=8 learning behavior. Formal Stage B/C should not start from this state. The recommended next step is to switch to a curriculum training recipe before entering mixed-load formal comparison.

## Omitted From GitHub

Large/generated binary artifacts are intentionally not committed:

- model checkpoints under the remote `checkpoints/` directory
- PNG/PDF plots
- Python cache files and process status files

The checkpoint paths remain recorded in CSV/summaries for auditability on the original AutoDL run directory.
