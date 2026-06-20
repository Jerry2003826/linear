# Linear State Capacity Experiments

This repository contains the code and recorded outputs for a synthetic key-value recall study:

> Can a linear/state-space sequence model store long content in a fixed-size state, and where does that capacity break?

The current work is intentionally gated. Full Stage B and Stage C have **not** been launched because the Transformer control is not yet stable enough under repeated reproduction.

## Current Status

- Stage A: environment and initial Transformer sanity checks completed.
- Stage A2: one Transformer N=8 run learned successfully; its checkpoint remained high on fresh re-evaluation.
- Stage B0: three fresh N=8 Transformer seeds failed to reproduce the fast A2 learning.
- Stage B0R: A2 artifact was audited and appears real, but current fresh training remains unstable.
- Stage B/C: blocked.

Latest gate category from `summaries/stage_b0r_summary.md`:

```text
HIGH_VARIANCE_UNSTABLE
Whether Stage B is allowed: NO
```

## Key Finding So Far

The A2.2 checkpoint is not a logging artifact:

- Fresh N=8 eval mean accuracy: `0.9750`
- Leak probes passed: normal eval is high, corrupted query/records/labels collapse toward baseline.

But fresh training under the current controlled code does not reliably reproduce it:

- B0.1 seed 0: `0.3215`
- B0.1 seed 1: `0.3119`
- B0.1 seed 2: `0.3315`
- B0R.5a seed 0 at 30k: `0.3276`

The practical conclusion is that Stage B should remain blocked until the training recipe is revised, most likely with curriculum training or another stabilizing intervention.

## Repository Layout

- `src/`: experiment code and model definitions.
- `configs/`: training strategy and run config JSONs.
- `summaries/`: human-readable stage reports and audits.
- `results/`: CSV/JSONL metrics.
- `plots/`: learning curves.
- `logs/`: stage logs needed to verify the run history.
- `status/`: terminal run status metadata for checkpointed B0/B0R jobs.

Large checkpoints are intentionally not committed. They were left on the AutoDL machine under:

```text
/root/mamba_recall_experiment/runs/20260620_180244/checkpoints/
```

## Main Files To Read

Start here:

- `summaries/stage_b0r_summary.md`
- `summaries/stage_b0r_checkpoint_reeval.md`
- `summaries/stage_b0r_leak_probes.md`
- `summaries/stage_b0_summary.md`
- `results/run_manifest.csv`

## Environment Notes

The code was run on AutoDL with a single RTX 4090 and the `state-spaces/mamba` community image.

Important environment constraints:

- Use existing conda env `mamba2`.
- Do not reinstall `mamba-ssm` or `causal-conv1d`.
- Do not change CUDA or PyTorch versions.
- Keep batch/model sizes conservative for 24GB VRAM.

## Running

The scripts are stage-specific and are intended to be run from the experiment root:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2
RUN_DIR=$PWD python -u src/stage_b0r.py
```

Do not launch full Stage B/C unless the gate summaries are updated and explicitly allow it.
