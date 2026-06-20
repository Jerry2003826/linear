# Training Strategy Overrides

## Stage B: Transformer Formal Comparison

Stage B uses mixed-load training. One Transformer model must cover the full
association-load curve. Do not train one model per `n_pairs`.

Training:

```text
train_n_pairs = [1,2,4,8,16,32,64]
```

Each step samples one `n_pairs` value uniformly. Evaluation runs fixed
conditions:

```text
eval_n_pairs = [1,2,4,8,16,32,64]
```

B1 LR screening is budget-aware:

- Run all LR candidates to at least 20k steps unless NaN/OOM/clear bug occurs.
- Between 30k and 50k, keep only the best LR for extension to 100k.
- Discarded LR runs are screening-only and do not enter final CI.

B2 final seeds:

- Add only seed=1 and seed=2 for the best LR.
- Reuse the B1 best-LR seed=0 run extended to 100k. Do not rerun seed=0.

## Stage C: Mamba Formal Comparison

Stage C uses the same mixed-load training as Stage B. One Mamba model covers the
full `n_pairs` curve. Do not train one model per `n_pairs`.

Training:

```text
train_n_pairs = [1,2,4,8,16,32,64]
```

Evaluation:

```text
eval_n_pairs = [1,2,4,8,16,32,64]
```

Before C0, make a Mamba dtype decision:

- Run bf16 smoke first.
- If bf16 is stable, formal Mamba uses bf16.
- If bf16 is unstable, formal Mamba falls back to fp32.
- fp16 is forbidden.

C1 d_state + LR screening is budget-aware:

- Run all d_state/LR candidates to at least 20k steps unless NaN/OOM/clear bug occurs.
- Between 30k and 50k, keep only the best LR per d_state for extension to 100k.
- Discarded LR runs are screening-only and do not enter final CI.

C2 final seeds:

- Add only seed=1 and seed=2 for each d_state best LR.
- Reuse the C1 best-LR seed=0 run extended to 100k. Do not rerun seed=0.
