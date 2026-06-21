# Linear-RAG Cost Plan

- GPU: single RTX 4090 24GB
- GPU_PRICE_PER_HOUR (env, default 2.0): 2.0
- MAX_GPU_HOURS_TOTAL: 10.0

## Per-stage budget
| stage | budget |
|-------|--------|
| R0 data gen | CPU, < 0.5h |
| R1 BM25+embedding | < 1 GPU-h |
| R2 zero-shot | < 2 GPU-h |
| R2.5 cross-encoder | < 1 GPU-h |
| R3 LoRA | < 6 GPU-h |
| TOTAL | 10.0 GPU-h |

Each GPU stage runs a dry-run (100-200 items) first, estimates full-stage time,
and shrinks sample size if over budget. runtime_profile.csv records actuals.
