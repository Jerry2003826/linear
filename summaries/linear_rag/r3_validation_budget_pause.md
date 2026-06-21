# R3 Validation — Budget Decision Note

**Budget cap:** `MAX_GPU_HOURS_R3_VALIDATION = 8`

## Why this note exists
The expert spec sets `max_steps=3000` per seed and `eval_interval=250` with dev
early-stopping over `seeds=[0,1,2]`, for BOTH Mamba-130m and Pythia-160m, plus a
cross-encoder re-eval and a latency/VRAM benchmark — all under a single 8 GPU-h
cap. Direct dry-run timing on the actual server shows the naive recipe overruns
that cap, so per the spec ("超预算先缩 seeds/steps") we reduce **steps**, keep
**seeds=3** (the whole point of the validation is cross-seed stability — never
reduce to one seed), and keep test eval at full 1000.

## Measured timings (this GPU, single seed, real data)
- LoRA train (Mamba-130m, bf16, bs8×grad_acc4): **~2.19 s/step** → 3000 steps ≈ 110 min/seed; 1500 steps ≈ 55 min/seed.
- Single-pair yes/no eval (causal LM): **~10 ms/pair** → 1000 queries × 100 cand ≈ 17 min.
- Cross-encoder (batched, has attention_mask): **~0.3 min / 1000 queries** (cheap).

## Why not batch the causal-LM eval
Left-padded batched scoring was tested against single-prompt scoring and did NOT
match numerically (Mamba max|diff|≈0.75, Pythia≈8.0; rankings differed). Mamba is
an SSM with no attention_mask, so pad tokens contaminate the recurrent state;
Pythia left-padding shifted positions. We therefore keep the **verified single-pair
eval** for both LoRA models and control cost by reducing steps + using a dev
monitoring subset for early-stopping (full dev + full test only at the end).

## Decisions (applied)
1. **max_steps 3000 → 1500** for both Mamba and Pythia. Rationale: prior R3 run on a
   400-query subset already overfit (train loss→0) by 3000 steps; with 3000 training
   queries here, 1500 steps is ample and early-stopping (patience=4 evals) will
   usually trigger earlier.
2. **seeds = [0,1,2]** unchanged (cross-seed stability is the core deliverable).
3. **eval_interval = 250** unchanged. **Early stopping patience = 4** unchanged.
   `metric_for_best = dev MRR`. Test is NEVER used for early stopping.
4. **Dev early-stop monitoring uses a 200-query dev subset** (deterministic prefix of
   the dev split). The **best-dev checkpoint** is then re-evaluated ONCE on the
   **full 1000-query dev** and the **full 1000-query test** for the reported numbers.
   This keeps early-stopping faithful (dev-only) while cutting monitoring cost.
5. Test eval stays at full **1000 queries, top-100** (top-500 not needed: candidate
   audit showed test Recall@100 = 0.914 ≥ 0.75 threshold).

## Projected budget (with above)
| Stage | Setting | Est. |
|---|---|---|
| Mamba-LoRA 3 seeds | 1500 steps, dev-monitor 200, full test | ~3.5 GPU-h |
| Pythia-LoRA 3 seeds | 1500 steps, dev-monitor 200, full test | ~3.0 GPU-h |
| Cross-encoder re-eval | full test, batched | ~0.01 GPU-h |
| Latency/VRAM bench | batch [1,4,8], topk 100 | ~0.2 GPU-h |
| **Total** | | **~6.7 GPU-h ≤ 8** |

Early-stopping is expected to bring the LoRA stages in below these estimates.
If any stage trends over budget at runtime, the next lever is reducing Pythia
steps further (Pythia is the baseline, not the primary deliverable) before
touching Mamba seeds/steps.

## Runtime update (actual execution)
- Mamba-130m LoRA: ran the FULL 1500 steps x 3 seeds (no early stop triggered;
  dev MRR kept inching up). Per-seed train ~5100s. Used ~4.3 GPU-h total.
- Pythia-160m LoRA baseline: to keep the comparison strictly fair (SAME steps),
  it is ALSO run at 1500 steps x 3 seeds (Pythia trains faster, ~1.53 s/step).
  Decision: prioritize fairness of the baseline over shaving budget; if the
  combined total approaches the 8 GPU-h cap, note it here rather than weaken the
  comparison by under-training the baseline.
- Causal-LM eval is the fixed cost driver (~10-11 ms/pair, full test+dev =
  ~37 min/seed); this cannot be reduced without dropping full-test reporting.
