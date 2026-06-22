# R3.7 — Efficiency Bottleneck Audit (Reranker Inference Profiling)

**Date:** 2026-06-22 · **GPU:** RTX 4090 24GB · **dtype:** bf16
**Sample:** 200 queries × top-100 candidates = 20,000 pairs · warmup=50, measured=200 iters
**Models profiled (reused checkpoints, no retraining):**
- `mamba-130m-lora` (seed0_best) — LM yes/no scoring
- `pythia-160m-lora-tuned` (seed0_best) — LM yes/no scoring
- `cross-encoder/ms-marco-MiniLM-L-6` — native cross-encoder scoring

> Scope note: per R3.7 spec, NO new models were trained and NO Mamba-370m / Mamba-2 / DeltaNet / EFLA / toy-KV were run. Existing checkpoints + the R3 validation test split only.

---

## Headline numbers (bs=8, max_len=512)

| Model | Scoring | ms/query | ms/candidate | Fwd tok/s | Peak VRAM (MB) | test R@5 |
|---|---|---:|---:|---:|---:|---:|
| Mamba-130m LoRA | LM yes/no | **218.7** | 2.19 | 30,091 | 551 | **0.700** |
| Pythia-160m LoRA | LM yes/no | 116.1 | 1.16 | 61,907 | 802 | 0.555 |
| CrossEncoder MiniLM-L6 | cross-enc | **62.0** | 0.62 | 83,209 | **103** | 0.700 |

(R@5 here is on the 200-query audit subsample; full-test R@5 from R3 = Mamba 0.745, CE 0.709.)

---

## Stage-level latency breakdown (bs=8, max_len=512)

| Stage | Mamba | Pythia | CrossEncoder |
|---|---:|---:|---:|
| Prompt build | 0.05% | 0.08% | 1.09% |
| Tokenize | 7.06% | 12.98% | 16.81% |
| Pad / collate | 0.97% | 1.88% | 0.00% |
| **Model forward** | **91.02%** | **83.35%** | **82.04%** |
| yes/no extract | 0.87% | 1.67% | — |
| Sort | 0.03% | 0.04% | 0.06% |

**Input-length stats (all models):** avg ≈ 60 tokens, p50=60, p90=65, p99=68. **padding_waste_ratio ≈ 0.0** (length-bucketing already eliminates padding waste; real inputs are far shorter than any of the tested max_len caps).

---

## The 7 diagnostic questions

**Q1 — Where is Mamba slow?**
Almost entirely in the **model forward pass: 91.0%** of total latency. Everything else is negligible (tokenize 7.1%, pad 1.0%, yes/no extract 0.9%, prompt+sort <0.1%). At the same batch/len, Mamba's per-candidate forward is **~2.3× slower than Pythia and ~2.6× slower than CrossEncoder**. This is consistent with R3: Mamba has a quality/stability edge but **no efficiency edge**.

**Q2 — Is the cost in tokenizer / prompt building / batching, or in the core forward?**
Core forward, unambiguously. Tokenizer is the only other non-trivial cost (7–17% across models, larger as a *share* for the faster models simply because their forward is cheaper). Prompt building and pad/collate are rounding error. So inference optimization must target the forward path, not the data pipeline.

**Q3 — How much does yes/no logprob extraction cost?**
**Only 0.87% (Mamba), 1.67% (Pythia)** of latency. Extraction itself is cheap. **Important nuance:** this measures only the final last-position logit gather. It does NOT capture the cost *hidden inside* the forward pass of projecting the last hidden state through the full **LM head over the ~50k vocab**. A classification head replaces that 50k-way projection with a 2-way `Linear(hidden,2)`, which removes vocab-projection FLOPs and memory from the forward — that is where a head can help, not in the extraction step itself.

**Q4 — Does increasing batch size help?**
Yes, dramatically, for the recurrent/transformer LMs. Mamba's forward is **~15 ms/batch and essentially flat from bs=1→16** (15.03 → 15.34 ms), so per-candidate latency collapses **15.10 → 0.96 ms (≈15.7×)** going bs=1→16. Pythia similarly 9.87 → 0.45 ms. CrossEncoder is already near its floor (3.74 → 0.42 ms). Takeaway: Mamba's fixed per-call overhead dominates at small batch; batching amortizes it almost perfectly. Inference should always run at the largest batch that fits.

**Q5 — Does reducing max_len help?**
**No.** Latency and VRAM are flat across max_len ∈ {256, 384, 512} for every model, because real inputs average ~60 tokens (p99=68) and length-bucketing already trims padding to ~0%. Lowering the cap below the actual length distribution would only risk truncation. max_len is not a lever here.

**Q6 — Is there any batch/len region where Mamba approaches or beats CrossEncoder?**
**No region beats CE on speed or VRAM.** At every batch/len, CE is faster per candidate (e.g. bs=16: CE 0.42 ms vs Mamba 0.96 ms) and far leaner (CE 366 MB vs Mamba 589 MB peak @ bs=16). Mamba *narrows* the gap to ~2.3× at large batch but never reaches parity. Mamba's only edge over Pythia is VRAM (lower) and accuracy/stability; over CE it is accuracy parity (R@5≈0.70–0.745) with worse efficiency.

**Q7 — Is it worth building a classification head?**
**Yes — worth testing (ENTER_R38).** Rationale:
1. The forward pass is the sole bottleneck, and a meaningful slice of that forward is the **50k-way LM-head projection** that a 2-way head eliminates — a legitimate, untested efficiency lever the yes/no profiling cannot see directly.
2. A head enables cleaner, fully-batched scoring (one logit pair per candidate) without vocab-token bookkeeping.
3. Mamba retains a real **accuracy + stability edge** (full-test R@5 0.745 ±0.0017 over 3 seeds vs Pythia 0.465 ±0.133) that justifies a fairer, architecture-agnostic scoring head before any further conclusions.

---

## Gate decision

**Gate criteria:** ENTER_R38 if (forward is not the *only* optimizable item) OR (yes/no / prompt / tokenization is optimizable) OR (accuracy edge worth a classifier head). STOP only if Mamba forward is slower across ALL batch/len AND higher VRAM AND no optimization space.

- Mamba forward **is** slower across all batch/len ✔ (would point toward STOP)…
- …**but** Mamba VRAM is **lower than Pythia** (551 vs 802 MB) — not "higher VRAM" universally, so the STOP precondition is not fully met.
- And there **is** optimization space: the LM-head 50k-vocab projection inside the forward can be removed by a classification head (untested), plus Mamba's accuracy/stability edge is worth a fairer head.

### → **DECISION: ENTER_R38**

Proceed to build and evaluate a binary classification-head reranker (Linear(hidden,2) + CE loss) on Mamba-130m (main) and Pythia-160m (fair control), keeping CE and the R3.6 yes/no Mamba as comparison baselines.

---

## Honest framing (do not overstate)
- Mamba has **no efficiency advantage** today: it is the slowest and not the leanest. CrossEncoder remains the efficiency frontier (fastest + lowest VRAM) at equal R@5.
- Mamba's case rests on **accuracy parity with CE + a large seed-stability edge over Pythia**, not on architectural speed superiority.
- R3.8 is a fair test of whether a classification head changes the efficiency picture — it is NOT assumed to win.

**Artifacts:**
`results/linear_rag/r37_efficiency_breakdown.csv`, `results/linear_rag/r37_latency_by_batch_and_len.csv`, `results/linear_rag/r37_efficiency_audit_meta.json`, `plots/linear_rag/r37_efficiency_breakdown.png`, `plots/linear_rag/r37_latency_vram_frontier.png`.
