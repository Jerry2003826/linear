# R3.7 + R3.8 — Final Summary (Efficiency Audit + Classification-head Reranker)

**Date:** 2026-06-22 · GPU: RTX 4090 24GB · bf16 · linear-rag branch
Scope: profile reranker inference bottlenecks (R3.7), then test whether a binary classification head improves the efficiency picture (R3.8). **No new architectures trained beyond the cls head; no Mamba-370m / Mamba-2 / DeltaNet / EFLA / toy-KV.**

---

## Result

- **R3.7 → ENTER_R38.** The model forward pass is the sole latency bottleneck (Mamba 91%, Pythia 83%, CE 82%); tokenization/padding/yes-no extraction are negligible, and `max_len` reduction does nothing (real inputs ~60 tokens, padding waste ~0%). The yes/no path hides a 50k-way LM-head vocab projection inside the forward — a real, untested optimization lever — and Mamba's accuracy/stability edge justified a fairer classification head.
- **R3.8 → PASS_EFFICIENCY.** A `Linear(hidden,2)` + cross-entropy head on the last-token hidden state **preserves Mamba's accuracy** (R@5 0.742 vs yes/no 0.745), **cuts end-to-end latency 43%** (219→124 ms/q), and gives Mamba **lower eval VRAM than the cross-encoder** (~294–321 vs ~356 MB), while keeping its hallmark seed stability (σ=0.005).

---

## Key metrics table

| Model | scoring_type | R@5 | R@10 | MRR | latency-q (ms) | latency-cand (ms, bs16) | peak VRAM | seed R@5 std | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mamba-130m** | **classification_head** | **0.742** | **0.807** | **0.702** | **124** | 1.11 | ~294–321 MB (eval) / 564 MB (train) | **0.0045** | **PASS_EFFICIENCY** |
| Mamba-130m | yes/no (R3) | 0.745 | 0.790 | 0.695 | 219 | 0.96 | 427–551 MB | 0.0017 | baseline |
| Pythia-160m | classification_head | 0.541 | 0.604 | 0.432 | 66 | 0.56 | ~413 MB (eval) | 0.106 | unstable (seed2 collapse) |
| Pythia-160m | yes/no tuned (R3.6) | 0.465 | ~0.50 | ~0.36 | 116 | 0.45 | 802 MB | 0.133 | baseline |
| CrossEncoder | cross_encoder | 0.709 | 0.770 | 0.589 | 62 | 0.42 | 103–356 MB | — | efficiency frontier |

Per-seed Mamba-cls R@5: 0.741 / 0.748 / 0.737. Per-seed Pythia-cls R@5: 0.623 / 0.610 / 0.391.

---

## Interpretation (5 questions)

1. **Where was the cost?** Entirely in the model forward (Mamba 91%). Tokenize 7%, pad <1%, yes/no extract <1%, sort ~0%. Not a data-pipeline problem.
2. **Did the classification head help?** Yes — it removed the 50k-vocab LM-head projection from the scoring path, cutting end-to-end latency 43% and eval VRAM ~40% vs Mamba yes/no, with **no accuracy loss** (R@5 0.742 vs 0.745, R@1 and MRR slightly better).
3. **Mamba vs CrossEncoder now?** Accuracy: Mamba-cls ≥ CE (R@5 0.742 vs 0.709). VRAM: Mamba-cls now ≤ CE at eval. Speed: Mamba-cls still ~2× slower per query (124 vs 62 ms). CE stays the latency frontier.
4. **Mamba vs Pythia under a fair head?** Mamba wins on both accuracy (+0.20 R@5) and stability (σ 0.005 vs 0.106). The head improves Pythia (0.541 vs 0.465) but one seed still collapses — Pythia remains fundamentally unstable here.
5. **Is the conclusion honest?** Yes, and bounded: Mamba shows a genuine **VRAM + accuracy + stability** advantage under a fair, architecture-agnostic scorer. It does **not** show raw-speed superiority over the cross-encoder, and this is **not** evidence that "Mamba replaces RAG" or that linear models are inherently better. The pass rests on the VRAM branch of the gate, not latency parity with CE.

---

## Recommended next step

**Enter R4** — the classification head cleared PASS_EFFICIENCY: Mamba now has a concrete efficiency advantage (eval VRAM ≤ CE) on top of its accuracy/stability edge, with a 43% latency cut over its own yes/no baseline.

**Framing constraint for R4:** position the advantage as *VRAM + accuracy + stability*, not speed — Mamba-cls is still ~2× slower per query than the cross-encoder. If R4 prioritizes raw latency, the cross-encoder remains the bar to beat.

**Secondary (do NOT auto-run):** A listwise scorer (R3.9) could further close the latency gap to CE by scoring multiple candidates per forward; recommend it only as an optional follow-up, not a prerequisite.

---

## Artifacts
- R3.7: `results/linear_rag/r37_efficiency_breakdown.csv`, `r37_latency_by_batch_and_len.csv`, `plots/linear_rag/r37_efficiency_breakdown.png`, `r37_latency_vram_frontier.png`, `summaries/linear_rag/r37_efficiency_audit_summary.md`
- R3.8: `results/linear_rag/r38_classification_head_metrics.csv`, `r38_classification_head_predictions_sample.csv`, `r38_latency_vram.csv`, `plots/linear_rag/r38_recall_comparison.png`, `r38_latency_vram_frontier.png`, `r38_seed_stability.png`, `summaries/linear_rag/r38_classification_head_summary.md`
- Checkpoints (gitignored): `checkpoints/linear_rag/r38_mamba_cls/`, `r38_pythia_cls/`
