# R3 Validation — Final Summary

Validating whether the R3 surprise (Mamba-130m + LoRA reranker beating a
cross-encoder) holds up under a rigorous, fair protocol: fixed disjoint
train/dev/test split, 3 seeds, best-dev checkpointing with early stopping, a
fair Transformer-LoRA baseline, a same-split cross-encoder re-eval, and a
unified latency/VRAM benchmark.

---

## Result

**The accuracy result is real and robust. The efficiency advantage is not — but the original efficiency gap was overstated by a measurement artifact (see R3.4 below).**

On an identical, held-out 1000-query test split that no model trained on,
Mamba-130m + LoRA reranks **more accurately than the ms-marco cross-encoder**,
and does so **stably across 3 seeds** with a negligible dev→test generalization
gap. However, the unified benchmark shows Mamba's reranking is **slower and uses
more VRAM** than the cross-encoder, so the earlier impression of a
latency/efficiency edge does **not** survive a like-for-like measurement.

---

## Gate: **PARTIAL**

| Gate criterion | Threshold | Measured | Pass? |
|---|---|---|---|
| Mamba mean test Recall@5 | ≥ 0.70 | **0.7453** | ✅ |
| Seed stability (std of R@5) | ≤ 0.08 | **0.0017** | ✅ |
| Dev→test R@5 gap | ≤ 0.05 | **0.0157** | ✅ |
| R@5 vs cross-encoder | ≥ 0.95× CE (0.674) | 0.745 (> CE 0.709) | ✅ |
| Latency **or** VRAM advantage | ≥ 25% better than CE | even with corrected batched scoring, CE is 2.7–5.4× faster and leaner | ❌ |
| Pythia does not dominate | Pythia R@5 < Mamba | 0.322 ≪ 0.745 | ✅ |

Per the spec's gate definition, beating the cross-encoder on accuracy with
cross-seed stability would be **STRONG_PASS** — but because the efficiency
advantage is gone, this lands at **PARTIAL**: Mamba clearly beats the zero-shot
and embedding baselines and edges out the cross-encoder on accuracy, but it is
**not** on the accuracy-efficiency Pareto frontier.

---

## Key metrics (identical 1000-query test split, top-100 candidates)

| Model | R@1 | R@5 (mean±std) | R@10 | MRR | NDCG@10 | cond R@5 | dev-test gap | VRAM (b1) | latency ms/q (b1 / b8) | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| **Mamba-130m LoRA (ours)** | 0.646 | **0.745 ± 0.0017** | 0.811 | 0.695 | 0.718 | 0.816 | 0.016 | 282 MB | 1425 / 55 (b1/b32, batched)¹ | 3 |
| Cross-encoder MiniLM-L6 | 0.510 | 0.709 | 0.786 | 0.600 | 0.640 | 0.776 | — | **97 MB** | **266 / 20 (b1/b32)** | 1 |
| Pythia-160m LoRA | 0.173 | 0.322 ± 0.102 | 0.398 | 0.253 | 0.276 | 0.353 | 0.028 | 646 MB | 723 / 937 | 3 |

Per-seed Mamba R@5: 0.746 / 0.743 / 0.747. Per-seed Pythia R@5: 0.368 / 0.418 / 0.181.
Candidate ceiling on this test split: Recall@100 = 0.914 (top-100 was sufficient;
top-500 not needed). "cond R@5" = Recall@5 restricted to the 914 queries whose
gold doc is within the top-100 candidates.

Mamba per-difficulty test R@5 (seed0): easy 0.645, medium 0.746, hard 0.844,
adversarial 0.781. Per-conditions: 1→1.00, 2→1.00, 3→0.72, 4+→0.65 (multi-condition
queries are the hard part; single/double-condition queries are near-solved).

¹ Latency numbers here use the **corrected batched scorer** from R3.4. The
earlier figures (2779/1607 ms) used a pseudo-batched path for Mamba (one
candidate per forward) while the cross-encoder used true batching — an unfair
comparison. See R3.4.

---

## Interpretation — the five questions

**1. Is the Mamba-LoRA reranker stable across seeds?**
Yes, strikingly so. Test R@5 = 0.7453 with std = 0.0017 across seeds {0,1,2}
(individual seeds within 0.004 of each other). This is far inside the ≤0.08
stability bar. The R3 result was not a single-seed fluke.

**2. Does it generalize under a strict train/dev/test split?**
Yes. With a fixed, disjoint, difficulty-and-conditions-stratified 3000/1000/1000
split and best-dev-checkpoint selection (early stopping on dev MRR, test touched
only once), the dev→test R@5 gap is just 0.016 (±0.004). No meaningful
overfitting to the evaluation set — the earlier 400-query / single-seed run's
train-loss→0 overfitting did not corrupt the held-out result.

**3. How does it compare to a fair Transformer-LoRA baseline?**
Decisively better. Under the *same* split, prompt, positives/negatives, LoRA
config, steps, seeds and early-stopping, Pythia-160m (a larger Transformer) only
reached R@5 = 0.322 ± 0.102 and trained unstably (dev MRR peaked at step 250 then
collapsed; all 3 seeds early-stopped). Caveat: both models used the *same*
hyperparameters, chosen around Mamba; lr = 2e-4 appears too high for Pythia and a
Pythia-specific sweep could narrow the gap. Still, "any small LM + LoRA works" is
not supported — the recipe that makes Mamba excel does not transfer to Pythia.

**4. What is the accuracy / latency / VRAM tradeoff?**
Mamba is the most *accurate* reranker but still not the most efficient — though
the gap is far smaller than first measured. After fixing the scoring path (R3.4:
correct right-padded, length-bucketed batched scoring, verified Spearman 0.9965
& 100% sign-agreement vs the single-pair scorer), Mamba's top-100 latency drops
from 1425 ms/query (batch 1) to **55 ms/query (batch 32)** — roughly a 2× gain at
batch 1 and an 8× gain at batch 8 over the old pseudo-batched numbers. But the
cross-encoder is *also* fast (266 ms→20 ms over the same batch range) and much
leaner on VRAM (97–123 MB vs Mamba's 285–936 MB, which grows with batch due to
padding). Net: the Mamba/CE latency ratio narrows from 5.4× (batch 1) to 2.7×
(batch 32) but never closes, and VRAM is consistently worse. So Mamba buys
+3.6 pts R@5 / +13.6 pts R@1 / +9.5 pts MRR at a real — but smaller than
previously reported — latency-and-memory *cost*.

**5. Should we proceed to R4 scaling?**
Not yet on the strength of an efficiency claim — that claim still does not hold,
even after the R3.4 fix. The defensible finding remains an *accuracy/quality*
edge, not a speed/footprint edge. The R3.4 work removed the unfair-measurement
objection (the comparison is now apples-to-apples batched throughput), and the
verdict is unchanged in direction: the cross-encoder stays on the
accuracy-efficiency Pareto frontier for latency and VRAM, Mamba for accuracy.
R4 scaling should be gated on either (a) a use case where the +3.6 pt R@5 is
worth a 2.7× latency cost, or (b) a further Mamba inference optimization (e.g. a
single-token relevance head, KV-free fixed-length scoring) that closes the
remaining gap.

---

## Recommended next step

**Do not jump to R4 scaling yet. Close the efficiency-measurement gap first.**

1. **[DONE in R3.4] Fixed Mamba inference efficiency measurement:** implemented a
   correct right-padded, length-bucketed batched scorer (left-padding is
   numerically wrong for SSMs; right-padding + reading the logit at each row's
   last real position is clean — verified Spearman 0.9965, 100% sign-agreement vs
   the single-pair scorer). Re-ran the benchmark: Mamba's latency improved
   substantially (1425→55 ms/query at batch 32) but the cross-encoder remains
   2.7–5.4× faster and leaner. The efficiency gap is real, not a measurement
   artifact — though much smaller than the original 6× figure. Next inference
   optimization to try if pursuing this: a single-token relevance head or
   fixed-length KV-free scoring to cut Mamba's padding-driven VRAM growth.
2. **Give Pythia a fair hyperparameter sweep** (lower lr, lr schedule) so the
   baseline comparison cannot be dismissed as "Pythia was mis-tuned."
3. Only if Mamba reaches a genuine latency-or-VRAM advantage at matched accuracy
   should R4 scaling (larger Mamba, more data) be greenlit.

### Honest framing (claims we can and cannot make)
- **Can say:** A 130M-parameter linear-attention (Mamba) model, fine-tuned with
  LoRA, is a **stable, well-generalizing reranker that matches/slightly exceeds a
  standard cross-encoder on accuracy** on this synthetic multi-condition RAG
  benchmark, and clearly beats a same-size Transformer-LoRA baseline under an
  identical recipe. Linear models show promise as rerankers on a *quality* basis.
- **Cannot say:** "Mamba replaces RAG", "linear models are inherently better", or
  "architecture superiority is proven." We also cannot claim a latency/VRAM
  advantage — even with corrected, fair batched scoring (R3.4), the cross-encoder
  is 2.7–5.4× faster and uses less VRAM. What we *can* now say is that the earlier
  6× latency gap was inflated by an unfair measurement; the true gap is smaller
  but still favors the cross-encoder.

---

## Budget

R3-validation compute (this stage, separate from prior 3.52 GPU-h): Mamba 3 seeds
(~4.3 GPU-h) + Pythia 3 seeds (~3.0 GPU-h) + cross-encoder re-eval + latency
benchmark (~0.3 GPU-h). To keep the baseline strictly fair, steps were held at
1500 for both LoRA models (reduced from the spec's 3000 after dry-run timing
showed 3000 would breach the 8 GPU-h cap; see r3_validation_budget_pause.md).
Total stayed within the 8 GPU-h cap. No CUDA/PyTorch/mamba-ssm reinstall; no
toy-KV or legacy Stage B/C runs; checkpoints/HF-cache/large parquet kept out of git.
The R3.4 follow-up added only inference benchmarking (~0.1 GPU-h), still within cap.

---

## R3.4 — Corrected batched-scoring efficiency re-measurement (follow-up)

**Motivation:** the original latency benchmark scored Mamba candidates one at a
time (pseudo-batch) while the cross-encoder used native batching — an unfair
comparison that inflated Mamba's latency gap to ~6×.

**Fix:** a correct batched yes/no scorer for causal LMs. Right-pad sequences
(pads at the tail), run one forward over `[B, T]`, and for each row read the
logit at position `L_i − 1` (the next-token distribution after that row's last
real token). For SSMs (no attention mask) this is numerically clean because the
recurrent state at `L_i − 1` has not yet consumed any pad. Candidates are
length-bucketed to minimize padding waste. **Validated:** vs the verified
single-pair scorer, Spearman = 0.9965 and sign-agreement = 1.00 on a 64-pair
sample — ranking is preserved (the only deviations are bf16 rounding, immaterial
to reranking order).

**Result (top-100, ms/query):**

| batch | Mamba (batched) | Cross-encoder | Mamba/CE ratio | Mamba VRAM | CE VRAM |
|---|---|---|---|---|---|
| 1  | 1425 | 266 | 5.4× | 285 MB | 98 MB |
| 8  | 183  | 48  | 3.8× | 438 MB | 102 MB |
| 16 | 98   | 30  | 3.3× | 611 MB | 110 MB |
| 32 | 55   | 20  | 2.7× | 936 MB | 123 MB |

**Takeaway:** batching + length-bucketing roughly halves Mamba's batch-1 latency
and cuts batch-8 latency ~8× vs the old pseudo-batch path, narrowing the gap to
the cross-encoder from 5.4× to 2.7× as batch grows. The gap narrows but does not
close, and Mamba's VRAM grows with batch (padding) while the cross-encoder's
stays flat. **Conclusion unchanged: cross-encoder remains the more efficient
reranker; Mamba remains the more accurate one.** (Note: the same batched scorer
applied to Pythia did *not* preserve ranking — Spearman 0.63 — likely a
position-id/padding interaction; Pythia is the failed baseline and its batched
efficiency is not decision-relevant, so this was not pursued further.)

Artifacts: `results/linear_rag/r3_validation_batched_latency.csv`,
`r3_validation_batched_summary.json`; refreshed
`plots/linear_rag/r3_validation_frontier.png` (3 panels: accuracy-vs-latency,
accuracy-vs-VRAM, latency-vs-batch); scripts `r3v_batch_score.py` (correctness
check), `r3v_batch_latency.py`, `r3v_refresh_frontier.py`.
