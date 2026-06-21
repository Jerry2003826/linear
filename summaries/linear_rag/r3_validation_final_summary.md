# R3 Validation — Final Summary

Validating whether the R3 surprise (Mamba-130m + LoRA reranker beating a
cross-encoder) holds up under a rigorous, fair protocol: fixed disjoint
train/dev/test split, 3 seeds, best-dev checkpointing with early stopping, a
fair Transformer-LoRA baseline, a same-split cross-encoder re-eval, and a
unified latency/VRAM benchmark.

---

## Result

**The accuracy result is real and robust. The efficiency advantage is not.**

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
| Latency **or** VRAM advantage | ≥ 25% better than CE | latency −6.2×, VRAM −1.9× | ❌ |
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
| **Mamba-130m LoRA (ours)** | 0.646 | **0.745 ± 0.0017** | 0.811 | 0.695 | 0.718 | 0.816 | 0.016 | 282 MB | 2779 / 1607 | 3 |
| Cross-encoder MiniLM-L6 | 0.510 | 0.709 | 0.786 | 0.600 | 0.640 | 0.776 | — | **97 MB** | **387 / 62** | 1 |
| Pythia-160m LoRA | 0.173 | 0.322 ± 0.102 | 0.398 | 0.253 | 0.276 | 0.353 | 0.028 | 646 MB | 723 / 937 | 3 |

Per-seed Mamba R@5: 0.746 / 0.743 / 0.747. Per-seed Pythia R@5: 0.368 / 0.418 / 0.181.
Candidate ceiling on this test split: Recall@100 = 0.914 (top-100 was sufficient;
top-500 not needed). "cond R@5" = Recall@5 restricted to the 914 queries whose
gold doc is within the top-100 candidates.

Mamba per-difficulty test R@5 (seed0): easy 0.645, medium 0.746, hard 0.844,
adversarial 0.781. Per-conditions: 1→1.00, 2→1.00, 3→0.72, 4+→0.65 (multi-condition
queries are the hard part; single/double-condition queries are near-solved).

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
This is where the optimistic story breaks. Mamba is the most *accurate* reranker
but the *slowest* and not the leanest: at batch 1 it is ~2.78 s/query (top-100)
vs the cross-encoder's 0.39 s/query (~7× slower), widening to ~26× at batch 8
because the cross-encoder batches efficiently while our HF Mamba scoring runs
candidates one at a time. VRAM: Mamba 282 MB vs cross-encoder 97 MB. So Mamba
buys +3.6 pts R@5 / +13.6 pts R@1 / +9.5 pts MRR over the cross-encoder at a
real latency-and-memory *cost*, not a saving.

**5. Should we proceed to R4 scaling?**
Not yet on the strength of an efficiency claim — that claim did not hold. The
defensible finding is an *accuracy/quality* edge, not a speed/footprint edge.
Before any scaling, the efficiency gap must be addressed at the implementation
level (batched/padded Mamba inference, or a single-forward scoring head) so the
comparison is apples-to-apples on throughput; only then is a fair efficiency
verdict possible.

---

## Recommended next step

**Do not jump to R4 scaling yet. Close the efficiency-measurement gap first.**

1. **Fix Mamba inference efficiency** before re-judging the tradeoff: implement a
   proper batched scoring path (the naive left-padding tried here is numerically
   wrong for SSMs, so it needs length-bucketed batching or a dedicated relevance
   head with a single forward), then re-run the latency/VRAM benchmark. The
   current verdict reflects an *un-optimized* Mamba inference path, not an
   architectural limit.
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
  "architecture superiority is proven." We also cannot currently claim a
  latency/VRAM advantage — measured head-to-head, the cross-encoder is faster and
  leaner.

---

## Budget

R3-validation compute (this stage, separate from prior 3.52 GPU-h): Mamba 3 seeds
(~4.3 GPU-h) + Pythia 3 seeds (~3.0 GPU-h) + cross-encoder re-eval + latency
benchmark (~0.3 GPU-h). To keep the baseline strictly fair, steps were held at
1500 for both LoRA models (reduced from the spec's 3000 after dry-run timing
showed 3000 would breach the 8 GPU-h cap; see r3_validation_budget_pause.md).
Total stayed within the 8 GPU-h cap. No CUDA/PyTorch/mamba-ssm reinstall; no
toy-KV or legacy Stage B/C runs; checkpoints/HF-cache/large parquet kept out of git.
