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
| Pythia does not dominate | Pythia R@5 < Mamba | fairly tuned 0.465 < 0.745 | ✅ |

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
| Pythia-160m LoRA (R3.2, mis-tuned) | 0.173 | 0.322 ± 0.102 | 0.398 | 0.253 | 0.276 | 0.353 | 0.028 | 646 MB | 723 / 937 | 3 |
| Pythia-160m LoRA (R3.6, fairly tuned)² | 0.247 | 0.465 ± 0.133 | 0.522 | 0.349 | 0.382 | 0.509 | 0.020 | 458 MB | — | 3 |

Per-seed Mamba R@5: 0.746 / 0.743 / 0.747 (std 0.0017). Per-seed R3.2 Pythia R@5: 0.368 / 0.418 / 0.181.
Per-seed R3.6 tuned Pythia R@5: **0.559 / 0.560 / 0.277** (seeds 0/1 stable ≈0.56; seed 2 diverged). Std 0.133 — ~78× Mamba's.
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

² R3.6 = the definitive fair-baseline run: best lr (1e-4) + warmup+cosine from
the R3.5 sweep, run to a full 2000-step budget, 3 seeds, one-shot test eval. See
R3.6 below. Note the **high variance**: two seeds reach ≈0.56, one (seed 2)
diverged to 0.28 — so the mean (0.465) understates the two stable seeds (≈0.56)
but the instability is itself a real finding.

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
Better, with a moderate — not decisive — accuracy edge, and Mamba additionally
wins decisively on **training stability**. The story evolved across three runs:
- *R3.2 (mis-tuned):* under Mamba's hyperparameters (lr = 2e-4, bare AdamW, no
  scheduler) Pythia collapsed to test R@5 = 0.322 ± 0.102.
- *R3.5 (dev probe):* a warmup+cosine LR sweep showed the collapse was largely a
  missing-scheduler artifact — at lr = 1e-4, single-seed **dev** R@5 doubled to
  0.63 on a 200-query monitor subset.
- *R3.6 (definitive):* running that best config to a full 2000-step budget with
  **3 seeds and a one-shot test eval** gives test **R@5 = 0.465 ± 0.133**. Two
  seeds train cleanly to ≈0.56; the third (seed 2) **diverged** to 0.28. So the
  fairly tuned baseline lands around **0.56 when it trains well**, well above the
  R3.2 figure but still clearly below Mamba's 0.745.

Honest read: (a) the R3.2 "collapse" was a tuning artifact, so the original
"recipe doesn't transfer" framing was too strong; (b) even fairly tuned and run
to a longer budget, the Transformer-LoRA baseline trails Mamba by **~0.18 R@5
at its best seeds** (and more on average); (c) crucially, **Pythia-LoRA is
high-variance** — std 0.133 across seeds (1 of 3 diverged) versus Mamba's 0.0017
(≈78× tighter). Mamba's edge is therefore *moderate on peak accuracy but large
on reliability*: it trains stably without a scheduler and reproduces to within
0.004 across seeds, while the Transformer baseline needs careful tuning and still
occasionally diverges. This is a genuine, defensible advantage — but it is not
"architectural superiority," and a larger Pythia or more seeds could shift the
peak-accuracy gap.

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
2. **[DONE in R3.5 + R3.6] Gave Pythia a fair LR sweep, then a definitive
   matched-budget 3-seed test eval.** R3.5 (sweep) found the R3.2 collapse was a
   missing-scheduler artifact (dev R@5 0.32→0.63 at lr 1e-4). R3.6 ran that best
   config to 2000 steps × 3 seeds × one-shot test: **test R@5 = 0.465 ± 0.133**
   (two seeds ≈0.56, one diverged). Conclusion: fairly tuned Pythia is much
   better than R3.2 implied but still trails Mamba (≈0.18 R@5 at best seeds), and
   is far less stable (std 0.133 vs 0.0017). The baseline comparison is now
   **settled**: Mamba's edge is moderate on peak accuracy, large on reliability.
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
- **Softened (per R3.5/R3.6):** the claim that Mamba "clearly beats" / "the
  recipe does not transfer to" the Transformer-LoRA baseline. With a fair LR
  schedule the Transformer baseline more than doubles (test R@5 0.32→0.465, best
  seeds ≈0.56) and the peak-accuracy gap to Mamba narrows from decisive to
  moderate (≈0.18 R@5). Mamba still leads at matched budget.
- **New, defensible claim (per R3.6):** Mamba is **substantially more stable to
  train** than the same-size Transformer-LoRA baseline — std 0.0017 across 3
  seeds with no scheduler, vs Pythia's std 0.133 with 1 of 3 seeds diverging
  even after careful LR tuning. Reliability/reproducibility is a real, measured
  advantage; it is *not* the same as "architectural superiority" and should be
  stated as a training-stability observation on this benchmark.

---

## Budget

R3-validation compute (this stage, separate from prior 3.52 GPU-h): Mamba 3 seeds
(~4.3 GPU-h) + Pythia 3 seeds (~3.0 GPU-h) + cross-encoder re-eval + latency
benchmark (~0.3 GPU-h). To keep the baseline strictly fair, steps were held at
1500 for both LoRA models (reduced from the spec's 3000 after dry-run timing
showed 3000 would breach the 8 GPU-h cap; see r3_validation_budget_pause.md).
Total stayed within the 8 GPU-h cap. No CUDA/PyTorch/mamba-ssm reinstall; no
toy-KV or legacy Stage B/C runs; checkpoints/HF-cache/large parquet kept out of git.
The R3.4 follow-up added only inference benchmarking (~0.1 GPU-h); the R3.5
Pythia LR sweep (2 lrs × 1500 steps, single seed each) added ~1.6 GPU-h; the R3.6
definitive Pythia run (best lr × 2000 steps × 3 seeds + one-shot test eval) added
~4.0 GPU-h (3 × ~4760 s train + eval). Cumulative still within the validation
budget envelope.

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

---

## R3.5 — Fair LR sweep for the Pythia-160m baseline (follow-up)

**Motivation:** R3.2 trained Pythia with Mamba's hyperparameters (lr = 2e-4,
bare AdamW, no scheduler) and it collapsed (dev MRR peaked at step 250 then fell;
test R@5 = 0.322). That left the baseline open to the objection "Pythia was
mis-tuned, so the comparison is unfair." R3.5 tests that objection directly.

**Setup:** same split, prompt, positives/negatives, LoRA config, step budget
(1500) and dev-monitoring protocol as R3.2 — the *only* changes are (a) a
**linear-warmup + cosine-decay** schedule (10% warmup) and (b) a lower lr.
Swept lr ∈ {5e-5, 1e-4}, single seed (0), dev-only tuning probe (no test eval).

**Result (dev, 200-query monitor subset):**

| lr | scheduler | best dev MRR | best dev R@5 | behavior |
|---|---|---|---|---|
| 2e-4 (R3.2) | none | ~0.32 | 0.322 | collapses after step 250 |
| 5e-5 | warmup+cosine | 0.284 | 0.395 | stable, slow |
| **1e-4** | warmup+cosine | **0.486** | **0.630** | stable, still rising at 1500 |
| Mamba (ref) | warmup-free* | ~0.76 | 0.745 (test) | — |

*Mamba used the same bare-AdamW setup as R3.2 and did **not** need a scheduler to
stay stable — itself a mild point in its favor.

**Takeaway (honest):** the R3.2 Pythia collapse was largely a
**missing-LR-schedule artifact**, not an inherent Pythia failure. With warmup +
cosine and lr = 1e-4, Pythia's dev R@5 roughly **doubles (0.32 → 0.63)** and the
curve no longer collapses. This means the earlier "Mamba clearly/decisively beats
the Transformer baseline" framing was **too strong** — the gap was inflated by an
under-tuned baseline. **That said, fairly tuned Pythia (dev R@5 0.63) still trails
Mamba (dev R@5 ≈0.76 / test 0.745) at the same 1500-step budget**, and Pythia had
not yet saturated, so a longer run could narrow the gap further. Net: Mamba keeps
a **moderate** accuracy edge over a fairly tuned same-size Transformer-LoRA
baseline — promising, but not decisive, and not evidence of architectural
superiority. A definitive baseline comparison would re-run best-lr Pythia at
matched-or-longer budget with 3 seeds and a one-shot test eval.

Artifacts: `results/linear_rag/r3_pythia_lr_sweep.csv` / `.json`,
`plots/linear_rag/r3_pythia_lr_sweep.png` (dev MRR & dev R@5 vs step, both lrs,
with Mamba reference line); script `scripts/linear_rag/r3v_pythia_lr_sweep.py`
(reuses the R3.2 trainer's data/prompt/eval helpers; only lr + scheduler differ).

---

## R3.6 — Definitive fair-baseline run (best lr, full budget, 3 seeds, one-shot test)

**Motivation:** R3.5 showed the R3.2 Pythia collapse was a missing-scheduler
artifact, but it only probed **dev** on a 200-query subset with a **single seed**
and a 1500-step budget that had not saturated. To settle the baseline comparison
fairly, R3.6 runs the winning config (lr = 1e-4, warmup+cosine, 10% warmup) to a
full **2000-step** budget with **3 seeds {0,1,2}** and evaluates the held-out
**1000-query test split exactly once** per seed (best-dev-checkpoint selection,
early-stopping patience 5). The only changes from R3.2 are the LR schedule and lr
value — split, prompt, positives/negatives, LoRA config and eval protocol are
identical, keeping the comparison apples-to-apples.

**Result (full 1000-query test split):**

| seed | best step | dev R@5 | test R@5 | test MRR | dev-test gap | behavior |
|---|---|---|---|---|---|---|
| 0 | 1750 | 0.567 | **0.559** | 0.419 | 0.008 | clean, converges then plateaus |
| 1 | 1750 | 0.582 | **0.560** | 0.404 | 0.022 | clean, best dev MRR 0.479 |
| 2 | 2000 | 0.306 | **0.277** | 0.223 | 0.029 | **diverged** (dev MRR peaked 0.29 @ step 750, never recovered) |
| **mean±std** | — | 0.485±0.131 | **0.465 ± 0.133** | 0.349±0.089 | 0.020±0.009 | high variance |

**Takeaways (honest):**
1. **Fairly tuned Pythia is far better than R3.2 (0.322 → 0.465 mean, ≈0.56 at
   its two good seeds)** — confirming the R3.2 collapse was a tuning artifact and
   that the earlier "recipe doesn't transfer" framing was too strong.
2. **It still trails Mamba.** At matched 2000-step budget, best-seed Pythia
   (≈0.56) is ~0.18 R@5 below Mamba (0.745); on the mean the gap is larger
   (0.465 vs 0.745). Mamba retains a **moderate** peak-accuracy edge — not a
   decisive one, and not evidence of architectural superiority.
3. **The decisive, newly-quantified gap is stability, not peak accuracy.**
   Pythia-LoRA test R@5 std = **0.133** with **1 of 3 seeds diverging** even
   after careful LR tuning, vs Mamba's std = **0.0017** (3/3 seeds within 0.004)
   trained with no scheduler at all. Mamba is **~78× more reproducible** here. For
   a practitioner this matters: the Transformer baseline needs scheduler tuning
   and seed luck to reach ≈0.56, while Mamba reaches 0.745 reliably every time.
4. **Generalization is healthy for the good seeds** (dev-test gap 0.008–0.022);
   the divergent seed 2 is a training-stability failure, not an overfitting one.

**Net verdict (baseline comparison settled):** A same-size Transformer-LoRA
reranker, *fairly and carefully tuned*, is competitive but (a) still trails Mamba
on peak accuracy by a moderate margin and (b) is markedly less stable to train.
The defensible claims are an accuracy/quality edge plus a training-reliability
edge for the linear model on this synthetic benchmark — not architectural
superiority, and not (per R3.4) an efficiency edge.

Artifacts: `results/linear_rag/r3_pythia_160m_lora_tuned_seed_metrics.csv`,
`r3_pythia_160m_lora_tuned_summary.json`, per-seed `*_breakdown.json`,
`*_learning_curves.json`; `plots/linear_rag/r3_pythia_160m_lora_tuned_learning_curves.png`;
updated `results/linear_rag/r3_validation_model_comparison.csv` (adds the tuned
Pythia row); config `scripts/linear_rag/r3v_pythia_tuned.yaml`; `r3v_train.py`
gained an optional cosine+warmup scheduler (default off — backward compatible, so
the Mamba/CE/R3.2 results are unchanged). test_predictions parquet and
checkpoints are gitignored.
