# R3.9 Real-Data Finetune — Decision

## Gate result: **PASS**

### Gate rubric (defined before eval)
| Gate | Criterion | Result |
|---|---|---|
| STRONG_PASS | Finetuned Mamba ≥ 95% of CrossEncoder R@5 **and** calibration PASS | ✗ (Mamba 0.557 = 76% of CE 0.733) |
| **PASS** | Finetuned Mamba clearly beats same-size Pythia **and** calibration PASS (ROC-AUC ≥ 0.80, separable pos/neg) | **✓** |
| PARTIAL | Finetune helps (R@5 rises materially over 0-shot) but loses to Pythia or fails calibration | n/a |
| FAIL | Finetune does not materially beat the synthetic 0-shot baseline | n/a |

### Why PASS (evidence)
1. **Beats same-size Pythia decisively** — SciFact test R@5 0.557 vs 0.180 (3.1×), NDCG@10 0.488 vs 0.139 (3.5×), ROC-AUC 0.906 vs 0.662; same direction on NFCorpus. Lower VRAM (786 vs 1165 MB).
2. **Calibration PASS** — pos/neg score distributions cleanly separated, ROC-AUC 0.906, PR-AUC high, mean margin 11.3 (0-shot was 0.05). The score is usable as a ranking signal, not a constant.
3. **OOD hypothesis confirmed** — finetune lifts Mamba R@5 from 0.04 → 0.557 and AUC 0.53 → 0.906 on the exact data where it had collapsed zero-shot. The zero-shot failure was synthetic→real distribution shift, **not** an architecture ceiling.

### Why NOT STRONG_PASS
- Finetuned Mamba reaches only 76% of CrossEncoder R@5 (0.557 vs 0.733) and 77% of NDCG@10.
- This is attributable to **data scale**, not architecture: only 200 SciFact training queries, and train loss → 0 (memorization regime), whereas the CrossEncoder was trained on hundreds of thousands of real MS-MARCO IR pairs. The recipe — not the model family — is the bottleneck.

## What this evidence CAN prove
- A small Mamba (130m) reranker, after a light LoRA finetune on real BEIR data, is a **calibrated, VRAM-efficient** reranker that substantially outperforms a same-size Transformer (Pythia-160m) under an identical recipe.
- The earlier zero-shot collapse was an OOD artifact, fully recoverable with real data.

## What this evidence CANNOT prove
- It does **not** show "Mamba replaces RAG" or that "linear models are inherently better."
- It is **not** a same-data PK vs CrossEncoder (CE has a real-IR-data training advantage).
- No real-data multi-seed stability claim (single seed 0 this stage).
- NFCorpus R@5 is structurally capped (BM25 top-100 candidate Recall = 0.235), so it is reference-only, not a gate.

## Single recommended next step
**Scale up the real-data training recipe** (more real BEIR train queries / mine more/better BM25 hard negatives / add a second real dataset to the train mix), then re-evaluate against the same SciFact test split.

Rationale: the only thing standing between PASS and STRONG_PASS is data scale, evidenced by train-loss→0 memorization on 200 queries. The cheapest, highest-information next experiment is to give the same architecture more real signal and see whether the CE gap closes.

**Explicitly NOT next:**
- ✗ Listwise / ranking-loss objectives — premature before the pointwise recipe is data-saturated.
- ✗ Architecture swaps or larger backbones — would confound the data-scale question.
- ✗ Moving to R4 — the R3.9 question (OOD vs architecture) is answered, but the PASS→STRONG_PASS data-scale question must close first.
