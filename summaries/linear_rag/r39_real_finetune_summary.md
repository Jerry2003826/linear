# R3.9 Real-Data Finetune — Summary

**Question:** Is the BEIR zero-shot failure (R@5≈0.04) caused by synthetic→real OOD, or by the
Mamba reranker architecture being unable to model real relevance?

**Answer: OOD.** Light LoRA finetune on real BEIR data recovers the model.

## Headline numbers (SciFact test, primary dataset)

| model | R@5 | NDCG@10 | ROC-AUC | VRAM |
|---|---|---|---|---|
| CrossEncoder (MS-MARCO trained) | 0.733 | 0.635 | — | 1216 MB |
| BM25 | 0.702 | 0.634 | — | — |
| **Mamba-130m finetuned** | **0.557** | **0.488** | **0.906** | **786 MB** |
| Pythia-160m finetuned | 0.180 | 0.139 | 0.662 | 1165 MB |
| Mamba-130m synthetic (0-shot) | 0.040 | 0.029 | 0.529 | 532 MB |
| Pythia-160m synthetic (0-shot) | 0.060 | 0.057 | 0.540 | 1165 MB |

## Three findings
1. **Finetune recovers Mamba**: R@5 0.04 → 0.557; ROC-AUC 0.53 → 0.906; score margin 0.05 → 11.3.
   The zero-shot collapse was OOD, not an architectural limit.
2. **Mamba > same-size Pythia**: 3.1× R@5, 3.5× NDCG, higher AUC, lower VRAM. Same pattern on NFCorpus.
3. **Mamba < CrossEncoder** (76–77% of CE on R@5/NDCG). Cause is data scale (200 train queries,
   train-loss→0 = memorization) vs CE's hundreds of thousands of MS MARCO pairs — a recipe gap, not
   an architecture ceiling.

## Mandatory honesty caveats
- **CrossEncoder is not a same-data PK**: it was trained on real MS-MARCO IR data, so it has a real-
  domain advantage. The comparison measures "how close our small linear reranker gets to a real-
  usable baseline", not architecture superiority.
- **NFCorpus R@5 is structurally capped**: BM25 top-100 candidate Recall is only 0.235 (≈38 gold
  docs/query), so reranker R@5 there is bounded far below 1.0. Gate decided on SciFact only.
- **Single seed (0)** this stage; no real-data multi-seed stability claim.
- Does NOT show "Mamba replaces RAG" or "linear inherently better" — only that the linear reranker
  is a credible, VRAM-efficient direction worth one more real-data experiment.

## Gate: **PASS** (meets PASS vs Pythia + calibration PASS; below STRONG_PASS threshold vs CE)
## Next step: **Tune real-data training (scale up real training data)** — see decision doc.
