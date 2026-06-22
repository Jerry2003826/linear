# R3.9b Decision — Bottleneck Localized; Recipe-First Next Step

## Result: bottleneck is **hard-negative count**, not real-query count

R3.9 attributed the Mamba<CrossEncoder gap to training-data scale. R3.9b's one-factor-at-a-
time ablation **refines that diagnosis**: with the same 200 SciFact training queries, raising
hard negatives from 4→15 lifts test R@5 0.557→0.688 (94% of CE) and NDCG@10 0.470→0.632
(99.5% of CE), while LoRA rank (+0.025 R@5) and steps (~0) barely move. The dominant, nearly
free lever is **more contrastive negatives per query**.

## Updated gate read
- R3.9 gate was **PASS** (beats Pythia + calibrated). R3.9b does not change the formal gate,
  but materially **strengthens the case toward STRONG_PASS**: best config reaches 94% of CE
  R@5 and 99.5% of CE NDCG@10 on a single seed, and is multi-seed stable (R@5 0.651±0.047,
  AUC 0.912±0.005).
- We deliberately do **not** declare STRONG_PASS: the 94% is single-seed; the 3-seed mean
  (0.651, ~89% of CE) is honestly below the ≥95% bar, and seed 2 (0.585) shows real variance.

## Single recommended next step
**Recipe-first, then decide on data scale.** Concretely, the next experiment is a focused
**hard-negative recipe study on the existing data**:
1. Push neg/pos beyond 15 toward the BM25 top-100 supply, and test **negative-mining quality**
   (e.g. score-banded negatives, dedup near-duplicates) rather than just count.
2. Add **multi-seed** (≥3) for every candidate config so the comparison is on means, not single
   seeds.
3. Only **after** the recipe saturates on current data, decide whether R4.0 "scale up real
   queries / add a second real dataset" is still needed to close the residual gap to CE.

Rationale: R3.9 would have sent us to expensive data-scaling; R3.9b shows the cheapest,
highest-information lever is sitting inside the data we already have. Spend GPU there first.

**Explicitly NOT next:**
- ✗ Jump straight to R4.0 data-scaling — premature before the negative recipe is saturated.
- ✗ Listwise / ranking losses — still a pointwise-recipe question.
- ✗ Architecture swaps or bigger backbones — rank32 showed capacity is not the bottleneck.

## Honest boundary (mandatory)
- This does **not** show "Mamba replaces RAG" or "linear models are inherently better."
- It shows: a 130m Mamba cls-head reranker, with a better hard-negative recipe on modest real
  data, becomes a **calibrated, VRAM-efficient reranker that lands within ~6–11% of a
  MS-MARCO-trained CrossEncoder on SciFact**, and decisively beats a same-size Transformer.
- The residual gap to CE may still require more/real data; that remains untested.
