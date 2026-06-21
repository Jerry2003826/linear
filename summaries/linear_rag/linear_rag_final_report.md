# Linear-RAG Interim Report

## 1. Project Vision
Linear-RAG evaluates linear-time sequence models (Mamba, Mamba-2, DeltaNet,
Gated DeltaNet, EFLA) as **internal scanner / reranker / reader** components of a
RAG pipeline, operating on candidates that BM25/embedding retrieval already
surfaced. The external knowledge base stays updatable, traceable, and auditable.
We do **not** claim linear models replace RAG or are inherently superior to
Transformers; we measure an efficiency-accuracy frontier. We stopped the toy
KV-recall line (Stages A2/B0/B0C-F) because from-scratch Transformer controls
were unstable and the task probed induction-circuit emergence rather than
RAG-internal retrieval value.

## 2. Literature Positioning
SSMs (Mamba/Mamba-2/Mamba-3); linear-attention delta-rule memory
(DeltaNet/Gated DeltaNet/EFLA); linear models for ranking (Mamba Retriever — a
dense encoder, not a state retriever; RankMamba; "SSMs are Strong Text
Rerankers"); memory-augmented RAG / model-as-index (RAG, MemoRAG, DSI as a
risk reference); recall stress tests (Zoology, MQAR — a risk boundary, not the
main experiment). See docs/literature_map.md.

## 3. Benchmark Design
synth_rag_v1: 10000 docs, 5000 queries,
multi-field conjunctive queries (two/three/four-condition, code-based,
organization+event), >= 25 hard negatives/query
(single/two-field overlap, near-synonym, swapped-entity, high-overlap), four
difficulty splits, deterministic (gold_unique_rate
1.0). See docs/benchmark_spec.md.

## 4. Baseline Results (R1, gate PASS)
- BM25 Recall@5: 0.4282, Recall@10: 0.545, MRR: 0.3468012748935157, lat/q: 7.944ms
- Embedding Recall@5: 0.3196, Recall@10: 0.429, MRR: 0.22637429158834863, lat/q: 0.13ms
- gold_in_top100: 0.9156


## 5. Zero-shot Linear Scanner Results (R2, gate PASS_SIGNAL)
- Mamba zero-shot Recall@5: 0.0426 (lat/q 878.875ms, VRAM 519.4MB)
- Pythia zero-shot Recall@5: 0.0680 (lat/q 510.996ms, VRAM 1202.6MB)
- Cross-encoder Recall@5: 0.7240 (lat/q 18.228ms)

## 6. Latency and VRAM
See results/linear_rag/latency_vram.csv and plots/linear_rag/. Per-model
forward-pass latency and peak VRAM recorded with CUDA-event timing (50 warmup +
200 measured).

## 7. Decision
- R2 signal: **PASS_SIGNAL**
- Entered R3 LoRA: **True**
- R3 result: PARTIAL [{'model': 'state-spaces/mamba-130m-hf', 'seed': 0, 'recall@1': 0.6875, 'recall@5': 0.7825, 'recall@10': 0.8475, 'mrr': 0.7331948733633515, 'ndcg@10': 0.757447897396216, 'n_queries': 400.0, 'train_time_s': 4518.3, 'peak_vram_mb': 372.4, 'steps': 3000, 'eval_queries': 400}]

## 8. Next Steps
R3 LoRA trial ran; compare LoRA reranker against cross-encoder on the accuracy-latency-VRAM frontier; if promising, scale to 3 seeds and add Mamba-2 / Gated DeltaNet candidates (R5).

---
Total GPU-hours used: 3.524 / 10.0
(approx cost 7.05 at 2.0/GPU-h).
Conclusions describe an efficiency-accuracy boundary; no claim of architectural
superiority or RAG replacement is made.
