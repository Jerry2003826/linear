# R2 Zero-shot Scanner/Reranker Summary

- gate: **PASS_SIGNAL**
- queries evaluated: 5000, top-k=100
- reference embedding/BM25 Recall@5: 0.4282
- elapsed: 7237.9s

## Per-model rerank metrics
- state-spaces/mamba-130m-hf: R@1=0.0108 R@5=0.0426 R@10=0.0814 MRR=0.0465 lat/q=878.875ms VRAM=519.4MB margin=-0.0098
- EleutherAI/pythia-160m: R@1=0.0132 R@5=0.0680 R@10=0.1292 MRR=0.0616 lat/q=510.996ms VRAM=1202.6MB margin=0.0386

Interpretation: a signal means Mamba reranking improved Recall@5 over the coarse
retrieval order, matched Pythia at lower latency/VRAM, or produced a positive
gold-vs-rest score margin. This is an efficiency-accuracy boundary observation,
not a claim of architectural superiority.
