# Linear-RAG: Project Vision

## 1. What Linear-RAG is

Linear-RAG is a research program that evaluates **linear-time sequence models**
(Mamba, Mamba-2, DeltaNet, Gated DeltaNet, EFLA, and related state-space /
linear-attention architectures) as **internal components of a Retrieval-Augmented
Generation (RAG) pipeline** — specifically as a *scanner*, *reranker*, *reader*,
or *memory layer* operating over candidate chunks that a conventional retriever
has already surfaced.

The external knowledge base remains a first-class, updatable, auditable, and
traceable store (BM25 index, embedding index, document store). Linear-RAG does
**not** try to fold the corpus into model weights or into a fixed recurrent
state. Instead it asks a narrower, falsifiable question:

> Once BM25 / embedding retrieval has produced top-k candidate chunks, can a
> linear-time sequence model scan, rerank, or read those candidates fast enough
> and accurately enough to shift the **efficiency–accuracy frontier** of the
> reranking / reading stage?

The output of this project is an *efficiency–accuracy boundary evaluation*, not
an architectural verdict.

## 2. Why we are moving away from the toy KV-recall line

The previous line of work (Stages A2, B0, B0C, B0D, B0E, B0F) trained small
Transformers and Mamba models from scratch on synthetic key-value recall
(MQAR-style) tasks. Its findings:

- From-scratch small-Transformer controls were **highly unstable** across seeds;
  fast learning seen once (A2) did not reproduce.
- Sparse single-query, dense multi-query, `key_next_value`, and enlarged
  Transformer controls all failed to establish a stable formal control.
- These experiments mostly probed whether an **induction / lookup circuit**
  *emerges from scratch* — a mechanistic question — rather than the question the
  project actually cares about: the **value of a linear model as an internal
  retrieval/reranking layer inside RAG**.
- Continuing to burn GPU hours on toy KV recall has poor expected value.

Consequently, the new stage explicitly:

- does **not** run the old Stage B / Stage C;
- does **not** train Transformer controls from scratch;
- does **not** continue B0G/B0H formatting fine-tunes;
- treats toy-KV / MQAR only as a **later mechanistic-analysis side channel**,
  never as the current main line.

The new main line is: build a **RAG-oriented synthetic benchmark**, then
evaluate BM25, embedding retrieval, a cross-encoder reranker, a Mamba scanner,
and a Pythia scanner, to judge whether linear models add value as an internal
retrieval layer.

## 3. Relationship between Linear-RAG and conventional RAG

Linear-RAG is **complementary to**, not a replacement for, conventional RAG:

| Conventional RAG stage          | Linear-RAG stance                                  |
|---------------------------------|----------------------------------------------------|
| External KB (docs, chunks)      | Kept as-is: updatable, traceable, auditable        |
| Coarse retrieval (BM25/embed)   | Kept as-is: provides top-k candidate chunks        |
| Reranking                       | **Candidate slot** for a linear scanner/reranker   |
| Reading / answer synthesis      | **Candidate slot** for a linear reader/memory layer|

The linear model is inserted *after* coarse retrieval, operating on a bounded
candidate set, where its linear-time, low-KV-cache, low-VRAM properties can in
principle reduce the cost of reading / reranking long candidate contexts.

## 4. Why we do NOT claim "Linear models replace RAG"

We deliberately avoid four overclaims:

1. "Mamba replaces RAG." — The external KB is retained precisely because
   updatability, sourcing, and auditability matter; weights/state are not a
   substitute for a maintained corpus.
2. "Linear models are inherently better than Transformers." — Linear attention
   has known recall limitations (Zoology / MQAR stress tests). Any advantage is
   a *tradeoff*, measured per-task, not an intrinsic superiority.
3. "A fixed state can remember unbounded content." — Fixed-size recurrent states
   have finite capacity; this is a risk boundary, not a feature.
4. "Architectural superiority is proven." — Nothing here is proven; we report an
   efficiency–accuracy frontier with explicit gates and honest negative results.

The honest framing is: **"Mamba / linear sequence models as an efficiency–accuracy
boundary evaluation for RAG-internal scanner / reranker / reader."**

## 5. Why the first stage targets scanner / reranker / reader

Reranking and reading operate on a **bounded candidate set** (e.g. top-100
chunks). This is the natural place for a linear model to demonstrate value:

- The candidate budget caps the sequence length, so latency/VRAM comparisons are
  controlled and reproducible.
- Reranking quality has clean metrics (Recall@k, MRR, NDCG@10) against gold.
- It avoids the unstable from-scratch training regime of the old toy-KV line; we
  can start **zero-shot** and only fine-tune (LoRA) if a signal appears.
- It isolates the project's real question (internal retrieval value) from the
  mechanistic question (does an induction circuit emerge).

## 6. Long-term trajectory: toward learned linear index / memory-state retrieval

The staged roadmap (R0–R6) is deliberately incremental:

- **R0** — Relocation + benchmark design (RAG-oriented synthetic benchmark).
- **R1** — BM25 / embedding coarse-retrieval baselines.
- **R2** — Zero-shot Mamba / Pythia scanner-reranker.
- **R2.5** — Cross-encoder reranker (strong accuracy baseline).
- **R3** — Mamba LoRA reranker (light fine-tune), only if R2 shows a signal.
- **R4** — top-k scaling, latency, VRAM frontier curves.
- **R5** — Stronger linear candidates: Mamba-2, DeltaNet, Gated DeltaNet, EFLA.
- **R6** — Learned linear index / memory-state retrieval exploration, where the
  recurrent state itself begins to act as a learned, queryable index — always
  alongside (not replacing) the auditable external KB.

The endpoint is a principled understanding of *where on the
efficiency–accuracy frontier* linear models help inside RAG, and whether a
learned linear/memory-state index can complement (never silently replace) an
explicit, auditable knowledge store.
