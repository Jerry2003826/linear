# Linear-RAG: Literature Map

This map situates Linear-RAG against related work. It is a structured reference,
not an exhaustive survey. Each entry notes *what it is* and *how it relates to
this project's specific claim* (linear models as RAG-internal
scanner / reranker / reader).

## A. Linear state-space models (SSMs)

- **Mamba** (Gu & Dao, 2023) — Selective state-space model with input-dependent
  SSM parameters and a hardware-aware parallel scan. Linear-time in sequence
  length, constant-size recurrent state. Our primary zero-shot / LoRA candidate
  (`state-spaces/mamba-130m-hf`, `mamba-370m-hf`).
- **Mamba-2** (Dao & Gu, 2024) — State-Space Duality (SSD): connects SSMs and
  linear attention, larger state, faster training. Planned for R5.
- **Mamba-3** — Successor-line SSM improvements (larger/structured state,
  improved recall). Tracked as a future R5 candidate; treat as forward-looking.

## B. Linear attention / delta-rule memory

- **DeltaNet** — Linear attention with a delta rule (online associative-memory
  update), improving associative recall over vanilla linear attention. R5
  candidate.
- **Gated DeltaNet** — Adds gating to the delta-rule update for better
  retention/forgetting control. R5 candidate.
- **Gated DeltaNet-2** — Further gated delta-rule refinement. R5 candidate.
- **EFLA** (Efficient/Expressive Forgetting Linear Attention) — Linear-attention
  family emphasizing controllable forgetting and expressive state updates. R5
  candidate.

These delta-rule / gated variants directly target the associative-recall
weakness of plain linear attention, which is the crux of whether a linear model
can rerank/read candidates reliably.

## C. Linear models for retrieval / reranking

- **Mamba Retriever** — Uses Mamba as a **dense retriever encoder** producing
  embeddings for ANN search. *Important distinction:* this is encoder-as-embedder,
  **not** an internal-state retriever and **not** a pairwise reranker over
  candidates. Our project differs: we use linear models as a
  **scanner / reranker / reader over already-retrieved candidates**.
- **RankMamba** — Investigates Mamba for document ranking / reranking; closest
  prior work to our reranker slot. We extend the question to an explicit
  efficiency–accuracy frontier vs. a cross-encoder, with VRAM/latency profiling.
- **"State Space Models are Strong Text Rerankers"** — Evidence that SSMs can be
  competitive text rerankers. Motivates our R2/R3 reranker experiments and frames
  the expected tradeoff (accuracy vs. cost).

## D. Memory-augmented RAG / model-as-index

- **RAG** (Lewis et al., 2020) — The canonical retrieve-then-read architecture;
  the external, updatable, auditable KB that Linear-RAG explicitly preserves.
- **MemoRAG** — Adds a memory model that forms global clues to guide retrieval
  over long contexts. Relevant to R6 (memory-state retrieval), but still pairs a
  memory model with an explicit store.
- **DSI (Differentiable Search Index)** — Encodes a corpus *into model
  parameters* and maps queries directly to doc-ids. This is the "model-as-index"
  extreme; we treat it as a **risk reference**: it sacrifices updatability /
  auditability, which Linear-RAG declines to do.
- **Internal retrieval / memory-state architectures** — Broader line on using a
  model's internal state as a retrieval mechanism; the long-horizon R6 direction,
  always alongside an explicit KB.

## E. Recall stress tests

- **Zoology** — Analysis showing where efficient (linear/gated) architectures
  fail on associative recall relative to attention.
- **MQAR (Multi-Query Associative Recall)** — Synthetic stress test for
  associative recall capacity of sequence models.

**Explicit framing for this project:**

- **MQAR is a risk boundary, not our main experiment.** It tells us *where* a
  fixed-state linear model can break on associative recall; we use it only as a
  later mechanistic-analysis side channel, not as the R0–R3 main line.
- **Mamba Retriever is a dense retriever encoder**, not an internal-state
  retriever. Citing it does not imply our claim; our claim is narrower and
  different.
- **This project's differentiator:** linear models as a RAG-**internal**
  scanner / reranker / reader over already-retrieved candidate chunks, evaluated
  on an explicit efficiency–accuracy frontier (Recall/MRR/NDCG vs.
  latency/VRAM), with honest gates and the option of a negative result.

> Note on citations: this map intentionally avoids fabricated venue/year/DOI
> strings for fast-moving recent work (DeltaNet variants, EFLA, Mamba-3,
> RankMamba). Names are given so the reader can locate the primary sources;
> exact bibliographic details should be confirmed against the original papers
> before external publication.
