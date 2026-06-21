# Linear-RAG: Benchmark Specification (synth_rag_v1)

## 1. Goal

A RAG-oriented synthetic retrieval benchmark whose queries require **multi-field
conjunctive matching** over structured-but-textual documents, with controllable
**hard negatives** and **difficulty splits**. This is deliberately closer to a
RAG reranking problem than to toy KV recall.

## 2. Synthetic documents schema

Each document is a JSON object (`data/synth_rag_v1/docs.jsonl`):

```json
{
  "doc_id": 17,
  "person": "Akira Tanaka",
  "location": "Kyoto",
  "object": "camera",
  "color": "red",
  "date": "2021-04-12",
  "event": "purchase",
  "organization": "Nikon Store",
  "numeric_code": "NC-48213",
  "text": "On 2021-04-12, Akira Tanaka completed a purchase of a red camera in Kyoto at Nikon Store (ref NC-48213).",
  "metadata": {"difficulty_pool": "...", "field_values": {...}}
}
```

Field vocabularies are fixed, deterministic pools (persons, locations, objects,
colors, dates, events, organizations). `numeric_code` is a structured code like
`NC-#####`. `text` is a templated natural-language realization of the fields so
that BM25 and embeddings have lexical/semantic signal. `metadata` stores the raw
field values for error analysis and hard-negative construction.

- `docs = 10000`, `seed = 42`.

## 3. Query schema

Each query (`data/synth_rag_v1/queries.jsonl`):

```json
{
  "query_id": 0,
  "query_text": "Who bought a red camera in Kyoto?",
  "gold_doc_id": 17,
  "query_type": "two-condition",
  "difficulty": "easy",
  "conditions": {"object": "camera", "color": "red", "location": "Kyoto"},
  "n_conditions": 3
}
```

- `queries = 5000`, `seed = 42`.
- Each query maps to a **unique** `gold_doc_id` (gold unique rate > 99.9%).

### Query types

- `two-condition` — two field constraints.
- `three-condition` — three field constraints.
- `four-condition` — four field constraints.
- `code-based` — constrained by `numeric_code` (+ optionally one field).
- `organization+event` — organization plus event (+ optionally another field).

## 4. Hard-negative types

For every query we construct >= 20 hard negatives drawn from docs that are
*plausible but wrong* (`data/synth_rag_v1/hard_negatives.jsonl`):

1. **single-field overlap** — shares exactly one field with the gold doc.
2. **two-field overlap** — shares exactly two fields, but is not the answer.
3. **near-synonym rewrite** — event/object/location lightly paraphrased.
4. **swapped entity** — person/location/date/object cross-mismatched.
5. **high-overlap distractor** — text nearly identical to gold, exactly one key
   field wrong.

No hard-negative may equal the gold doc_id (enforced by tests).

## 5. Difficulty splits

- `easy` — typically two-condition, lexically obvious gold.
- `medium` — three-condition.
- `hard` — four-condition / code-based with more overlapping distractors.
- `adversarial` — queries whose top hard negatives include high-overlap
  distractors and swapped-entity negatives.

Splits are recorded per-query so metrics can be reported per split.

## 6. Files

- `data/synth_rag_v1/docs.jsonl`
- `data/synth_rag_v1/queries.jsonl`
- `data/synth_rag_v1/qrels.tsv`        (`query_id \t doc_id \t relevance`)
- `data/synth_rag_v1/hard_negatives.jsonl`
- `data/synth_rag_v1/README.md`
- `summaries/linear_rag/r0_data_summary.md`

## 7. Determinism

Generation is fully deterministic given `seed=42`: a stable content hash over
docs+queries+qrels is written to the data README and asserted by tests
(`test_synth_rag_data.py`). Re-running with the same seed reproduces the same
hash.

## 8. Evaluation metrics

Computed against `qrels` (single relevant gold per query):

- `Recall@k` for `k in {1,5,10,50,100,500}` — fraction of queries whose gold
  appears in the top-k ranked candidates.
- `MRR` — mean reciprocal rank of the gold.
- `NDCG@10` — single-relevant DCG normalized by ideal (gold at rank 1).

All metrics are unit-tested on hand-checkable small samples
(`test_retrieval_metrics.py`).

## 9. Latency / VRAM measurement methodology

For every GPU inference stage (R2, R2.5, R3):

- 50 warmup iterations, then 200 measured iterations.
- `torch.cuda.Event` start/stop timing (GPU-side).
- `torch.cuda.max_memory_allocated()` for peak VRAM (reset before measurement).
- `batch_size in {1,4,8}` (run whichever fit), `topk in {50,100,500}` (top-100
  first), recorded `dtype`, `max_length`, `model_name`.

Output: `results/linear_rag/latency_vram.csv` with fields:
`model_name, stage, batch_size, topk, max_length, dtype,
latency_ms_per_query, latency_ms_per_candidate, tokens_per_sec, peak_vram_mb,
queries_per_second, notes`.

## 10. Stage gates (summary; full logic in experiment_protocol.md)

- **R0**: docs=10000, queries=5000, gold-unique > 99.9%, all tests pass → PASS.
- **R1**: BM25 + embedding both run, top100/top500 candidates written, metric
  tests pass → PASS; BM25-only (embedding download failed) → PARTIAL.
- **R2**: PASS_SIGNAL / WEAK_SIGNAL / NO_SIGNAL / FAIL per rerank deltas and
  latency/VRAM advantage.
- **R2.5**: cross-encoder runs, metrics + latency/VRAM recorded → PASS;
  too-slow/partial → PARTIAL.
- **R3**: entered only if R2 >= WEAK_SIGNAL, no data/metric bugs, dry-run <= 6
  GPU-hours, training data constructible from R1 top-100.
