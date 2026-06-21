# synth_rag_v1

Deterministic synthetic RAG retrieval benchmark (seed=42).

- docs: 10000
- queries: 5000
- gold_unique_rate: 1.0
- min_hard_negatives per query: 25
- content_hash (docs+queries+qrels sha256): `6090c9da57fc973edb7127c41f5f40ecdcbbd18e991cc1b8d5d7f8808a7cbef5`

## Files
- docs.jsonl, queries.jsonl, qrels.tsv, hard_negatives.jsonl

## Query type counts
{'two-condition': 1452, 'organization+event': 789, 'three-condition': 1222, 'four-condition': 763, 'code-based': 774}

## Difficulty counts
{'easy': 1452, 'medium': 1710, 'hard': 1313, 'adversarial': 525}

Re-running gen_synth_rag with the same seed reproduces the same content_hash.
