# R3 Candidate Upper-Bound Audit

Candidate recall = fraction of queries whose gold doc is present in the
BM25 candidate set (reranking can never exceed this ceiling).

- **Test Recall@100 = 0.9140**, Test Recall@500 = 0.9970
- top100 ceiling sufficient (>=0.75)

## Full table

| scope   | subset                 |   n_queries |   candidate_recall@100 |   candidate_recall@500 |
|:--------|:-----------------------|------------:|-----------------------:|-----------------------:|
| all5000 | overall                |        5000 |                 0.9156 |                 0.9976 |
| test    | overall                |        1000 |                 0.914  |                 0.997  |
| test    | difficulty=easy        |         290 |                 0.869  |                 0.9931 |
| test    | difficulty=medium      |         343 |                 0.9009 |                 0.9971 |
| test    | difficulty=hard        |         262 |                 0.9733 |                 1      |
| test    | difficulty=adversarial |         105 |                 0.9333 |                 1      |
| test    | conditions=1           |          76 |                 1      |                 1      |
| test    | conditions=2           |          78 |                 1      |                 1      |
| test    | conditions=3           |         598 |                 0.8896 |                 0.995  |
| test    | conditions=4+          |         248 |                 0.9194 |                 1      |
