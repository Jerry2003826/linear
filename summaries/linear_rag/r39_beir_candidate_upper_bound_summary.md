# R3.9 BEIR Candidate Upper Bound (BM25 top-100)

Reranker Recall@k is capped by candidate recall. Critical caveat below.

| dataset   |   k |   candidate_recall@k |
|:----------|----:|---------------------:|
| scifact   |  10 |               0.7757 |
| scifact   |  50 |               0.8691 |
| scifact   | 100 |               0.8731 |
| nfcorpus  |  10 |               0.1522 |
| nfcorpus  |  50 |               0.2117 |
| nfcorpus  | 100 |               0.2349 |


## Caveat

- **NFCorpus candidate Recall@100 is very low** (many relevant docs per query, single-stage BM25 cannot surface them). Any reranker's Recall@k on NFCorpus is structurally capped — NFCorpus is therefore treated as a SECONDARY/reference dataset; the R3.9 gate decision is anchored on **SciFact**, whose candidate Recall@100 is high.

- SciFact: single-gold-ish, high candidate ceiling => clean signal for whether a reranker can learn real relevance.
