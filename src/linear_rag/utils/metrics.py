from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(ranked_doc_ids: Sequence[int], gold_doc_id: int, k: int) -> float:
    """1.0 if gold is within the top-k of the ranked list, else 0.0 (single gold)."""
    return 1.0 if gold_doc_id in list(ranked_doc_ids)[:k] else 0.0


def reciprocal_rank(ranked_doc_ids: Sequence[int], gold_doc_id: int) -> float:
    for i, d in enumerate(ranked_doc_ids):
        if d == gold_doc_id:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_doc_ids: Sequence[int], gold_doc_id: int, k: int = 10) -> float:
    """Single-relevant NDCG@k. IDCG = 1 (gold at rank 1)."""
    for i, d in enumerate(list(ranked_doc_ids)[:k]):
        if d == gold_doc_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def aggregate_metrics(
    rankings: dict[int, Sequence[int]],
    gold: dict[int, int],
    topk_list: Sequence[int] = (1, 5, 10, 50, 100, 500),
    ndcg_k: int = 10,
) -> dict[str, float]:
    """Mean metrics over all query_ids present in `rankings`.

    rankings: query_id -> ranked list of doc_ids (best first)
    gold:     query_id -> gold doc_id
    """
    qids = list(rankings.keys())
    n = len(qids)
    if n == 0:
        raise ValueError("aggregate_metrics: empty rankings")
    out: dict[str, float] = {}
    for k in topk_list:
        out[f"recall@{k}"] = sum(
            recall_at_k(rankings[q], gold[q], k) for q in qids
        ) / n
    out["mrr"] = sum(reciprocal_rank(rankings[q], gold[q]) for q in qids) / n
    out[f"ndcg@{ndcg_k}"] = sum(
        ndcg_at_k(rankings[q], gold[q], ndcg_k) for q in qids
    ) / n
    out["n_queries"] = float(n)
    return out
