from linear_rag.utils.metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    aggregate_metrics,
)
import math


def test_recall_at_k():
    ranked = [3, 7, 1, 9, 2]
    assert recall_at_k(ranked, 7, 1) == 0.0
    assert recall_at_k(ranked, 3, 1) == 1.0
    assert recall_at_k(ranked, 1, 3) == 1.0
    assert recall_at_k(ranked, 2, 4) == 0.0
    assert recall_at_k(ranked, 2, 5) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([5, 6, 7], 5) == 1.0
    assert reciprocal_rank([5, 6, 7], 6) == 0.5
    assert abs(reciprocal_rank([5, 6, 7], 7) - 1.0 / 3) < 1e-9
    assert reciprocal_rank([5, 6, 7], 99) == 0.0


def test_ndcg_at_k():
    # gold at rank 1 -> 1/log2(2) = 1.0
    assert abs(ndcg_at_k([4, 1, 2], 4, 10) - 1.0) < 1e-9
    # gold at rank 2 -> 1/log2(3)
    assert abs(ndcg_at_k([1, 4, 2], 4, 10) - 1.0 / math.log2(3)) < 1e-9
    # gold beyond k
    assert ndcg_at_k([1, 2, 3], 4, 10) == 0.0


def test_aggregate_metrics():
    rankings = {0: [10, 11, 12], 1: [21, 20, 22]}
    gold = {0: 10, 1: 20}
    m = aggregate_metrics(rankings, gold, topk_list=[1, 5], ndcg_k=10)
    # q0 gold at rank1, q1 gold at rank2
    assert abs(m["recall@1"] - 0.5) < 1e-9
    assert abs(m["recall@5"] - 1.0) < 1e-9
    assert abs(m["mrr"] - (1.0 + 0.5) / 2) < 1e-9
    assert m["n_queries"] == 2.0
