import pytest
from linear_rag.eval.candidates import (
    candidates_to_rows,
    validate_candidates,
)


def test_valid_candidates_pass():
    rankings = {0: [5, 6, 7], 1: [9, 8, 7]}
    rows = candidates_to_rows(rankings)
    validate_candidates(rows, expected_topk=3)
    # ranks contiguous, doc_ids unique per query
    q0 = [r for r in rows if r["query_id"] == 0]
    assert sorted(r["rank"] for r in q0) == [0, 1, 2]


def test_duplicate_doc_id_fails():
    rows = candidates_to_rows({0: [5, 5, 7]})
    with pytest.raises(ValueError):
        validate_candidates(rows)


def test_gold_in_candidate_stat():
    rankings = {0: [5, 6, 7], 1: [9, 8, 7]}
    gold = {0: 6, 1: 1}
    hit = sum(1 for q in rankings if gold[q] in rankings[q])
    assert hit == 1
