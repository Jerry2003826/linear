import pytest
from linear_rag.data.prompts import build_pairwise_prompt


def test_query_doc_order_not_swapped():
    p = build_pairwise_prompt("WHO BOUGHT CAMERA", "DOCTEXT123")
    qi = p.index("WHO BOUGHT CAMERA")
    di = p.index("DOCTEXT123")
    assert p.index("Query:") < qi < p.index("Document:") < di
    assert "Answer yes or no" in p


def test_none_raises():
    with pytest.raises(ValueError):
        build_pairwise_prompt(None, "x")
    with pytest.raises(ValueError):
        build_pairwise_prompt("x", None)
