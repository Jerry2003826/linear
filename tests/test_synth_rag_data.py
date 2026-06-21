import random
import tempfile
from pathlib import Path

import yaml

from linear_rag.data import gen_synth_rag as G
from linear_rag.utils.io import read_jsonl, content_hash


def _small_cfg(tmp, n_docs=400, n_queries=120):
    return {
        "stage": "R0",
        "seed": 42,
        "n_docs": n_docs,
        "n_queries": n_queries,
        "min_hard_negatives": 20,
        "out_dir": str(tmp),
        "query_type_weights": {
            "two-condition": 0.3,
            "three-condition": 0.25,
            "four-condition": 0.15,
            "code-based": 0.15,
            "organization+event": 0.15,
        },
        "difficulty_map": {
            "two-condition": "easy",
            "three-condition": "medium",
            "four-condition": "hard",
            "code-based": "hard",
            "organization+event": "medium",
        },
        "adversarial_fraction": 0.15,
    }


def _run(tmp, **kw):
    cfg = _small_cfg(tmp, **kw)
    cfgp = Path(tmp) / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(cfg))
    return G.main(str(cfgp)), tmp


def test_counts_and_gold_unique():
    with tempfile.TemporaryDirectory() as tmp:
        stats, _ = _run(tmp)
        docs = list(read_jsonl(Path(tmp) / "docs.jsonl"))
        queries = list(read_jsonl(Path(tmp) / "queries.jsonl"))
        assert len(docs) == 400
        assert len(queries) == 120
        # gold unique within this small set should be very high
        assert stats["gold_unique_rate"] > 0.999


def test_gold_doc_exists_and_in_corpus():
    with tempfile.TemporaryDirectory() as tmp:
        _run(tmp)
        doc_ids = {d["doc_id"] for d in read_jsonl(Path(tmp) / "docs.jsonl")}
        for q in read_jsonl(Path(tmp) / "queries.jsonl"):
            assert q["gold_doc_id"] in doc_ids


def test_qrels_alignment():
    with tempfile.TemporaryDirectory() as tmp:
        _run(tmp)
        queries = {q["query_id"]: q["gold_doc_id"]
                   for q in read_jsonl(Path(tmp) / "queries.jsonl")}
        lines = (Path(tmp) / "qrels.tsv").read_text().strip().splitlines()
        assert len(lines) == len(queries)
        for line in lines:
            qid, did, rel = line.split("\t")
            assert queries[int(qid)] == int(did)
            assert rel == "1"


def test_hard_negatives_exclude_gold_and_min_count():
    with tempfile.TemporaryDirectory() as tmp:
        _run(tmp)
        for r in read_jsonl(Path(tmp) / "hard_negatives.jsonl"):
            assert r["n_hard_negatives"] >= 20
            neg_ids = {n["doc_id"] for n in r["hard_negatives"]}
            assert r["gold_doc_id"] not in neg_ids


def test_deterministic_hash():
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        s1, _ = _run(t1)
        s2, _ = _run(t2)
        assert s1["content_hash"] == s2["content_hash"]
        h1 = content_hash(Path(t1) / "docs.jsonl", Path(t1) / "queries.jsonl",
                          Path(t1) / "qrels.tsv")
        assert h1 == s1["content_hash"]
