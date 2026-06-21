from __future__ import annotations

"""Candidate file helpers. A candidate file is a parquet with columns:
   query_id (int), rank (int, 0-based), doc_id (int), score (float).
Each (query_id) has exactly `topk` rows with unique doc_ids, rank 0..topk-1.
"""

from pathlib import Path
from typing import Dict, List


def candidates_to_rows(
    rankings: Dict[int, List[int]],
    scores: Dict[int, List[float]] | None = None,
) -> list[dict]:
    rows = []
    for qid, dids in rankings.items():
        for rank, did in enumerate(dids):
            sc = scores[qid][rank] if scores is not None else float(-rank)
            rows.append(
                {"query_id": int(qid), "rank": int(rank),
                 "doc_id": int(did), "score": float(sc)}
            )
    return rows


def validate_candidates(rows: list[dict], expected_topk: int | None = None) -> None:
    by_q: dict[int, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["query_id"], []).append(r)
    for qid, rs in by_q.items():
        dids = [r["doc_id"] for r in rs]
        if len(dids) != len(set(dids)):
            raise ValueError(f"duplicate doc_id in candidates for query {qid}")
        ranks = sorted(r["rank"] for r in rs)
        if ranks != list(range(len(ranks))):
            raise ValueError(f"non-contiguous ranks for query {qid}")
        if expected_topk is not None and len(rs) > expected_topk:
            raise ValueError(f"too many candidates for query {qid}")


def load_candidates(path: str | Path) -> Dict[int, List[int]]:
    import pandas as pd

    df = pd.read_parquet(path)
    out: dict[int, list[int]] = {}
    for qid, grp in df.sort_values("rank").groupby("query_id"):
        out[int(qid)] = grp["doc_id"].astype(int).tolist()
    return out
