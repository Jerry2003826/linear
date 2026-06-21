from __future__ import annotations

"""R3.3: Re-evaluate cross-encoder/ms-marco-MiniLM-L-6-v2 on the SAME test split.

Reranks top-100 candidates for the 1000 test queries. Reports end-to-end and
conditional (gold in candidates) metrics + per-difficulty / per-conditions
breakdown, on the identical split used by the LoRA models. (+top500 if needed.)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.linear_rag.utils.io import read_jsonl, write_json
from src.linear_rag.utils.metrics import aggregate_metrics
from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb
from src.linear_rag.eval.candidates import load_candidates


def breakdown(rankings, gold, qmap, by_field):
    groups = {}
    for qid in rankings:
        if by_field == "conditions":
            n = qmap[qid].get("n_conditions", 0)
            key = str(n) if n <= 3 else "4+"
        else:
            key = qmap[qid].get("difficulty", "?")
        groups.setdefault(key, []).append(qid)
    out = {}
    for key, qids in groups.items():
        sub = {q: rankings[q] for q in qids}
        m = aggregate_metrics(sub, {q: gold[q] for q in qids},
                              topk_list=[1, 5, 10], ndcg_k=10)
        out[key] = {"n": len(qids), "recall@5": m["recall@5"],
                    "recall@1": m["recall@1"], "mrr": m["mrr"]}
    return out


def main(topk=100, batch_size=64):
    import torch
    from sentence_transformers import CrossEncoder

    data_dir = Path("data/synth_rag_v1")
    out_dir = Path("results/linear_rag"); out_dir.mkdir(parents=True, exist_ok=True)
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    docs = {d["doc_id"]: d["text"] for d in read_jsonl(data_dir / "docs.jsonl")}
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    qmap = {q["query_id"]: q for q in queries}
    gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
    candidates = load_candidates("results/linear_rag/r1_candidates_top100.parquet")
    if topk > 100:
        candidates = load_candidates("results/linear_rag/r1_candidates_top500.parquet")
    split = json.loads(Path(
        "data/synth_rag_v1/splits/r3_validation_split.json").read_text())
    test_ids = split["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder(model_name, device=device, max_length=512)

    reset_peak_memory()
    rankings = {}
    t0 = time.time()
    n_cand_total = 0
    for qid in test_ids:
        cand_ids = candidates.get(qid, [])[:topk]
        pairs = [[qmap[qid]["query_text"], docs[c]] for c in cand_ids]
        scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        n_cand_total += len(cand_ids)
        order = np.argsort(-np.array(scores))
        rankings[qid] = [cand_ids[i] for i in order]
    dt = time.time() - t0
    peak = peak_vram_mb()

    m_e2e = aggregate_metrics(rankings, {q: gold[q] for q in rankings},
                              topk_list=[1, 5, 10], ndcg_k=10)
    cond = {q: r for q, r in rankings.items()
            if gold[q] in candidates.get(q, [])[:topk]}
    m_cond = aggregate_metrics(cond, {q: gold[q] for q in cond},
                               topk_list=[1, 5, 10], ndcg_k=10)
    bd_diff = breakdown(rankings, gold, qmap, "difficulty")
    bd_cond = breakdown(rankings, gold, qmap, "conditions")

    row = {
        "model": model_name, "tag": "cross_encoder", "topk": topk,
        "test_recall@1": m_e2e["recall@1"], "test_recall@5": m_e2e["recall@5"],
        "test_recall@10": m_e2e["recall@10"], "test_mrr": m_e2e["mrr"],
        "test_ndcg@10": m_e2e["ndcg@10"],
        "test_cond_n": int(m_cond["n_queries"]),
        "test_cond_recall@5": m_cond["recall@5"],
        "test_cond_mrr": m_cond["mrr"],
        "peak_vram_mb": round(peak, 1),
        "eval_latency_per_q_ms": round(dt / len(test_ids) * 1000, 3),
        "eval_latency_per_cand_ms": round(dt / n_cand_total * 1000, 3),
    }
    pd.DataFrame([row]).to_csv(
        out_dir / "r3_cross_encoder_same_split.csv", index=False)
    write_json(out_dir / "r3_cross_encoder_breakdown.json",
               {"difficulty": bd_diff, "conditions": bd_cond,
                "e2e": m_e2e, "conditional": m_cond})
    print(json.dumps(row, indent=2))
    print(f"TEST R@5={m_e2e['recall@5']:.4f} R@1={m_e2e['recall@1']:.4f} "
          f"MRR={m_e2e['mrr']:.4f} NDCG@10={m_e2e['ndcg@10']:.4f} "
          f"vram={peak:.0f}MB lat/q={row['eval_latency_per_q_ms']}ms")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    main(topk=args.topk, batch_size=args.batch_size)
