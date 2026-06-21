from __future__ import annotations

"""R2.5: cross-encoder reranker baseline."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..utils.io import read_jsonl, write_json
from ..utils.metrics import aggregate_metrics
from ..utils.seeds import seed_everything
from ..utils.gpu import reset_peak_memory, peak_vram_mb
from .candidates import load_candidates


def main(config_path: str, n_queries: int | None = None) -> dict:
    import torch
    from sentence_transformers import CrossEncoder

    cfg = yaml.safe_load(Path(config_path).read_text())
    seed_everything(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(cfg["data_dir"])
    cand_dir = Path(cfg["candidates_dir"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    topk = cfg.get("topk", 100)
    max_len = cfg.get("max_length", 512)

    docs = list(read_jsonl(data_dir / "docs.jsonl"))
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
    candidates = load_candidates(cand_dir / f"r1_candidates_top{max(topk,100)}.parquet")

    nq = n_queries or cfg.get("queries_sample_initial", 500)
    q_subset = queries[:nq]

    model = CrossEncoder(cfg["model"], max_length=max_len, device=device)
    reset_peak_memory()

    rankings = {}
    t0 = time.time()
    total_cands = 0
    for q in q_subset:
        qid = q["query_id"]
        cand_ids = candidates.get(qid, [])[:topk]
        pairs = [[q["query_text"], doc_text_map[c]] for c in cand_ids]
        scores = model.predict(pairs, batch_size=cfg.get("batch_size", 64),
                               show_progress_bar=False)
        order = np.argsort(-np.array(scores))
        rankings[qid] = [cand_ids[i] for i in order]
        total_cands += len(cand_ids)
    elapsed = time.time() - t0
    peak = peak_vram_mb()

    m = aggregate_metrics(rankings, {q: gold[q] for q in rankings},
                          topk_list=[1, 5, 10], ndcg_k=10)
    lat_q = elapsed / max(len(q_subset), 1)
    row = {
        "model": cfg["model"], "n_queries": len(q_subset), "topk": topk, **m,
        "latency_ms_per_query": round(lat_q * 1000, 3),
        "latency_ms_per_candidate": round(elapsed / max(total_cands, 1) * 1000, 4),
        "peak_vram_mb": round(peak, 1),
        "elapsed_s": round(elapsed, 2),
        "est_gpu_hours_full_5000": round(lat_q * 5000 / 3600, 4),
    }
    pd.DataFrame([row]).to_csv(out_dir / "r25_cross_encoder_metrics.csv", index=False)
    write_json(out_dir / "r25_summary.json", row)
    print(f"[R2.5] {cfg['model']}: R@5={row['recall@5']:.4f} "
          f"lat/q={row['latency_ms_per_query']}ms vram={row['peak_vram_mb']}MB")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r25_cross_encoder.yaml")
    ap.add_argument("--n_queries", type=int, default=None)
    args = ap.parse_args()
    print(main(args.config, n_queries=args.n_queries))
