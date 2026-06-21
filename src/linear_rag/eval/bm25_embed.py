from __future__ import annotations

"""R1: BM25 + embedding retrieval baselines over synth_rag_v1."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..utils.io import read_jsonl, write_json
from ..utils.metrics import aggregate_metrics
from ..utils.seeds import seed_everything
from .candidates import candidates_to_rows, validate_candidates


def load_data(data_dir: Path):
    docs = list(read_jsonl(data_dir / "docs.jsonl"))
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
    return docs, queries, doc_ids, doc_texts, gold


def run_bm25(doc_ids, doc_texts, queries, topk_list, max_k):
    from rank_bm25 import BM25Okapi

    t0 = time.time()
    tokenized = [t.lower().replace("(", " ").replace(")", " ").split()
                 for t in doc_texts]
    bm25 = BM25Okapi(tokenized)
    build_time = time.time() - t0
    doc_ids_arr = np.array(doc_ids)

    rankings = {}
    lat = []
    for q in queries:
        qt = q["query_text"].lower().replace("?", " ").split()
        t1 = time.time()
        scores = bm25.get_scores(qt)
        idx = np.argsort(-scores)[:max_k]
        lat.append(time.time() - t1)
        rankings[q["query_id"]] = doc_ids_arr[idx].tolist()
    return rankings, build_time, float(np.mean(lat))


def run_embedding(doc_ids, doc_texts, queries, cfg, topk_list, max_k):
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(cfg["embedding_model"], device=device)
    t0 = time.time()
    doc_emb = model.encode(
        doc_texts, batch_size=cfg.get("embed_batch_size", 256),
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32")
    encode_docs_time = time.time() - t0

    dim = doc_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    t1 = time.time()
    index.add(doc_emb)
    build_time = time.time() - t1

    q_texts = [q["query_text"] for q in queries]
    q_emb = model.encode(
        q_texts, batch_size=cfg.get("embed_batch_size", 256),
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32")

    doc_ids_arr = np.array(doc_ids)
    t2 = time.time()
    _, I = index.search(q_emb, max_k)
    search_time = time.time() - t2

    rankings = {}
    for i, q in enumerate(queries):
        rankings[q["query_id"]] = doc_ids_arr[I[i]].tolist()
    lat_per_q = search_time / max(len(queries), 1)
    return rankings, build_time, encode_docs_time, lat_per_q


def compute_and_save(name, rankings, gold, topk_list, out_dir, extra):
    m = aggregate_metrics(rankings, gold, topk_list=topk_list, ndcg_k=10)
    row = {"method": name, **m, **extra}
    df = pd.DataFrame([row])
    df.to_csv(out_dir / f"r1_{name}_results.csv", index=False)
    return row


def save_candidates(rankings, scores, out_dir, topk):
    rows = candidates_to_rows(
        {q: r[:topk] for q, r in rankings.items()},
        scores={q: s[:topk] for q, s in scores.items()} if scores else None,
    )
    validate_candidates(rows, expected_topk=topk)
    pd.DataFrame(rows).to_parquet(out_dir / f"r1_candidates_top{topk}.parquet",
                                  index=False)


def main(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed_everything(int(cfg["seed"]))
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    topk_list = cfg["topk_list"]
    max_k = max(topk_list)

    _, queries, doc_ids, doc_texts, gold = load_data(data_dir)

    summary = {"topk_list": topk_list}

    # BM25
    bm25_rank, bm25_build, bm25_lat = run_bm25(
        doc_ids, doc_texts, queries, topk_list, max_k
    )
    summary["bm25"] = compute_and_save(
        "bm25", bm25_rank, gold, topk_list, out_dir,
        {"index_build_s": round(bm25_build, 3),
         "latency_ms_per_query": round(bm25_lat * 1000, 3),
         "throughput_qps": round(1.0 / bm25_lat, 2) if bm25_lat > 0 else 0},
    )

    # Embedding (may fail if model download fails)
    emb_ok = True
    emb_err = ""
    try:
        emb_rank, emb_build, emb_enc, emb_lat = run_embedding(
            doc_ids, doc_texts, queries, cfg, topk_list, max_k
        )
        summary["embedding"] = compute_and_save(
            "embedding", emb_rank, gold, topk_list, out_dir,
            {"index_build_s": round(emb_build, 3),
             "encode_docs_s": round(emb_enc, 3),
             "latency_ms_per_query": round(emb_lat * 1000, 3),
             "throughput_qps": round(1.0 / emb_lat, 2) if emb_lat > 0 else 0},
        )
    except Exception as e:
        emb_ok = False
        emb_err = f"{type(e).__name__}: {e}"
        summary["embedding_error"] = emb_err

    # candidate files: choose the source with the better top-100 recall ceiling
    # (reranking cannot recover gold that is absent from the candidate set).
    cand_topk = max(cfg.get("candidate_topk_save", [100, 500]) + [100])
    bm25_ceiling = sum(1 for q in bm25_rank if gold[q] in bm25_rank[q][:100]) / len(bm25_rank)
    if emb_ok:
        emb_ceiling = sum(1 for q in emb_rank if gold[q] in emb_rank[q][:100]) / len(emb_rank)
        if bm25_ceiling >= emb_ceiling:
            cand_rank, cand_source = bm25_rank, "bm25"
        else:
            cand_rank, cand_source = emb_rank, "embedding"
        summary["bm25_gold_in_top100"] = round(bm25_ceiling, 6)
        summary["embedding_gold_in_top100"] = round(emb_ceiling, 6)
    else:
        cand_rank, cand_source = bm25_rank, "bm25"
        summary["bm25_gold_in_top100"] = round(bm25_ceiling, 6)
    for k in cfg.get("candidate_topk_save", [100, 500]):
        save_candidates(cand_rank, None, out_dir, k)
    summary["candidate_source"] = cand_source

    # gold-in-candidate stats
    for k in cfg.get("candidate_topk_save", [100, 500]):
        hit = sum(1 for q in cand_rank if gold[q] in cand_rank[q][:k])
        summary[f"gold_in_top{k}"] = round(hit / len(cand_rank), 6)

    write_json(out_dir / "r1_summary.json", summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1_baseline.yaml")
    args = ap.parse_args()
    print(main(args.config))
