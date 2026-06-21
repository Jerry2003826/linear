from __future__ import annotations

"""R2: zero-shot LM scanner/reranker (pairwise yes/no relevance)."""

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
from ..data.prompts import build_pairwise_prompt
from .candidates import load_candidates
from .scoring import sequence_logprob_for_answer


def load_models(model_name, dtype, device):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}.get(dtype, torch.float32)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    ).to(device).eval()
    return tok, model


def score_query(model, tok, query_text, cand_docs, device, yes_tok, no_tok, max_len):
    """Return list of scores aligned with cand_docs (list of (doc_id, text))."""
    if not cand_docs:
        raise ValueError("empty candidate list for query")
    scores = []
    for _, dtext in cand_docs:
        prompt = build_pairwise_prompt(query_text, dtext)
        yl = sequence_logprob_for_answer(model, tok, prompt, yes_tok, device, max_len)
        nl = sequence_logprob_for_answer(model, tok, prompt, no_tok, device, max_len)
        scores.append(yl - nl)
    return scores


def run_model(model_name, queries, doc_text_map, candidates, gold, cfg, device,
              n_queries, topk):
    import torch

    dtype = cfg.get("dtype", "bfloat16")
    yes_tok = cfg.get("yes_token", " yes")
    no_tok = cfg.get("no_token", " no")
    max_len = cfg.get("max_length", 512)

    tok, model = load_models(model_name, dtype, device)
    reset_peak_memory()

    q_subset = queries[:n_queries]
    rankings = {}
    score_rows = []
    t0 = time.time()
    total_cands = 0
    for q in q_subset:
        qid = q["query_id"]
        cand_ids = candidates.get(qid, [])[:topk]
        cand_docs = [(c, doc_text_map[c]) for c in cand_ids]
        scores = score_query(model, tok, q["query_text"], cand_docs, device,
                             yes_tok, no_tok, max_len)
        order = np.argsort(-np.array(scores))
        ranked = [cand_ids[i] for i in order]
        rankings[qid] = ranked
        total_cands += len(cand_ids)
        for rank_pos, i in enumerate(order):
            score_rows.append({"query_id": qid, "doc_id": cand_ids[i],
                               "score": float(scores[i]), "rank": rank_pos})
    elapsed = time.time() - t0
    peak = peak_vram_mb()

    m = aggregate_metrics(rankings, {q: gold[q] for q in rankings},
                          topk_list=[1, 5, 10], ndcg_k=10)
    lat_per_q = elapsed / max(len(q_subset), 1)
    lat_per_cand = elapsed / max(total_cands, 1)

    # score margin: positive (gold) vs hard-negative separation proxy
    margins = []
    for qid in rankings:
        srows = [r for r in score_rows if r["query_id"] == qid]
        gscore = next((r["score"] for r in srows if r["doc_id"] == gold[qid]), None)
        others = [r["score"] for r in srows if r["doc_id"] != gold[qid]]
        if gscore is not None and others:
            margins.append(gscore - float(np.mean(others)))
    score_margin = float(np.mean(margins)) if margins else 0.0

    del model
    torch.cuda.empty_cache()

    metrics = {
        "model": model_name, "n_queries": len(q_subset), "topk": topk,
        **m,
        "latency_ms_per_query": round(lat_per_q * 1000, 3),
        "latency_ms_per_candidate": round(lat_per_cand * 1000, 4),
        "peak_vram_mb": round(peak, 1),
        "elapsed_s": round(elapsed, 2),
        "score_margin_gold_vs_rest": round(score_margin, 4),
        "est_gpu_hours_full_5000": round(lat_per_q * 5000 / 3600, 4),
    }
    return metrics, rankings, score_rows


def main(config_path: str, n_queries: int | None = None,
         models: list | None = None) -> dict:
    import torch

    cfg = yaml.safe_load(Path(config_path).read_text())
    seed_everything(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(cfg["data_dir"])
    cand_dir = Path(cfg["candidates_dir"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    topk = cfg.get("topk", 100)

    docs = list(read_jsonl(data_dir / "docs.jsonl"))
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    gold = {q["query_id"]: q["gold_doc_id"] for q in queries}

    cand_path = cand_dir / f"r1_candidates_top{max(topk,100)}.parquet"
    candidates = load_candidates(cand_path)

    nq = n_queries or cfg.get("queries_sample_initial", 500)
    model_list = models or cfg.get("models_batch1", [])

    all_metrics = []
    all_scores = []
    for mn in model_list:
        try:
            metrics, rankings, score_rows = run_model(
                mn, queries, doc_text_map, candidates, gold, cfg, device, nq, topk
            )
            all_metrics.append(metrics)
            for r in score_rows:
                r["model"] = mn
            all_scores.extend(score_rows)
            print(f"[R2] {mn}: R@5={metrics['recall@5']:.4f} "
                  f"lat/q={metrics['latency_ms_per_query']}ms "
                  f"vram={metrics['peak_vram_mb']}MB")
        except Exception as e:
            all_metrics.append({"model": mn, "error": f"{type(e).__name__}: {e}"})
            print(f"[R2] {mn} FAILED: {e}")

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(
            out_dir / "r2_zero_shot_metrics.csv", index=False)
    if all_scores:
        pd.DataFrame(all_scores).to_parquet(
            out_dir / "r2_zero_shot_scores.parquet", index=False)
    write_json(out_dir / "r2_summary.json",
               {"n_queries": nq, "topk": topk, "metrics": all_metrics})
    return {"metrics": all_metrics, "n_queries": nq}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r2_zero_shot.yaml")
    ap.add_argument("--n_queries", type=int, default=None)
    args = ap.parse_args()
    print(main(args.config, n_queries=args.n_queries))
