#!/usr/bin/env python3
"""Consolidate R3.8 results: load each best ckpt (Mamba + Pythia, seeds 0/1/2),
run TEST-only eval ONCE, write full 3-seed metrics CSV + latency/vram CSV +
predictions sample. Reuses ClsReranker from the training module."""
import argparse, json, time, os
from pathlib import Path
import numpy as np, pandas as pd, torch

import sys
sys.path.insert(0, "/root/autodl-tmp/linear")
sys.path.insert(0, "/root/autodl-tmp/linear/src")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "r38mod", "/root/autodl-tmp/linear/scripts/linear_rag/r38_cls_train.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import yaml
from src.linear_rag.utils.metrics import aggregate_metrics
from src.linear_rag.utils.io import read_jsonl
from src.linear_rag.eval.candidates import load_candidates


def load_data(cfg):
    data_dir = Path(cfg["data_dir"]); cand_dir = Path(cfg["candidates_dir"])
    docs = list(read_jsonl(data_dir / "docs.jsonl"))
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    qmap = {q["query_id"]: q for q in queries}
    gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
    candidates = load_candidates(cand_dir / "r1_candidates_top100.parquet")
    train_ids, dev_ids, test_ids, _ = m.load_split(cfg["split_path"])
    return dict(qmap=qmap, candidates=candidates, doc_text_map=doc_text_map,
                gold=gold, test_ids=test_ids, dev_ids=dev_ids)


def run(config_path, tag_models):
    rows = []; pred_rows = []
    device = "cuda"
    for cfg_path, tag, model_name, ckpt_root in tag_models:
        cfg = yaml.safe_load(open(cfg_path))
        tcfg = cfg["train"]
        max_len = tcfg["max_len"]; eval_bs = tcfg.get("eval_batch_size", 16)
        topk = cfg.get("eval_topk", 100)
        dtype = torch.bfloat16 if tcfg.get("bf16", True) else torch.float32
        data = load_data(cfg)
        qmap = data["qmap"]; candidates = data["candidates"]
        doc_text_map = data["doc_text_map"]; gold = data["gold"]
        test_ids = data["test_ids"]; dev_ids = data["dev_ids"]
        for seed in [0, 1, 2]:
            best = Path(ckpt_root) / f"seed{seed}_best"
            if not best.exists():
                print(f"[skip] {tag} seed{seed} missing"); continue
            rr = m.ClsReranker(model_name, cfg["lora"], device, dtype)
            rr.load_head(best)
            rr.backbone.load_adapter(str(best), adapter_name="default")
            rr.backbone.set_adapter("default")
            m.reset_peak_memory()
            test_rank, lat_q, lat_c = m.rerank_eval(
                rr, test_ids, qmap, candidates, doc_text_map, gold,
                max_len, eval_bs, topk)
            peak = m.peak_vram_mb()
            tm = aggregate_metrics(test_rank, {q: gold[q] for q in test_rank},
                                   topk_list=[1, 5, 10], ndcg_k=10)
            cond = m.conditional_split(test_rank, gold, candidates, topk)
            cm = aggregate_metrics(cond, {q: gold[q] for q in cond},
                                   topk_list=[1, 5, 10], ndcg_k=10)
            dev_rank = m.rerank_eval(rr, dev_ids, qmap, candidates, doc_text_map,
                                     gold, max_len, eval_bs, topk)[0]
            dfm = aggregate_metrics(dev_rank, {q: gold[q] for q in dev_rank},
                                    topk_list=[1, 5, 10], ndcg_k=10)
            gap = abs(dfm["recall@5"] - tm["recall@5"])
            rows.append({
                "model": model_name, "tag": tag, "seed": seed,
                "scoring_type": "classification_head",
                "test_recall@1": round(tm["recall@1"], 4),
                "test_recall@5": round(tm["recall@5"], 4),
                "test_recall@10": round(tm["recall@10"], 4),
                "test_mrr": round(tm["mrr"], 4),
                "test_ndcg@10": round(tm["ndcg@10"], 4),
                "test_cond_recall@5": round(cm["recall@5"], 4),
                "dev_recall@5": round(dfm["recall@5"], 4),
                "dev_mrr": round(dfm["mrr"], 4),
                "dev_test_r5_gap": round(gap, 4),
                "peak_vram_mb": round(peak, 1),
                "eval_latency_per_q_ms": round(lat_q * 1000, 3),
                "eval_latency_per_cand_ms": round(lat_c * 1000, 4),
            })
            for qid in test_ids[:5]:
                pred_rows.append({"tag": tag, "seed": seed, "query_id": qid,
                                  "gold": gold[qid],
                                  "top5": ",".join(map(str, test_rank[qid][:5]))})
            print(f"[{tag} seed{seed}] R@1={tm['recall@1']:.4f} R@5={tm['recall@5']:.4f} "
                  f"R@10={tm['recall@10']:.4f} MRR={tm['mrr']:.4f} "
                  f"NDCG@10={tm['ndcg@10']:.4f} vram={peak:.0f}MB lat/q={lat_q*1000:.1f}ms",
                  flush=True)
            import gc
            del rr; gc.collect(); torch.cuda.empty_cache()
    out = Path("/root/autodl-tmp/linear/results/linear_rag")
    df = pd.DataFrame(rows)
    df.to_csv(out / "r38_classification_head_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "r38_classification_head_predictions_sample.csv", index=False)
    print("wrote r38_classification_head_metrics.csv ({} rows)".format(len(rows)))
    # aggregate
    for tag in df["tag"].unique():
        sub = df[df["tag"] == tag]
        print(f"  {tag}: R@5 {sub['test_recall@5'].mean():.4f} ± {sub['test_recall@5'].std(ddof=0):.4f} "
              f"MRR {sub['test_mrr'].mean():.4f} ± {sub['test_mrr'].std(ddof=0):.4f}")


if __name__ == "__main__":
    tag_models = [
        ("/root/autodl-tmp/linear/scripts/linear_rag/r38_mamba_cls.yaml",
         "r38_mamba_cls", "state-spaces/mamba-130m-hf",
         "/root/autodl-tmp/linear/checkpoints/linear_rag/r38_mamba_cls"),
        ("/root/autodl-tmp/linear/scripts/linear_rag/r38_pythia_cls.yaml",
         "r38_pythia_cls", "EleutherAI/pythia-160m",
         "/root/autodl-tmp/linear/checkpoints/linear_rag/r38_pythia_cls"),
    ]
    run(None, tag_models)
