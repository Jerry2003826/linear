#!/usr/bin/env python
"""R3.9 evaluation: evaluate all rerankers on each dataset's TEST split once,
using the SAME BM25 top-100 candidates. Plus calibration (ROC/PR-AUC, score
distributions) for the trained cls models.

Models per dataset:
  BM25 (candidate order)
  CrossEncoder ms-marco-MiniLM-L-6-v2
  synthetic-trained Mamba-cls  (R3.8 ckpt, if present)
  synthetic-trained Pythia-cls (R3.8 ckpt, if present)
  BEIR-finetuned Mamba-cls      (R3.9 ckpt)
  BEIR-finetuned Pythia-cls     (R3.9 ckpt)

Metrics (multi-gold): Recall@1/5/10, MRR@10, NDCG@10, latency/q, latency/cand,
peak VRAM, candidate Recall@100 ceiling, conditional Recall@5 (gold in top100).

Outputs:
  results/linear_rag/r39_real_finetune_metrics.csv
  results/linear_rag/r39_real_finetune_predictions_sample.csv
  results/linear_rag/r39_real_finetune_latency.csv
  results/linear_rag/r39_real_finetune_calibration.csv
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path("/root/autodl-tmp/linear")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "linear_rag"))

import numpy as np
import pandas as pd
from r38_cls_train import ClsReranker
import r39_beir_eval as BE
import r39_finetune as FT

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SYN_CKPT = {"mamba": REPO / "checkpoints/linear_rag/r38_mamba_cls/seed0_best",
            "pythia": REPO / "checkpoints/linear_rag/r38_pythia_cls/seed0_best"}


def cls_scores_for_pairs(rr, qids, qtext, cand, doc_text, max_len=512, eval_bs=16):
    """Return dict qid->(cids, scores) and timing."""
    import torch
    out = {}
    t0 = time.time(); n = 0
    rr.eval()
    with torch.no_grad():
        for qid in qids:
            qt = qtext[qid]; cids = cand[qid]
            sc = []
            for bs in range(0, len(cids), eval_bs):
                chunk = cids[bs:bs + eval_bs]
                items = [(qt, doc_text[c]) for c in chunk]
                ids, last = BE.encode_batch_beir(rr, items, max_len)
                logits = rr.forward_scores(ids, last)
                sc.extend((logits[:, 1] - logits[:, 0]).detach().cpu().tolist())
            out[qid] = (cids, sc); n += len(cids)
    dt = time.time() - t0
    return out, dt / max(1, len(qids)), dt / max(1, n)


def rankings_from(scored):
    rk = {}
    for qid, (cids, sc) in scored.items():
        order = np.argsort(-np.array(sc))
        rk[qid] = [cids[i] for i in order]
    return rk


def load_cls(model_key, ckpt, device, dtype):
    cfg = FT.MODEL_CFG[model_key]
    rr = ClsReranker(cfg["name"], cfg["lora"], device, dtype)
    rr.load_head(ckpt)
    rr.backbone.load_adapter(str(ckpt), adapter_name="default")
    rr.backbone.set_adapter("default")
    rr.eval()
    return rr


def calibration(scored, gold):
    """Compute pos/neg score stats + ROC-AUC + PR-AUC over all candidate pairs."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    ys, ss = [], []
    for qid, (cids, sc) in scored.items():
        g = gold[qid]
        for c, s in zip(cids, sc):
            ys.append(1 if c in g else 0); ss.append(s)
    ys = np.array(ys); ss = np.array(ss)
    pos = ss[ys == 1]; neg = ss[ys == 0]
    pos_rate = ys.mean()
    try:
        roc = roc_auc_score(ys, ss) if len(np.unique(ys)) > 1 else float("nan")
        pr = average_precision_score(ys, ss) if len(np.unique(ys)) > 1 else float("nan")
    except Exception:
        roc = pr = float("nan")
    return {"pos_score_mean": float(pos.mean()) if len(pos) else float("nan"),
            "neg_score_mean": float(neg.mean()) if len(neg) else float("nan"),
            "pos_score_std": float(pos.std()) if len(pos) else float("nan"),
            "neg_score_std": float(neg.std()) if len(neg) else float("nan"),
            "score_margin": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else float("nan"),
            "roc_auc": float(roc), "pr_auc": float(pr), "pos_rate": float(pos_rate),
            "_ys": ys, "_ss": ss}  # arrays for plotting (popped before csv)


def cond_recall5(rankings, cand, gold, topk=100):
    """Recall@5 over queries whose gold is reachable in top-k candidates."""
    qs = [q for q in rankings if len(set(cand[q][:topk]) & gold[q]) > 0]
    if not qs:
        return float("nan")
    return sum(BE.recall_at_k(rankings[q], gold[q], 5) for q in qs) / len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="scifact,nfcorpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", default="results/linear_rag/r39_real_finetune")
    args = ap.parse_args()

    import torch
    from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    datasets = args.datasets.split(",")

    rows = []; lat_rows = []; cal_rows = []; pred_rows = []; cal_arrays = {}
    for ds in datasets:
        print(f"\n===== EVAL {ds} =====", flush=True)
        doc_text, qtext, gold, cand, split, _ = FT.load_prepped(ds)
        test_qids = split["test"]
        gold_eval = {q: gold[q] for q in test_qids}
        ceil100 = sum(BE.recall_at_k(cand[q], gold_eval[q], 100) for q in test_qids) / len(test_qids)

        # BM25 baseline
        m = BE.aggregate_multi({q: cand[q] for q in test_qids}, gold_eval)
        rows.append({"dataset": ds, "model": "BM25", "scoring": "bm25", **{k: m[k] for k in
                    ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]},
                    "cand_recall@100": ceil100,
                    "cond_recall@5": cond_recall5({q: cand[q] for q in test_qids}, cand, gold_eval),
                    "latency_per_q_ms": None, "vram_mb": None, "n_queries": len(test_qids)})

        # CrossEncoder
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(CE_MODEL, max_length=512, device=device)
        reset_peak_memory(); t0 = time.time(); n = 0
        ce_rk = {}
        for qid in test_qids:
            cids = cand[qid]; pairs = [[qtext[qid], doc_text[c]] for c in cids]
            sc = ce.predict(pairs, batch_size=64, show_progress_bar=False)
            ce_rk[qid] = [cids[i] for i in np.argsort(-np.array(sc))]; n += len(cids)
        ce_dt = time.time() - t0; ce_vram = peak_vram_mb()
        m = BE.aggregate_multi(ce_rk, gold_eval)
        rows.append({"dataset": ds, "model": "CrossEncoder-MiniLM-L6", "scoring": "cross_encoder",
                    **{k: m[k] for k in ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]},
                    "cand_recall@100": ceil100, "cond_recall@5": cond_recall5(ce_rk, cand, gold_eval),
                    "latency_per_q_ms": ce_dt / len(test_qids) * 1000, "vram_mb": ce_vram,
                    "n_queries": len(test_qids)})
        del ce; torch.cuda.empty_cache()

        # cls models: synthetic + finetuned, mamba + pythia
        variants = []
        for mk in ["mamba", "pythia"]:
            syn = SYN_CKPT[mk]
            if syn.exists():
                variants.append((f"{mk}-synthetic", mk, syn))
            else:
                print(f"  [skip] {mk}-synthetic ckpt missing: {syn}", flush=True)
            ft = REPO / f"checkpoints/linear_rag/r39_{mk}_cls/{ds}_seed{args.seed}_best"
            if ft.exists():
                variants.append((f"{mk}-finetuned", mk, ft))
            else:
                print(f"  [skip] {mk}-finetuned ckpt missing: {ft}", flush=True)

        for tag, mk, ckpt in variants:
            reset_peak_memory()
            rr = load_cls(mk, ckpt, device, dtype)
            scored, lat_q, lat_c = cls_scores_for_pairs(rr, test_qids, qtext, cand, doc_text)
            vram = peak_vram_mb()
            rk = rankings_from(scored)
            m = BE.aggregate_multi(rk, gold_eval)
            model_label = ("Mamba-130m" if mk == "mamba" else "Pythia-160m") + \
                          ("-cls(finetuned)" if "finetuned" in tag else "-cls(synthetic)")
            rows.append({"dataset": ds, "model": model_label, "scoring": "cls_head",
                        **{k: m[k] for k in ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]},
                        "cand_recall@100": ceil100, "cond_recall@5": cond_recall5(rk, cand, gold_eval),
                        "latency_per_q_ms": lat_q * 1000, "vram_mb": vram, "n_queries": len(test_qids)})
            lat_rows.append({"dataset": ds, "model": model_label,
                            "latency_per_q_ms": lat_q * 1000, "latency_per_cand_ms": lat_c * 1000,
                            "vram_mb": vram})
            # calibration only for finetuned (the question of interest) + synthetic for contrast
            cal = calibration(scored, gold_eval)
            cal_arrays[(ds, model_label)] = (cal.pop("_ys"), cal.pop("_ss"))
            cal_rows.append({"dataset": ds, "model": model_label, **cal})
            # predictions sample
            if "finetuned" in tag and mk == "mamba":
                for qid in test_qids[:10]:
                    cids, sc = scored[qid]
                    top = rk[qid][:3]
                    pred_rows.append({"dataset": ds, "model": model_label, "query_id": qid,
                                      "query_text": qtext[qid][:100],
                                      "gold_in_top5": int(BE.recall_at_k(rk[qid], gold_eval[qid], 5) > 0),
                                      "top3_doc_ids": ",".join(top)})
            del rr; torch.cuda.empty_cache()
            print(f"  [{model_label}] R@5={m['recall@5']:.4f} nDCG@10={m['ndcg@10']:.4f} "
                  f"ROC-AUC={cal['roc_auc']:.3f} {lat_q*1000:.0f}ms/q VRAM={vram:.0f}MB", flush=True)

        pd.DataFrame(rows).to_csv(REPO / (args.out_prefix + "_metrics.csv"), index=False)

    # macro-average over datasets (SciFact + NFCorpus), and SciFact-only marker
    df = pd.DataFrame(rows)
    df.to_csv(REPO / (args.out_prefix + "_metrics.csv"), index=False)
    pd.DataFrame(lat_rows).to_csv(REPO / (args.out_prefix + "_latency.csv"), index=False)
    pd.DataFrame(cal_rows).to_csv(REPO / (args.out_prefix + "_calibration.csv"), index=False)
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(REPO / (args.out_prefix + "_predictions_sample.csv"), index=False)
    # dump calibration arrays for plotting
    np.savez(REPO / "results/linear_rag/r39_calibration_arrays.npz",
             **{f"{d}|{m}|ys": a[0] for (d, m), a in cal_arrays.items()},
             **{f"{d}|{m}|ss": a[1] for (d, m), a in cal_arrays.items()})

    print("\n=== METRICS ===")
    cols = ["dataset", "model", "recall@5", "ndcg@10", "mrr@10", "cond_recall@5",
            "latency_per_q_ms", "vram_mb"]
    print(df[cols].to_string(index=False))
    print("\n=== CALIBRATION ===")
    print(pd.DataFrame(cal_rows)[["dataset", "model", "roc_auc", "pr_auc",
          "score_margin", "pos_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
