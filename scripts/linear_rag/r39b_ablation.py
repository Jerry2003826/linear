#!/usr/bin/env python
"""R3.9b ablation: locate the bottleneck behind the Mamba<CrossEncoder gap on
SciFact. Fixes Mamba-130m cls-head reranker; sweeps three knobs ONE AT A TIME
around the R3.9 reference config (neg_per_pos=4, lora_r=16, max_steps=1500):

  (a) hard-neg count   : 4 / 8 / 15   (rebuilt from existing BM25 top-100, NO redownload)
  (b) LoRA rank/capacity: r16 / r32
  (c) train steps      : 750 / 1500 / 3000

Reuses r39_finetune (MODEL_CFG, load_prepped, dev_ndcg, train loop pieces) and
r39_eval_all (cls_scores_for_pairs, rankings_from, calibration). Touches the
SciFact TEST split once per config for the final number. Records dev/test
NDCG@10/R@5 + train-loss terminal value + AUC + train time/VRAM to a CSV row.

Usage:
  python r39b_ablation.py --runs ref negs8 negs15 rank32 steps750 steps3000 [--seed 0] [--dry_run]
  python r39b_ablation.py --seeds 0 1 2 --config ref   # multi-seed stability on best config
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
import r39_eval_all as EV

DS = "scifact"
OUT_CSV = REPO / "results/linear_rag/r39b_ablation_metrics.csv"

# config presets: each overrides (neg_per_pos, lora_r, max_steps) from the R3.9 reference
PRESETS = {
    "ref":     dict(neg_per_pos=4,  lora_r=16, max_steps=1500),
    "negs8":   dict(neg_per_pos=8,  lora_r=16, max_steps=1500),
    "negs15":  dict(neg_per_pos=15, lora_r=16, max_steps=1500),
    "rank32":  dict(neg_per_pos=4,  lora_r=32, max_steps=1500),
    "steps750":  dict(neg_per_pos=4, lora_r=16, max_steps=750),
    "steps3000": dict(neg_per_pos=4, lora_r=16, max_steps=3000),
}


def build_pairs_dynamic(train_qids, qtext, cand, gold, doc_text, neg_per_pos, seed):
    """Rebuild 1-pos / N-neg pairs from EXISTING BM25 top-100 candidates.
    No data download, no candidate rebuild — just re-samples negatives."""
    rng = np.random.default_rng(seed)
    rows = []
    for qid in train_qids:
        g = gold[qid]
        cand_list = cand[qid]
        pos = [d for d in g if d in doc_text]
        if not pos:
            continue
        p = rng.choice(pos)
        rows.append({"query_id": qid, "doc_id": str(p), "label": 1})
        negs = [d for d in cand_list if d not in g]
        rng.shuffle(negs)
        for nd in negs[:neg_per_pos]:
            rows.append({"query_id": qid, "doc_id": nd, "label": 0})
    return rows


def run_one(run_name, preset, seed, dry_run):
    import torch
    import torch.nn as nn
    from src.linear_rag.utils.seeds import seed_everything
    from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb

    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    neg_per_pos = preset["neg_per_pos"]
    lora_r = preset["lora_r"]
    max_steps = preset["max_steps"] if not dry_run else 30
    eval_interval = 250 if not dry_run else 15
    patience = 3
    batch_size, grad_acc, max_len = 8, 4, 512

    # build cfg with overridden lora rank
    base = FT.MODEL_CFG["mamba"]
    cfg = {**base, "lora": {**base["lora"], "r": lora_r, "alpha": 2 * lora_r}}

    doc_text, qtext, gold, cand, split, _ = FT.load_prepped(DS)
    train_qids = split["train"]
    dev_qids = split["dev"]
    test_qids = split["test"]
    if dry_run:
        train_qids = train_qids[:20]; dev_qids = dev_qids[:15]; test_qids = test_qids[:15]

    train_pairs = build_pairs_dynamic(train_qids, qtext, cand, gold, doc_text, neg_per_pos, seed)
    print(f"[{run_name} seed{seed}] neg={neg_per_pos} r={lora_r} steps={max_steps} "
          f"pairs={len(train_pairs)} ({sum(p['label'] for p in train_pairs)} pos) "
          f"dev={len(dev_qids)} test={len(test_qids)}", flush=True)

    rr = ClsReranker(cfg["name"], cfg["lora"], device, dtype)
    opt = torch.optim.AdamW(rr.trainable_params(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    ckpt_dir = REPO / f"checkpoints/linear_rag/r39b_abl/{run_name}_seed{seed}_best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_batch(bs):
        sel = np.random.randint(0, len(train_pairs), size=bs)
        items, labels = [], []
        for j in sel:
            p = train_pairs[j]
            items.append((qtext[p["query_id"]], doc_text[p["doc_id"]]))
            labels.append(p["label"])
        return items, torch.tensor(labels, device=device)

    best_ndcg, best_step, bad = -1.0, 0, 0
    curve = []
    reset_peak_memory()
    t0 = time.time()
    rr.train(); opt.zero_grad()
    last_loss = float("nan")
    for step in range(1, max_steps + 1):
        total = 0.0
        for _ in range(grad_acc):
            items, labels = get_batch(batch_size)
            ids, last = BE.encode_batch_beir(rr, items, max_len)
            logits = rr.forward_scores(ids, last)
            loss = loss_fn(logits, labels) / grad_acc
            loss.backward()
            total += loss.item()
        opt.step(); opt.zero_grad()
        last_loss = total
        if step % eval_interval == 0 or step == max_steps:
            nd, mrr, r5 = FT.dev_ndcg(rr, dev_qids, qtext, cand, doc_text, gold, max_len)
            curve.append({"step": step, "dev_ndcg@10": nd, "train_loss": total})
            print(f"  step {step}/{max_steps} loss={total:.4f} dev nDCG@10={nd:.4f} R@5={r5:.4f}", flush=True)
            if nd > best_ndcg:
                best_ndcg, best_step, bad = nd, step, 0
                rr.save(ckpt_dir)
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stop @ {step} (best dev nDCG={best_ndcg:.4f} @ {best_step})", flush=True)
                    break
            rr.train()
    train_time = time.time() - t0
    peak = peak_vram_mb()

    # === final TEST eval (touch test split once) on best-dev checkpoint ===
    rr.load_head(ckpt_dir)
    scored, lat_q, _ = EV.cls_scores_for_pairs(rr, test_qids, qtext, cand, doc_text)
    rk = EV.rankings_from(scored)
    gold_eval = {q: gold[q] for q in test_qids}
    m = BE.aggregate_multi(rk, gold_eval)
    cal = EV.calibration(scored, gold_eval)

    row = {
        "run": run_name, "seed": seed,
        "neg_per_pos": neg_per_pos, "lora_r": lora_r, "max_steps": max_steps,
        "best_step": best_step, "best_dev_ndcg@10": round(best_ndcg, 4),
        "test_ndcg@10": round(m["ndcg@10"], 4), "test_recall@5": round(m["recall@5"], 4),
        "test_mrr@10": round(m["mrr@10"], 4),
        "test_roc_auc": round(cal.get("roc_auc", float("nan")), 4),
        "test_pr_auc": round(cal.get("pr_auc", float("nan")), 4),
        "score_margin": round(cal.get("score_margin", float("nan")), 4),
        "train_loss_final": round(last_loss, 5),
        "train_time_s": round(train_time, 1), "peak_vram_mb": round(peak, 1),
        "n_train_pairs": len(train_pairs),
    }
    print(f"[DONE {run_name} seed{seed}] test NDCG@10={row['test_ndcg@10']} "
          f"R@5={row['test_recall@5']} AUC={row['test_roc_auc']} "
          f"loss_final={row['train_loss_final']} time={train_time:.0f}s", flush=True)
    del rr
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["ref", "negs8", "negs15", "rank32", "steps750", "steps3000"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = []
    if OUT_CSV.exists() and not args.dry_run:
        try:
            rows = pd.read_csv(OUT_CSV).to_dict("records")
        except Exception:
            rows = []
    for run_name in args.runs:
        if run_name not in PRESETS:
            print(f"!! unknown run {run_name}, skip"); continue
        for seed in args.seeds:
            row = run_one(run_name, PRESETS[run_name], seed, args.dry_run)
            rows.append(row)
            if not args.dry_run:
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"  -> wrote {OUT_CSV} ({len(rows)} rows)", flush=True)
    if args.dry_run:
        print("[DRY RUN OK]")
        print(pd.DataFrame(rows).to_string())
    else:
        print(f"[ALL DONE] {len(rows)} rows in {OUT_CSV}")


if __name__ == "__main__":
    main()
