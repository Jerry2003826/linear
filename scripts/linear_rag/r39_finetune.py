#!/usr/bin/env python
"""R3.9 Real-Data Finetune: LoRA-finetune Mamba-130m / Pythia-160m cls-head
rerankers on BEIR (SciFact primary, NFCorpus secondary) training splits.

Reuses ClsReranker (LoRA backbone + last-token Linear(H,2) head) and the
doc-truncating encode_batch_beir from r39_beir_eval.py. Trains with CE loss on
1-pos/4-neg pairs, selects best checkpoint on dev NDCG@10, early-stops
(patience=3). Saves best adapter+head per (dataset,model,seed).

Does NOT touch eval baselines — evaluation is a separate script (r39_eval_all.py)
so test set is touched exactly once after best-dev is fixed.

Usage:
  python r39_finetune.py --dataset scifact --model mamba --seed 0 [--dry_run]
"""
from __future__ import annotations
import os, sys, json, time, argparse, math
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path("/root/autodl-tmp/linear")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "linear_rag"))

import numpy as np
from r38_cls_train import ClsReranker
import r39_beir_eval as BE  # encode_batch_beir, aggregate_multi, recall/ndcg

MODEL_CFG = {
    "mamba": {"name": "state-spaces/mamba-130m-hf",
              "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
              "lr": 2e-4, "scheduler": None},
    "pythia": {"name": "EleutherAI/pythia-160m",
               "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                        "target_modules": ["query_key_value", "dense"]},
               "lr": 1e-4, "scheduler": "cosine"},
}


def load_prepped(ds):
    base = REPO / "data/beir"
    doc_text, qtext, gold = {}, {}, {}
    for line in (base / ds / "corpus.jsonl").read_text().splitlines():
        r = json.loads(line); doc_text[r["doc_id"]] = r["text"]
    for line in (base / ds / "queries.jsonl").read_text().splitlines():
        r = json.loads(line); qtext[r["query_id"]] = r["text"]
    for line in (base / ds / "qrels_test.jsonl").read_text().splitlines():
        r = json.loads(line); gold[r["query_id"]] = set(r["gold"])
    cand = json.load(open(REPO / f"data/beir_cand/{ds}_bm25_top100.json"))
    split = json.load(open(REPO / f"data/beir_splits/{ds}_r39_split.json"))
    pairs = json.load(open(REPO / f"data/beir_splits/{ds}_r39_train_pairs.json"))
    return doc_text, qtext, gold, cand, split, pairs


def dev_ndcg(rr, dev_qids, qtext, cand, doc_text, gold, max_len=512, eval_bs=16):
    import torch
    rankings = {}
    rr.eval()
    with torch.no_grad():
        for qid in dev_qids:
            qt = qtext[qid]; cids = cand[qid]
            scores = []
            for bs in range(0, len(cids), eval_bs):
                chunk = cids[bs:bs + eval_bs]
                items = [(qt, doc_text[c]) for c in chunk]
                ids, last = BE.encode_batch_beir(rr, items, max_len)
                logits = rr.forward_scores(ids, last)
                scores.extend((logits[:, 1] - logits[:, 0]).detach().cpu().tolist())
            order = np.argsort(-np.array(scores))
            rankings[qid] = [cids[i] for i in order]
    m = BE.aggregate_multi(rankings, {q: gold[q] for q in dev_qids})
    return m["ndcg@10"], m["mrr@10"], m["recall@5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=["mamba", "pythia"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_acc", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from src.linear_rag.utils.seeds import seed_everything
    from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    cfg = MODEL_CFG[args.model]

    if args.dry_run:
        args.max_steps = 30; args.eval_interval = 15

    doc_text, qtext, gold, cand, split, pairs = load_prepped(args.dataset)
    train_pairs = pairs
    dev_qids = split["dev"]
    if args.dry_run:
        train_pairs = pairs[:60]; dev_qids = dev_qids[:15]
    print(f"[{args.dataset}/{args.model}/seed{args.seed}] train_pairs={len(train_pairs)} "
          f"dev={len(dev_qids)} steps={args.max_steps}", flush=True)

    rr = ClsReranker(cfg["name"], cfg["lora"], device, dtype)
    opt = torch.optim.AdamW(rr.trainable_params(), lr=cfg["lr"])
    sched = None
    if cfg["scheduler"] == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        warmup = int(0.1 * args.max_steps)
        sched = CosineAnnealingLR(opt, T_max=max(1, args.max_steps - warmup))
    loss_fn = nn.CrossEntropyLoss()

    ckpt_dir = REPO / f"checkpoints/linear_rag/r39_{args.model}_cls/{args.dataset}_seed{args.seed}_best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # build per-step batches by cycling through shuffled pairs
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(train_pairs))

    def get_batch(bsz):
        sel = rng.choice(idx, size=bsz, replace=len(idx) < bsz)
        items, labels = [], []
        for j in sel:
            p = train_pairs[j]
            items.append((qtext[p["query_id"]], doc_text[p["doc_id"]]))
            labels.append(p["label"])
        return items, torch.tensor(labels, device=device)

    best_ndcg = -1.0; best_step = 0; bad = 0; curve = []
    reset_peak_memory()
    t0 = time.time()
    rr.train()
    opt.zero_grad()
    warmup = int(0.1 * args.max_steps)
    for step in range(1, args.max_steps + 1):
        # gradient accumulation
        total = 0.0
        for micro in range(args.grad_acc):
            items, labels = get_batch(args.batch_size)
            ids, last = BE.encode_batch_beir(rr, items, args.max_len)
            logits = rr.forward_scores(ids, last)
            loss = loss_fn(logits, labels) / args.grad_acc
            loss.backward()
            total += loss.item()
        if cfg["scheduler"] == "cosine" and step <= warmup:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * step / max(1, warmup)
        opt.step(); opt.zero_grad()
        if sched and step > warmup:
            sched.step()

        if step % args.eval_interval == 0 or step == args.max_steps:
            nd, mrr, r5 = dev_ndcg(rr, dev_qids, qtext, cand, doc_text, gold, args.max_len)
            curve.append({"step": step, "dev_ndcg@10": nd, "dev_mrr@10": mrr,
                          "dev_recall@5": r5, "train_loss": total})
            print(f"  step {step}/{args.max_steps} loss={total:.4f} "
                  f"dev nDCG@10={nd:.4f} MRR={mrr:.4f} R@5={r5:.4f}", flush=True)
            if nd > best_ndcg:
                best_ndcg = nd; best_step = step; bad = 0
                rr.save(ckpt_dir)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"  early stop at step {step} (best dev nDCG={best_ndcg:.4f} @ {best_step})",
                          flush=True)
                    break
            rr.train()

    train_time = time.time() - t0
    peak = peak_vram_mb()
    meta = {"dataset": args.dataset, "model": args.model, "seed": args.seed,
            "model_name": cfg["name"], "best_step": best_step,
            "best_dev_ndcg@10": best_ndcg, "train_time_s": round(train_time, 1),
            "peak_train_vram_mb": round(peak, 1), "max_steps": args.max_steps,
            "lr": cfg["lr"], "dry_run": args.dry_run, "curve": curve}
    (ckpt_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[DONE] {args.dataset}/{args.model}/seed{args.seed} "
          f"best dev nDCG@10={best_ndcg:.4f} @step{best_step} "
          f"time={train_time:.0f}s vram={peak:.0f}MB ckpt={ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
