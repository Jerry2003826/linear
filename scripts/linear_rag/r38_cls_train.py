from __future__ import annotations
"""R3.8 Classification-head reranker.

Replaces LM yes/no logprob scoring with a binary classification head on top of
the backbone's LAST-TOKEN hidden state:

    h = backbone(input_ids).last_hidden_state[:, last_real_pos, :]   # [B, H]
    logits = Linear(H, 2)(h)                                          # [B, 2]
    loss = CrossEntropy(logits, label)        label 1=relevant, 0=irrelevant
    score = logits[:, 1] - logits[:, 0]       (used for reranking)

Why last-token (not mean-pool, not a special <SCORE> token): it is the simplest
stable option that works identically for the Mamba SSM and the Pythia
Transformer, needs no tokenizer/vocab changes, and matches how the yes/no
reranker already reads the final-position state. Documented in the summary.

Prompt (per spec):
    Query:
    {query}

    Document:
    {doc}

    Relevance:

Same split / candidates / positives / hard-negatives as R3.6. LoRA on the
backbone + a trainable Linear head (head always trained, full precision).

Protocol: best-dev-checkpoint (dev MRR) with early stopping; test evaluated
ONCE per seed with the best-dev checkpoint. Reports end-to-end + conditional
metrics, per-seed, latency/VRAM.

Usage:
  python r38_cls_train.py --config r38_mamba_cls.yaml [--seeds 0] [--dry_run]
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.linear_rag.utils.io import read_jsonl, write_json
from src.linear_rag.utils.metrics import aggregate_metrics
from src.linear_rag.utils.seeds import seed_everything
from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb
from src.linear_rag.eval.candidates import load_candidates
from src.linear_rag.train.rerank_lora import find_lora_targets

CLS_TEMPLATE = "Query:\n{q}\n\nDocument:\n{d}\n\nRelevance:"


def build_cls_prompt(q, d):
    return CLS_TEMPLATE.format(q=q, d=d)


def load_split(path):
    d = json.loads(Path(path).read_text())
    return d["train"], d["dev"], d["test"], d


def build_examples(qids, qmap, candidates, hard_neg_map, doc_text_map, gold,
                   neg_per_q, rng):
    examples = []
    for qid in qids:
        q = qmap[qid]; g = gold[qid]
        if g not in doc_text_map:
            continue
        examples.append((q["query_text"], doc_text_map[g], 1))
        cand = [c for c in candidates.get(qid, []) if c != g]
        negs = [n["doc_id"] for n in hard_neg_map.get(qid, [])
                if n["doc_id"] in doc_text_map and n["doc_id"] != g]
        pool = [c for c in negs if c in cand] or cand or negs
        rng.shuffle(pool)
        for nd in pool[:neg_per_q]:
            examples.append((q["query_text"], doc_text_map[nd], 0))
    rng.shuffle(examples)
    return examples


class ClsReranker:
    """Backbone (LoRA) + last-token Linear(H,2) head."""
    def __init__(self, model_name, lora_cfg, device, dtype, scheduler_cfg=None):
        import torch, torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        from peft import LoraConfig, get_peft_model
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # AutoModel => backbone that exposes last_hidden_state (no LM head)
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
        targets = lora_cfg.get("target_modules") or find_lora_targets(self.backbone)
        self.targets = targets
        lora = LoraConfig(r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
                          lora_dropout=lora_cfg["dropout"],
                          target_modules=targets, bias="none")
        self.backbone = get_peft_model(self.backbone, lora)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, 2).to(device).float()  # head in fp32
        self.hidden = hidden

    def trainable_params(self):
        ps = [p for p in self.backbone.parameters() if p.requires_grad]
        ps += list(self.head.parameters())
        return ps

    def forward_scores(self, input_ids, last_pos):
        """input_ids [B,L], last_pos [B] -> logits [B,2] (fp32)."""
        torch = self.torch
        out = self.backbone(input_ids).last_hidden_state  # [B,L,H]
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        h = out[rows, last_pos]              # [B,H]
        return self.head(h.float())          # [B,2]

    def encode_batch(self, items, max_len):
        """items: list of (q,d). Returns input_ids [B,L], last_pos [B]."""
        torch = self.torch
        seqs = []
        for q, d in items:
            ids = self.tok.encode(build_cls_prompt(q, d), add_special_tokens=False)
            if len(ids) > max_len:
                ids = ids[-max_len:]
            seqs.append(ids)
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        batch = torch.full((len(seqs), L), pad, dtype=torch.long)
        last = []
        for r, s in enumerate(seqs):
            batch[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            last.append(len(s) - 1)
        return batch.to(self.device), torch.tensor(last, device=self.device)

    def train(self):
        self.backbone.train(); self.head.train()

    def eval(self):
        self.backbone.eval(); self.head.eval()

    def save(self, path):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(p))
        self.torch.save(self.head.state_dict(), str(p / "cls_head.pt"))

    def load_head(self, path):
        sd = self.torch.load(str(Path(path) / "cls_head.pt"), map_location=self.device)
        self.head.load_state_dict(sd)


def rerank_eval(rr, eval_qids, qmap, candidates, doc_text_map, gold,
                max_len, eval_bs, topk=100):
    torch = rr.torch
    rankings = {}
    t0 = time.time(); n_cand = 0
    rr.eval()
    with torch.no_grad():
        for qid in eval_qids:
            qt = qmap[qid]["query_text"]
            cand_ids = candidates.get(qid, [])[:topk]
            scores = []
            for bs in range(0, len(cand_ids), eval_bs):
                chunk = cand_ids[bs:bs + eval_bs]
                items = [(qt, doc_text_map[c]) for c in chunk]
                ids, last = rr.encode_batch(items, max_len)
                logits = rr.forward_scores(ids, last)  # [b,2]
                s = (logits[:, 1] - logits[:, 0]).detach().cpu().tolist()
                scores.extend(s)
            n_cand += len(cand_ids)
            order = np.argsort(-np.array(scores))
            rankings[qid] = [cand_ids[i] for i in order]
    dt = time.time() - t0
    return rankings, dt / max(1, len(eval_qids)), dt / max(1, n_cand)


def dev_metric_value(rankings, gold, metric_for_best):
    m = aggregate_metrics(rankings, {q: gold[q] for q in rankings},
                          topk_list=[1, 5, 10], ndcg_k=10)
    return (m["mrr"] if metric_for_best == "mrr" else m["recall@5"]), m


def conditional_split(rankings, gold, candidates, topk=100):
    return {qid: r for qid, r in rankings.items()
            if gold[qid] in candidates.get(qid, [])[:topk]}


def main(config_path, seeds=None, dry_run=False):
    import torch
    import torch.nn.functional as F

    cfg = yaml.safe_load(Path(config_path).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(cfg["data_dir"]); cand_dir = Path(cfg["candidates_dir"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg["model"]; tag = cfg["tag"]; tcfg = cfg["train"]
    max_len = tcfg["max_len"]; topk = cfg.get("eval_topk", 100)
    eval_bs = tcfg.get("eval_batch_size", 16)

    docs = list(read_jsonl(data_dir / "docs.jsonl"))
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    qmap = {q["query_id"]: q for q in queries}
    gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
    candidates = load_candidates(cand_dir / "r1_candidates_top100.parquet")
    hard_neg_map = {r["query_id"]: r["hard_negatives"]
                    for r in read_jsonl(data_dir / "hard_negatives.jsonl")}
    train_ids, dev_ids, test_ids, _ = load_split(cfg["split_path"])
    dev_monitor_n = tcfg.get("dev_monitor_n", len(dev_ids))
    dev_eval_ids = dev_ids[:dev_monitor_n]
    if dry_run:
        dev_eval_ids = dev_ids[:60]; test_ids = test_ids[:60]

    seeds = seeds or tcfg.get("seeds", [0])
    seed_rows = []; pred_rows = []; curves = {}

    for seed in seeds:
        seed_everything(seed)
        rng = np.random.RandomState(seed)
        dtype = torch.bfloat16 if tcfg.get("bf16", True) else torch.float32
        rr = ClsReranker(model_name, cfg["lora"], device, dtype)
        opt = torch.optim.AdamW(rr.trainable_params(), lr=tcfg["lr"])

        examples = build_examples(train_ids, qmap, candidates, hard_neg_map,
                                  doc_text_map, gold,
                                  cfg.get("negatives_per_query", 4), rng)
        steps = 30 if dry_run else tcfg["steps"]
        bs = tcfg["batch_size"]; grad_acc = tcfg["grad_acc"]
        sched = None
        if tcfg.get("scheduler") == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            warm = int(tcfg.get("warmup_ratio", 0.1) * steps)
            sched = get_cosine_schedule_with_warmup(opt, warm, steps)
        eval_interval = 30 if dry_run else tcfg.get("eval_interval", 250)
        patience = tcfg.get("early_stopping_patience", 4)
        metric_for_best = tcfg.get("metric_for_best", "mrr")

        best_dev = -1.0; best_step = 0; no_improve = 0
        best_ckpt = ckpt_dir / f"seed{seed}_best"
        curve = []
        reset_peak_memory()
        t0 = time.time(); ei = 0; step = 0
        opt.zero_grad(); rr.train(); stopped = False
        while step < steps:
            for _ in range(grad_acc):
                items = []; labels = []
                for _ in range(bs):
                    qt, dt_, lab = examples[ei % len(examples)]; ei += 1
                    items.append((qt, dt_)); labels.append(lab)
                ids, last = rr.encode_batch(items, max_len)
                logits = rr.forward_scores(ids, last)
                y = torch.tensor(labels, device=device)
                loss = F.cross_entropy(logits, y) / grad_acc
                loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad(); step += 1
            if step % eval_interval == 0 or step == steps:
                rankings, _, _ = rerank_eval(rr, dev_eval_ids, qmap, candidates,
                                             doc_text_map, gold, max_len, eval_bs, topk)
                dv, dm = dev_metric_value(rankings, gold, metric_for_best)
                curve.append({"step": step, "dev_mrr": dm["mrr"],
                              "dev_recall@5": dm["recall@5"]})
                print(f"[{tag} seed{seed}] step {step}/{steps} "
                      f"dev_mrr={dm['mrr']:.4f} dev_r5={dm['recall@5']:.4f} "
                      f"(best={best_dev:.4f}@{best_step})", flush=True)
                if dv > best_dev:
                    best_dev = dv; best_step = step; no_improve = 0
                    rr.save(best_ckpt)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        stopped = True; break
                rr.train()
            if stopped:
                break
        train_time = time.time() - t0
        peak_train = peak_vram_mb()

        # load best, eval test ONCE
        from peft import PeftModel
        rr.load_head(best_ckpt)
        # reload adapter weights into backbone
        rr.backbone.load_adapter(str(best_ckpt), adapter_name="default")
        rr.backbone.set_adapter("default")

        reset_peak_memory()
        test_rank, lat_q, lat_c = rerank_eval(rr, test_ids, qmap, candidates,
                                              doc_text_map, gold, max_len, eval_bs, topk)
        peak_eval = peak_vram_mb()
        tm = aggregate_metrics(test_rank, {q: gold[q] for q in test_rank},
                               topk_list=[1, 5, 10], ndcg_k=10)
        cond = conditional_split(test_rank, gold, candidates, topk)
        cm = aggregate_metrics(cond, {q: gold[q] for q in cond},
                               topk_list=[1, 5, 10], ndcg_k=10)
        dev_full, dvm = rerank_eval(rr, dev_ids if not dry_run else dev_eval_ids,
                                    qmap, candidates, doc_text_map, gold,
                                    max_len, eval_bs, topk)[0], None
        dfm = aggregate_metrics(dev_full, {q: gold[q] for q in dev_full},
                                topk_list=[1, 5, 10], ndcg_k=10)
        gap = abs(dfm["recall@5"] - tm["recall@5"])

        row = {
            "model": model_name, "tag": tag, "seed": seed,
            "scoring_type": "classification_head",
            "best_step": best_step, "stopped_early": stopped,
            "test_recall@1": tm["recall@1"], "test_recall@5": tm["recall@5"],
            "test_recall@10": tm["recall@10"], "test_mrr": tm["mrr"],
            "test_ndcg@10": tm["ndcg@10"],
            "test_cond_n": int(cm["n_queries"]),
            "test_cond_recall@5": cm["recall@5"], "test_cond_mrr": cm["mrr"],
            "dev_recall@5": dfm["recall@5"], "dev_mrr": dfm["mrr"],
            "dev_test_r5_gap": round(gap, 4),
            "train_time_s": round(train_time, 1),
            "peak_vram_mb": round(max(peak_train, peak_eval), 1),
            "eval_latency_per_q_ms": round(lat_q * 1000, 3),
            "eval_latency_per_cand_ms": round(lat_c * 1000, 4),
            "best_dev_metric": round(best_dev, 4),
            "lora_targets": ",".join(rr.targets),
        }
        seed_rows.append(row)
        curves[f"seed{seed}"] = curve
        # sample predictions (first 5 test queries)
        for qid in test_ids[:5]:
            pred_rows.append({"tag": tag, "seed": seed, "query_id": qid,
                              "gold": gold[qid],
                              "top5": ",".join(map(str, test_rank[qid][:5]))})
        print(f"[{tag} seed{seed}] TEST R@5={tm['recall@5']:.4f} "
              f"cond_R@5={cm['recall@5']:.4f} dev_R@5={dfm['recall@5']:.4f} "
              f"gap={gap:.4f} vram={max(peak_train,peak_eval):.0f}MB "
              f"lat/q={lat_q*1000:.1f}ms best_step={best_step}", flush=True)
        del rr, opt; import gc; gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(seed_rows)
    csv_path = out_dir / f"r38_{tag}_seed_metrics.csv"
    df.to_csv(csv_path, index=False)
    write_json(out_dir / f"r38_{tag}_curves.json", curves)
    pd.DataFrame(pred_rows).to_csv(out_dir / f"r38_{tag}_predictions_sample.csv", index=False)

    if len(df) >= 1:
        agg = {f"{c}_mean": round(float(df[c].mean()), 4)
               for c in ["test_recall@1", "test_recall@5", "test_recall@10",
                         "test_mrr", "test_ndcg@10", "peak_vram_mb",
                         "eval_latency_per_q_ms", "dev_test_r5_gap"]}
        agg.update({f"{c}_std": round(float(df[c].std(ddof=0)), 4)
                    for c in ["test_recall@5", "test_mrr"]})
        write_json(out_dir / f"r38_{tag}_summary.json",
                   {"aggregate": agg, "per_seed": seed_rows, "dry_run": dry_run})
        print(f"[{tag}] test R@5 = {agg['test_recall@5_mean']} "
              f"± {agg.get('test_recall@5_std', 0)}")
    return {"rows": seed_rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    main(args.config, seeds=seeds, dry_run=args.dry_run)
