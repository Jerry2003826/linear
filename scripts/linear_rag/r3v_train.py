from __future__ import annotations

"""R3 Validation trainer: LoRA pairwise yes/no reranker with rigorous protocol.

Key protocol (per expert spec):
- Fixed train/dev/test split (r3_validation_split.json), SAME for all models.
- Per training query: 1 positive (gold) + >=neg_per_q hard negatives from top100.
- LoRA r=16 alpha=32 dropout=0.05, target Mamba proj modules (or model default).
- Eval on dev every eval_interval steps; track best-dev (metric_for_best).
- Early stopping: stop if no dev improvement for `patience` evals.
- Save best-dev checkpoint per seed. Eval test ONCE with best-dev checkpoint.
- Report BOTH end-to-end (all test) AND conditional (gold in candidates) metrics.
- Per-difficulty / per-conditions breakdown on test.
- Learning curves (dev metric vs step) saved per seed.

Works for both Mamba-130m-hf (R3.1) and Pythia-160m (R3.2) via --config.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.linear_rag.utils.io import read_jsonl, write_json
from src.linear_rag.utils.metrics import aggregate_metrics
from src.linear_rag.utils.seeds import seed_everything
from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb
from src.linear_rag.data.prompts import build_pairwise_prompt
from src.linear_rag.eval.candidates import load_candidates
from src.linear_rag.eval.scoring import yes_no_score_fast
from src.linear_rag.train.rerank_lora import find_lora_targets


def load_split(path):
    d = json.loads(Path(path).read_text())
    return d["train"], d["dev"], d["test"], d


def build_examples(qids, qmap, candidates, hard_neg_map, doc_text_map, gold,
                   neg_per_q, rng):
    """1 positive + neg_per_q hard negatives (from top100) per query."""
    examples = []
    for qid in qids:
        q = qmap[qid]
        g = gold[qid]
        if g not in doc_text_map:
            continue
        examples.append((q["query_text"], doc_text_map[g], 1))
        cand = [c for c in candidates.get(qid, []) if c != g]
        negs = [n["doc_id"] for n in hard_neg_map.get(qid, [])
                if n["doc_id"] in doc_text_map and n["doc_id"] != g]
        # prefer hard negatives that are also in candidate set (realistic)
        pool = [c for c in negs if c in cand] or cand or negs
        rng.shuffle(pool)
        for nd in pool[:neg_per_q]:
            examples.append((q["query_text"], doc_text_map[nd], 0))
    rng.shuffle(examples)
    return examples


def rerank_eval(model, tok, eval_qids, qmap, candidates, doc_text_map, gold,
                device, max_len, yes_tok, no_tok, topk=100):
    """Return (rankings, per_query_latency_s). End-to-end: all eval_qids."""
    rankings = {}
    t0 = time.time()
    n_cand_total = 0
    with __import__("torch").no_grad():
        for qid in eval_qids:
            q = qmap[qid]
            cand_ids = candidates.get(qid, [])[:topk]
            scores = []
            for c in cand_ids:
                prompt = build_pairwise_prompt(q["query_text"], doc_text_map[c])
                scores.append(
                    yes_no_score_fast(model, tok, prompt, device,
                                      yes_tok, no_tok, max_len))
            n_cand_total += len(cand_ids)
            order = np.argsort(-np.array(scores))
            rankings[qid] = [cand_ids[i] for i in order]
    dt = time.time() - t0
    lat_per_q = dt / max(1, len(eval_qids))
    lat_per_cand = dt / max(1, n_cand_total)
    return rankings, lat_per_q, lat_per_cand


def dev_metric_value(rankings, gold, metric_for_best):
    m = aggregate_metrics(rankings, {q: gold[q] for q in rankings},
                          topk_list=[1, 5, 10], ndcg_k=10)
    if metric_for_best == "mrr":
        return m["mrr"], m
    return m["recall@5"], m


def conditional_split(rankings, gold, candidates, topk=100):
    """Split test rankings into conditional subset (gold in candidates)."""
    cond = {}
    for qid, r in rankings.items():
        if gold[qid] in candidates.get(qid, [])[:topk]:
            cond[qid] = r
    return cond


def breakdown(rankings, gold, qmap, by_field):
    """Per-group recall@5 / mrr breakdown."""
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


def main(config_path, seeds=None, dry_run=False, dry_steps=30, dry_eval=200):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    cfg = yaml.safe_load(Path(config_path).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(cfg["data_dir"])
    cand_dir = Path(cfg["candidates_dir"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(cfg["plot_dir"]); plot_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg["model"]
    tag = cfg["tag"]  # e.g. "mamba_130m_lora" / "pythia_160m_lora"
    tcfg = cfg["train"]
    max_len = tcfg["max_len"]
    topk = cfg.get("eval_topk", 100)

    docs = list(read_jsonl(data_dir / "docs.jsonl"))
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    qmap = {q["query_id"]: q for q in queries}
    gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
    candidates = load_candidates(cand_dir / "r1_candidates_top100.parquet")
    hard_neg_map = {r["query_id"]: r["hard_negatives"]
                    for r in read_jsonl(data_dir / "hard_negatives.jsonl")}

    train_ids, dev_ids, test_ids, split_meta = load_split(cfg["split_path"])
    # dev monitoring subset for early-stopping (budget): full dev only at the end.
    dev_monitor_n = tcfg.get("dev_monitor_n", len(dev_ids))
    dev_eval_ids = dev_ids[:dev_monitor_n]
    if dry_run:
        dev_eval_ids = dev_ids[:dry_eval]
        test_ids = test_ids[:dry_eval]

    seeds = seeds or tcfg.get("seeds", [0])
    yes_tok, no_tok = " yes", " no"

    seed_rows = []
    test_pred_rows = []
    all_curves = {}

    for seed in seeds:
        seed_everything(seed)
        rng = np.random.RandomState(seed)
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if tcfg.get("bf16", True) else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device)
        targets = cfg["lora"].get("target_modules") or find_lora_targets(model)
        lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                          lora_dropout=cfg["lora"]["dropout"],
                          target_modules=targets, bias="none")
        model = get_peft_model(model, lora)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=tcfg["lr"])

        examples = build_examples(
            train_ids, qmap, candidates, hard_neg_map, doc_text_map, gold,
            cfg.get("negatives_per_query", 4), rng)

        yes_ids = tok.encode(yes_tok, add_special_tokens=False)
        no_ids = tok.encode(no_tok, add_special_tokens=False)

        steps = dry_steps if dry_run else tcfg["steps"]
        bs = tcfg["batch_size"]
        grad_acc = tcfg["grad_acc"]
        eval_interval = dry_eval if dry_run else tcfg.get("eval_interval", 250)
        patience = tcfg.get("early_stopping_patience", 4)
        metric_for_best = tcfg.get("metric_for_best", "mrr")

        best_dev = -1.0
        best_step = 0
        no_improve = 0
        best_ckpt = ckpt_dir / f"seed{seed}_best"
        curve = []

        reset_peak_memory()
        t0 = time.time()
        ei = 0
        step = 0
        opt.zero_grad()
        model.train()
        stopped_early = False
        while step < steps:
            loss_accum = 0.0
            for _ in range(grad_acc):
                for _ in range(bs):
                    qt, dt_, label = examples[ei % len(examples)]; ei += 1
                    ans_ids = yes_ids if label == 1 else no_ids
                    p_ids = tok.encode(build_pairwise_prompt(qt, dt_),
                                       add_special_tokens=False)
                    keep = max_len - len(ans_ids)
                    if len(p_ids) > keep:
                        p_ids = p_ids[-keep:]
                    inp = torch.tensor([p_ids + ans_ids], device=device)
                    out = model(inp).logits
                    lp = F.log_softmax(out.float(), dim=-1)
                    tloss = 0.0
                    for j, t in enumerate(ans_ids):
                        pos = len(p_ids) + j - 1
                        tloss = tloss - lp[0, pos, t]
                    (tloss / (bs * grad_acc)).backward()
                    loss_accum += float(tloss.item())
            opt.step(); opt.zero_grad()
            step += 1
            if step % eval_interval == 0 or step == steps:
                model.eval()
                dev_rank, _, _ = rerank_eval(
                    model, tok, dev_eval_ids, qmap, candidates, doc_text_map,
                    gold, device, max_len, yes_tok, no_tok, topk)
                dev_val, dev_full = dev_metric_value(dev_rank, gold,
                                                     metric_for_best)
                curve.append({"step": step,
                              "train_loss": loss_accum / (bs * grad_acc),
                              "dev_recall@5": dev_full["recall@5"],
                              "dev_recall@1": dev_full["recall@1"],
                              "dev_mrr": dev_full["mrr"],
                              "dev_metric_for_best": dev_val})
                print(f"[{tag} seed{seed}] step {step}/{steps} "
                      f"loss={loss_accum/(bs*grad_acc):.4f} "
                      f"dev_{metric_for_best}={dev_val:.4f} "
                      f"(best={best_dev:.4f}@{best_step})", flush=True)
                if dev_val > best_dev + 1e-5:
                    best_dev = dev_val
                    best_step = step
                    no_improve = 0
                    model.save_pretrained(str(best_ckpt))
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"[{tag} seed{seed}] EARLY STOP at step {step} "
                              f"(no improve {no_improve} evals)", flush=True)
                        stopped_early = True
                        break
                model.train()
        train_time = time.time() - t0
        peak = peak_vram_mb()
        all_curves[seed] = curve

        # ---- load best-dev checkpoint, eval TEST once ----
        del model
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device)
        from peft import PeftModel
        if best_ckpt.exists():
            best_model = PeftModel.from_pretrained(base, str(best_ckpt)).to(device)
        else:
            # never improved (shouldn't happen): use base+lora untrained
            best_model = get_peft_model(base, lora)
        best_model.eval()

        test_rank, lat_q, lat_cand = rerank_eval(
            best_model, tok, test_ids, qmap, candidates, doc_text_map, gold,
            device, max_len, yes_tok, no_tok, topk)

        # end-to-end (all test)
        m_e2e = aggregate_metrics(test_rank, {q: gold[q] for q in test_rank},
                                  topk_list=[1, 5, 10], ndcg_k=10)
        # conditional (gold in candidates)
        cond_rank = conditional_split(test_rank, gold, candidates, topk)
        m_cond = aggregate_metrics(cond_rank, {q: gold[q] for q in cond_rank},
                                   topk_list=[1, 5, 10], ndcg_k=10)
        # dev best metrics at best_step (re-eval full dev with best ckpt)
        final_dev_ids = dev_ids if not dry_run else dev_ids[:dry_eval]
        dev_rank_best, _, _ = rerank_eval(
            best_model, tok, final_dev_ids, qmap, candidates, doc_text_map, gold,
            device, max_len, yes_tok, no_tok, topk)
        m_dev = aggregate_metrics(dev_rank_best,
                                  {q: gold[q] for q in dev_rank_best},
                                  topk_list=[1, 5, 10], ndcg_k=10)

        bd_diff = breakdown(test_rank, gold, qmap, "difficulty")
        bd_cond = breakdown(test_rank, gold, qmap, "conditions")

        row = {
            "model": model_name, "tag": tag, "seed": seed,
            "best_step": best_step, "stopped_early": stopped_early,
            "total_steps_run": step,
            # end-to-end test
            "test_recall@1": m_e2e["recall@1"],
            "test_recall@5": m_e2e["recall@5"],
            "test_recall@10": m_e2e["recall@10"],
            "test_mrr": m_e2e["mrr"], "test_ndcg@10": m_e2e["ndcg@10"],
            # conditional test
            "test_cond_n": int(m_cond["n_queries"]),
            "test_cond_recall@1": m_cond["recall@1"],
            "test_cond_recall@5": m_cond["recall@5"],
            "test_cond_recall@10": m_cond["recall@10"],
            "test_cond_mrr": m_cond["mrr"], "test_cond_ndcg@10": m_cond["ndcg@10"],
            # dev (best ckpt)
            "dev_recall@5": m_dev["recall@5"], "dev_mrr": m_dev["mrr"],
            # gap
            "dev_test_r5_gap": m_dev["recall@5"] - m_e2e["recall@5"],
            # efficiency
            "train_time_s": round(train_time, 1),
            "peak_vram_mb": round(peak, 1),
            "eval_latency_per_q_ms": round(lat_q * 1000, 3),
            "eval_latency_per_cand_ms": round(lat_cand * 1000, 3),
            "best_dev_metric": round(best_dev, 4),
        }
        seed_rows.append(row)
        # store breakdowns
        write_json(out_dir / f"r3_{tag}_seed{seed}_breakdown.json",
                   {"difficulty": bd_diff, "conditions": bd_cond,
                    "dev_metrics": m_dev, "e2e": m_e2e, "conditional": m_cond})
        # test predictions (top10 only to keep small)
        for qid in test_ids:
            for rank, did in enumerate(test_rank[qid][:10]):
                test_pred_rows.append({"seed": seed, "query_id": qid,
                                       "rank": rank, "doc_id": did,
                                       "gold": gold[qid]})
        print(f"[{tag} seed{seed}] TEST R@5={m_e2e['recall@5']:.4f} "
              f"cond_R@5={m_cond['recall@5']:.4f} "
              f"dev_R@5={m_dev['recall@5']:.4f} gap={row['dev_test_r5_gap']:.4f} "
              f"vram={peak:.0f}MB best_step={best_step}", flush=True)

        del best_model, base
        torch.cuda.empty_cache()

    # ---- aggregate over seeds ----
    df = pd.DataFrame(seed_rows)
    df.to_csv(out_dir / f"r3_{tag}_seed_metrics.csv", index=False)
    pd.DataFrame(test_pred_rows).to_parquet(
        out_dir / f"r3_{tag}_test_predictions.parquet", index=False)
    write_json(out_dir / f"r3_{tag}_learning_curves.json", all_curves)

    # learning curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for seed, curve in all_curves.items():
            xs = [c["step"] for c in curve]
            ax[0].plot(xs, [c["train_loss"] for c in curve],
                       marker="o", label=f"seed{seed}")
            ax[1].plot(xs, [c["dev_recall@5"] for c in curve],
                       marker="o", label=f"seed{seed}")
        ax[0].set_title(f"{tag} train loss"); ax[0].set_xlabel("step")
        ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].set_title(f"{tag} dev Recall@5"); ax[1].set_xlabel("step")
        ax[1].set_ylabel("dev R@5"); ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"r3_{tag}_learning_curves.png", dpi=130)
        print(f"saved plot {plot_dir}/r3_{tag}_learning_curves.png")
    except Exception as e:
        print("plot failed:", e)

    # summary
    agg = {}
    for col in ["test_recall@1", "test_recall@5", "test_recall@10",
                "test_mrr", "test_ndcg@10", "test_cond_recall@5",
                "dev_test_r5_gap", "peak_vram_mb", "eval_latency_per_q_ms",
                "eval_latency_per_cand_ms"]:
        agg[col + "_mean"] = float(df[col].mean())
        agg[col + "_std"] = float(df[col].std(ddof=0))
    write_json(out_dir / f"r3_{tag}_summary.json",
               {"seeds": seeds, "per_seed": seed_rows, "aggregate": agg,
                "dry_run": dry_run})
    print(f"\n=== {tag} AGGREGATE (n={len(seeds)} seeds) ===")
    print(f"test R@5 = {agg['test_recall@5_mean']:.4f} "
          f"± {agg['test_recall@5_std']:.4f}")
    print(f"test MRR = {agg['test_mrr_mean']:.4f} ± {agg['test_mrr_std']:.4f}")
    print(f"dev-test R@5 gap = {agg['dev_test_r5_gap_mean']:.4f}")
    print(f"peak VRAM = {agg['peak_vram_mb_mean']:.0f}MB  "
          f"lat/q = {agg['eval_latency_per_q_ms_mean']:.2f}ms")
    return {"aggregate": agg, "per_seed": seed_rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dry_steps", type=int, default=30)
    ap.add_argument("--dry_eval", type=int, default=200)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    main(args.config, seeds=seeds, dry_run=args.dry_run,
         dry_steps=args.dry_steps, dry_eval=args.dry_eval)
