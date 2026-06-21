from __future__ import annotations

"""R3.5: Fair learning-rate sweep for the Pythia-160m LoRA baseline.

Goal: remove the "Pythia was mis-tuned" caveat. The R3.2 run used lr=2e-4 (chosen
around Mamba) with a bare AdamW (no warmup/decay) and Pythia's dev MRR peaked at
step 250 then collapsed -- a classic too-high-lr signature. Here we sweep lower
lrs WITH linear warmup + cosine decay, single seed each (seed 0), monitoring the
dev-MRR curve. If Pythia still cannot approach Mamba's dev MRR (~0.76) under its
best lr, the baseline comparison is solid. If it improves a lot, we report that
honestly.

Reuses the exact data/prompt/eval logic from r3v_train.py (same split, same
build_examples, same rerank_eval) so the comparison stays fair -- the ONLY thing
we change vs R3.2 is lr and the scheduler (warmup+cosine).

Single seed, dev-only monitoring (no test eval here -- this is a tuning probe).
Writes results/linear_rag/r3_pythia_lr_sweep.{json,csv} + a learning-curve plot.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/linear/src")
sys.path.insert(0, "/root/autodl-tmp/linear")
sys.path.insert(0, "/root/autodl-tmp/linear/scripts/linear_rag")

import r3v_train as T  # reuse build_examples, rerank_eval, dev_metric_value
from src.linear_rag.utils.io import read_jsonl, write_json
from src.linear_rag.utils.seeds import seed_everything
from src.linear_rag.eval.candidates import load_candidates
from src.linear_rag.data.prompts import build_pairwise_prompt


def run_one_lr(lr, warmup_ratio, steps, seed, data, cfg, dry=False):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    (qmap, gold, candidates, hard_neg_map, doc_text_map,
     train_ids, dev_eval_ids) = data
    device = "cuda"
    model_name = cfg["model"]
    tcfg = cfg["train"]
    max_len = tcfg["max_len"]
    topk = cfg.get("eval_topk", 100)
    bs = tcfg["batch_size"]; grad_acc = tcfg["grad_acc"]
    eval_interval = 250 if not dry else 30
    yes_tok, no_tok = " yes", " no"

    seed_everything(seed)
    rng = np.random.RandomState(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16).to(device)
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"],
                      target_modules=cfg["lora"]["target_modules"], bias="none")
    model = get_peft_model(model, lora)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    warmup = int(warmup_ratio * steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup, steps)

    examples = T.build_examples(train_ids, qmap, candidates, hard_neg_map,
                                doc_text_map, gold,
                                cfg.get("negatives_per_query", 4), rng)
    yes_ids = tok.encode(yes_tok, add_special_tokens=False)
    no_ids = tok.encode(no_tok, add_special_tokens=False)

    curve = []
    best_dev = -1.0; best_step = 0
    ei = 0; step = 0
    opt.zero_grad(); model.train()
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
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        if step % eval_interval == 0 or step == steps:
            model.eval()
            dev_rank, _, _ = T.rerank_eval(
                model, tok, dev_eval_ids, qmap, candidates, doc_text_map,
                gold, device, max_len, yes_tok, no_tok, topk)
            dev_val, dev_full = T.dev_metric_value(dev_rank, gold, "mrr")
            curve.append({"step": step, "lr": lr,
                          "train_loss": loss_accum / (bs * grad_acc),
                          "dev_mrr": dev_full["mrr"],
                          "dev_recall@5": dev_full["recall@5"],
                          "dev_recall@1": dev_full["recall@1"],
                          "cur_lr": sched.get_last_lr()[0]})
            if dev_val > best_dev:
                best_dev = dev_val; best_step = step
            print(f"[pythia lr={lr:.0e}] step {step}/{steps} "
                  f"loss={loss_accum/(bs*grad_acc):.4f} dev_mrr={dev_val:.4f} "
                  f"dev_r5={dev_full['recall@5']:.4f} (best={best_dev:.4f}@{best_step})",
                  flush=True)
            model.train()
    del model
    torch.cuda.empty_cache()
    return {"lr": lr, "warmup_ratio": warmup_ratio, "steps": steps,
            "seed": seed, "best_dev_mrr": best_dev, "best_step": best_step,
            "final_dev_mrr": curve[-1]["dev_mrr"],
            "final_dev_recall@5": curve[-1]["dev_recall@5"],
            "peak_dev_recall@5": max(c["dev_recall@5"] for c in curve),
            "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", default="5e-5,1e-4,2e-4")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev_n", type=int, default=200)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(
        "scripts/linear_rag/r3v_pythia.yaml").read_text())
    data_dir = Path(cfg["data_dir"])
    docs = list(read_jsonl(data_dir / "docs.jsonl"))
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    qmap = {q["query_id"]: q for q in queries}
    gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
    candidates = load_candidates(
        Path(cfg["candidates_dir"]) / "r1_candidates_top100.parquet")
    hard_neg_map = {r["query_id"]: r["hard_negatives"]
                    for r in read_jsonl(data_dir / "hard_negatives.jsonl")}
    split = json.loads(Path(cfg["split_path"]).read_text())
    train_ids = split["train"]
    dev_eval_ids = split["dev"][:args.dev_n]
    if args.dry:
        train_ids = train_ids
        dev_eval_ids = split["dev"][:80]

    data = (qmap, gold, candidates, hard_neg_map, doc_text_map,
            train_ids, dev_eval_ids)
    lrs = [float(x) for x in args.lrs.split(",")]
    steps = 60 if args.dry else args.steps

    results = []
    for lr in lrs:
        print(f"\n===== sweeping lr={lr:.0e} (warmup={args.warmup}, "
              f"steps={steps}, cosine) =====", flush=True)
        r = run_one_lr(lr, args.warmup, steps, args.seed, data, cfg, dry=args.dry)
        results.append(r)

    out_dir = Path("results/linear_rag")
    # summary rows (without full curve)
    rows = [{k: v for k, v in r.items() if k != "curve"} for r in results]
    pd.DataFrame(rows).to_csv(out_dir / "r3_pythia_lr_sweep.csv", index=False)
    write_json(out_dir / "r3_pythia_lr_sweep.json",
               {"results": results,
                "mamba_dev_mrr_reference": 0.76,
                "r3_2_pythia_lr": 2e-4,
                "note": "single seed; warmup+cosine added vs R3.2 bare AdamW; "
                        "dev-only tuning probe, no test eval"})
    print("\n=== SWEEP SUMMARY ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # plot dev_mrr curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for r in results:
            c = r["curve"]
            xs = [p["step"] for p in c]
            ax[0].plot(xs, [p["dev_mrr"] for p in c], marker="o",
                       label=f"lr={r['lr']:.0e}")
            ax[1].plot(xs, [p["dev_recall@5"] for p in c], marker="o",
                       label=f"lr={r['lr']:.0e}")
        ax[0].axhline(0.76, ls="--", color="gray", label="Mamba dev MRR ~0.76")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("dev MRR")
        ax[0].set_title("Pythia-160m LoRA lr sweep: dev MRR")
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].set_xlabel("step"); ax[1].set_ylabel("dev Recall@5")
        ax[1].set_title("Pythia-160m LoRA lr sweep: dev Recall@5")
        ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig("plots/linear_rag/r3_pythia_lr_sweep.png", dpi=130)
        print("saved plots/linear_rag/r3_pythia_lr_sweep.png")
    except Exception as e:
        print("plot failed:", e)


if __name__ == "__main__":
    main()
