from __future__ import annotations

"""R3: Mamba LoRA reranker (binary yes/no relevance fine-tune).

Trains a LoRA adapter on a causal Mamba LM to prefer ' yes' for the gold doc and
' no' for hard negatives, then re-evaluates reranking on top-100 candidates.
"""

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
from ..eval.candidates import load_candidates
from ..eval.scoring import yes_no_score_fast


def find_lora_targets(model):
    import torch.nn as nn
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            short = n.split(".")[-1]
            if short in ("in_proj", "x_proj", "out_proj", "dt_proj",
                         "embed_out", "lm_head"):
                names.add(short)
    # prefer projection layers; drop lm_head/embed_out to keep it light
    targets = [t for t in names if t in ("in_proj", "x_proj", "out_proj", "dt_proj")]
    return targets or list(names)


def build_training_examples(queries, candidates, hard_neg_map, doc_text_map,
                            gold, neg_per_q, rng):
    """Yield (query_text, doc_text, label) where label 1=relevant(gold), 0=neg."""
    examples = []
    for q in queries:
        qid = q["query_id"]
        g = gold[qid]
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


def main(config_path: str, seeds: list | None = None,
         train_queries: int = 800, eval_queries: int = 500) -> dict:
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
    model_name = cfg["model"]
    tcfg = cfg["train"]
    max_len = tcfg["max_len"]

    docs = list(read_jsonl(data_dir / "docs.jsonl"))
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
    candidates = load_candidates(cand_dir / "r1_candidates_top100.parquet")
    hard_neg_map = {r["query_id"]: r["hard_negatives"]
                    for r in read_jsonl(data_dir / "hard_negatives.jsonl")}

    seeds = seeds or tcfg.get("seeds", [0])
    yes_tok, no_tok = " yes", " no"
    all_metrics = []

    for seed in seeds:
        seed_everything(seed)
        rng = np.random.RandomState(seed)
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if tcfg.get("bf16", True) else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        targets = find_lora_targets(model)
        lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                          lora_dropout=cfg["lora"]["dropout"],
                          target_modules=targets, bias="none")
        model = get_peft_model(model, lora)
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=tcfg["lr"])

        tr_q = queries[:train_queries]
        examples = build_training_examples(
            tr_q, candidates, hard_neg_map, doc_text_map, gold,
            cfg.get("negatives_per_query", 7), rng)

        reset_peak_memory()
        t0 = time.time()
        steps = tcfg["steps"]
        bs = tcfg["batch_size"]
        grad_acc = tcfg["grad_acc"]
        yes_ids = tok.encode(yes_tok, add_special_tokens=False)
        no_ids = tok.encode(no_tok, add_special_tokens=False)
        step = 0
        opt.zero_grad()
        ei = 0
        while step < steps:
            loss_accum = 0.0
            for _ in range(grad_acc):
                batch = []
                for _ in range(bs):
                    batch.append(examples[ei % len(examples)]); ei += 1
                for qt, dt, label in batch:
                    prompt = build_pairwise_prompt(qt, dt)
                    ans_ids = yes_ids if label == 1 else no_ids
                    p_ids = tok.encode(prompt, add_special_tokens=False)
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
            if step % tcfg.get("eval_interval", 200) == 0:
                print(f"[R3 seed{seed}] step {step}/{steps} "
                      f"loss={loss_accum/(bs*grad_acc):.4f}")
        train_time = time.time() - t0
        peak = peak_vram_mb()

        # eval rerank
        model.eval()
        ev_q = queries[:eval_queries]
        rankings = {}
        with torch.no_grad():
            for q in ev_q:
                qid = q["query_id"]
                cand_ids = candidates.get(qid, [])[:100]
                scores = []
                for c in cand_ids:
                    prompt = build_pairwise_prompt(q["query_text"], doc_text_map[c])
                    scores.append(
                        yes_no_score_fast(model, tok, prompt, device, yes_tok, no_tok, max_len)
                    )
                order = np.argsort(-np.array(scores))
                rankings[qid] = [cand_ids[i] for i in order]
        m = aggregate_metrics(rankings, {q: gold[q] for q in rankings},
                              topk_list=[1, 5, 10], ndcg_k=10)
        row = {"model": model_name, "seed": seed, **m,
               "train_time_s": round(train_time, 1),
               "peak_vram_mb": round(peak, 1),
               "steps": steps, "eval_queries": eval_queries}
        all_metrics.append(row)
        model.save_pretrained(str(ckpt_dir / f"seed{seed}"))
        print(f"[R3 seed{seed}] R@5={m['recall@5']:.4f} vram={peak:.0f}MB "
              f"time={train_time:.0f}s")
        del model
        torch.cuda.empty_cache()

    pd.DataFrame(all_metrics).to_csv(out_dir / "r3_lora_metrics.csv", index=False)
    write_json(out_dir / "r3_summary.json", {"metrics": all_metrics})
    return {"metrics": all_metrics}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r3_lora.yaml")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--train_queries", type=int, default=800)
    ap.add_argument("--eval_queries", type=int, default=500)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    print(main(args.config, seeds=seeds, train_queries=args.train_queries,
               eval_queries=args.eval_queries))
