"""R3 dry-run estimator: time a tiny train+eval slice, estimate full-run GPU hours.

Writes results/linear_rag/r3_dry_estimate.json with per-phase timing and the
projected full-run hours. Does NOT do the full run.
"""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from linear_rag.utils.io import read_jsonl
from linear_rag.utils.seeds import seed_everything
from linear_rag.data.prompts import build_pairwise_prompt
from linear_rag.eval.candidates import load_candidates
from linear_rag.eval.scoring import yes_no_score_fast
from linear_rag.train.rerank_lora import find_lora_targets, build_training_examples

CFG = "configs/r3_lora_smallsample.yaml"
DRY_STEPS = 30
DRY_EVAL_Q = 40
FULL_STEPS = 600
FULL_EVAL_Q = 500
TRAIN_Q = 800

cfg = yaml.safe_load(Path(CFG).read_text())
tcfg = cfg["train"]
device = "cuda"
data_dir = Path(cfg["data_dir"]); cand_dir = Path(cfg["candidates_dir"])
max_len = tcfg["max_len"]

docs = list(read_jsonl(data_dir / "docs.jsonl"))
doc_text_map = {d["doc_id"]: d["text"] for d in docs}
queries = list(read_jsonl(data_dir / "queries.jsonl"))
gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
candidates = load_candidates(cand_dir / "r1_candidates_top100.parquet")
hard_neg_map = {r["query_id"]: r["hard_negatives"] for r in read_jsonl(data_dir / "hard_negatives.jsonl")}

seed_everything(0); rng = np.random.RandomState(0)
tok = AutoTokenizer.from_pretrained(cfg["model"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32).to(device)
targets = find_lora_targets(model)
lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                  lora_dropout=cfg["lora"]["dropout"], target_modules=targets, bias="none")
model = get_peft_model(model, lora); model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tcfg["lr"])

tr_q = queries[:TRAIN_Q]
examples = build_training_examples(tr_q, candidates, hard_neg_map, doc_text_map, gold,
                                   cfg.get("negatives_per_query", 7), rng)
yes_tok, no_tok = " yes", " no"
yes_ids = tok.encode(yes_tok, add_special_tokens=False)
no_ids = tok.encode(no_tok, add_special_tokens=False)
bs = tcfg["batch_size"]; grad_acc = tcfg["grad_acc"]

# time DRY_STEPS training steps
t0 = time.time(); ei = 0; step = 0; opt.zero_grad()
while step < DRY_STEPS:
    for _ in range(grad_acc):
        for _ in range(bs):
            qt, dt, label = examples[ei % len(examples)]; ei += 1
            ans_ids = yes_ids if label == 1 else no_ids
            p_ids = tok.encode(build_pairwise_prompt(qt, dt), add_special_tokens=False)
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
    opt.step(); opt.zero_grad(); step += 1
train_per_step = (time.time() - t0) / DRY_STEPS

# time DRY_EVAL_Q eval queries
model.eval()
t1 = time.time()
with torch.no_grad():
    for q in queries[:DRY_EVAL_Q]:
        cand_ids = candidates.get(q["query_id"], [])[:100]
        for c in cand_ids:
            yes_no_score_fast(model, tok, build_pairwise_prompt(q["query_text"], doc_text_map[c]),
                              device, yes_tok, no_tok, max_len)
eval_per_q = (time.time() - t1) / DRY_EVAL_Q

est_train_s = train_per_step * FULL_STEPS
est_eval_s = eval_per_q * FULL_EVAL_Q
est_total_h = (est_train_s + est_eval_s) / 3600.0
out = {
    "train_per_step_s": round(train_per_step, 3),
    "eval_per_q_s": round(eval_per_q, 3),
    "full_steps": FULL_STEPS, "full_eval_q": FULL_EVAL_Q,
    "est_train_s": round(est_train_s, 1), "est_eval_s": round(est_eval_s, 1),
    "est_total_gpu_h": round(est_total_h, 3),
    "budget_gpu_h": cfg["budget_gpu_hours"],
    "within_budget": est_total_h <= cfg["budget_gpu_hours"],
}
Path("results/linear_rag").mkdir(parents=True, exist_ok=True)
Path("results/linear_rag/r3_dry_estimate.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
