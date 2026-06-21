from __future__ import annotations

"""R3.4 efficiency fix: correct BATCHED yes/no pairwise scoring for causal LMs,
plus a correctness check against the verified single-pair fast scorer.

Key idea (handles Mamba which has NO attention_mask):
  - RIGHT-pad sequences (pads at the tail), record true length L_i per row.
  - One forward over [B, T]. For each row read logits[i, L_i - 1, :] -- the
    next-token distribution after the LAST REAL token. Pads sit AFTER position
    L_i-1, so for an SSM the recurrent state at position L_i-1 has not yet seen
    any pad -> the read is numerically clean. For attention models we also pass
    attention_mask so pads are masked anyway.
  - score = logit[yes] - logit[no] at that position (single-token answers).

This module is import-safe; run as a script to do the correctness check.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_model(model_name, ckpt):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ckpt).to("cuda").eval()
    return tok, model


def single_score(model, tok, prompt, yes_id, no_id, max_length=512):
    """Verified single-pair scorer (matches yes_no_score_fast)."""
    ids = tok.encode(prompt, add_special_tokens=False)[:max_length]
    inp = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits = model(inp).logits
    lp = F.log_softmax(logits[0, -1].float(), dim=-1)
    return float(lp[yes_id].item() - lp[no_id].item())


def batch_score(model, tok, prompts, yes_id, no_id, max_length=512,
                pass_attention_mask=True):
    """Right-padded batched scorer. Returns list of (yes-no) logprob diffs."""
    enc = [tok.encode(p, add_special_tokens=False)[:max_length] for p in prompts]
    lens = [len(e) for e in enc]
    T = max(lens)
    pad_id = tok.pad_token_id
    B = len(enc)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long, device="cuda")
    attn = torch.zeros((B, T), dtype=torch.long, device="cuda")
    for i, e in enumerate(enc):
        input_ids[i, :lens[i]] = torch.tensor(e, device="cuda")
        attn[i, :lens[i]] = 1
    kwargs = {}
    if pass_attention_mask:
        # Mamba forward ignores attention_mask; pass only if model accepts it.
        import inspect
        sig = inspect.signature(model.forward)
        if "attention_mask" in sig.parameters:
            kwargs["attention_mask"] = attn
    with torch.no_grad():
        logits = model(input_ids, **kwargs).logits  # [B, T, V]
    idx = torch.tensor([l - 1 for l in lens], device="cuda")
    # gather logits at each row's last-real position
    last = logits[torch.arange(B, device="cuda"), idx, :].float()  # [B, V]
    lp = F.log_softmax(last, dim=-1)
    diff = (lp[:, yes_id] - lp[:, no_id]).cpu().tolist()
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)  # mamba | pythia
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "/root/autodl-tmp/linear/src")
    sys.path.insert(0, "/root/autodl-tmp/linear")
    from src.linear_rag.utils.io import read_jsonl
    from src.linear_rag.data.prompts import build_pairwise_prompt
    from src.linear_rag.eval.candidates import load_candidates

    if args.model == "mamba":
        mn = "state-spaces/mamba-130m-hf"
        seed = 2  # best test R@5 per seed_metrics
        ck = f"checkpoints/linear_rag/r3_validation/mamba_130m_lora/seed{seed}_best"
    else:
        mn = "EleutherAI/pythia-160m"
        seed = 1
        ck = f"checkpoints/linear_rag/r3_validation/pythia_160m_lora/seed{seed}_best"

    data_dir = Path("data/synth_rag_v1")
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(data_dir / "docs.jsonl")}
    queries = {q["query_id"]: q for q in read_jsonl(data_dir / "queries.jsonl")}
    candidates = load_candidates("results/linear_rag/r1_candidates_top100.parquet")
    split = json.loads(Path(
        "data/synth_rag_v1/splits/r3_validation_split.json").read_text())
    test_ids = split["test"]

    tok, model = load_model(mn, ck)
    yes_id = tok.encode(" yes", add_special_tokens=False)[0]
    no_id = tok.encode(" no", add_special_tokens=False)[0]

    # build a sample of prompts with VARIED lengths (mix of short/long docs)
    prompts = []
    for qid in test_ids[:8]:
        for c in candidates[qid][:8]:
            prompts.append(build_pairwise_prompt(
                queries[qid]["query_text"], docs[c]))
    print(f"correctness check on {len(prompts)} prompts, model={args.model}")

    # single (ground truth)
    single = [single_score(model, tok, p, yes_id, no_id) for p in prompts]
    # batched (one big batch)
    batched = batch_score(model, tok, prompts, yes_id, no_id)

    single = np.array(single); batched = np.array(batched)
    abs_err = np.abs(single - batched)
    # ranking correlation matters most for reranking
    from scipy.stats import spearmanr
    rho = spearmanr(single, batched).correlation
    print(f"max|abs err|={abs_err.max():.4f} mean={abs_err.mean():.4f} "
          f"spearman={rho:.5f}")
    # decision sign agreement (yes vs no)
    sign_agree = np.mean(np.sign(single) == np.sign(batched))
    print(f"sign agreement={sign_agree:.4f}")
    print("first 8 single :", np.round(single[:8], 3).tolist())
    print("first 8 batched:", np.round(batched[:8], 3).tolist())


if __name__ == "__main__":
    main()
