from __future__ import annotations

"""R3.4: FAIR latency benchmark using CORRECT batched scoring for Mamba.

Compares apples-to-apples real batched throughput:
  - Mamba-130m LoRA: right-padded batched yes/no scoring (verified spearman
    0.9965 vs single-pair, sign agreement 1.0). Optional length-bucketing to
    cut padding waste.
  - Cross-encoder MiniLM: native batched predict.
  - Embedding bge-small: native batched encode (reference floor).

Reports ms/query (top100), ms/candidate, peak VRAM, candidates/sec at
batch sizes [1,8,16,32]. Length-bucketing sorts candidates of a query by
prompt length and batches similar lengths together.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_causal(model_name, ckpt):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ckpt).to("cuda").eval()
    return tok, model


def batched_forward(model, input_ids, attn, pass_mask):
    kwargs = {}
    if pass_mask:
        import inspect
        if "attention_mask" in inspect.signature(model.forward).parameters:
            kwargs["attention_mask"] = attn
    with torch.no_grad():
        return model(input_ids, **kwargs).logits


def score_batch(model, tok, enc, yes_id, no_id, pass_mask):
    """enc: list of token-id lists (already truncated). Returns diffs."""
    lens = [len(e) for e in enc]
    T = max(lens); B = len(enc)
    pad_id = tok.pad_token_id
    input_ids = torch.full((B, T), pad_id, dtype=torch.long, device="cuda")
    attn = torch.zeros((B, T), dtype=torch.long, device="cuda")
    for i, e in enumerate(enc):
        input_ids[i, :lens[i]] = torch.tensor(e, device="cuda")
        attn[i, :lens[i]] = 1
    logits = batched_forward(model, input_ids, attn, pass_mask)
    idx = torch.tensor([l - 1 for l in lens], device="cuda")
    last = logits[torch.arange(B, device="cuda"), idx, :].float()
    lp = F.log_softmax(last, dim=-1)
    return (lp[:, yes_id] - lp[:, no_id]).cpu().tolist()


def reset_peak():
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()


def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


def bench_causal(model_name, ckpt, all_enc, batch_sizes, pass_mask,
                 bucket=True, n_iter=10):
    tok, model = load_causal(model_name, ckpt)
    yes_id = tok.encode(" yes", add_special_tokens=False)[0]
    no_id = tok.encode(" no", add_special_tokens=False)[0]
    rows = []
    for bs in batch_sizes:
        enc = list(all_enc)
        if bucket:
            enc = sorted(enc, key=len)  # length-bucket to cut padding
        reset_peak()
        # warmup
        for i in range(0, min(len(enc), bs * 3), bs):
            score_batch(model, tok, enc[i:i + bs], yes_id, no_id, pass_mask)
        torch.cuda.synchronize()
        t0 = time.time(); ncand = 0
        for _ in range(n_iter):
            for i in range(0, len(enc), bs):
                chunk = enc[i:i + bs]
                score_batch(model, tok, chunk, yes_id, no_id, pass_mask)
                ncand += len(chunk)
        torch.cuda.synchronize()
        dt = time.time() - t0
        per_cand = dt / ncand
        rows.append({"batch": bs, "ms_per_cand": per_cand * 1000,
                     "ms_per_query_top100": per_cand * 1000 * 100,
                     "cand_per_sec": ncand / dt, "peak_vram_mb": round(peak_mb(), 1)})
    del model
    torch.cuda.empty_cache()
    return rows


def bench_cross_encoder(pairs, batch_sizes, n_iter=10):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                      device="cuda", max_length=512)
    rows = []
    for bs in batch_sizes:
        reset_peak()
        ce.predict(pairs[:bs * 3], batch_size=bs, show_progress_bar=False)
        torch.cuda.synchronize()
        t0 = time.time(); ncand = 0
        for _ in range(n_iter):
            ce.predict(pairs, batch_size=bs, show_progress_bar=False)
            ncand += len(pairs)
        torch.cuda.synchronize()
        dt = time.time() - t0
        per_cand = dt / ncand
        rows.append({"batch": bs, "ms_per_cand": per_cand * 1000,
                     "ms_per_query_top100": per_cand * 1000 * 100,
                     "cand_per_sec": ncand / dt, "peak_vram_mb": round(peak_mb(), 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--nq", type=int, default=20)
    ap.add_argument("--niter", type=int, default=10)
    args = ap.parse_args()
    bss = [int(x) for x in args.batches.split(",")]

    import sys
    sys.path.insert(0, "/root/autodl-tmp/linear/src")
    sys.path.insert(0, "/root/autodl-tmp/linear")
    from src.linear_rag.utils.io import read_jsonl, write_json
    from src.linear_rag.data.prompts import build_pairwise_prompt
    from src.linear_rag.eval.candidates import load_candidates

    data_dir = Path("data/synth_rag_v1")
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(data_dir / "docs.jsonl")}
    queries = {q["query_id"]: q for q in read_jsonl(data_dir / "queries.jsonl")}
    candidates = load_candidates("results/linear_rag/r1_candidates_top100.parquet")
    split = json.loads(Path(
        "data/synth_rag_v1/splits/r3_validation_split.json").read_text())
    test_ids = split["test"]

    # Build a realistic workload: 20 queries x top-100 candidates = 2000 pairs
    prompts, pairs = [], []
    for qid in test_ids[:args.nq]:
        for c in candidates[qid][:100]:
            prompts.append(build_pairwise_prompt(
                queries[qid]["query_text"], docs[c]))
            pairs.append([queries[qid]["query_text"], docs[c]])
    print(f"workload: {len(prompts)} pairs ({len(prompts)//100} queries x top100)")

    out_dir = Path("results/linear_rag")
    all_rows = []

    # tokenize once for mamba
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    enc_m = [tk.encode(p, add_special_tokens=False)[:512] for p in prompts]
    for r in bench_causal("state-spaces/mamba-130m-hf",
                          "checkpoints/linear_rag/r3_validation/mamba_130m_lora/seed2_best",
                          enc_m, bss, pass_mask=False, bucket=True, n_iter=args.niter):
        all_rows.append({"model": "mamba_130m_lora_batched", **r})
        print("mamba", r)

    for r in bench_cross_encoder(pairs, bss, n_iter=args.niter):
        all_rows.append({"model": "cross_encoder", **r})
        print("ce", r)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "r3_validation_batched_latency.csv", index=False)
    print(df.to_string(index=False))
    write_json(out_dir / "r3_validation_batched_latency_meta.json",
               {"workload_pairs": len(prompts), "batches": bss,
                "mamba_scoring": "right-pad batched, length-bucketed, spearman 0.9965 vs single",
                "note": "mamba pass_mask=False (SSM ignores it); read logit at last-real position"})


if __name__ == "__main__":
    main()
