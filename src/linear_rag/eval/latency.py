from __future__ import annotations

"""Latency / VRAM profiling for reranker forward passes."""

import argparse
from pathlib import Path

import pandas as pd

from ..utils.gpu import benchmark, reset_peak_memory
from ..data.prompts import build_pairwise_prompt


def profile_causal_lm(model_name, batch_sizes, max_length, dtype, device, stage):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(dtype, torch.float32)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=td).to(device).eval()

    prompt = build_pairwise_prompt(
        "Who bought a red camera in Kyoto?",
        "On 2021-04-12, Akira Tanaka purchased a red camera in Kyoto at Nikon Store.",
    ) + " yes"
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=max_length).input_ids.to(device)

    rows = []
    for bs in batch_sizes:
        try:
            batch = ids.repeat(bs, 1)
            reset_peak_memory()

            def fn():
                with torch.no_grad():
                    model(batch)

            res = benchmark(fn, warmup=50, measured=200)
            seq_len = batch.shape[1]
            tps = bs * seq_len / (res.latency_ms_per_iter / 1000.0)
            rows.append({
                "model_name": model_name, "stage": stage, "batch_size": bs,
                "topk": 100, "max_length": seq_len, "dtype": dtype,
                "latency_ms_per_query": round(res.latency_ms_per_iter / bs, 4),
                "latency_ms_per_candidate": round(res.latency_ms_per_iter / bs, 4),
                "tokens_per_sec": round(tps, 1),
                "peak_vram_mb": round(res.peak_vram_mb, 1),
                "queries_per_second": round(1000.0 / (res.latency_ms_per_iter / bs), 2),
                "notes": "single fwd pass per query",
            })
        except RuntimeError as e:
            rows.append({"model_name": model_name, "stage": stage, "batch_size": bs,
                         "topk": 100, "max_length": max_length, "dtype": dtype,
                         "notes": f"OOM_or_error: {e}"})
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return rows


def append_rows(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", default="latency")
    ap.add_argument("--out", default="results/linear_rag/latency_vram.csv")
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = profile_causal_lm(args.model, [1, 4, 8], 512, "bfloat16", dev, args.stage)
    append_rows(rows, Path(args.out))
    print(rows)
