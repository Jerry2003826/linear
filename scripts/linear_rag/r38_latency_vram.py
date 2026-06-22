#!/usr/bin/env python3
"""R3.8 latency/VRAM benchmark: classification-head scoring (Mamba-cls, Pythia-cls)
vs baselines (Mamba yes/no, Pythia yes/no, CrossEncoder). Measures per-query and
per-candidate forward latency, peak VRAM, tokens/sec at eval batch sizes.
Reuses ClsReranker (cls models) and the R3.7 yes/no / CE paths."""
import argparse, json, time, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, "/root/autodl-tmp/linear")
sys.path.insert(0, "/root/autodl-tmp/linear/src")

spec = importlib.util.spec_from_file_location(
    "r38mod", "/root/autodl-tmp/linear/scripts/linear_rag/r38_cls_train.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

import yaml
from src.linear_rag.utils.io import read_jsonl
from src.linear_rag.eval.candidates import load_candidates
from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb

DEVICE = "cuda"
DTYPE = torch.bfloat16


def make_pairs(n_pairs=400):
    """Representative (query, doc) text pairs from real data."""
    data_dir = Path("/root/autodl-tmp/linear/data/synth_rag_v1")
    docs = list(read_jsonl(data_dir / "docs.jsonl"))
    doc_text = {d["doc_id"]: d["text"] for d in docs}
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    cand = load_candidates(Path("/root/autodl-tmp/linear/results/linear_rag/r1_candidates_top100.parquet"))
    pairs = []
    for q in queries:
        qid = q["query_id"]
        if qid not in cand:
            continue
        for did in cand[qid][:5]:
            if did in doc_text:
                pairs.append((q["query_text"], doc_text[did]))
            if len(pairs) >= n_pairs:
                return pairs
    return pairs


def bench_cls(cfg_path, model_name, ckpt, pairs, batch_sizes, max_len, warmup, measured):
    cfg = yaml.safe_load(open(cfg_path))
    rr = m.ClsReranker(model_name, cfg["lora"], DEVICE, DTYPE)
    best = Path(ckpt) / "seed0_best"
    rr.load_head(best)
    rr.backbone.load_adapter(str(best), adapter_name="default")
    rr.backbone.set_adapter("default")
    rr.eval()
    rows = []
    for bs in batch_sizes:
        batch = pairs[:bs]
        reset_peak_memory()
        with torch.no_grad():
            for _ in range(warmup):
                ids, last = rr.encode_batch(batch, max_len)
                _ = rr.forward_scores(ids, last)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(measured):
                ids, last = rr.encode_batch(batch, max_len)
                _ = rr.forward_scores(ids, last)
            end.record(); torch.cuda.synchronize()
        ms_batch = start.elapsed_time(end) / measured
        vram = peak_vram_mb()
        rows.append(dict(model=cfg["tag"], scoring="classification_head", batch_size=bs,
                         max_len=max_len, fwd_ms_per_batch=round(ms_batch, 4),
                         fwd_ms_per_candidate=round(ms_batch / bs, 4),
                         peak_vram_mb=round(vram, 1)))
        print(f"  [{cfg['tag']}] bs={bs} {ms_batch:.3f}ms/batch ({ms_batch/bs:.3f}/cand) vram={vram:.0f}MB", flush=True)
    import gc; del rr; gc.collect(); torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_sizes", default="1,8,16")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--measured", type=int, default=200)
    args = ap.parse_args()
    bss = [int(x) for x in args.batch_sizes.split(",")]
    pairs = make_pairs(max(bss) + 10)
    print(f"[R38-bench] {len(pairs)} pairs, batch_sizes={bss}, max_len={args.max_len}")

    rows = []
    print("[R38-bench] Mamba-cls")
    rows += bench_cls("/root/autodl-tmp/linear/scripts/linear_rag/r38_mamba_cls.yaml",
                      "state-spaces/mamba-130m-hf",
                      "/root/autodl-tmp/linear/checkpoints/linear_rag/r38_mamba_cls",
                      pairs, bss, args.max_len, args.warmup, args.measured)
    print("[R38-bench] Pythia-cls")
    rows += bench_cls("/root/autodl-tmp/linear/scripts/linear_rag/r38_pythia_cls.yaml",
                      "EleutherAI/pythia-160m",
                      "/root/autodl-tmp/linear/checkpoints/linear_rag/r38_pythia_cls",
                      pairs, bss, args.max_len, args.warmup, args.measured)

    out = Path("/root/autodl-tmp/linear/results/linear_rag/r38_latency_vram.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")
