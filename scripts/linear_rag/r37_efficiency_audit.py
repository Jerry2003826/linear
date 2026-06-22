from __future__ import annotations
"""R3.7 Efficiency Bottleneck Audit.

Profiles the END-TO-END reranking pipeline for the existing rerankers, broken
into stages, on a FIXED batch of (query, candidate) pairs. No training.

Models profiled:
  1. Mamba-130m + LoRA (best ckpt, seed0_best)  -- yes/no logprob reranker
  2. Pythia-160m + LoRA tuned (best ckpt, seed0_best) -- yes/no logprob reranker
  3. cross-encoder MiniLM-L6 (sentence-transformers)

Stages timed per query (sum of its `topk` candidate pairs):
  - prompt construction      (CPU, perf_counter)
  - tokenization             (CPU, perf_counter)
  - padding / collation      (CPU, perf_counter)
  - model forward            (GPU, cuda.Event)
  - yes/no logprob extraction (GPU/CPU)
  - sorting / reranking      (CPU)
Plus: total latency/query, /candidate, tokens/sec, peak VRAM, CUDA mem
allocated, input-length stats (avg, p50/p90/p99), padding waste ratio.

Grid: batch_size in [1,4,8,16], max_len in [256,384,512]. dtype bf16.
Fixed query/candidate sample for fairness. >=50 warmup, >=200 measured iters
(an "iter" = one batch forward) for the forward-time micro-benchmark; the
stage breakdown is measured over the full query sample once (end-to-end).

Outputs:
  results/linear_rag/r37_efficiency_breakdown.csv      (per-model stage split)
  results/linear_rag/r37_latency_by_batch_and_len.csv  (grid: forward latency/VRAM)
  results/linear_rag/r37_efficiency_audit_meta.json
"""
import argparse, json, time, statistics
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.linear_rag.utils.io import read_jsonl
from src.linear_rag.data.prompts import build_pairwise_prompt
from src.linear_rag.eval.candidates import load_candidates
from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb

DATA_DIR = Path("data/synth_rag_v1")
CAND = Path("results/linear_rag/r1_candidates_top100.parquet")
SPLIT = DATA_DIR / "splits/r3_validation_split.json"
OUT = Path("results/linear_rag"); OUT.mkdir(parents=True, exist_ok=True)

MAMBA = "state-spaces/mamba-130m-hf"
PYTHIA = "EleutherAI/pythia-160m"
CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAMBA_CKPT = "checkpoints/linear_rag/r3_validation/mamba_130m_lora/seed0_best"
PYTHIA_CKPT = "checkpoints/linear_rag/r3_validation/pythia_160m_lora_tuned/seed0_best"


def load_data():
    docs = list(read_jsonl(DATA_DIR / "docs.jsonl"))
    doc_text = {d["doc_id"]: d["text"] for d in docs}
    queries = list(read_jsonl(DATA_DIR / "queries.jsonl"))
    qmap = {q["query_id"]: q for q in queries}
    cands = load_candidates(CAND)
    split = json.loads(SPLIT.read_text())
    test_ids = split["test"]
    return doc_text, qmap, cands, test_ids


def build_pairs(qids, qmap, cands, doc_text, topk):
    """Fixed flat list of (qid, query_text, doc_text) pairs."""
    pairs = []
    per_q = {}
    for qid in qids:
        qt = qmap[qid]["query_text"]
        cs = cands.get(qid, [])[:topk]
        per_q[qid] = cs
        for c in cs:
            pairs.append((qid, qt, doc_text[c]))
    return pairs, per_q


# ---------------- LM yes/no reranker (Mamba / Pythia) ----------------

def load_lm(model_name, ckpt, device, dtype):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, ckpt).to(device)
    model.eval()
    return tok, model


def profile_lm_stages(tag, model_name, ckpt, pairs, per_q, gold, device,
                      dtype, max_len, batch_size):
    """Full end-to-end stage breakdown over the whole pair sample, batched.
    Returns a dict of stage times (s), totals, length stats, vram, metrics."""
    import torch
    import torch.nn.functional as F
    from src.linear_rag.utils.metrics import aggregate_metrics

    tok, model = load_lm(model_name, ckpt, device, dtype)
    yes_id = tok.encode(" yes", add_special_tokens=False)
    no_id = tok.encode(" no", add_special_tokens=False)
    assert len(yes_id) == 1 and len(no_id) == 1, "yes/no must be single token"
    yes_id, no_id = yes_id[0], no_id[0]

    # ---- Stage 1: prompt construction (CPU) ----
    t = time.perf_counter()
    prompts = [build_pairwise_prompt(qt, dt) for (_, qt, dt) in pairs]
    t_prompt = time.perf_counter() - t

    # ---- Stage 2: tokenization (CPU, no padding) ----
    t = time.perf_counter()
    token_lists = []
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) > max_len:
            ids = ids[-max_len:]
        token_lists.append(ids)
    t_tok = time.perf_counter() - t

    raw_lens = [len(x) for x in token_lists]

    reset_peak_memory()
    scores = [0.0] * len(pairs)
    t_pad = 0.0
    t_fwd = 0.0
    t_extract = 0.0
    padded_tokens = 0
    real_tokens = 0
    n_batches = 0

    fwd_events = []  # per-batch GPU ms for micro stats

    order = list(range(len(token_lists)))
    # length-bucket to reduce padding waste (sort by length, keep mapping)
    order.sort(key=lambda i: raw_lens[i])

    for bstart in range(0, len(order), batch_size):
        bidx = order[bstart:bstart + batch_size]
        seqs = [token_lists[i] for i in bidx]
        # ---- Stage 3: padding / collation (CPU) ----
        tp = time.perf_counter()
        L = max(len(s) for s in seqs)
        pad_id = tok.pad_token_id
        # RIGHT-pad (correct for SSM: state at last real token unaffected)
        batch_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        last_pos = []
        for r, s in enumerate(seqs):
            batch_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            last_pos.append(len(s) - 1)
            padded_tokens += (L - len(s))
            real_tokens += len(s)
        batch_ids = batch_ids.to(device)
        last_pos_t = torch.tensor(last_pos, device=device)
        t_pad += time.perf_counter() - tp

        # ---- Stage 4: model forward (GPU, cuda.Event) ----
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        with torch.no_grad():
            logits = model(batch_ids).logits  # [B, L, V]
        ev1.record()
        torch.cuda.synchronize()
        fwd_ms = ev0.elapsed_time(ev1)
        t_fwd += fwd_ms / 1000.0
        fwd_events.append(fwd_ms)

        # ---- Stage 5: yes/no logprob extraction (GPU->CPU) ----
        torch.cuda.synchronize()
        ev2 = torch.cuda.Event(enable_timing=True)
        ev3 = torch.cuda.Event(enable_timing=True)
        ev2.record()
        with torch.no_grad():
            rows = torch.arange(len(seqs), device=device)
            last_logits = logits[rows, last_pos_t]  # [B, V]
            lp = F.log_softmax(last_logits.float(), dim=-1)
            diff = (lp[:, yes_id] - lp[:, no_id]).detach().cpu().tolist()
        ev3.record()
        torch.cuda.synchronize()
        t_extract += ev2.elapsed_time(ev3) / 1000.0

        for j, i in enumerate(bidx):
            scores[i] = diff[j]
        n_batches += 1

    peak = peak_vram_mb()
    mem_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # ---- Stage 6: sorting / reranking (CPU) ----
    t = time.perf_counter()
    # rebuild per-query scores
    pair_qids = [p[0] for p in pairs]
    qid_to_scores = {}
    qid_to_cands = {q: list(cs) for q, cs in per_q.items()}
    idx = 0
    flat_idx_by_q = {}
    for i, (qid, _, _) in enumerate(pairs):
        flat_idx_by_q.setdefault(qid, []).append(i)
    rankings = {}
    for qid, cs in per_q.items():
        s = np.array([scores[i] for i in flat_idx_by_q[qid]])
        o = np.argsort(-s)
        rankings[qid] = [cs[k] for k in o]
    t_sort = time.perf_counter() - t

    m = aggregate_metrics(rankings, {q: gold[q] for q in rankings},
                          topk_list=[1, 5, 10], ndcg_k=10)

    n_pairs = len(pairs)
    n_q = len(per_q)
    total_s = t_prompt + t_tok + t_pad + t_fwd + t_extract + t_sort
    tokens_sec = real_tokens / max(t_fwd, 1e-9)

    del model
    torch.cuda.empty_cache()

    return {
        "model": tag, "scoring_type": "lm_yesno",
        "batch_size": batch_size, "max_len": max_len,
        "n_queries": n_q, "n_pairs": n_pairs,
        "t_prompt_s": round(t_prompt, 4),
        "t_tokenize_s": round(t_tok, 4),
        "t_pad_collate_s": round(t_pad, 4),
        "t_forward_s": round(t_fwd, 4),
        "t_extract_s": round(t_extract, 4),
        "t_sort_s": round(t_sort, 4),
        "t_total_s": round(total_s, 4),
        "pct_prompt": round(100 * t_prompt / total_s, 2),
        "pct_tokenize": round(100 * t_tok / total_s, 2),
        "pct_pad": round(100 * t_pad / total_s, 2),
        "pct_forward": round(100 * t_fwd / total_s, 2),
        "pct_extract": round(100 * t_extract / total_s, 2),
        "pct_sort": round(100 * t_sort / total_s, 2),
        "latency_ms_per_query": round(1000 * total_s / n_q, 3),
        "latency_ms_per_candidate": round(1000 * total_s / n_pairs, 4),
        "tokens_per_sec_forward": round(tokens_sec, 1),
        "peak_vram_mb": round(peak, 1),
        "cuda_mem_alloc_mb": round(mem_alloc, 1),
        "avg_input_len": round(float(np.mean(raw_lens)), 1),
        "p50_len": int(np.percentile(raw_lens, 50)),
        "p90_len": int(np.percentile(raw_lens, 90)),
        "p99_len": int(np.percentile(raw_lens, 99)),
        "padding_waste_ratio": round(padded_tokens / max(real_tokens + padded_tokens, 1), 4),
        "recall@1": round(m["recall@1"], 4),
        "recall@5": round(m["recall@5"], 4),
        "recall@10": round(m["recall@10"], 4),
        "mrr": round(m["mrr"], 4),
        "ndcg@10": round(m["ndcg@10"], 4),
        "fwd_ms_per_batch_mean": round(statistics.mean(fwd_events), 3),
    }


# ---------------- cross-encoder ----------------

def profile_ce_stages(pairs, per_q, gold, device, max_len, batch_size):
    import torch
    from sentence_transformers import CrossEncoder
    from src.linear_rag.utils.metrics import aggregate_metrics

    model = CrossEncoder(CE, max_length=max_len, device=device)
    tok = model.tokenizer

    # Stage 1: prompt construction == building [q, d] pairs (cheap)
    t = time.perf_counter()
    st_pairs = [[qt, dt] for (_, qt, dt) in pairs]
    t_prompt = time.perf_counter() - t

    # Stage 2: tokenization (length stats), CE truncates internally
    t = time.perf_counter()
    lens = []
    for qt, dt in st_pairs:
        enc = tok(qt, dt, truncation=True, max_length=max_len)
        lens.append(len(enc["input_ids"]))
    t_tok = time.perf_counter() - t

    reset_peak_memory()
    # Stage 4: forward (CE.predict handles tokenize+pad+forward internally;
    # we measure the whole predict as "forward+extract" since it's a fused path)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    scores = model.predict(st_pairs, batch_size=batch_size, show_progress_bar=False)
    ev1.record(); torch.cuda.synchronize()
    t_fwd = ev0.elapsed_time(ev1) / 1000.0
    peak = peak_vram_mb()
    mem_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Stage 6: sorting
    t = time.perf_counter()
    flat_idx_by_q = {}
    for i, (qid, _, _) in enumerate(pairs):
        flat_idx_by_q.setdefault(qid, []).append(i)
    rankings = {}
    for qid, cs in per_q.items():
        s = np.array([scores[i] for i in flat_idx_by_q[qid]])
        o = np.argsort(-s)
        rankings[qid] = [cs[k] for k in o]
    t_sort = time.perf_counter() - t

    m = aggregate_metrics(rankings, {q: gold[q] for q in rankings},
                          topk_list=[1, 5, 10], ndcg_k=10)
    n_pairs = len(pairs); n_q = len(per_q)
    real_tokens = sum(lens)
    total_s = t_prompt + t_tok + t_fwd + t_sort
    return {
        "model": "cross-encoder-MiniLM-L6", "scoring_type": "cross_encoder",
        "batch_size": batch_size, "max_len": max_len,
        "n_queries": n_q, "n_pairs": n_pairs,
        "t_prompt_s": round(t_prompt, 4),
        "t_tokenize_s": round(t_tok, 4),
        "t_pad_collate_s": 0.0,
        "t_forward_s": round(t_fwd, 4),
        "t_extract_s": 0.0,
        "t_sort_s": round(t_sort, 4),
        "t_total_s": round(total_s, 4),
        "pct_prompt": round(100 * t_prompt / total_s, 2),
        "pct_tokenize": round(100 * t_tok / total_s, 2),
        "pct_pad": 0.0,
        "pct_forward": round(100 * t_fwd / total_s, 2),
        "pct_extract": 0.0,
        "pct_sort": round(100 * t_sort / total_s, 2),
        "latency_ms_per_query": round(1000 * total_s / n_q, 3),
        "latency_ms_per_candidate": round(1000 * total_s / n_pairs, 4),
        "tokens_per_sec_forward": round(real_tokens / max(t_fwd, 1e-9), 1),
        "peak_vram_mb": round(peak, 1),
        "cuda_mem_alloc_mb": round(mem_alloc, 1),
        "avg_input_len": round(float(np.mean(lens)), 1),
        "p50_len": int(np.percentile(lens, 50)),
        "p90_len": int(np.percentile(lens, 90)),
        "p99_len": int(np.percentile(lens, 99)),
        "padding_waste_ratio": None,
        "recall@1": round(m["recall@1"], 4),
        "recall@5": round(m["recall@5"], 4),
        "recall@10": round(m["recall@10"], 4),
        "mrr": round(m["mrr"], 4),
        "ndcg@10": round(m["ndcg@10"], 4),
        "fwd_ms_per_batch_mean": None,
    }


# ---------------- forward-only micro-benchmark grid ----------------

def micro_forward_grid(tag, model_name, ckpt, sample_lens, device, dtype,
                       batch_sizes, max_lens, warmup, measured):
    """Pure forward latency vs (batch, max_len) on synthetic fixed-length
    batches. Uses real input-length distribution clipped to max_len."""
    import torch
    rows = []
    tok, model = load_lm(model_name, ckpt, device, dtype)
    pad_id = tok.pad_token_id
    for ml in max_lens:
        # representative length for this max_len = min(p90, ml)
        rep_len = int(min(np.percentile(sample_lens, 90), ml))
        rep_len = max(rep_len, 8)
        for bs in batch_sizes:
            inp = torch.full((bs, rep_len), pad_id, dtype=torch.long, device=device)
            def fn():
                with torch.no_grad():
                    model(inp).logits
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize(); reset_peak_memory()
            ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(measured):
                fn()
            ev1.record(); torch.cuda.synchronize()
            ms_per_batch = ev0.elapsed_time(ev1) / measured
            peak = peak_vram_mb()
            rows.append({
                "model": tag, "batch_size": bs, "max_len": ml, "rep_len": rep_len,
                "fwd_ms_per_batch": round(ms_per_batch, 4),
                "fwd_ms_per_candidate": round(ms_per_batch / bs, 4),
                "peak_vram_mb": round(peak, 1),
                "tokens_per_sec": round(bs * rep_len / (ms_per_batch / 1000.0), 1),
            })
            print(f"  [{tag}] bs={bs} ml={ml} rep_len={rep_len} "
                  f"{ms_per_batch:.2f}ms/batch vram={peak:.0f}MB", flush=True)
    del model; torch.cuda.empty_cache()
    return rows


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch_sizes", default="1,4,8,16")
    ap.add_argument("--max_lens", default="256,384,512")
    ap.add_argument("--breakdown_bs", type=int, default=8)
    ap.add_argument("--breakdown_ml", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--measured", type=int, default=200)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    max_lens = [int(x) for x in args.max_lens.split(",")]

    doc_text, qmap, cands, test_ids = load_data()
    gold = {q: qmap[q]["gold_doc_id"] for q in qmap}

    nq = args.queries
    if args.dry_run:
        nq = min(8, nq)
        batch_sizes = [1, 4]
        max_lens = [256, 512]
        args.warmup = 3; args.measured = 5
    qids = test_ids[:nq]
    pairs, per_q = build_pairs(qids, qmap, cands, doc_text, args.topk)
    sample_lens = None

    print(f"[R37] {len(qids)} queries x topk{args.topk} = {len(pairs)} pairs", flush=True)

    # ---- 1) stage breakdown at a fixed (bs, ml) for the 3 models ----
    breakdown_rows = []
    bs0, ml0 = args.breakdown_bs, args.breakdown_ml
    print(f"[R37] stage breakdown @ bs={bs0} ml={ml0}", flush=True)
    r = profile_lm_stages("mamba-130m-lora", MAMBA, MAMBA_CKPT, pairs, per_q,
                          gold, device, dtype, ml0, bs0)
    breakdown_rows.append(r); print("  mamba done", r["latency_ms_per_query"], "ms/q", flush=True)
    sample_lens = [r["avg_input_len"]]  # placeholder; real lens computed below

    r2 = profile_lm_stages("pythia-160m-lora-tuned", PYTHIA, PYTHIA_CKPT, pairs,
                           per_q, gold, device, dtype, ml0, bs0)
    breakdown_rows.append(r2); print("  pythia done", r2["latency_ms_per_query"], "ms/q", flush=True)

    rce = profile_ce_stages(pairs, per_q, gold, device, ml0, bs0)
    breakdown_rows.append(rce); print("  CE done", rce["latency_ms_per_query"], "ms/q", flush=True)

    pd.DataFrame(breakdown_rows).to_csv(OUT / "r37_efficiency_breakdown.csv", index=False)

    # recover real token lengths for the micro grid from the mamba pass:
    # re-tokenize quickly once with mamba tokenizer
    from transformers import AutoTokenizer
    mtok = AutoTokenizer.from_pretrained(MAMBA)
    real_lens = []
    for (_, qt, dt) in pairs:
        real_lens.append(len(mtok.encode(build_pairwise_prompt(qt, dt),
                                         add_special_tokens=False)))

    # ---- 2) forward-only micro grid (batch x max_len) ----
    grid_rows = []
    print("[R37] micro forward grid (Mamba)", flush=True)
    grid_rows += micro_forward_grid("mamba-130m-lora", MAMBA, MAMBA_CKPT,
                                    real_lens, device, dtype, batch_sizes,
                                    max_lens, args.warmup, args.measured)
    print("[R37] micro forward grid (Pythia)", flush=True)
    grid_rows += micro_forward_grid("pythia-160m-lora-tuned", PYTHIA, PYTHIA_CKPT,
                                    real_lens, device, dtype, batch_sizes,
                                    max_lens, args.warmup, args.measured)
    # cross-encoder micro grid via its own forward
    print("[R37] micro forward grid (CE)", flush=True)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CE, max_length=max(max_lens), device=device)
    cetok = ce.tokenizer
    for ml in max_lens:
        rep_len = int(min(np.percentile(real_lens, 90), ml)); rep_len = max(rep_len, 8)
        for bs in batch_sizes:
            txt = ["query text here"] * bs
            txt2 = ["document text " * 20] * bs
            st = [[a, b] for a, b in zip(txt, txt2)]
            for _ in range(args.warmup):
                ce.predict(st, batch_size=bs, show_progress_bar=False)
            torch.cuda.synchronize(); reset_peak_memory()
            ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(args.measured):
                ce.predict(st, batch_size=bs, show_progress_bar=False)
            ev1.record(); torch.cuda.synchronize()
            ms = ev0.elapsed_time(ev1) / args.measured
            peak = peak_vram_mb()
            grid_rows.append({
                "model": "cross-encoder-MiniLM-L6", "batch_size": bs, "max_len": ml,
                "rep_len": rep_len, "fwd_ms_per_batch": round(ms, 4),
                "fwd_ms_per_candidate": round(ms / bs, 4),
                "peak_vram_mb": round(peak, 1),
                "tokens_per_sec": None})
            print(f"  [CE] bs={bs} ml={ml} {ms:.2f}ms/batch vram={peak:.0f}MB", flush=True)

    pd.DataFrame(grid_rows).to_csv(OUT / "r37_latency_by_batch_and_len.csv", index=False)

    meta = {
        "queries_sample": len(qids), "topk": args.topk,
        "batch_sizes": batch_sizes, "max_lens": max_lens,
        "warmup": args.warmup, "measured": args.measured,
        "dtype": "bf16", "device": torch.cuda.get_device_name(0),
        "mamba_ckpt": MAMBA_CKPT, "pythia_ckpt": PYTHIA_CKPT,
        "dry_run": args.dry_run,
    }
    (OUT / "r37_efficiency_audit_meta.json").write_text(json.dumps(meta, indent=2))
    print("[R37] DONE")


if __name__ == "__main__":
    main()
