from __future__ import annotations

"""R3 validation: unified latency / VRAM benchmark + accuracy-efficiency frontier.

Benchmarks reranking throughput for:
  - Mamba-130m LoRA (best-dev checkpoint, seed picked by best test R@5)
  - Pythia-160m LoRA (best-dev checkpoint)
  - cross-encoder/ms-marco-MiniLM-L-6-v2
  - embedding bge-small (bi-encoder, scores candidates by cos sim)
over batch sizes [1,4,8], topk 100. Reports ms/query, ms/candidate, peak VRAM,
tokens/sec (approx). Pairs accuracy (from seed_metrics) with efficiency to draw
the accuracy-latency and accuracy-VRAM frontier.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.linear_rag.utils.io import read_jsonl, write_json
from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb
from src.linear_rag.data.prompts import build_pairwise_prompt
from src.linear_rag.eval.candidates import load_candidates


def pick_best_seed(csv_path):
    """Return (seed, ckpt_subdir) of the seed with best test R@5."""
    df = pd.read_csv(csv_path)
    best = df.loc[df["test_recall@5"].idxmax()]
    return int(best["seed"]), float(best["test_recall@5"])


def bench_causal_lora(model_name, ckpt, sample_prompts, batch_sizes, n_iter=40):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ckpt).to("cuda").eval()
    yes_id = tok.encode(" yes", add_special_tokens=False)[0]
    no_id = tok.encode(" no", add_special_tokens=False)[0]
    # tokenize sample prompts once
    enc = [tok.encode(p, add_special_tokens=False)[:512] for p in sample_prompts]
    rows = []
    for bs in batch_sizes:
        # single-prompt scoring repeated bs times == one "batch" of bs candidates
        reset_peak_memory()
        # warmup
        for _ in range(5):
            for ids in enc[:bs]:
                inp = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    model(inp)
        torch.cuda.synchronize()
        t0 = time.time()
        ntok = 0
        for _ in range(n_iter):
            for ids in enc[:bs]:
                inp = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    model(inp)
                ntok += len(ids)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = peak_vram_mb()
        per_cand = dt / (n_iter * bs)
        rows.append({"batch": bs, "ms_per_cand": per_cand * 1000,
                     "ms_per_query_top100": per_cand * 1000 * 100,
                     "tokens_per_sec": ntok / dt, "peak_vram_mb": round(peak, 1)})
    del model, base
    torch.cuda.empty_cache()
    return rows


def bench_cross_encoder(sample_pairs, batch_sizes, n_iter=40):
    import torch
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                      device="cuda", max_length=512)
    rows = []
    for bs in batch_sizes:
        reset_peak_memory()
        ce.predict(sample_pairs[:bs], batch_size=bs, show_progress_bar=False)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            ce.predict(sample_pairs[:bs], batch_size=bs, show_progress_bar=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = peak_vram_mb()
        per_cand = dt / (n_iter * bs)
        rows.append({"batch": bs, "ms_per_cand": per_cand * 1000,
                     "ms_per_query_top100": per_cand * 1000 * 100,
                     "tokens_per_sec": float("nan"),
                     "peak_vram_mb": round(peak, 1)})
    return rows


def bench_embedding(sample_q, sample_d, batch_sizes, n_iter=40):
    import torch
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    rows = []
    for bs in batch_sizes:
        reset_peak_memory()
        st.encode(sample_d[:bs], batch_size=bs, show_progress_bar=False,
                  convert_to_numpy=True)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            st.encode(sample_d[:bs], batch_size=bs, show_progress_bar=False,
                      convert_to_numpy=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = peak_vram_mb()
        per_cand = dt / (n_iter * bs)
        rows.append({"batch": bs, "ms_per_cand": per_cand * 1000,
                     "ms_per_query_top100": per_cand * 1000 * 100,
                     "tokens_per_sec": float("nan"),
                     "peak_vram_mb": round(peak, 1)})
    return rows


def main(batch_sizes=(1, 4, 8)):
    out_dir = Path("results/linear_rag")
    plot_dir = Path("plots/linear_rag"); plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/synth_rag_v1")
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(data_dir / "docs.jsonl")}
    queries = {q["query_id"]: q for q in read_jsonl(data_dir / "queries.jsonl")}
    candidates = load_candidates("results/linear_rag/r1_candidates_top100.parquet")
    split = json.loads(Path(
        "data/synth_rag_v1/splits/r3_validation_split.json").read_text())
    test_ids = split["test"]
    # sample prompts/pairs from test
    sp_prompts, sp_pairs, sp_q, sp_d = [], [], [], []
    for qid in test_ids[:20]:
        for c in candidates[qid][:10]:
            sp_prompts.append(build_pairwise_prompt(queries[qid]["query_text"],
                                                    docs[c]))
            sp_pairs.append([queries[qid]["query_text"], docs[c]])
            sp_q.append(queries[qid]["query_text"]); sp_d.append(docs[c])

    all_rows = []
    # accuracy lookup
    acc = {}
    mamba_csv = out_dir / "r3_mamba_130m_lora_seed_metrics.csv"
    pythia_csv = out_dir / "r3_pythia_160m_lora_seed_metrics.csv"
    ce_csv = out_dir / "r3_cross_encoder_same_split.csv"

    if mamba_csv.exists():
        seed, r5 = pick_best_seed(mamba_csv)
        acc["mamba_130m_lora"] = r5
        ck = f"checkpoints/linear_rag/r3_validation/mamba_130m_lora/seed{seed}_best"
        for r in bench_causal_lora("state-spaces/mamba-130m-hf", ck,
                                   sp_prompts, batch_sizes):
            all_rows.append({"model": "mamba_130m_lora", **r})
    if pythia_csv.exists():
        seed, r5 = pick_best_seed(pythia_csv)
        acc["pythia_160m_lora"] = r5
        ck = f"checkpoints/linear_rag/r3_validation/pythia_160m_lora/seed{seed}_best"
        for r in bench_causal_lora("EleutherAI/pythia-160m", ck,
                                   sp_prompts, batch_sizes):
            all_rows.append({"model": "pythia_160m_lora", **r})
    if ce_csv.exists():
        acc["cross_encoder"] = float(pd.read_csv(ce_csv)["test_recall@5"].iloc[0])
    for r in bench_cross_encoder(sp_pairs, batch_sizes):
        all_rows.append({"model": "cross_encoder", **r})
    for r in bench_embedding(sp_q, sp_d, batch_sizes):
        all_rows.append({"model": "embedding_bge_small", **r})

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "r3_validation_latency_vram.csv", index=False)
    write_json(out_dir / "r3_validation_latency_vram_accuracy.json", acc)
    print(df.to_string(index=False))

    # frontier plot: accuracy vs latency (batch=1) and accuracy vs VRAM
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        b1 = df[df["batch"] == 1].set_index("model")
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        names = {"mamba_130m_lora": "Mamba-130m LoRA",
                 "pythia_160m_lora": "Pythia-160m LoRA",
                 "cross_encoder": "Cross-encoder MiniLM",
                 "embedding_bge_small": "Embedding bge-small"}
        for mdl in b1.index:
            if mdl not in acc:
                continue
            x = b1.loc[mdl, "ms_per_query_top100"]
            y = acc[mdl]
            ax[0].scatter(x, y, s=90)
            ax[0].annotate(names.get(mdl, mdl), (x, y),
                           textcoords="offset points", xytext=(6, 5))
            v = b1.loc[mdl, "peak_vram_mb"]
            ax[1].scatter(v, y, s=90)
            ax[1].annotate(names.get(mdl, mdl), (v, y),
                           textcoords="offset points", xytext=(6, 5))
        ax[0].set_xlabel("latency ms/query (top100, batch=1)")
        ax[0].set_ylabel("test Recall@5"); ax[0].set_xscale("log")
        ax[0].set_title("Accuracy vs Latency"); ax[0].grid(alpha=.3)
        ax[1].set_xlabel("peak VRAM (MB)"); ax[1].set_ylabel("test Recall@5")
        ax[1].set_title("Accuracy vs VRAM"); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "r3_validation_frontier.png", dpi=130)
        print("saved", plot_dir / "r3_validation_frontier.png")
    except Exception as e:
        print("frontier plot failed:", e)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,4,8")
    args = ap.parse_args()
    bs = tuple(int(x) for x in args.batches.split(","))
    main(batch_sizes=bs)
