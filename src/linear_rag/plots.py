from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def r1_recall_curves(r1_summary: dict, out_path: Path):
    topks = r1_summary["topk_list"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ("bm25", "embedding"):
        if method in r1_summary:
            ys = [r1_summary[method].get(f"recall@{k}", None) for k in topks]
            ax.plot(topks, ys, marker="o", label=method)
    ax.set_xscale("log"); ax.set_xlabel("top-k"); ax.set_ylabel("Recall@k")
    ax.set_title("R1 Recall@k: BM25 vs Embedding"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def r1_latency(r1_summary: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    methods, lats = [], []
    for method in ("bm25", "embedding"):
        if method in r1_summary:
            methods.append(method)
            lats.append(r1_summary[method].get("latency_ms_per_query", 0))
    ax.bar(methods, lats, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("latency ms/query"); ax.set_title("R1 retrieval latency")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def r2_recall_comparison(r1_summary, r2_metrics, r25_row, out_path: Path):
    labels, r5 = [], []
    if r1_summary.get("embedding"):
        labels.append("embedding"); r5.append(r1_summary["embedding"].get("recall@5", 0))
    if r1_summary.get("bm25"):
        labels.append("bm25"); r5.append(r1_summary["bm25"].get("recall@5", 0))
    for m in r2_metrics:
        if "recall@5" in m:
            labels.append(m["model"].split("/")[-1]); r5.append(m["recall@5"])
    if r25_row and "recall@5" in r25_row:
        labels.append("cross-encoder"); r5.append(r25_row["recall@5"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, r5, color="#55A868")
    ax.set_ylabel("Recall@5"); ax.set_title("Reranking Recall@5 comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def r2_latency_vram(r2_metrics, r25_row, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in r2_metrics:
        if "recall@5" in m:
            ax.scatter(m["latency_ms_per_query"], m["recall@5"], s=80,
                       label=m["model"].split("/")[-1])
            ax.annotate(f"{m.get('peak_vram_mb',0):.0f}MB",
                        (m["latency_ms_per_query"], m["recall@5"]))
    if r25_row and "recall@5" in r25_row:
        ax.scatter(r25_row["latency_ms_per_query"], r25_row["recall@5"], s=120,
                   marker="*", label="cross-encoder")
    ax.set_xlabel("latency ms/query"); ax.set_ylabel("Recall@5")
    ax.set_title("Accuracy vs latency (label=peak VRAM)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def latency_vram_tradeoff(csv_path: Path, out_path: Path):
    if not Path(csv_path).exists():
        return
    df = pd.read_csv(csv_path)
    if "latency_ms_per_query" not in df or df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, grp in df.groupby("model_name"):
        g = grp.dropna(subset=["latency_ms_per_query", "peak_vram_mb"])
        if not g.empty:
            ax.scatter(g["latency_ms_per_query"], g["peak_vram_mb"],
                       label=str(name).split("/")[-1], s=60)
    ax.set_xlabel("latency ms/query"); ax.set_ylabel("peak VRAM MB")
    ax.set_title("Latency / VRAM tradeoff"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
