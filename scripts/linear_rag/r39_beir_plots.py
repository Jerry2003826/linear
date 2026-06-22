#!/usr/bin/env python
"""R3.9 BEIR zero-shot comparison plots."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/root/autodl-tmp/linear")
CSV = REPO / "results/linear_rag/r39_beir_metrics.csv"
PLOT = REPO / "plots/linear_rag"
PLOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
datasets = ["scifact", "nfcorpus", "fiqa"]
models = ["BM25", "CrossEncoder-MiniLM-L6", "Mamba-130m-cls", "Pythia-160m-cls"]
labels = {"BM25": "BM25", "CrossEncoder-MiniLM-L6": "CrossEncoder\n(real-data trained)",
          "Mamba-130m-cls": "Mamba-130m-cls\n(synthetic, 0-shot)",
          "Pythia-160m-cls": "Pythia-160m-cls\n(synthetic, 0-shot)"}
colors = {"BM25": "#888888", "CrossEncoder-MiniLM-L6": "#2c7fb8",
          "Mamba-130m-cls": "#d95f02", "Pythia-160m-cls": "#7570b3"}


def grouped_bar(metric, title, fname, ceiling_col=None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(datasets))
    w = 0.2
    for i, m in enumerate(models):
        vals = [df[(df.dataset == ds) & (df.model == m)][metric].values[0] for ds in datasets]
        ax.bar(x + (i - 1.5) * w, vals, w, label=labels[m], color=colors[m])
    if ceiling_col:
        ceil = [df[(df.dataset == ds) & (df.model == "BM25")][ceiling_col].values[0] for ds in datasets]
        for xi, c in zip(x, ceil):
            ax.hlines(c, xi - 2 * w, xi + 2 * w, colors="red", linestyles="--", linewidth=1.4)
        ax.plot([], [], "r--", label="BM25 cand recall ceiling@100")
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(PLOT / fname, dpi=130)
    plt.close(fig)
    print("saved", fname)


grouped_bar("recall@5",
            "BEIR Zero-shot Recall@5: synthetic-trained linear rerankers collapse on real data",
            "r39_beir_recall_comparison.png", ceiling_col="cand_recall_ceiling@topk")
grouped_bar("ndcg@10",
            "BEIR Zero-shot NDCG@10 (synthetic-trained Mamba/Pythia vs real-data baselines)",
            "r39_beir_ndcg_comparison.png")

# latency / VRAM frontier (reranker models only)
fig, ax = plt.subplots(figsize=(8, 6))
sub = df[df.model.isin(["CrossEncoder-MiniLM-L6", "Mamba-130m-cls", "Pythia-160m-cls"])]
agg = sub.groupby("model").agg(lat=("latency_per_q_ms", "mean"),
                               vram=("vram_mb", "mean"),
                               r5=("recall@5", "mean")).reset_index()
for _, r in agg.iterrows():
    ax.scatter(r.lat, r.vram, s=260, color=colors[r.model], edgecolor="k", zorder=3)
    ax.annotate(f"{labels[r.model].splitlines()[0]}\nmean R@5={r.r5:.3f}",
                (r.lat, r.vram), textcoords="offset points", xytext=(10, 6), fontsize=8)
ax.set_xlabel("latency / query (ms, mean over 3 datasets)")
ax.set_ylabel("peak eval VRAM (MB)")
ax.set_title("BEIR reranker efficiency frontier (zero-shot)\nNote: accuracy not comparable — synthetic models have no real signal",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT / "r39_beir_latency_vram.png", dpi=130)
plt.close(fig)
print("saved r39_beir_latency_vram.png")
