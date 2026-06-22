#!/usr/bin/env python
"""R3.9b ablation plots:
  1) ablation bar chart — marginal effect of each knob (hard-neg count / LoRA rank /
     steps) on SciFact test R@5 + NDCG@10, with CrossEncoder + BM25 reference lines.
  2) multi-seed stability — best config (negs15) across seeds 0/1/2, mean +- std error bars.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/root/autodl-tmp/linear")
RES = REPO / "results/linear_rag"
PLOT = REPO / "plots/linear_rag"; PLOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RES / "r39b_ablation_metrics.csv")
# R3.9 reference baselines (from r39_real_finetune_metrics.csv)
CE_R5, CE_ND = 0.733, 0.635
BM25_R5, BM25_ND = 0.702, 0.634

# seed0 ablation grid (single seed per config)
g = df[df.seed == 0].set_index("run")

# ===== Plot 1: ablation marginal effects =====
# group configs by the knob they vary, relative to ref(neg4,r16,1500)
panels = [
    ("Hard-neg count (rank16, 1500 steps)", ["steps750", "ref", "negs8", "negs15"],
     {"steps750": "n=4*", "ref": "n=4", "negs8": "n=8", "negs15": "n=15"}),
    ("LoRA rank (n=4, 1500 steps)", ["ref", "rank32"], {"ref": "r=16", "rank32": "r=32"}),
    ("Train steps (n=4, rank16)", ["steps750", "ref", "steps3000"],
     {"steps750": "750", "ref": "1500", "steps3000": "3000"}),
]
# fix: steps750 belongs only to the steps panel; for hard-neg panel use only ref/negs8/negs15
panels[0] = ("Hard-neg count (rank16, 1500 steps)", ["ref", "negs8", "negs15"],
             {"ref": "n=4", "negs8": "n=8", "negs15": "n=15"})

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, (title, runs, lbl) in zip(axes, panels):
    runs = [r for r in runs if r in g.index]
    x = np.arange(len(runs)); w = 0.38
    r5 = [g.loc[r, "test_recall@5"] for r in runs]
    nd = [g.loc[r, "test_ndcg@10"] for r in runs]
    ax.bar(x - w/2, r5, w, label="R@5", color="#d95f02")
    ax.bar(x + w/2, nd, w, label="NDCG@10", color="#2c7fb8")
    ax.axhline(CE_R5, ls="--", c="#d95f02", alpha=0.6, lw=1)
    ax.axhline(CE_ND, ls="--", c="#2c7fb8", alpha=0.6, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([lbl[r] for r in runs])
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(r5, nd)):
        ax.text(i - w/2, a + 0.008, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + w/2, b + 0.008, f"{b:.3f}", ha="center", fontsize=7)
axes[0].set_ylabel("SciFact test metric")
axes[0].legend(loc="upper left", fontsize=8)
axes[-1].text(0.98, CE_R5 + 0.01, "CrossEncoder R@5=0.733", ha="right", fontsize=7,
              color="#d95f02", transform=axes[-1].get_yaxis_transform())
fig.suptitle("R3.9b ablation (SciFact test): hard-neg count is the dominant lever; "
             "LoRA rank & steps are not. n=15 reaches ~94% of CrossEncoder R@5",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT / "r39b_ablation_bars.png", dpi=130); plt.close(fig)
print("saved r39b_ablation_bars.png")

# ===== Plot 2: multi-seed stability of best config (negs15) =====
seedrows = df[df.run == "negs15"].sort_values("seed")
if len(seedrows) >= 2:
    r5 = seedrows["test_recall@5"].values
    nd = seedrows["test_ndcg@10"].values
    auc = seedrows["test_roc_auc"].values
    seeds = seedrows["seed"].values
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    metrics = {"R@5": r5, "NDCG@10": nd, "ROC-AUC": auc}
    colors = {"R@5": "#d95f02", "NDCG@10": "#2c7fb8", "ROC-AUC": "#1b9e77"}
    x = np.arange(len(metrics)); 
    means = [v.mean() for v in metrics.values()]
    stds = [v.std(ddof=0) for v in metrics.values()]
    ax.bar(x, means, yerr=stds, capsize=8, color=[colors[k] for k in metrics],
           alpha=0.85, width=0.55)
    for i, (k, v) in enumerate(metrics.items()):
        ax.text(i, means[i] + stds[i] + 0.01,
                f"{means[i]:.3f}\n±{stds[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        # scatter individual seeds
        ax.scatter([i]*len(v), v, color="black", zorder=5, s=22)
    ax.axhline(CE_R5, ls="--", c="#d95f02", alpha=0.5, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylabel("SciFact test")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"R3.9b multi-seed stability — best config (n=15, r16, 1500) "
                 f"seeds {list(seeds)}\nCrossEncoder R@5=0.733 (dashed)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT / "r39b_seed_stability.png", dpi=130); plt.close(fig)
    print("saved r39b_seed_stability.png")
else:
    print("multi-seed rows not ready, skipping seed stability plot")
