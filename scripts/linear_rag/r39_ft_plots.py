#!/usr/bin/env python
"""R3.9 finetune plots: recall/ndcg comparison, latency/vram, score dist, PR/ROC."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

REPO = Path("/root/autodl-tmp/linear")
RES = REPO / "results/linear_rag"
PLOT = REPO / "plots/linear_rag"; PLOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RES / "r39_real_finetune_metrics.csv")
cal = pd.read_csv(RES / "r39_real_finetune_calibration.csv")
arr = np.load(RES / "r39_calibration_arrays.npz")

ORDER = ["BM25", "CrossEncoder-MiniLM-L6",
         "Mamba-130m-cls(synthetic)", "Mamba-130m-cls(finetuned)",
         "Pythia-160m-cls(synthetic)", "Pythia-160m-cls(finetuned)"]
SHORT = {"BM25": "BM25", "CrossEncoder-MiniLM-L6": "CrossEncoder",
         "Mamba-130m-cls(synthetic)": "Mamba 0-shot",
         "Mamba-130m-cls(finetuned)": "Mamba finetuned",
         "Pythia-160m-cls(synthetic)": "Pythia 0-shot",
         "Pythia-160m-cls(finetuned)": "Pythia finetuned"}
CLR = {"BM25": "#888888", "CrossEncoder-MiniLM-L6": "#2c7fb8",
       "Mamba-130m-cls(synthetic)": "#fdae6b", "Mamba-130m-cls(finetuned)": "#d95f02",
       "Pythia-160m-cls(synthetic)": "#bcbddc", "Pythia-160m-cls(finetuned)": "#7570b3"}
DSS = ["scifact", "nfcorpus"]


def comp_bar(metric, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, ds in zip(axes, DSS):
        sub = df[df.dataset == ds].set_index("model")
        vals = [sub.loc[m, metric] if m in sub.index else 0 for m in ORDER]
        ax.bar([SHORT[m] for m in ORDER], vals, color=[CLR[m] for m in ORDER])
        ax.set_title(f"{ds.upper()}", fontweight="bold")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    axes[0].set_ylabel(metric)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT / fname, dpi=130); plt.close(fig); print("saved", fname)


comp_bar("recall@5", "r39_real_finetune_recall_comparison.png",
         "R3.9 Recall@5 (test): finetune recovers Mamba from 0-shot; Mamba > Pythia; still < CrossEncoder")
comp_bar("ndcg@10", "r39_real_finetune_ndcg_comparison.png",
         "R3.9 Real-Data Finetune — NDCG@10 (test)")

# latency / vram (reranker models only)
fig, ax = plt.subplots(figsize=(8.5, 6))
sub = df[df.vram_mb.notna()].groupby("model").agg(
    lat=("latency_per_q_ms", "mean"), vram=("vram_mb", "mean"),
    r5=("recall@5", "mean")).reset_index()
for _, r in sub.iterrows():
    ax.scatter(r.lat, r.vram, s=260, color=CLR.get(r.model, "#333"), edgecolor="k", zorder=3)
    ax.annotate(f"{SHORT.get(r.model, r.model)}\nmean R@5={r.r5:.3f}",
                (r.lat, r.vram), textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.set_xlabel("latency / query (ms, mean over datasets)")
ax.set_ylabel("peak eval VRAM (MB)")
ax.set_title("R3.9 reranker efficiency frontier (test)\nMamba finetuned: lower VRAM than Pythia & CE, but ~2x slower/query",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(PLOT / "r39_real_finetune_latency_vram.png", dpi=130); plt.close(fig)
print("saved r39_real_finetune_latency_vram.png")

# score distributions (finetuned models, scifact)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mk in zip(axes, ["Mamba-130m-cls(finetuned)", "Pythia-160m-cls(finetuned)"]):
    key = f"scifact|{mk}"
    ys = arr[f"{key}|ys"]; ss = arr[f"{key}|ss"]
    ax.hist(ss[ys == 0], bins=40, alpha=0.6, label="negative", color="#888", density=True)
    ax.hist(ss[ys == 1], bins=20, alpha=0.7, label="positive (relevant)", color="#d95f02", density=True)
    auc = cal[(cal.dataset == "scifact") & (cal.model == mk)].roc_auc.values[0]
    ax.set_title(f"{SHORT[mk]} — SciFact (ROC-AUC={auc:.3f})", fontweight="bold")
    ax.set_xlabel("relevance score (logit[1]-logit[0])"); ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("R3.9 Score Distribution — finetuned models separate pos/neg (esp. Mamba)",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(PLOT / "r39_score_distribution.png", dpi=130); plt.close(fig)
print("saved r39_score_distribution.png")

# ROC + PR curves (scifact, all 4 cls variants)
variants = ["Mamba-130m-cls(finetuned)", "Pythia-160m-cls(finetuned)",
            "Mamba-130m-cls(synthetic)", "Pythia-160m-cls(synthetic)"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
for mk in variants:
    key = f"scifact|{mk}"
    ys = arr[f"{key}|ys"]; ss = arr[f"{key}|ss"]
    fpr, tpr, _ = roc_curve(ys, ss); prec, rec, _ = precision_recall_curve(ys, ss)
    a1.plot(fpr, tpr, color=CLR[mk], label=SHORT[mk])
    a2.plot(rec, prec, color=CLR[mk], label=SHORT[mk])
a1.plot([0, 1], [0, 1], "k--", alpha=0.4); a1.set_xlabel("FPR"); a1.set_ylabel("TPR")
a1.set_title("ROC — SciFact", fontweight="bold"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
pos_rate = cal[cal.dataset == "scifact"].pos_rate.values[0]
a2.axhline(pos_rate, color="r", ls="--", alpha=0.5, label=f"pos rate={pos_rate:.3f}")
a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
a2.set_title("PR — SciFact", fontweight="bold"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(PLOT / "r39_pr_roc_curve.png", dpi=130); plt.close(fig)
print("saved r39_pr_roc_curve.png")
