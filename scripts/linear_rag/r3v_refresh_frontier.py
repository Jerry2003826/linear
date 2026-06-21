from __future__ import annotations
"""Refresh the accuracy-latency / accuracy-VRAM frontier using the CORRECTED
batched scoring numbers for Mamba. Adds a throughput-vs-batch panel showing how
the Mamba/CE latency gap narrows but does not close."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("/root/autodl-tmp/linear")
res = root / "results/linear_rag"
plot = root / "plots/linear_rag"

bl = pd.read_csv(res / "r3_validation_batched_latency.csv")
# accuracy (test Recall@5) from finalize
acc = json.loads((res / "r3_validation_latency_vram_accuracy.json").read_text())
# acc keys: mamba_130m_lora, pythia_160m_lora, cross_encoder
mamba_r5 = acc.get("mamba_130m_lora", 0.7453)
ce_r5 = acc.get("cross_encoder", 0.709)

def pick(model, bs, col):
    r = bl[(bl.model == model) & (bl.batch == bs)]
    return float(r[col].iloc[0]) if len(r) else float("nan")

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

# Panel 1: accuracy vs latency at batch=32 (best throughput config)
for model, label, r5, color in [
    ("mamba_130m_lora_batched", "Mamba-130m LoRA\n(batched, len-bucket)", mamba_r5, "#2563eb"),
    ("cross_encoder", "Cross-encoder\nMiniLM", ce_r5, "#dc2626")]:
    x = pick(model, 32, "ms_per_query_top100")
    ax[0].scatter(x, r5, s=160, color=color, zorder=3)
    ax[0].annotate(label, (x, r5), textcoords="offset points",
                   xytext=(-12, -28), fontsize=10, ha="right")
ax[0].set_xlabel("latency ms/query (top-100, batch=32)")
ax[0].set_ylabel("test Recall@5")
ax[0].set_xscale("log")
ax[0].set_xlim(15, 90)
ax[0].set_ylim(0.700, 0.752)
ax[0].set_title("Accuracy vs Latency (corrected, batched)")
ax[0].grid(alpha=.3)

# Panel 2: accuracy vs peak VRAM at batch=32
for model, label, r5, color in [
    ("mamba_130m_lora_batched", "Mamba-130m LoRA", mamba_r5, "#2563eb"),
    ("cross_encoder", "Cross-encoder MiniLM", ce_r5, "#dc2626")]:
    v = pick(model, 32, "peak_vram_mb")
    ax[1].scatter(v, r5, s=160, color=color, zorder=3)
    ax[1].annotate(label, (v, r5), textcoords="offset points",
                   xytext=(8, -4), fontsize=10)
ax[1].set_xlabel("peak VRAM (MB, batch=32)")
ax[1].set_ylabel("test Recall@5")
ax[1].set_title("Accuracy vs VRAM (batched)")
ax[1].grid(alpha=.3)

# Panel 3: latency vs batch size (the gap narrows but never closes)
for model, label, color, marker in [
    ("mamba_130m_lora_batched", "Mamba-130m LoRA", "#2563eb", "o"),
    ("cross_encoder", "Cross-encoder MiniLM", "#dc2626", "s")]:
    sub = bl[bl.model == model].sort_values("batch")
    ax[2].plot(sub.batch, sub.ms_per_query_top100, marker=marker,
               color=color, label=label, linewidth=2, markersize=8)
ax[2].set_xlabel("batch size")
ax[2].set_ylabel("latency ms/query (top-100)")
ax[2].set_yscale("log")
ax[2].set_xticks([1, 8, 16, 32])
ax[2].set_title("Latency vs Batch: gap narrows, not closed")
ax[2].legend()
ax[2].grid(alpha=.3)

fig.tight_layout()
out = plot / "r3_validation_frontier.png"
fig.savefig(out, dpi=130)
print("saved", out)

# also write a tidy comparison table for the summary
gap = {bs: round(pick("mamba_130m_lora_batched", bs, "ms_per_query_top100") /
                  pick("cross_encoder", bs, "ms_per_query_top100"), 2)
       for bs in [1, 8, 16, 32]}
print("mamba/ce latency ratio by batch:", gap)
(res / "r3_validation_batched_summary.json").write_text(json.dumps({
    "mamba_ms_per_query": {bs: round(pick("mamba_130m_lora_batched", bs, "ms_per_query_top100"), 1) for bs in [1,8,16,32]},
    "ce_ms_per_query": {bs: round(pick("cross_encoder", bs, "ms_per_query_top100"), 1) for bs in [1,8,16,32]},
    "mamba_vram_mb": {bs: pick("mamba_130m_lora_batched", bs, "peak_vram_mb") for bs in [1,8,16,32]},
    "ce_vram_mb": {bs: pick("cross_encoder", bs, "peak_vram_mb") for bs in [1,8,16,32]},
    "latency_ratio_mamba_over_ce": gap,
    "mamba_scoring": "right-pad batched, length-bucketed; spearman 0.9965 & sign-agreement 1.0 vs verified single-pair",
}, indent=2))
print("wrote r3_validation_batched_summary.json")
