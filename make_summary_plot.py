"""Summary plot: Recall@5 across all methods + accuracy-latency frontier."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

outdir = Path("plots/linear_rag"); outdir.mkdir(parents=True, exist_ok=True)

# (label, recall@5, latency_ms_per_query_batch1, color)
rows = [
    ("Mamba-130m\nzero-shot", 0.0426, 878.9, "#bdbdbd"),
    ("Embedding\n(bge-small)", 0.3196, 0.13, "#90caf9"),
    ("BM25", 0.4282, 7.94, "#64b5f6"),
    ("Cross-encoder\n(MiniLM)", 0.7240, 18.2, "#ffb74d"),
    ("Mamba-130m\n+ LoRA (R3)", 0.7825, 12.3, "#66bb6a"),
]

# --- Bar chart: Recall@5 ---
fig, ax = plt.subplots(figsize=(9, 5.2))
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
colors = [r[3] for r in rows]
bars = ax.bar(labels, vals, color=colors, edgecolor="#333", linewidth=0.8)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Recall@5", fontsize=12)
ax.set_ylim(0, 0.88)
ax.set_title("Linear-RAG: Reranking Recall@5 by Method (synth_rag_v1, 5000 queries)",
             fontsize=12.5, fontweight="bold")
ax.axhline(0.4282, ls="--", lw=1, color="#1976d2", alpha=0.6)
ax.text(0.02, 0.445, "BM25 baseline", transform=ax.get_yaxis_transform(),
        color="#1976d2", fontsize=9, alpha=0.8)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
fig.savefig(outdir / "summary_recall5_by_method.png", dpi=140)
print("wrote summary_recall5_by_method.png")

# --- Frontier: accuracy vs latency (log x) ---
fig2, ax2 = plt.subplots(figsize=(8.5, 5.2))
for label, r5, lat, col in rows:
    ax2.scatter(lat, r5, s=160, color=col, edgecolor="#333", zorder=3)
    ax2.annotate(label.replace("\n", " "), (lat, r5),
                 textcoords="offset points", xytext=(8, 6), fontsize=9)
ax2.set_xscale("log")
ax2.set_xlabel("Latency per query (ms, batch=1, log scale)", fontsize=12)
ax2.set_ylabel("Recall@5", fontsize=12)
ax2.set_title("Accuracy-Latency Frontier (higher-left is better)",
              fontsize=12.5, fontweight="bold")
ax2.grid(alpha=0.25, which="both")
plt.tight_layout()
fig2.savefig(outdir / "summary_accuracy_latency_frontier.png", dpi=140)
print("wrote summary_accuracy_latency_frontier.png")
