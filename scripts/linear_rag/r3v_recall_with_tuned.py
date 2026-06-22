"""Refresh the Recall@5 comparison plot to include the R3.6 fairly-tuned Pythia,
showing both mean and error bars (std) to highlight the stability gap."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib

OUT = pathlib.Path("/root/autodl-tmp/linear/plots/linear_rag")
OUT.mkdir(parents=True, exist_ok=True)

# (label, mean R@5, std or None, n_seeds, color)
models = [
    ("Mamba-130m\nLoRA (ours)",            0.7453, 0.0017, 3, "#2563eb"),
    ("Cross-encoder\nMiniLM-L6",            0.709,  None,   1, "#6b7280"),
    ("Pythia-160m LoRA\n(R3.6 fair-tuned)", 0.4653, 0.1332, 3, "#f59e0b"),
    ("Pythia-160m LoRA\n(R3.2 mis-tuned)",  0.3223, 0.102,  3, "#d1d5db"),
]

fig, ax = plt.subplots(figsize=(8.2, 5.2))
x = np.arange(len(models))
means = [m[1] for m in models]
errs  = [m[2] if m[2] is not None else 0 for m in models]
colors = [m[4] for m in models]
labels = [m[0] for m in models]

bars = ax.bar(x, means, yerr=errs, capsize=6, color=colors,
              edgecolor="#1f2937", linewidth=0.8,
              error_kw=dict(ecolor="#111827", lw=1.6))

for i, m in enumerate(models):
    txt = f"{m[1]:.3f}"
    if m[2] is not None:
        txt += f"\n±{m[2]:.3f}"
    ax.text(i, m[1] + (errs[i] if errs[i] else 0) + 0.015, txt,
            ha="center", va="bottom", fontsize=9.5,
            fontweight="bold" if i == 0 else "normal")

ax.axhline(0.70, ls="--", lw=1, color="#9ca3af")
ax.text(len(models) - 0.45, 0.705, "gate ≥ 0.70", fontsize=8.5, color="#6b7280")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Test Recall@5 (held-out 1000 queries)", fontsize=10.5)
ax.set_ylim(0, 0.86)
ax.set_title("Reranker accuracy on identical test split\n(error bars = std across 3 seeds)",
             fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)

# annotate the stability story
ax.annotate("std 0.0017\n(3/3 seeds within 0.004)",
            xy=(0, 0.7453), xytext=(0.55, 0.80),
            fontsize=8, color="#2563eb",
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1))
ax.annotate("std 0.133 — 1 of 3\nseeds diverged",
            xy=(2, 0.4653 + 0.1332), xytext=(1.9, 0.70),
            fontsize=8, color="#b45309",
            arrowprops=dict(arrowstyle="->", color="#b45309", lw=1))

fig.tight_layout()
dest = OUT / "r3_validation_recall_comparison.png"
fig.savefig(dest, dpi=150, bbox_inches="tight")
print(f"Saved {dest}")
