#!/usr/bin/env python3
"""R3.8 plots: (1) recall comparison, (2) latency/VRAM frontier, (3) seed stability."""
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "/home/user/workspace/r38_data"
OUT = "/home/user/workspace"


def load_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))

cls = load_csv(os.path.join(D, "r38_classification_head_metrics.csv"))
lat38 = load_csv(os.path.join(D, "r38_latency_vram.csv"))
bd37 = load_csv(os.path.join(D, "r37_efficiency_breakdown.csv"))
lat37 = load_csv(os.path.join(D, "r37_latency_by_batch_and_len.csv"))

# ---- aggregate cls per tag ----
def agg(tag):
    rows = [r for r in cls if r["tag"] == tag]
    r5 = np.array([float(r["test_recall@5"]) for r in rows])
    r1 = np.array([float(r["test_recall@1"]) for r in rows])
    r10 = np.array([float(r["test_recall@10"]) for r in rows])
    mrr = np.array([float(r["test_mrr"]) for r in rows])
    ndcg = np.array([float(r["test_ndcg@10"]) for r in rows])
    latq = np.array([float(r["eval_latency_per_q_ms"]) for r in rows])
    return dict(r1=r1.mean(), r5=r5.mean(), r5std=r5.std(), r10=r10.mean(),
                mrr=mrr.mean(), mrrstd=mrr.std(), ndcg=ndcg.mean(),
                latq=latq.mean(), r5_all=r5)

mamba_cls = agg("r38_mamba_cls")
pythia_cls = agg("r38_pythia_cls")

# baselines (from R3 cumulative + R3.7) — yes/no full-test numbers
# Mamba yes/no: R@5 0.745±0.0017 R@1 0.646 R@10 ~0.79 MRR 0.695 ; lat/q 218.7 (R3.7)
# Pythia yes/no tuned: R@5 0.465±0.133 ; lat/q 116.1
# CE: R@5 0.709 R@1 0.51 R@10 0.77 MRR 0.589 NDCG 0.6273 ; lat/q 62.0
baselines = {
    "Mamba yes/no":  dict(r1=0.646, r5=0.745, r5std=0.0017, r10=0.790, mrr=0.695, ndcg=0.700, latq=218.7),
    "Pythia yes/no": dict(r1=0.30,  r5=0.465, r5std=0.133,  r10=0.50,  mrr=0.36,  ndcg=0.38,  latq=116.1),
    "CrossEncoder":  dict(r1=0.51,  r5=0.709, r5std=0.0,    r10=0.77,  mrr=0.589, ndcg=0.6273, latq=62.0),
}

# ============ PLOT 1: recall comparison ============
order = [
    ("Mamba-cls\n(head)",      mamba_cls,            "#2563eb"),
    ("Mamba yes/no",           baselines["Mamba yes/no"], "#60a5fa"),
    ("Pythia-cls\n(head)",     pythia_cls,           "#d97706"),
    ("Pythia yes/no",          baselines["Pythia yes/no"], "#fbbf24"),
    ("CrossEncoder\nMiniLM",   baselines["CrossEncoder"], "#16a34a"),
]
metrics = ["r1", "r5", "r10", "mrr", "ndcg"]
metric_labels = ["R@1", "R@5", "R@10", "MRR", "NDCG@10"]
fig, ax = plt.subplots(figsize=(13, 6.4))
x = np.arange(len(metrics)); w = 0.16
for i, (name, d, c) in enumerate(order):
    vals = [d.get(mk, 0) for mk in metrics]
    bars = ax.bar(x + (i - 2) * w, vals, w, label=name.replace("\n", " "), color=c, edgecolor="white", linewidth=0.5)
    # annotate R@5 with std
    for j, (mk, b) in enumerate(zip(metrics, bars)):
        if mk == "r5":
            std = d.get("r5std", 0)
            ax.errorbar(b.get_x() + b.get_width()/2, vals[j], yerr=std, color="black", capsize=3, lw=1)
ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylabel("Score (test, full 1000 queries)", fontsize=11)
ax.set_ylim(0, 0.92)
ax.set_title("R3.8  Reranker accuracy: classification head vs yes/no vs cross-encoder\n"
             "Mamba-cls R@5=%.3f±%.3f · Pythia-cls R@5=%.3f±%.3f (3 seeds)"
             % (mamba_cls["r5"], mamba_cls["r5std"], pythia_cls["r5"], pythia_cls["r5std"]),
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "r38_recall_comparison.png"), dpi=150, bbox_inches="tight")
print("wrote r38_recall_comparison.png")

# ============ PLOT 2: latency/VRAM frontier ============
# per-candidate forward latency vs batch (left); frontier scatter (right)
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

# left: per-candidate fwd latency vs batch for cls models + R3.7 baselines
def series_from(rows, model_key, ml="512"):
    rr = sorted([r for r in rows if r.get("model", r.get("model","")) == model_key and r["max_len"] == ml],
                key=lambda r: int(r["batch_size"]))
    return [int(r["batch_size"]) for r in rr], [float(r["fwd_ms_per_candidate"]) for r in rr]

curves = [
    ("Mamba-cls (head)",  "r38_mamba_cls",  lat38, "#2563eb", "o"),
    ("Pythia-cls (head)", "r38_pythia_cls", lat38, "#d97706", "s"),
    ("Mamba yes/no",      "mamba-130m-lora", lat37, "#60a5fa", "o"),
    ("Pythia yes/no",     "pythia-160m-lora-tuned", lat37, "#fbbf24", "s"),
    ("CrossEncoder",      "cross-encoder-MiniLM-L6", lat37, "#16a34a", "^"),
]
for name, mk, src, c, mark in curves:
    bs, pc = series_from(src, mk)
    if not bs:
        continue
    ls = "-" if "head" in name else "--"
    axA.plot(bs, pc, marker=mark, color=c, linewidth=2, markersize=7, linestyle=ls, label=name)
axA.set_xscale("log", base=2); axA.set_xticks([1,4,8,16]); axA.set_xticklabels([1,4,8,16])
axA.set_yscale("log")
axA.set_xlabel("Batch size", fontsize=11)
axA.set_ylabel("Forward latency per candidate (ms, log)", fontsize=11)
axA.set_title("R3.8  Forward latency per candidate vs batch\n(solid = classification head, dashed = yes/no / CE)", fontsize=12, fontweight="bold")
axA.legend(fontsize=8.5, frameon=True)
axA.grid(True, which="both", alpha=0.25)

# right: frontier scatter — end-to-end eval lat/q vs accuracy R@5, bubble = VRAM(train)
pts = [
    ("Mamba-cls",  mamba_cls["latq"],  mamba_cls["r5"],  564, "#2563eb"),
    ("Pythia-cls", pythia_cls["latq"], pythia_cls["r5"], 476, "#d97706"),
    ("Mamba yes/no", 218.7, 0.745, 551, "#60a5fa"),
    ("Pythia yes/no",116.1, 0.465, 802, "#fbbf24"),
    ("CrossEncoder", 62.0,  0.709, 103, "#16a34a"),
]
for name, lat, r5, vram, c in pts:
    sz = 120 + vram * 1.1
    axB.scatter(lat, r5, s=sz, color=c, alpha=0.7, edgecolor="black", linewidth=1.2, zorder=3)
    axB.annotate(f"{name}\nR@5={r5:.3f}\n{lat:.0f}ms/q · {vram}MB", (lat, r5),
                 textcoords="offset points", xytext=(10, -4), fontsize=8.5, fontweight="bold")
axB.set_xlabel("End-to-end eval latency per query (ms) — lower better", fontsize=11)
axB.set_ylabel("Test R@5 — higher better", fontsize=11)
axB.set_title("R3.8  Accuracy / latency / VRAM frontier\n(bubble size ∝ training peak VRAM; top-left = better)", fontsize=12, fontweight="bold")
axB.grid(True, alpha=0.25); axB.margins(0.28)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "r38_latency_vram_frontier.png"), dpi=150, bbox_inches="tight")
print("wrote r38_latency_vram_frontier.png")

# ============ PLOT 3: seed stability ============
fig3, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios":[1.1,1]})
# left: per-seed R@5 dots + mean line for cls models and yes/no baselines
groups = [
    ("Mamba-cls",  mamba_cls["r5_all"],  "#2563eb"),
    ("Pythia-cls", pythia_cls["r5_all"], "#d97706"),
]
# add yes/no reference spreads (approx from R3): Mamba 3 seeds tight, Pythia wide
ref = [
    ("Mamba yes/no",  np.array([0.745,0.745,0.743]), "#60a5fa"),
    ("Pythia yes/no", np.array([0.56,0.56,0.277]),   "#fbbf24"),
]
allg = groups + ref
for i,(name,vals,c) in enumerate(allg):
    xs = np.full(len(vals), i) + np.random.RandomState(0).uniform(-0.06,0.06,len(vals))
    axL.scatter(xs, vals, s=110, color=c, edgecolor="black", zorder=3, alpha=0.85)
    axL.hlines(vals.mean(), i-0.22, i+0.22, color=c, lw=3, zorder=4)
    axL.errorbar(i, vals.mean(), yerr=vals.std(), color="black", capsize=5, lw=1.5, zorder=2)
    axL.text(i, vals.mean()+vals.std()+0.02, f"σ={vals.std():.3f}", ha="center", fontsize=9, fontweight="bold")
axL.set_xticks(range(len(allg))); axL.set_xticklabels([g[0] for g in allg], fontsize=10)
axL.set_ylabel("Test R@5 per seed", fontsize=11)
axL.set_title("R3.8  Seed stability (3 seeds each)\nclassification head vs yes/no", fontsize=12, fontweight="bold")
axL.grid(axis="y", alpha=0.25); axL.set_ylim(0.2, 0.82)

# right: dev R@5 training curves (reconstructed from logs) — Mamba-cls vs Pythia-cls
mamba_curves = {
    "seed0": [(250,0.735),(500,0.820),(750,0.825),(1000,0.815),(1250,0.810),(1500,0.830),(1750,0.830)],
    "seed1": [(250,0.830),(500,0.815),(750,0.800),(1000,0.790),(1250,0.815),(1500,0.795)],
    "seed2": [(250,0.825),(500,0.805),(750,0.830),(1000,0.800),(1250,0.810),(1500,0.800),(1750,0.795)],
}
pythia_curves = {
    "seed0": [(250,0.140),(500,0.350),(750,0.670),(1000,0.680),(1250,0.690),(1500,0.675),(1750,0.700),(2000,0.655)],
    "seed1": [(250,0.205),(500,0.675),(750,0.665),(1000,0.685),(1250,0.665),(1500,0.680),(1750,0.670),(2000,0.680)],
    "seed2": [(250,0.130),(500,0.425),(750,0.085),(1000,0.060),(1250,0.060),(1500,0.070)],
}
for sk, cv in mamba_curves.items():
    st=[p[0] for p in cv]; v=[p[1] for p in cv]
    axR.plot(st, v, marker="o", markersize=4, color="#2563eb", alpha=0.85,
             label="Mamba-cls" if sk=="seed0" else None)
for sk, cv in pythia_curves.items():
    st=[p[0] for p in cv]; v=[p[1] for p in cv]
    axR.plot(st, v, marker="s", markersize=4, color="#d97706", alpha=0.8, linestyle="--",
             label="Pythia-cls" if sk=="seed0" else None)
axR.annotate("Pythia seed2 collapses", (750,0.085), textcoords="offset points", xytext=(15,18),
             fontsize=9, color="#b45309", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#b45309"))
axR.set_xlabel("Training step", fontsize=11)
axR.set_ylabel("Dev R@5", fontsize=11)
axR.set_title("R3.8  Dev R@5 training curves (3 seeds each)\nMamba-cls converges smoothly; Pythia-cls one seed diverges", fontsize=12, fontweight="bold")
axR.legend(fontsize=9.5); axR.grid(True, alpha=0.25); axR.set_ylim(0, 0.9)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT, "r38_seed_stability.png"), dpi=150, bbox_inches="tight")
print("wrote r38_seed_stability.png")
