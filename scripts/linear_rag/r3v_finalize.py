from __future__ import annotations

"""R3 validation finalize: model comparison CSV, recall comparison plot, gate decision."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results/linear_rag")
P = Path("plots/linear_rag")


def load_lora(tag):
    s = json.loads((R / f"r3_{tag}_summary.json").read_text())
    a = s["aggregate"]
    df = pd.read_csv(R / f"r3_{tag}_seed_metrics.csv")
    return a, df


def main():
    mamba_a, mamba_df = load_lora("mamba_130m_lora")
    pythia_a, pythia_df = load_lora("pythia_160m_lora")
    ce = pd.read_csv(R / "r3_cross_encoder_same_split.csv").iloc[0]
    lat = pd.read_csv(R / "r3_validation_latency_vram.csv")

    def lat_of(model, batch, col):
        r = lat[(lat.model == model) & (lat.batch == batch)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    # ---- model comparison table ----
    rows = []
    rows.append({
        "model": "Mamba-130m LoRA (ours)",
        "test_R@1": round(mamba_a["test_recall@1_mean"], 4),
        "test_R@5": round(mamba_a["test_recall@5_mean"], 4),
        "test_R@5_std": round(mamba_a["test_recall@5_std"], 4),
        "test_R@10": round(mamba_a["test_recall@10_mean"], 4),
        "test_MRR": round(mamba_a["test_mrr_mean"], 4),
        "test_NDCG@10": round(mamba_a["test_ndcg@10_mean"], 4),
        "cond_R@5": round(mamba_a["test_cond_recall@5_mean"], 4),
        "dev_test_gap": round(mamba_a["dev_test_r5_gap_mean"], 4),
        "peak_VRAM_MB": round(lat_of("mamba_130m_lora", 1, "peak_vram_mb"), 0),
        "lat_ms_per_q_b1": round(lat_of("mamba_130m_lora", 1, "ms_per_query_top100"), 1),
        "lat_ms_per_q_b8": round(lat_of("mamba_130m_lora", 8, "ms_per_query_top100"), 1),
        "n_seeds": len(mamba_df),
    })
    rows.append({
        "model": "Cross-encoder MiniLM-L6 (baseline)",
        "test_R@1": round(float(ce["test_recall@1"]), 4),
        "test_R@5": round(float(ce["test_recall@5"]), 4),
        "test_R@5_std": np.nan,
        "test_R@10": round(float(ce["test_recall@10"]), 4),
        "test_MRR": round(float(ce["test_mrr"]), 4),
        "test_NDCG@10": round(float(ce["test_ndcg@10"]), 4),
        "cond_R@5": round(float(ce["test_cond_recall@5"]), 4),
        "dev_test_gap": np.nan,
        "peak_VRAM_MB": round(lat_of("cross_encoder", 1, "peak_vram_mb"), 0),
        "lat_ms_per_q_b1": round(lat_of("cross_encoder", 1, "ms_per_query_top100"), 1),
        "lat_ms_per_q_b8": round(lat_of("cross_encoder", 8, "ms_per_query_top100"), 1),
        "n_seeds": 1,
    })
    rows.append({
        "model": "Pythia-160m LoRA (baseline)",
        "test_R@1": round(pythia_a["test_recall@1_mean"], 4),
        "test_R@5": round(pythia_a["test_recall@5_mean"], 4),
        "test_R@5_std": round(pythia_a["test_recall@5_std"], 4),
        "test_R@10": round(pythia_a["test_recall@10_mean"], 4),
        "test_MRR": round(pythia_a["test_mrr_mean"], 4),
        "test_NDCG@10": round(pythia_a["test_ndcg@10_mean"], 4),
        "cond_R@5": round(pythia_a["test_cond_recall@5_mean"], 4),
        "dev_test_gap": round(pythia_a["dev_test_r5_gap_mean"], 4),
        "peak_VRAM_MB": round(lat_of("pythia_160m_lora", 1, "peak_vram_mb"), 0),
        "lat_ms_per_q_b1": round(lat_of("pythia_160m_lora", 1, "ms_per_query_top100"), 1),
        "lat_ms_per_q_b8": round(lat_of("pythia_160m_lora", 8, "ms_per_query_top100"), 1),
        "n_seeds": len(pythia_df),
    })
    # prior-stage references (full 5000-query, not same split — labeled)
    comp = pd.DataFrame(rows)
    comp.to_csv(R / "r3_validation_model_comparison.csv", index=False)
    print(comp.to_string(index=False))

    # ---- recall comparison plot (R@1/R@5/R@10 grouped bars) ----
    models = ["Mamba-130m\nLoRA (ours)", "Cross-encoder\nMiniLM", "Pythia-160m\nLoRA"]
    r1 = [mamba_a["test_recall@1_mean"], float(ce["test_recall@1"]),
          pythia_a["test_recall@1_mean"]]
    r5 = [mamba_a["test_recall@5_mean"], float(ce["test_recall@5"]),
          pythia_a["test_recall@5_mean"]]
    r10 = [mamba_a["test_recall@10_mean"], float(ce["test_recall@10"]),
           pythia_a["test_recall@10_mean"]]
    r5err = [mamba_a["test_recall@5_std"], 0, pythia_a["test_recall@5_std"]]
    x = np.arange(len(models)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w, r1, w, label="Recall@1", color="#4C72B0")
    ax.bar(x, r5, w, yerr=r5err, capsize=4, label="Recall@5", color="#DD8452")
    ax.bar(x + w, r10, w, label="Recall@10", color="#55A868")
    ax.axhline(float(ce["test_recall@5"]), ls="--", color="gray", lw=1,
               label="CE Recall@5 ref")
    for i, v in enumerate(r5):
        ax.text(x[i], v + 0.015, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Recall"); ax.set_ylim(0, 0.95)
    ax.set_title("R3 Validation — Reranker Recall on identical 1000-query test split\n"
                 "(3000/1000/1000 split, stratified, mean over 3 seeds)")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(P / "r3_validation_recall_comparison.png", dpi=130)
    print("saved", P / "r3_validation_recall_comparison.png")

    # ---- GATE decision ----
    m_r5 = mamba_a["test_recall@5_mean"]
    m_std = mamba_a["test_recall@5_std"]
    m_gap = mamba_a["dev_test_r5_gap_mean"]
    ce_r5 = float(ce["test_recall@5"])
    ce_vram = lat_of("cross_encoder", 1, "peak_vram_mb")
    ce_lat = lat_of("cross_encoder", 1, "ms_per_query_top100")
    m_vram = lat_of("mamba_130m_lora", 1, "peak_vram_mb")
    m_lat = lat_of("mamba_130m_lora", 1, "ms_per_query_top100")
    # efficiency advantage: lower is better
    vram_better = (ce_vram - m_vram) / ce_vram  # negative => worse
    lat_better = (ce_lat - m_lat) / ce_lat
    eff_25 = (vram_better >= 0.25) or (lat_better >= 0.25)

    cond_pass = (m_r5 >= 0.70)
    stable = (m_std <= 0.08)
    gap_ok = (m_gap <= 0.05)
    beats_ce = (m_r5 >= 0.95 * ce_r5)
    pythia_dominates = pythia_a["test_recall@5_mean"] > m_r5

    if pythia_dominates or (m_std > 0.08) or (m_gap > 0.05):
        gate = "FAIL"
    elif cond_pass and stable and gap_ok and eff_25:
        gate = "STRONG_PASS" if (m_r5 >= 0.95 * ce_r5) else "PASS"
    elif beats_ce and stable and gap_ok and not eff_25:
        # accuracy meets/exceeds CE & stable, but efficiency advantage gone
        gate = "PARTIAL"
    elif cond_pass and stable and gap_ok:
        gate = "PASS"
    else:
        gate = "PARTIAL"

    decision = {
        "gate": gate,
        "mamba_test_R5_mean": round(m_r5, 4), "mamba_test_R5_std": round(m_std, 4),
        "mamba_dev_test_gap": round(m_gap, 4),
        "cross_encoder_R5": round(ce_r5, 4),
        "pythia_R5_mean": round(pythia_a["test_recall@5_mean"], 4),
        "pythia_R5_std": round(pythia_a["test_recall@5_std"], 4),
        "mamba_vs_ce_R5_abs": round(m_r5 - ce_r5, 4),
        "vram_better_frac_vs_ce": round(vram_better, 3),
        "latency_better_frac_vs_ce": round(lat_better, 3),
        "efficiency_25pct_better": bool(eff_25),
        "checks": {"R5>=0.70": bool(cond_pass), "std<=0.08": bool(stable),
                   "gap<=0.05": bool(gap_ok), "R5>=0.95xCE": bool(beats_ce),
                   "eff>=25%": bool(eff_25), "pythia_dominates": bool(pythia_dominates)},
    }
    (R / "r3_validation_gate_decision.json").write_text(json.dumps(decision, indent=2))
    print("\n=== GATE:", gate, "===")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
