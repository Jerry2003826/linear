"""Append the R3.6 fairly-tuned Pythia row to the model comparison CSV."""
import csv, pathlib

p = pathlib.Path("/root/autodl-tmp/linear/results/linear_rag/r3_validation_model_comparison.csv")
rows = list(csv.reader(p.read_text().splitlines()))
header = rows[0]
data = rows[1:]

# Drop any prior tuned row (idempotent re-run)
data = [r for r in data if "fairly tuned" not in r[0]]

# R3.6 tuned Pythia aggregate (from r3_pythia_160m_lora_tuned_summary.json)
tuned = {
    "model": "Pythia-160m LoRA (R3.6, fairly tuned)",
    "test_R@1": 0.2467,
    "test_R@5": 0.4653,
    "test_R@5_std": 0.1332,
    "test_R@10": 0.5220,
    "test_MRR": 0.3489,
    "test_NDCG@10": 0.3824,
    "cond_R@5": 0.5091,
    "dev_test_gap": 0.0197,
    "peak_VRAM_MB": 457.6,
    "lat_ms_per_q_b1": "",   # not re-benchmarked at batch level for tuned run
    "lat_ms_per_q_b8": "",
    "n_seeds": 3,
}
new_row = [tuned[c] for c in header]
data.append(new_row)

with p.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(data)

print("Updated. Final contents:")
print(p.read_text())
