"""R3 validation: create a fixed, reproducible, stratified train/dev/test split.

Stratify jointly by (difficulty, conditions_bucket).
- conditions_bucket: 1, 2, 3, 4+ (n_conditions 5/6 folded into 4+ since they are rare)
Sizes: train=3000, dev=1000, test=1000 (total 5000).
Saves data/synth_rag_v1/splits/r3_validation_split.json and a summary md.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

SEED = 42
DATA = Path("data/synth_rag_v1")
SPLIT_DIR = DATA / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
SUM = Path("summaries/linear_rag"); SUM.mkdir(parents=True, exist_ok=True)

N_TRAIN, N_DEV, N_TEST = 3000, 1000, 1000

queries = [json.loads(l) for l in open(DATA / "queries.jsonl")]
assert len(queries) == N_TRAIN + N_DEV + N_TEST, f"expected 5000, got {len(queries)}"


def cond_bucket(n):
    return str(n) if n <= 3 else "4+"


# group by stratum
strata = defaultdict(list)
for q in queries:
    key = (q["difficulty"], cond_bucket(q["n_conditions"]))
    strata[key].append(q["query_id"])

rng = np.random.RandomState(SEED)
# proportional allocation per stratum with deterministic shuffling
frac_dev = N_DEV / len(queries)
frac_test = N_TEST / len(queries)

train_ids, dev_ids, test_ids = [], [], []
for key in sorted(strata.keys()):
    ids = sorted(strata[key])
    rng.shuffle(ids)
    n = len(ids)
    n_dev = int(round(n * frac_dev))
    n_test = int(round(n * frac_test))
    dev_ids += ids[:n_dev]
    test_ids += ids[n_dev:n_dev + n_test]
    train_ids += ids[n_dev + n_test:]

# fix rounding drift to hit exact sizes by moving from the largest pool (train)
def rebalance(target_train, target_dev, target_test):
    global train_ids, dev_ids, test_ids
    # move excess/deficit using train as buffer
    rng2 = np.random.RandomState(SEED + 1)
    pools = {"train": train_ids, "dev": dev_ids, "test": test_ids}
    targets = {"train": target_train, "dev": target_dev, "test": target_test}
    # first top up dev/test from train if short
    for name in ("dev", "test"):
        while len(pools[name]) < targets[name]:
            pools[name].append(pools["train"].pop())
        while len(pools[name]) > targets[name]:
            pools["train"].append(pools[name].pop())
    # train auto-correct
    train_ids[:] = pools["train"]; dev_ids[:] = pools["dev"]; test_ids[:] = pools["test"]

rebalance(N_TRAIN, N_DEV, N_TEST)
train_ids = sorted(train_ids); dev_ids = sorted(dev_ids); test_ids = sorted(test_ids)

# integrity: disjoint + complete
s_tr, s_dv, s_te = set(train_ids), set(dev_ids), set(test_ids)
assert len(s_tr & s_dv) == 0 and len(s_tr & s_te) == 0 and len(s_dv & s_te) == 0, "overlap!"
assert len(s_tr | s_dv | s_te) == len(queries), "not complete!"
assert len(train_ids) == N_TRAIN and len(dev_ids) == N_DEV and len(test_ids) == N_TEST

split = {
    "seed": SEED, "n_total": len(queries),
    "sizes": {"train": N_TRAIN, "dev": N_DEV, "test": N_TEST},
    "stratify_by": ["difficulty", "conditions_bucket(1,2,3,4+)"],
    "train": train_ids, "dev": dev_ids, "test": test_ids,
}
(SPLIT_DIR / "r3_validation_split.json").write_text(json.dumps(split))
print("wrote", SPLIT_DIR / "r3_validation_split.json")

# summary tables
qmap = {q["query_id"]: q for q in queries}
def dist(ids, field):
    if field == "cond":
        return Counter(cond_bucket(qmap[i]["n_conditions"]) for i in ids)
    return Counter(qmap[i][field] for i in ids)

lines = ["# R3 Validation Split Summary\n",
         f"- seed: {SEED} (reproducible)",
         f"- sizes: train={N_TRAIN}, dev={N_DEV}, test={N_TEST}",
         "- stratified jointly by (difficulty, conditions_bucket)",
         "- disjoint & complete verified; same split used by all models\n",
         "## Difficulty distribution (count / fraction)\n",
         "| split | easy | medium | hard | adversarial |",
         "|---|---|---|---|---|"]
for name, ids in [("train", train_ids), ("dev", dev_ids), ("test", test_ids)]:
    d = dist(ids, "difficulty"); n = len(ids)
    lines.append(f"| {name} | {d['easy']} ({d['easy']/n:.1%}) | {d['medium']} ({d['medium']/n:.1%}) | "
                 f"{d['hard']} ({d['hard']/n:.1%}) | {d['adversarial']} ({d['adversarial']/n:.1%}) |")
lines += ["\n## Conditions-per-query distribution\n",
          "| split | 1 | 2 | 3 | 4+ |", "|---|---|---|---|---|"]
for name, ids in [("train", train_ids), ("dev", dev_ids), ("test", test_ids)]:
    d = dist(ids, "cond"); n = len(ids)
    lines.append(f"| {name} | {d['1']} ({d['1']/n:.1%}) | {d['2']} ({d['2']/n:.1%}) | "
                 f"{d['3']} ({d['3']/n:.1%}) | {d['4+']} ({d['4+']/n:.1%}) |")
(SUM / "r3_validation_split_summary.md").write_text("\n".join(lines) + "\n")
print("wrote", SUM / "r3_validation_split_summary.md")
print("sizes:", len(train_ids), len(dev_ids), len(test_ids))
