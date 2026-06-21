"""R3 validation: candidate upper-bound audit.

Computes candidate Recall@100 / Recall@500 (is gold present in the candidate
set?) overall and stratified by difficulty and conditions_bucket, for the test
split (and also reports overall on all 5000 for context).
"""
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd

DATA = Path("data/synth_rag_v1")
RES = Path("results/linear_rag"); RES.mkdir(parents=True, exist_ok=True)
SUM = Path("summaries/linear_rag"); SUM.mkdir(parents=True, exist_ok=True)

queries = [json.loads(l) for l in open(DATA / "queries.jsonl")]
gold = {q["query_id"]: q["gold_doc_id"] for q in queries}
qmap = {q["query_id"]: q for q in queries}
split = json.loads((DATA / "splits" / "r3_validation_split.json").read_text())
test_ids = set(split["test"])


def cond_bucket(n):
    return str(n) if n <= 3 else "4+"


def load_cand(path):
    df = pd.read_parquet(path)
    out = defaultdict(list)
    for qid, grp in df.sort_values("rank").groupby("query_id"):
        out[int(qid)] = grp["doc_id"].astype(int).tolist()
    return out

cand100 = load_cand(RES / "r1_candidates_top100.parquet")
cand500 = load_cand(RES / "r1_candidates_top500.parquet")


def gold_present(cand, ids):
    if not ids:
        return float("nan")
    return sum(1 for q in ids if gold[q] in cand.get(q, [])) / len(ids)


rows = []
def add(scope, subset, ids):
    rows.append({
        "scope": scope, "subset": subset, "n_queries": len(ids),
        "candidate_recall@100": round(gold_present(cand100, ids), 4),
        "candidate_recall@500": round(gold_present(cand500, ids), 4),
    })

# overall (all 5000) and test split
all_ids = [q["query_id"] for q in queries]
add("all5000", "overall", all_ids)
add("test", "overall", sorted(test_ids))

# stratified on test
for diff in ["easy", "medium", "hard", "adversarial"]:
    ids = [q for q in test_ids if qmap[q]["difficulty"] == diff]
    add("test", f"difficulty={diff}", sorted(ids))
for cb in ["1", "2", "3", "4+"]:
    ids = [q for q in test_ids if cond_bucket(qmap[q]["n_conditions"]) == cb]
    add("test", f"conditions={cb}", sorted(ids))

df = pd.DataFrame(rows)
df.to_csv(RES / "r3_candidate_upper_bound.csv", index=False)
print(df.to_string(index=False))

test_r100 = df[(df.scope == "test") & (df.subset == "overall")]["candidate_recall@100"].iloc[0]
test_r500 = df[(df.scope == "test") & (df.subset == "overall")]["candidate_recall@500"].iloc[0]
use_top500 = test_r100 < 0.75

lines = ["# R3 Candidate Upper-Bound Audit\n",
         "Candidate recall = fraction of queries whose gold doc is present in the",
         "BM25 candidate set (reranking can never exceed this ceiling).\n",
         f"- **Test Recall@100 = {test_r100:.4f}**, Test Recall@500 = {test_r500:.4f}",
         f"- top100 ceiling {'INSUFFICIENT (<0.75): also evaluate top500' if use_top500 else 'sufficient (>=0.75)'}\n",
         "## Full table\n",
         df.to_markdown(index=False)]
(SUM / "r3_candidate_upper_bound_summary.md").write_text("\n".join(lines) + "\n")
print("\nwrote audit csv + summary; use_top500 =", use_top500)
