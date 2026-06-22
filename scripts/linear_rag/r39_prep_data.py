#!/usr/bin/env python
"""R3.9 data prep: SciFact + NFCorpus -> stats, BM25 top100 candidates,
candidate upper bound, query-level splits, training pairs.

Caches everything to disk so the training/eval scripts can reuse without
re-downloading or re-running BM25.

Outputs:
  data/beir/<ds>/{corpus.jsonl,queries.jsonl,qrels_test.jsonl}   (cached raw)
  data/beir_cand/<ds>_bm25_top100.json   {qid: [doc_ids...]}  (all test qids)
  data/beir_splits/<ds>_r39_split.json    {train:[qids],dev:[],test:[],seed}
  results/linear_rag/r39_beir_candidate_upper_bound.csv
  results/linear_rag/r39_training_pairs_sample.csv
  summaries/linear_rag/r39_beir_data_summary.md
  summaries/linear_rag/r39_beir_candidate_upper_bound_summary.md
"""
from __future__ import annotations
import os, sys, json, time, re, random
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path("/root/autodl-tmp/linear")
sys.path.insert(0, str(REPO / "src"))
import numpy as np

DATASETS = ["scifact", "nfcorpus"]
TOPK = 100
SEED = 0
_TOK = re.compile(r"[A-Za-z0-9]+")


def tok(s):
    return _TOK.findall(s.lower())


def load_beir(name):
    from datasets import load_dataset
    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{name}-qrels", split="test")
    doc_text = {}
    for r in corpus:
        t = (r.get("title") or "").strip(); b = (r.get("text") or "").strip()
        doc_text[str(r["_id"])] = (t + " " + b).strip() if t else b
    qtext = {str(r["_id"]): (r.get("text") or "").strip() for r in queries}
    gold = {}
    for r in qrels:
        if int(r["score"]) <= 0:
            continue
        gold.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    gold = {q: s for q, s in gold.items() if q in qtext}
    return doc_text, qtext, gold


def cache_raw(name, doc_text, qtext, gold):
    d = REPO / "data/beir" / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "corpus.jsonl", "w") as f:
        for did, t in doc_text.items():
            f.write(json.dumps({"doc_id": did, "text": t}) + "\n")
    with open(d / "queries.jsonl", "w") as f:
        for qid, t in qtext.items():
            f.write(json.dumps({"query_id": qid, "text": t}) + "\n")
    with open(d / "qrels_test.jsonl", "w") as f:
        for qid, s in gold.items():
            f.write(json.dumps({"query_id": qid, "gold": sorted(s)}) + "\n")


def bm25_topk(doc_text, qtext, gold, topk):
    from rank_bm25 import BM25Okapi
    doc_ids = list(doc_text.keys())
    print(f"    BM25 tokenizing {len(doc_ids)} docs ...", flush=True)
    bm25 = BM25Okapi([tok(doc_text[d]) for d in doc_ids])
    cand = {}
    qids = list(gold.keys())
    for i, qid in enumerate(qids):
        sc = bm25.get_scores(tok(qtext[qid]))
        idx = np.argsort(-sc)[:topk]
        cand[qid] = [doc_ids[j] for j in idx]
    return cand, qids


def ceil_at(cand, gold, qids, k):
    return sum(len(set(cand[q][:k]) & gold[q]) / len(gold[q]) for q in qids) / len(qids)


def make_split(qids, seed, train=0.70, dev=0.15, min_eval=50):
    rng = random.Random(seed)
    qs = list(qids); rng.shuffle(qs)
    n = len(qs)
    n_dev = max(min_eval, int(round(n * dev)))
    n_test = max(min_eval, int(round(n * dev)))
    # ensure train non-empty
    if n_dev + n_test >= n:
        n_dev = n_test = max(1, (n - max(1, int(n * train))) // 2)
    test = qs[:n_test]
    dev_ = qs[n_test:n_test + n_dev]
    train_ = qs[n_test + n_dev:]
    return {"train": train_, "dev": dev_, "test": test, "seed": seed,
            "sizes": {"train": len(train_), "dev": len(dev_), "test": len(test)}}


def build_pairs(train_qids, qtext, cand, gold, doc_text, neg_per_pos=4, seed=0):
    rng = random.Random(seed)
    rows = []
    for qid in train_qids:
        g = gold[qid]
        cand_list = cand[qid]
        rank_of = {d: i for i, d in enumerate(cand_list)}
        pos = [d for d in g if d in doc_text]
        if not pos:
            continue
        p = rng.choice(pos)  # sample 1 positive
        rows.append({"query_id": qid, "doc_id": p, "label": 1,
                     "bm25_rank": rank_of.get(p, -1)})
        negs = [d for d in cand_list if d not in g]  # hard negs = bm25 top, not relevant
        rng.shuffle(negs)
        for nd in negs[:neg_per_pos]:
            rows.append({"query_id": qid, "doc_id": nd, "label": 0,
                         "bm25_rank": rank_of.get(nd, -1)})
    return rows


def doc_len_stats(doc_text):
    ls = np.array([len(tok(t)) for t in list(doc_text.values())[:8000]])
    return float(ls.mean()), float(np.percentile(ls, 50)), float(np.percentile(ls, 90))


def main():
    import pandas as pd
    (REPO / "data/beir_cand").mkdir(parents=True, exist_ok=True)
    (REPO / "data/beir_splits").mkdir(parents=True, exist_ok=True)
    (REPO / "results/linear_rag").mkdir(parents=True, exist_ok=True)
    (REPO / "summaries/linear_rag").mkdir(parents=True, exist_ok=True)

    data_rows = []
    ceil_rows = []
    pair_sample = []
    for ds in DATASETS:
        print(f"\n===== {ds} =====", flush=True)
        doc_text, qtext, gold = load_beir(ds)
        cache_raw(ds, doc_text, qtext, gold)
        cand, qids = bm25_topk(doc_text, qtext, gold, TOPK)
        json.dump({q: cand[q] for q in qids},
                  open(REPO / f"data/beir_cand/{ds}_bm25_top100.json", "w"))

        pos_per_q = np.array([len(gold[q]) for q in qids])
        dl_mean, dl_p50, dl_p90 = doc_len_stats(doc_text)
        ql = np.array([len(tok(qtext[q])) for q in qids])
        data_rows.append({"dataset": ds, "corpus_size": len(doc_text),
                          "query_count": len(qids), "qrels_count": int(pos_per_q.sum()),
                          "pos_per_query_mean": float(pos_per_q.mean()),
                          "pos_per_query_median": float(np.median(pos_per_q)),
                          "pos_per_query_max": int(pos_per_q.max()),
                          "avg_doc_len_tok": round(dl_mean, 1),
                          "doc_len_p90": round(dl_p90, 1),
                          "avg_query_len_tok": round(float(ql.mean()), 1)})

        for k in (10, 50, 100):
            ceil_rows.append({"dataset": ds, "k": k,
                              "candidate_recall@k": round(ceil_at(cand, gold, qids, k), 4)})

        split = make_split(qids, SEED)
        json.dump(split, open(REPO / f"data/beir_splits/{ds}_r39_split.json", "w"), indent=2)
        print(f"    split sizes: {split['sizes']}", flush=True)

        pairs = build_pairs(split["train"], qtext, cand, gold, doc_text, seed=SEED)
        # sample for csv (first 40 rows w/ text)
        for r in pairs[:40]:
            pair_sample.append({"dataset": ds, **r,
                                "query_text": qtext[r["query_id"]][:120],
                                "doc_text": doc_text[r["doc_id"]][:160]})
        # save full pairs for training reuse
        json.dump(pairs, open(REPO / f"data/beir_splits/{ds}_r39_train_pairs.json", "w"))
        print(f"    train pairs: {len(pairs)} ({sum(p['label'] for p in pairs)} pos)", flush=True)

    df_data = pd.DataFrame(data_rows)
    df_ceil = pd.DataFrame(ceil_rows)
    df_ceil.to_csv(REPO / "results/linear_rag/r39_beir_candidate_upper_bound.csv", index=False)
    pd.DataFrame(pair_sample).to_csv(REPO / "results/linear_rag/r39_training_pairs_sample.csv", index=False)

    # data summary md
    dm = ["# R3.9 BEIR Data Summary\n",
          "Datasets prepared for real-data finetune validation (SciFact primary, NFCorpus secondary).\n",
          df_data.to_markdown(index=False), "\n"]
    (REPO / "summaries/linear_rag/r39_beir_data_summary.md").write_text("\n".join(dm))

    # candidate upper bound summary md
    cm = ["# R3.9 BEIR Candidate Upper Bound (BM25 top-100)\n",
          "Reranker Recall@k is capped by candidate recall. Critical caveat below.\n",
          df_ceil.to_markdown(index=False), "\n",
          "## Caveat\n",
          "- **NFCorpus candidate Recall@100 is very low** (many relevant docs per query, "
          "single-stage BM25 cannot surface them). Any reranker's Recall@k on NFCorpus is "
          "structurally capped — NFCorpus is therefore treated as a SECONDARY/reference dataset; "
          "the R3.9 gate decision is anchored on **SciFact**, whose candidate Recall@100 is high.\n",
          "- SciFact: single-gold-ish, high candidate ceiling => clean signal for whether a "
          "reranker can learn real relevance.\n"]
    (REPO / "summaries/linear_rag/r39_beir_candidate_upper_bound_summary.md").write_text("\n".join(cm))

    print("\n=== DATA STATS ===")
    print(df_data.to_string(index=False))
    print("\n=== CANDIDATE CEILING ===")
    print(df_ceil.to_string(index=False))


if __name__ == "__main__":
    main()
