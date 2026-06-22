#!/usr/bin/env python
"""R3.9 BEIR zero-shot validation.

Validates R3.8 rerankers (Mamba-cls, Pythia-cls) + cross-encoder on REAL BEIR
retrieval datasets (SciFact, NFCorpus, FiQA), WITHOUT installing beir/pyserini.

Pipeline (per dataset):
  1. Load corpus / queries / qrels(test) via `datasets` from HF mirror.
  2. BM25 (rank_bm25, CPU) retrieves top-K candidates per test query.
  3. Rerank candidates with: Mamba-cls (seed0 ckpt), Pythia-cls (seed0 ckpt),
     cross-encoder (cross-encoder/ms-marco-MiniLM-L-6).
  4. Multi-gold metrics: Recall@{1,5,10}, MRR@10, NDCG@10 (handles >=1 relevant
     docs per query, unlike the single-gold aggregate_metrics in src).
  5. Latency/query + peak eval VRAM per model.

HONEST-FRAMING NOTES (echoed into output meta):
  - cross-encoder was trained on MS MARCO (real IR data) => unfair zero-shot
    advantage on BEIR. This measures "how far our synthetic-trained pipeline is
    from a real-usable reranker", NOT a fair architecture PK.
  - BM25 candidate recall@K caps every reranker's achievable Recall@k. We report
    BM25 candidate recall as the ceiling.

Reuses: ClsReranker / build_cls_prompt from scripts/linear_rag/r38_cls_train.py.
"""
from __future__ import annotations
import os, sys, json, time, argparse, math, re
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path("/root/autodl-tmp/linear")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "linear_rag"))

import numpy as np
import pandas as pd

# Reuse R3.8 reranker + prompt
from r38_cls_train import ClsReranker, build_cls_prompt  # noqa: E402

DATASETS = ["scifact", "nfcorpus", "fiqa"]
DATASET_CKPT_TOPK = 100

# ckpt dirs (seed0 best)
MAMBA_CKPT = REPO / "checkpoints/linear_rag/r38_mamba_cls/seed0_best"
PYTHIA_CKPT = REPO / "checkpoints/linear_rag/r38_pythia_cls/seed0_best"
MAMBA_MODEL = "state-spaces/mamba-130m-hf"
PYTHIA_MODEL = "EleutherAI/pythia-160m"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # trained on MS MARCO real data
LORA_CFG_MAMBA = {"r": 16, "alpha": 32, "dropout": 0.05}
LORA_CFG_PYTHIA = {"r": 16, "alpha": 32, "dropout": 0.05,
                   "target_modules": ["query_key_value", "dense"]}

_TOK = re.compile(r"[A-Za-z0-9]+")


def tok(s):
    return _TOK.findall(s.lower())


# ---- BEIR-safe encoding: keep the QUERY, truncate the DOCUMENT ----
# R3.8 trained on short synthetic docs, so naive right-truncation ids[-max_len:]
# was fine. BEIR docs are long (p90~444, max~955 tokens) => right-truncation
# can cut the entire "Query:" prefix, leaving the model blind to the query.
# We instead truncate ONLY the document so the query + "Relevance:" survive.
CLS_TEMPLATE = "Query:\n{q}\n\nDocument:\n{d}\n\nRelevance:"


def encode_batch_beir(rr, items, max_len, query_budget_frac=0.4):
    """Like ClsReranker.encode_batch but doc-truncating so query is preserved.

    Builds prompt as: Query:\n{q}\n\nDocument:\n{d}\n\nRelevance:
    Token budget for the document is whatever remains after the fixed template
    tokens + (capped) query tokens. Query capped at query_budget_frac*max_len.
    """
    torch = rr.torch
    t = rr.tok
    # static template piece token counts (encode the literal scaffolding once)
    pre_q = t.encode("Query:\n", add_special_tokens=False)
    mid = t.encode("\n\nDocument:\n", add_special_tokens=False)
    post = t.encode("\n\nRelevance:", add_special_tokens=False)
    overhead = len(pre_q) + len(mid) + len(post)
    q_cap = max(8, int(max_len * query_budget_frac))
    seqs = []
    for q, d in items:
        q_ids = t.encode(q, add_special_tokens=False)[:q_cap]
        doc_budget = max_len - overhead - len(q_ids)
        d_ids = t.encode(d, add_special_tokens=False)
        if doc_budget < 1:
            d_ids = []
        elif len(d_ids) > doc_budget:
            d_ids = d_ids[:doc_budget]  # head of document
        ids = pre_q + q_ids + mid + d_ids + post
        seqs.append(ids)
    L = max(len(s) for s in seqs)
    pad = t.pad_token_id
    batch = torch.full((len(seqs), L), pad, dtype=torch.long)
    last = []
    for r, s in enumerate(seqs):
        batch[r, :len(s)] = torch.tensor(s, dtype=torch.long)
        last.append(len(s) - 1)
    return batch.to(rr.device), torch.tensor(last, device=rr.device)


# ---------------- multi-gold metrics ----------------
def recall_at_k(ranked, gold_set, k):
    """frac of relevant docs retrieved in top-k (standard BEIR recall)."""
    if not gold_set:
        return 0.0
    topk = set(ranked[:k])
    return len(topk & gold_set) / len(gold_set)


def mrr_at_k(ranked, gold_set, k=10):
    for i, d in enumerate(ranked[:k]):
        if d in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, gold_set, k=10):
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in gold_set:
            dcg += 1.0 / math.log2(i + 2)
    # IDCG: all relevant at top (binary rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_multi(rankings, gold, ks=(1, 5, 10), ndcg_k=10, mrr_k=10):
    qids = list(rankings.keys())
    n = len(qids)
    out = {}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(rankings[q], gold[q], k) for q in qids) / n
    out[f"mrr@{mrr_k}"] = sum(mrr_at_k(rankings[q], gold[q], mrr_k) for q in qids) / n
    out[f"ndcg@{ndcg_k}"] = sum(ndcg_at_k(rankings[q], gold[q], ndcg_k) for q in qids) / n
    out["n_queries"] = float(n)
    return out


# ---------------- data loading ----------------
def load_beir(name):
    from datasets import load_dataset
    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{name}-qrels", split="test")

    doc_text = {}
    for r in corpus:
        t = (r.get("title") or "").strip()
        b = (r.get("text") or "").strip()
        doc_text[str(r["_id"])] = (t + " " + b).strip() if t else b
    qtext = {str(r["_id"]): (r.get("text") or "").strip() for r in queries}

    gold = {}
    for r in qrels:
        if int(r["score"]) <= 0:
            continue
        qid = str(r["query-id"]); did = str(r["corpus-id"])
        gold.setdefault(qid, set()).add(did)
    # only queries with >=1 positive qrel AND query text present
    gold = {q: s for q, s in gold.items() if q in qtext}
    return doc_text, qtext, gold


# ---------------- BM25 candidate generation ----------------
def bm25_candidates(doc_text, qtext, gold, topk, max_queries=None):
    from rank_bm25 import BM25Okapi
    doc_ids = list(doc_text.keys())
    print(f"    tokenizing {len(doc_ids)} docs for BM25 ...", flush=True)
    corpus_tok = [tok(doc_text[d]) for d in doc_ids]
    bm25 = BM25Okapi(corpus_tok)
    qids = list(gold.keys())
    if max_queries:
        qids = qids[:max_queries]
    cand = {}
    t0 = time.time()
    for i, qid in enumerate(qids):
        scores = bm25.get_scores(tok(qtext[qid]))
        idx = np.argsort(-scores)[:topk]
        cand[qid] = [doc_ids[j] for j in idx]
        if (i + 1) % 100 == 0:
            print(f"    bm25 {i+1}/{len(qids)} ({(time.time()-t0):.0f}s)", flush=True)
    return cand, qids


# ---------------- rerankers ----------------
def make_cls_reranker(model_name, ckpt, lora_cfg, device, dtype):
    rr = ClsReranker(model_name, lora_cfg, device, dtype)
    rr.load_head(ckpt)
    rr.backbone.load_adapter(str(ckpt), adapter_name="default")
    rr.backbone.set_adapter("default")
    rr.eval()
    return rr


def rerank_cls(rr, qids, qtext, cand, doc_text, max_len=512, eval_bs=16):
    import torch
    from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb
    reset_peak_memory()
    rankings = {}
    t0 = time.time(); n_cand = 0
    with torch.no_grad():
        for qid in qids:
            qt = qtext[qid]
            cids = cand[qid]
            scores = []
            for bs in range(0, len(cids), eval_bs):
                chunk = cids[bs:bs + eval_bs]
                items = [(qt, doc_text[c]) for c in chunk]
                ids, last = encode_batch_beir(rr, items, max_len)
                logits = rr.forward_scores(ids, last)
                s = (logits[:, 1] - logits[:, 0]).detach().cpu().tolist()
                scores.extend(s)
            n_cand += len(cids)
            order = np.argsort(-np.array(scores))
            rankings[qid] = [cids[i] for i in order]
    dt = time.time() - t0
    vram = peak_vram_mb()
    return rankings, dt / max(1, len(qids)), dt / max(1, n_cand), vram


def rerank_ce(qids, qtext, cand, doc_text, device, eval_bs=64):
    import torch
    from sentence_transformers import CrossEncoder
    from src.linear_rag.utils.gpu import reset_peak_memory, peak_vram_mb
    ce = CrossEncoder(CE_MODEL, max_length=512, device=device)
    reset_peak_memory()
    rankings = {}
    t0 = time.time(); n_cand = 0
    for qid in qids:
        qt = qtext[qid]
        cids = cand[qid]
        pairs = [[qt, doc_text[c]] for c in cids]
        scores = ce.predict(pairs, batch_size=eval_bs, show_progress_bar=False)
        n_cand += len(cids)
        order = np.argsort(-np.array(scores))
        rankings[qid] = [cids[i] for i in order]
    dt = time.time() - t0
    vram = peak_vram_mb()
    return rankings, dt / max(1, len(qids)), dt / max(1, n_cand), vram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--topk", type=int, default=DATASET_CKPT_TOPK)
    ap.add_argument("--max_queries", type=int, default=0, help="0=all")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--models", default="bm25,mamba,pythia,ce")
    ap.add_argument("--out", default="results/linear_rag/r39_beir_metrics.csv")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    datasets = args.datasets.split(",")
    models = args.models.split(",")
    maxq = args.max_queries or None
    if args.dry_run:
        maxq = min(maxq or 20, 20)

    rows = []
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load cls rerankers once (reused across datasets)
    rr_mamba = rr_pythia = None
    if "mamba" in models:
        print("[load] Mamba-cls ...", flush=True)
        rr_mamba = make_cls_reranker(MAMBA_MODEL, MAMBA_CKPT, LORA_CFG_MAMBA, device, dtype)
    if "pythia" in models:
        print("[load] Pythia-cls ...", flush=True)
        rr_pythia = make_cls_reranker(PYTHIA_MODEL, PYTHIA_CKPT, LORA_CFG_PYTHIA, device, dtype)

    for ds in datasets:
        print(f"\n========== {ds} ==========", flush=True)
        doc_text, qtext, gold = load_beir(ds)
        print(f"  corpus={len(doc_text)} queries(test,with-qrel)={len(gold)}", flush=True)
        cand, qids = bm25_candidates(doc_text, qtext, gold, args.topk, maxq)
        gold_eval = {q: gold[q] for q in qids}

        # BM25 ceiling (candidate recall = recall of the candidate SET, order=bm25)
        if "bm25" in models:
            m = aggregate_multi(cand, gold_eval)
            # candidate-set recall ceiling at topk
            ceil = sum(recall_at_k(cand[q], gold_eval[q], args.topk) for q in qids) / len(qids)
            rows.append({"dataset": ds, "model": "BM25", "scoring": "bm25",
                         "recall@1": m["recall@1"], "recall@5": m["recall@5"],
                         "recall@10": m["recall@10"], "mrr@10": m["mrr@10"],
                         "ndcg@10": m["ndcg@10"], "cand_recall_ceiling@topk": ceil,
                         "latency_per_q_ms": None, "vram_mb": None,
                         "n_queries": len(qids), "topk": args.topk})
            print(f"  [BM25] R@5={m['recall@5']:.4f} nDCG@10={m['ndcg@10']:.4f} "
                  f"ceiling@{args.topk}={ceil:.4f}", flush=True)

        ceil = sum(recall_at_k(cand[q], gold_eval[q], args.topk) for q in qids) / len(qids)

        for mkey, rr in [("mamba", rr_mamba), ("pythia", rr_pythia)]:
            if mkey not in models or rr is None:
                continue
            rk, lat_q, lat_c, vram = rerank_cls(rr, qids, qtext, cand, doc_text)
            m = aggregate_multi(rk, gold_eval)
            tag = "Mamba-130m-cls" if mkey == "mamba" else "Pythia-160m-cls"
            rows.append({"dataset": ds, "model": tag, "scoring": "cls_head",
                         "recall@1": m["recall@1"], "recall@5": m["recall@5"],
                         "recall@10": m["recall@10"], "mrr@10": m["mrr@10"],
                         "ndcg@10": m["ndcg@10"], "cand_recall_ceiling@topk": ceil,
                         "latency_per_q_ms": lat_q * 1000, "vram_mb": vram,
                         "n_queries": len(qids), "topk": args.topk})
            print(f"  [{tag}] R@5={m['recall@5']:.4f} nDCG@10={m['ndcg@10']:.4f} "
                  f"{lat_q*1000:.0f}ms/q VRAM={vram:.0f}MB", flush=True)

        if "ce" in models:
            rk, lat_q, lat_c, vram = rerank_ce(qids, qtext, cand, doc_text, device)
            m = aggregate_multi(rk, gold_eval)
            rows.append({"dataset": ds, "model": "CrossEncoder-MiniLM-L6",
                         "scoring": "cross_encoder",
                         "recall@1": m["recall@1"], "recall@5": m["recall@5"],
                         "recall@10": m["recall@10"], "mrr@10": m["mrr@10"],
                         "ndcg@10": m["ndcg@10"], "cand_recall_ceiling@topk": ceil,
                         "latency_per_q_ms": lat_q * 1000, "vram_mb": vram,
                         "n_queries": len(qids), "topk": args.topk})
            print(f"  [CE] R@5={m['recall@5']:.4f} nDCG@10={m['ndcg@10']:.4f} "
                  f"{lat_q*1000:.0f}ms/q VRAM={vram:.0f}MB", flush=True)

        # incremental save
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    meta = {
        "datasets": datasets, "models": models, "topk": args.topk,
        "max_queries": maxq, "dry_run": args.dry_run,
        "ce_model": CE_MODEL,
        "honest_notes": [
            "cross-encoder was trained on MS MARCO real IR data => unfair zero-shot "
            "advantage on BEIR; this measures distance of our synthetic pipeline from "
            "real-usable, NOT a fair architecture PK.",
            "BM25 candidate recall@topk caps every reranker's achievable recall; "
            "reported as cand_recall_ceiling@topk.",
            "Mamba/Pythia cls heads were trained ONLY on synthetic data (synth_rag_v1); "
            "BEIR is fully zero-shot / out-of-distribution for them.",
        ],
    }
    meta_path = out_path.with_name("r39_beir_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print("\n=== SAVED ===", out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
