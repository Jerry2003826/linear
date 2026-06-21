# Linear-RAG: Experiment Protocol

## 0. Principles

- **Fully automatic, but gated.** The orchestrator advances stage-to-stage
  without per-stage user confirmation, but obeys hard gates and a GPU-hour
  budget.
- **Automation is not blind.** Gate outcomes drive control flow:
  - `PASS` → advance to next stage.
  - `PARTIAL` → continue with cheaper diagnostics / alternative baseline.
  - `FAIL` → stop downstream high-cost stages; write summary + recommended next
    step.
  - `budget exceeded` → pause immediately; write `budget_pause.md`.
  - `data / metric / alignment error` → stop immediately; do **not** keep running
    models.
- **No overclaiming.** Reports describe an efficiency–accuracy frontier, never
  architectural superiority.

## 1. Auto-advance rules

```
R0 PASS                      -> run R1
R1 PASS or PARTIAL           -> run R2 (small sample first)
R2 done (any non-FAIL)       -> run R2.5 cross-encoder baseline
R2.5 done                    -> evaluate R3 entry conditions
R3 conditions met            -> run R3 (small-scale)
R3 conditions NOT met        -> skip R3, write final report + next steps
any FAIL on R0/R1            -> stop, write summary
data/metric/alignment error  -> stop immediately
```

## 2. Budget (hard gate)

Defaults (override via env `MAX_GPU_HOURS_TOTAL`):

| Stage | Budget |
|-------|--------|
| R0 data gen | CPU, < 0.5 h |
| R1 BM25+embedding | < 1 GPU-hour |
| R2 zero-shot | < 2 GPU-hours |
| R2.5 cross-encoder | < 1 GPU-hour |
| R3 LoRA | < 6 GPU-hours |
| **TOTAL** | **MAX_GPU_HOURS_TOTAL = 10** |

Cost discipline per GPU stage:

1. Run a **dry-run benchmark** on 100–200 query/candidate pairs first.
2. Estimate full-stage time from the dry-run.
3. If estimate > stage budget → **shrink sample size** automatically (do not run
   full).
4. If still over budget after shrinking → stop, write `budget_pause.md`.
5. If actual time exceeds estimate by > 1.5x → pause downstream, write
   `cost_overrun_summary.md`.

All long jobs run under `nohup` with a saved `pid`, log file, and `status.json`.

Cost outputs:
- `summaries/linear_rag/cost_plan.md`
- `summaries/linear_rag/cost_overrun_summary.md` (if triggered)
- `summaries/linear_rag/budget_pause.md` (if triggered)
- `results/linear_rag/runtime_profile.csv`

GPU price: read from env `GPU_PRICE_PER_HOUR` (default documented in cost_plan).
Cost = elapsed_gpu_hours * price.

## 3. Stage protocols

### R0 — Synthetic RAG benchmark
Generate docs(10000)/queries(5000)/qrels/hard_negatives deterministically
(seed=42). Run `test_synth_rag_data.py`. Gate: counts correct, gold-unique
> 99.9%, all tests pass.

### R1 — BM25 + embedding retrieval baseline
BM25 via `rank-bm25`; embeddings via `BAAI/bge-small-en-v1.5` + FAISS. Retrieve
top-k in {1,5,10,50,100,500}. Record Recall@k, MRR, NDCG@10, latency/query,
throughput, index build time, memory. Write top100 & top500 candidate parquet
files. Run metric + alignment tests. Gate: both run (PASS) or BM25-only
(PARTIAL); embedding Recall@5 far below BM25 must be explained.

### R2 — Zero-shot LM scanner/reranker
Candidates from embedding top100 (else BM25 top100). Prompt format A (pairwise
yes/no): `score = logprob(" yes") - logprob(" no")` using full token-sequence
logprob if multi-token. No generation, no training. Models batch 1:
`state-spaces/mamba-130m-hf`, `EleutherAI/pythia-160m`; batch 2 (budget
permitting): `mamba-370m-hf`, `pythia-410m`. Start 500 queries x top100; auto
expand to 2000 then 5000 if a single model's full run is estimated < 1.5
GPU-hour and signal is stable. Never auto-run > 1B models. Record reranked
Recall@1/5/10, MRR, NDCG@10, latency, tokens/sec, peak VRAM, cost/1000 queries.
Gate: PASS_SIGNAL / WEAK_SIGNAL / NO_SIGNAL / FAIL.

### R2.5 — Cross-encoder reranker baseline
`cross-encoder/ms-marco-MiniLM-L-6-v2` over R1 embedding top100 (else BM25).
Start 500 queries; expand toward R2's query count if cheap; do not exceed 1
GPU-hour unless dry-run is very low. Gate: PASS / PARTIAL / FAIL.

### R3 — Mamba LoRA reranker (conditional)
Enter **only if all**: no data/metric bug in R0–R2.5; R2 Mamba >= WEAK_SIGNAL;
cross-encoder clearly more accurate but slower / higher-VRAM; R3 dry-run total
<= 6 GPU-hours; training data constructible from R1 top100. Model
`mamba-130m-hf` (optionally 370m). LoRA r=16/alpha=32/dropout=0.05; max_len=512,
bf16 if stable, batch=8, grad_acc=4, lr=2e-4, steps=3000, eval/save=200,
seeds=[0,1,2] (fall back to seed 0 if 3-seed estimate > 6 GPU-hours). Gate:
PASS / PARTIAL / FAIL.

## 4. Failure-stop conditions

Stop immediately (no further model runs) on:
- gold not unique / qrels misaligned / determinism failure (R0);
- candidate query_id/doc_id misalignment, metric miscomputation, untrustworthy
  gold-in-candidate stats (R1);
- scoring-logic error, incomplete outputs, non-reproducible metrics (R2/R2.5);
- training instability or cost blow-up (R3).

## 5. Writing style for conclusions

Allowed: "Mamba / linear models show promise as internal scanner/reranker
modules inside RAG **if** they improve reranking or offer a better
latency–accuracy tradeoff."

Forbidden: "Mamba replaces RAG." / "Linear models are inherently better than
Transformers." / "RAG is dead." / "Architecture-level superiority is proven."

## 6. Determinism & logging

All stochastic steps seeded. Each long job writes `status.json`, a `.pid` file,
its resolved `config.yaml`, and `git_sha.txt`. Logs in `logs/linear_rag/*.log`.
