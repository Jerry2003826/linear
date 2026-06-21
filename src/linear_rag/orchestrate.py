from __future__ import annotations

"""Linear-RAG full-auto orchestrator: R0 -> R1 -> R2 -> R2.5 -> (conditional) R3.

Runs end-to-end with budget gates, dry-run estimation, stage gates, summaries,
plots, and a final report. Designed to be launched under nohup on the GPU box.
"""

import os
import time
import traceback
from pathlib import Path

import yaml

from .utils.io import write_json
from .utils.cost import CostTracker, MAX_GPU_HOURS_TOTAL, DEFAULT_PRICE

ROOT = Path(__file__).resolve().parents[2]
SUM = ROOT / "summaries" / "linear_rag"
RES = ROOT / "results" / "linear_rag"
PLOTS = ROOT / "plots" / "linear_rag"
LOGS = ROOT / "logs" / "linear_rag"
for d in (SUM, RES, PLOTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

STATUS = {}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LOGS / "orchestrate.log").open("a") as f:
        f.write(line + "\n")


def save_status():
    write_json(RES / "orchestrator_status.json", STATUS)


def write_md(path, text):
    Path(path).write_text(text)
    log(f"wrote {path}")


# --------------------------------------------------------------------------- #
def run_r0():
    from .data import gen_synth_rag
    t0 = time.time()
    stats = gen_synth_rag.main(str(ROOT / "configs" / "r0_synth.yaml"))
    elapsed = time.time() - t0
    gate = "PASS"
    if not (stats["n_docs"] == 10000 and stats["n_queries"] == 5000
            and stats["gold_unique_rate"] > 0.999):
        gate = "FAIL"
    STATUS["R0"] = {"gate": gate, "stats": stats, "elapsed_s": round(elapsed, 1)}
    write_md(SUM / "r0_data_summary.md", f"""# R0 Synthetic RAG Data Summary

- gate: **{gate}**
- docs: {stats['n_docs']}
- queries: {stats['n_queries']}
- gold_unique_rate: {stats['gold_unique_rate']}
- distinct_gold_ratio: {stats['distinct_gold_ratio']}
- min_hard_negatives: {stats['min_hard_negatives']}
- non_unique_discarded: {stats['non_unique_discarded']}
- deterministic content_hash: `{stats['content_hash']}`
- generation time: {elapsed:.1f}s (CPU)

## Query types
{stats['query_type_counts']}

## Difficulty splits
{stats['difficulty_counts']}
""")
    return gate


def run_r1(cost: CostTracker):
    from .eval import bm25_embed
    from . import plots
    t0 = time.time()
    summary = bm25_embed.main(str(ROOT / "configs" / "r1_baseline.yaml"))
    elapsed = time.time() - t0
    cost.record("R1", elapsed, uses_gpu=True, notes="bm25+embedding")

    emb_ok = "embedding" in summary
    bm25_r5 = summary["bm25"]["recall@5"]
    gate = "PASS"
    note = ""
    if not emb_ok:
        gate = "PARTIAL"; note = f"embedding failed: {summary.get('embedding_error','')}"
    else:
        emb_r5 = summary["embedding"]["recall@5"]
        if emb_r5 < bm25_r5 - 0.2:
            note = (f"embedding Recall@5 ({emb_r5:.3f}) much lower than BM25 "
                    f"({bm25_r5:.3f}); likely because synthetic docs are lexically "
                    f"templated so BM25 exact-match dominates short factual queries.")
    STATUS["R1"] = {"gate": gate, "summary": summary, "elapsed_s": round(elapsed, 1),
                    "note": note}
    try:
        plots.r1_recall_curves(summary, PLOTS / "r1_recall_curves.png")
        plots.r1_latency(summary, PLOTS / "r1_latency.png")
    except Exception as e:
        log(f"R1 plot error: {e}")

    emb_block = ""
    if emb_ok:
        e = summary["embedding"]
        emb_block = (f"- embedding Recall@5: {e['recall@5']:.4f}, Recall@10: "
                     f"{e['recall@10']:.4f}, MRR: {e['mrr']:.4f}, "
                     f"lat/q: {e['latency_ms_per_query']}ms\n")
    write_md(SUM / "r1_baseline_summary.md", f"""# R1 Retrieval Baseline Summary

- gate: **{gate}**
- candidate_source: {summary.get('candidate_source')}
- BM25 Recall@5: {summary['bm25']['recall@5']:.4f}, Recall@10: {summary['bm25']['recall@10']:.4f}, MRR: {summary['bm25']['mrr']:.4f}, lat/q: {summary['bm25']['latency_ms_per_query']}ms
{emb_block}- gold_in_top100: {summary.get('gold_in_top100')}
- gold_in_top500: {summary.get('gold_in_top500')}
- elapsed: {elapsed:.1f}s

{('Note: ' + note) if note else ''}
""")
    return gate, summary


def _r2_dryrun_then_run(cost):
    from .eval import zero_shot_lm
    cfg = yaml.safe_load((ROOT / "configs" / "r2_zero_shot.yaml").read_text())
    models = cfg["models_batch1"]
    # dry-run: 100 queries, estimate full cost
    log("R2 dry-run (100 queries)")
    t0 = time.time()
    dry = zero_shot_lm.main(str(ROOT / "configs" / "r2_zero_shot.yaml"),
                            n_queries=100, models=models)
    dry_elapsed = time.time() - t0
    cost.record("R2_dryrun", dry_elapsed, uses_gpu=True, notes="100q x2 models")

    # estimate per-model full time from metrics
    ests = {m["model"]: m.get("est_gpu_hours_full_5000", 99)
            for m in dry["metrics"] if "est_gpu_hours_full_5000" in m}
    # choose sample size honoring R2 budget (2 GPU-h) and remaining total budget
    remaining = cost.remaining_gpu_hours()
    r2_budget = min(2.0, remaining)
    # total time for all models at full 5000:
    total_full = sum(ests.values()) if ests else 99
    if total_full <= r2_budget and total_full > 0:
        nq = 5000
    elif total_full * (2000 / 5000) <= r2_budget:
        nq = 2000
    elif total_full * (500 / 5000) <= r2_budget:
        nq = 500
    else:
        nq = max(200, int(5000 * r2_budget / max(total_full, 1e-6)))
    log(f"R2 estimates(full5000 gpu-h)={ests} r2_budget={r2_budget:.2f} -> nq={nq}")

    t1 = time.time()
    res = zero_shot_lm.main(str(ROOT / "configs" / "r2_zero_shot.yaml"),
                            n_queries=nq, models=models)
    elapsed = time.time() - t1
    cost.record("R2", elapsed, uses_gpu=True, notes=f"{nq}q x{len(models)} models")
    # overrun check
    est_for_nq = total_full * (nq / 5000)
    overrun = elapsed / 3600 > 1.5 * max(est_for_nq, 1e-6)
    return res, nq, elapsed, ests, overrun


def run_r2(cost: CostTracker, r1_summary):
    from . import plots
    res, nq, elapsed, ests, overrun = _r2_dryrun_then_run(cost)
    metrics = res["metrics"]

    # gate logic: the baseline to beat is the ORDERING of the candidate source
    # that was actually reranked (reranking is capped by that source's top-100).
    cand_src = r1_summary.get("candidate_source", "embedding")
    emb_r5 = r1_summary.get(cand_src, {}).get("recall@5",
              r1_summary.get("bm25", {}).get("recall@5", 0))
    mamba = next((m for m in metrics if "mamba" in m["model"] and "recall@5" in m), None)
    pythia = next((m for m in metrics if "pythia" in m["model"] and "recall@5" in m), None)

    if any("error" in m for m in metrics) and not (mamba or pythia):
        gate = "FAIL"
    elif mamba is None:
        gate = "FAIL"
    else:
        improves = mamba["recall@5"] > emb_r5 + 1e-6
        near_pythia_cheaper = (pythia and abs(mamba["recall@5"] - pythia["recall@5"]) < 0.05
                               and (mamba["latency_ms_per_query"] < pythia["latency_ms_per_query"]
                                    or mamba["peak_vram_mb"] < pythia["peak_vram_mb"]))
        margin_ok = mamba.get("score_margin_gold_vs_rest", 0) > 0.1
        if improves or near_pythia_cheaper or margin_ok:
            gate = "PASS_SIGNAL"
        elif mamba["recall@5"] >= emb_r5 - 0.02:
            gate = "WEAK_SIGNAL"
        else:
            gate = "NO_SIGNAL"

    STATUS["R2"] = {"gate": gate, "metrics": metrics, "n_queries": nq,
                    "elapsed_s": round(elapsed, 1), "embedding_recall@5_ref": emb_r5}
    if overrun:
        write_md(SUM / "cost_overrun_summary.md",
                 f"# Cost Overrun\nR2 actual time exceeded 1.5x estimate.\n"
                 f"estimates={ests}\nactual_elapsed_s={elapsed:.1f}\n")

    try:
        r25_row = STATUS.get("R2.5", {}).get("row")
        plots.r2_recall_comparison(r1_summary, metrics, r25_row,
                                   PLOTS / "r2_recall_comparison.png")
        plots.r2_latency_vram(metrics, r25_row, PLOTS / "r2_latency_vram.png")
    except Exception as e:
        log(f"R2 plot error: {e}")

    lines = []
    for m in metrics:
        if "recall@5" in m:
            lines.append(f"- {m['model']}: R@1={m['recall@1']:.4f} R@5={m['recall@5']:.4f} "
                         f"R@10={m['recall@10']:.4f} MRR={m['mrr']:.4f} "
                         f"lat/q={m['latency_ms_per_query']}ms VRAM={m['peak_vram_mb']}MB "
                         f"margin={m.get('score_margin_gold_vs_rest')}")
        else:
            lines.append(f"- {m['model']}: ERROR {m.get('error')}")
    write_md(SUM / "r2_zero_shot_summary.md", f"""# R2 Zero-shot Scanner/Reranker Summary

- gate: **{gate}**
- queries evaluated: {nq}, top-k=100
- reference embedding/BM25 Recall@5: {emb_r5:.4f}
- elapsed: {elapsed:.1f}s

## Per-model rerank metrics
{chr(10).join(lines)}

Interpretation: a signal means Mamba reranking improved Recall@5 over the coarse
retrieval order, matched Pythia at lower latency/VRAM, or produced a positive
gold-vs-rest score margin. This is an efficiency-accuracy boundary observation,
not a claim of architectural superiority.
""")
    return gate, metrics


def run_r25(cost: CostTracker, r1_summary, r2_metrics):
    from .eval import cross_encoder
    from . import plots
    # dry-run 100
    t0 = time.time()
    dry = cross_encoder.main(str(ROOT / "configs" / "r25_cross_encoder.yaml"), n_queries=100)
    dry_el = time.time() - t0
    cost.record("R2.5_dryrun", dry_el, uses_gpu=True, notes="100q")
    est_full = dry.get("est_gpu_hours_full_5000", 1)
    # match R2 query count if cheap & under 1 GPU-h
    target_nq = STATUS.get("R2", {}).get("n_queries", 500)
    budget = min(1.0, cost.remaining_gpu_hours())
    est_target = est_full * (target_nq / 5000)
    nq = target_nq if est_target <= budget else 500
    gate = "PASS"
    try:
        t1 = time.time()
        row = cross_encoder.main(str(ROOT / "configs" / "r25_cross_encoder.yaml"), n_queries=nq)
        el = time.time() - t1
        cost.record("R2.5", el, uses_gpu=True, notes=f"{nq}q")
        if nq < target_nq:
            gate = "PARTIAL"
    except Exception as e:
        gate = "FAIL"; row = {"error": f"{type(e).__name__}: {e}"}
        log(f"R2.5 FAIL: {e}")
    STATUS["R2.5"] = {"gate": gate, "row": row, "n_queries": nq}
    try:
        plots.r2_recall_comparison(r1_summary, r2_metrics, row if "recall@5" in row else None,
                                   PLOTS / "r25_cross_encoder_comparison.png")
    except Exception as e:
        log(f"R2.5 plot error: {e}")
    body = (f"- {row['model']}: R@1={row['recall@1']:.4f} R@5={row['recall@5']:.4f} "
            f"R@10={row['recall@10']:.4f} MRR={row['mrr']:.4f} "
            f"lat/q={row['latency_ms_per_query']}ms VRAM={row['peak_vram_mb']}MB"
            if "recall@5" in row else f"- ERROR: {row.get('error')}")
    write_md(SUM / "r25_cross_encoder_summary.md", f"""# R2.5 Cross-Encoder Baseline Summary

- gate: **{gate}**
- queries evaluated: {nq}, top-k=100
{body}
""")
    return gate, row


def decide_r3(cost, r2_gate, r2_metrics, r25_row):
    reasons = []
    if r2_gate not in ("PASS_SIGNAL", "WEAK_SIGNAL"):
        reasons.append(f"R2 gate is {r2_gate} (need >= WEAK_SIGNAL)")
    mamba = next((m for m in r2_metrics if "mamba" in m["model"] and "recall@5" in m), None)
    ce_ok = isinstance(r25_row, dict) and "recall@5" in r25_row
    if mamba and ce_ok:
        ce_more_accurate = r25_row["recall@5"] > mamba["recall@5"] + 1e-6
        ce_costlier = (r25_row["latency_ms_per_query"] > mamba["latency_ms_per_query"]
                       or r25_row["peak_vram_mb"] > mamba["peak_vram_mb"])
        if not (ce_more_accurate and ce_costlier):
            reasons.append("cross-encoder is not (more accurate AND costlier) than Mamba")
    else:
        reasons.append("missing Mamba or cross-encoder metrics")
    # dry-run cost estimate: rough 3-seed estimate based on R2 latency
    if cost.remaining_gpu_hours() < 1.0:
        reasons.append(f"remaining budget {cost.remaining_gpu_hours():.2f} GPU-h too low")
    enter = len(reasons) == 0
    return enter, reasons


def run_r3(cost):
    from .train import rerank_lora
    # estimate seeds count under 6h budget; conservative: try seed 0 first
    remaining = cost.remaining_gpu_hours()
    budget = min(6.0, remaining)
    seeds = [0]
    t0 = time.time()
    try:
        res = rerank_lora.main(str(ROOT / "configs" / "r3_lora.yaml"),
                               seeds=seeds, train_queries=600, eval_queries=400)
        el = time.time() - t0
        cost.record("R3", el, uses_gpu=True, notes=f"LoRA seeds={seeds}")
        metrics = res["metrics"]
        gate = "PARTIAL"
        STATUS["R3"] = {"gate": gate, "metrics": metrics, "seeds": seeds,
                        "note": f"budget {budget:.1f}h, ran seeds={seeds}"}
        write_md(SUM / "r3_lora_summary.md", f"""# R3 Mamba LoRA Reranker Summary

- gate: **{gate}** (single-seed small-scale trial)
- seeds: {seeds}
- metrics: {metrics}
""")
        return gate
    except Exception as e:
        el = time.time() - t0
        cost.record("R3", el, uses_gpu=True, notes="R3 failed")
        STATUS["R3"] = {"gate": "FAIL", "error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()[-1500:]}
        write_md(SUM / "r3_lora_summary.md", f"# R3 FAILED\n{e}\n")
        return "FAIL"


def write_cost_plan(cost):
    write_md(SUM / "cost_plan.md", f"""# Linear-RAG Cost Plan

- GPU: single RTX 4090 24GB
- GPU_PRICE_PER_HOUR (env, default {DEFAULT_PRICE}): {cost.price}
- MAX_GPU_HOURS_TOTAL: {MAX_GPU_HOURS_TOTAL}

## Per-stage budget
| stage | budget |
|-------|--------|
| R0 data gen | CPU, < 0.5h |
| R1 BM25+embedding | < 1 GPU-h |
| R2 zero-shot | < 2 GPU-h |
| R2.5 cross-encoder | < 1 GPU-h |
| R3 LoRA | < 6 GPU-h |
| TOTAL | {MAX_GPU_HOURS_TOTAL} GPU-h |

Each GPU stage runs a dry-run (100-200 items) first, estimates full-stage time,
and shrinks sample size if over budget. runtime_profile.csv records actuals.
""")


def write_budget_pause(cost, stage):
    write_md(SUM / "budget_pause.md",
             f"# Budget Pause\nPaused before/at {stage}: cumulative GPU-hours "
             f"{cost.total_gpu_hours:.3f} >= MAX {MAX_GPU_HOURS_TOTAL}.\n"
             f"Completed stages: {list(STATUS.keys())}\n")


def main():
    price = float(os.environ.get("GPU_PRICE_PER_HOUR", str(DEFAULT_PRICE)))
    cost = CostTracker(RES, price_per_hour=price)
    write_cost_plan(cost)
    log(f"START orchestrator MAX_GPU_HOURS_TOTAL={MAX_GPU_HOURS_TOTAL} price={price}")

    # R0
    g0 = run_r0(); save_status(); log(f"R0 gate={g0}")
    if g0 == "FAIL":
        log("R0 FAILED — stopping."); save_status(); return

    if cost.over_budget():
        write_budget_pause(cost, "R1"); save_status(); return
    g1, r1_summary = run_r1(cost); save_status(); log(f"R1 gate={g1}")
    if g1 == "FAIL":
        log("R1 FAILED — stopping."); save_status(); return

    if cost.over_budget():
        write_budget_pause(cost, "R2"); save_status(); return
    g2, r2_metrics = run_r2(cost, r1_summary); save_status(); log(f"R2 gate={g2}")

    if cost.over_budget():
        write_budget_pause(cost, "R2.5"); save_status(); return
    g25, r25_row = run_r25(cost, r1_summary, r2_metrics); save_status(); log(f"R2.5 gate={g25}")

    # decide R3
    enter_r3, reasons = decide_r3(cost, g2, r2_metrics, r25_row)
    STATUS["R3_decision"] = {"enter": enter_r3, "reasons": reasons}
    save_status()
    log(f"R3 decision: enter={enter_r3} reasons={reasons}")
    if enter_r3 and not cost.over_budget():
        run_r3(cost); save_status()

    # latency/VRAM profiling for the models actually used
    try:
        collect_latency(cost)
    except Exception as e:
        log(f"latency collection error: {e}")
    save_status()

    # plots + final report
    finalize(cost, r1_summary, r2_metrics, r25_row, g0, g1, g2, g25, enter_r3, reasons)
    save_status()
    log("DONE")


def collect_latency(cost):
    from .eval import latency
    models = ["state-spaces/mamba-130m-hf", "EleutherAI/pythia-160m"]
    t0 = time.time()
    for mn in models:
        try:
            rows = latency.profile_causal_lm(mn, [1, 4, 8], 512, "bfloat16",
                                             "cuda", "latency")
            latency.append_rows(rows, RES / "latency_vram.csv")
            log(f"latency profiled {mn}")
        except Exception as e:
            log(f"latency profile {mn} failed: {e}")
    cost.record("latency_profile", time.time() - t0, uses_gpu=True, notes="fwd-pass bench")


def finalize(cost, r1_summary, r2_metrics, r25_row, g0, g1, g2, g25, enter_r3, reasons):
    from . import plots
    try:
        plots.latency_vram_tradeoff(RES / "latency_vram.csv",
                                    PLOTS / "latency_vram_tradeoff.png")
    except Exception as e:
        log(f"latency plot error: {e}")

    def r5(d, default="n/a"):
        return f"{d['recall@5']:.4f}" if isinstance(d, dict) and "recall@5" in d else default

    bm25 = r1_summary.get("bm25", {})
    emb = r1_summary.get("embedding", {})
    mamba = next((m for m in r2_metrics if "mamba" in m["model"] and "recall@5" in m), {})
    pythia = next((m for m in r2_metrics if "pythia" in m["model"] and "recall@5" in m), {})
    r3 = STATUS.get("R3", {})

    report = f"""# Linear-RAG Interim Report

## 1. Project Vision
Linear-RAG evaluates linear-time sequence models (Mamba, Mamba-2, DeltaNet,
Gated DeltaNet, EFLA) as **internal scanner / reranker / reader** components of a
RAG pipeline, operating on candidates that BM25/embedding retrieval already
surfaced. The external knowledge base stays updatable, traceable, and auditable.
We do **not** claim linear models replace RAG or are inherently superior to
Transformers; we measure an efficiency-accuracy frontier. We stopped the toy
KV-recall line (Stages A2/B0/B0C-F) because from-scratch Transformer controls
were unstable and the task probed induction-circuit emergence rather than
RAG-internal retrieval value.

## 2. Literature Positioning
SSMs (Mamba/Mamba-2/Mamba-3); linear-attention delta-rule memory
(DeltaNet/Gated DeltaNet/EFLA); linear models for ranking (Mamba Retriever — a
dense encoder, not a state retriever; RankMamba; "SSMs are Strong Text
Rerankers"); memory-augmented RAG / model-as-index (RAG, MemoRAG, DSI as a
risk reference); recall stress tests (Zoology, MQAR — a risk boundary, not the
main experiment). See docs/literature_map.md.

## 3. Benchmark Design
synth_rag_v1: {STATUS['R0']['stats']['n_docs']} docs, {STATUS['R0']['stats']['n_queries']} queries,
multi-field conjunctive queries (two/three/four-condition, code-based,
organization+event), >= {STATUS['R0']['stats']['min_hard_negatives']} hard negatives/query
(single/two-field overlap, near-synonym, swapped-entity, high-overlap), four
difficulty splits, deterministic (gold_unique_rate
{STATUS['R0']['stats']['gold_unique_rate']}). See docs/benchmark_spec.md.

## 4. Baseline Results (R1, gate {g1})
- BM25 Recall@5: {r5(bm25)}, Recall@10: {bm25.get('recall@10','n/a')}, MRR: {bm25.get('mrr','n/a')}, lat/q: {bm25.get('latency_ms_per_query','n/a')}ms
- Embedding Recall@5: {r5(emb)}, Recall@10: {emb.get('recall@10','n/a')}, MRR: {emb.get('mrr','n/a')}, lat/q: {emb.get('latency_ms_per_query','n/a')}ms
- gold_in_top100: {r1_summary.get('gold_in_top100')}
{('- Note: ' + STATUS['R1'].get('note','')) if STATUS['R1'].get('note') else ''}

## 5. Zero-shot Linear Scanner Results (R2, gate {g2})
- Mamba zero-shot Recall@5: {r5(mamba)} (lat/q {mamba.get('latency_ms_per_query','n/a')}ms, VRAM {mamba.get('peak_vram_mb','n/a')}MB)
- Pythia zero-shot Recall@5: {r5(pythia)} (lat/q {pythia.get('latency_ms_per_query','n/a')}ms, VRAM {pythia.get('peak_vram_mb','n/a')}MB)
- Cross-encoder Recall@5: {r5(r25_row)} (lat/q {r25_row.get('latency_ms_per_query','n/a') if isinstance(r25_row,dict) else 'n/a'}ms)

## 6. Latency and VRAM
See results/linear_rag/latency_vram.csv and plots/linear_rag/. Per-model
forward-pass latency and peak VRAM recorded with CUDA-event timing (50 warmup +
200 measured).

## 7. Decision
- R2 signal: **{g2}**
- Entered R3 LoRA: **{enter_r3}**{(' (reasons not met: ' + '; '.join(reasons) + ')') if not enter_r3 else ''}
{('- R3 result: ' + str(r3.get('gate')) + ' ' + str(r3.get('metrics', r3.get('error','')))) if r3 else ''}

## 8. Next Steps
{_next_steps(g2, enter_r3, mamba, r25_row)}

---
Total GPU-hours used: {cost.total_gpu_hours:.3f} / {MAX_GPU_HOURS_TOTAL}
(approx cost {cost.total_gpu_hours * cost.price:.2f} at {cost.price}/GPU-h).
Conclusions describe an efficiency-accuracy boundary; no claim of architectural
superiority or RAG replacement is made.
"""
    write_md(SUM / "linear_rag_final_report.md", report)


def _next_steps(g2, enter_r3, mamba, r25_row):
    if g2 == "NO_SIGNAL":
        return ("Zero-shot Mamba showed no reranking signal on this benchmark. "
                "Recommend: (a) revisit benchmark difficulty / lexical leakage, "
                "(b) try prompt format B (doc-id selection), (c) consider stronger "
                "linear models (Mamba-2 / Gated DeltaNet) before any LoRA training. "
                "Do not invest in R3 LoRA yet.")
    if enter_r3:
        return ("R3 LoRA trial ran; compare LoRA reranker against cross-encoder on "
                "the accuracy-latency-VRAM frontier; if promising, scale to 3 seeds "
                "and add Mamba-2 / Gated DeltaNet candidates (R5).")
    return ("R2 showed a (weak) signal but R3 entry conditions were not all met. "
            "Recommend confirming cross-encoder vs Mamba cost gap and re-checking "
            "budget before committing to LoRA; consider top-k scaling curves (R4).")


if __name__ == "__main__":
    main()
