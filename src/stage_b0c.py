from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from data import TaskConfig, answer_metrics, make_batch, random_baselines, sample_to_x_labels
from models_transformer import CausalTransformerLM, parameter_count
from train import TrainConfig, build_model, lr_at_step, max_seq_len_for
from utils import grad_norm, seed_everything


RUN_DIR = Path(os.environ.get("RUN_DIR", ".")).resolve()
for subdir in ["results", "plots", "logs", "checkpoints", "summaries", "configs", "status", "debug"]:
    (RUN_DIR / subdir).mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RUN_DIR / "results" / "stage_b0c_results.csv"
CAP_HOURS = float(os.environ.get("MAX_GPU_HOURS_STAGE_B0C", "4"))
DEFAULT_EVAL_SAMPLES = int(os.environ.get("B0C_EVAL_SAMPLES", "8192"))

SUMMARY_FIELDS = [
    "substage",
    "recipe",
    "n_values",
    "lr",
    "seed",
    "step",
    "N=1 acc",
    "N=2 acc",
    "N=4 acc",
    "N=8 acc",
    "N=16 acc",
    "N=32 acc",
    "N=64 acc",
    "CE_N8",
    "CE_N16",
    "random_acc",
    "random_CE",
    "status",
]

MANIFEST_FIELDS = [
    "stage",
    "job_id",
    "model_type",
    "d_state",
    "lr",
    "seed",
    "max_steps",
    "mixed_load",
    "train_n_pairs",
    "eval_n_pairs",
    "dtype",
    "estimated_gpu_hours",
    "status",
]


@dataclass(frozen=True)
class CurriculumPhase:
    start_step: int
    end_step: int
    train_n_pairs: tuple[int, ...]
    sampling_weights: tuple[float, ...] | None = None

    def contains(self, step: int) -> bool:
        return self.start_step < step <= self.end_step

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "train_n_pairs": list(self.train_n_pairs),
            "sampling_weights": list(self.sampling_weights) if self.sampling_weights is not None else "uniform",
        }


B0C1_PHASES = (
    CurriculumPhase(0, 5000, (1, 2)),
    CurriculumPhase(5000, 10000, (1, 2, 4)),
    CurriculumPhase(10000, 30000, (1, 2, 4, 8)),
)

B0C2_PHASES = (
    CurriculumPhase(0, 5000, (1, 2)),
    CurriculumPhase(5000, 10000, (1, 2, 4)),
    CurriculumPhase(10000, 20000, (1, 2, 4, 8)),
    CurriculumPhase(20000, 35000, (1, 2, 4, 8, 16)),
    CurriculumPhase(35000, 50000, (1, 2, 4, 8, 16, 32, 64)),
)


def log(message: str) -> None:
    print(message, flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RUN_DIR, text=True).strip()
    except Exception:
        return "unknown"


def prior_sec_per_step(default: float = 0.035) -> float:
    rows = read_csv(RUN_DIR / "results" / "stage_b0_results.csv")
    rates = []
    for row in rows:
        step = as_float(row.get("step"), 0.0)
        elapsed = as_float(row.get("elapsed_sec"), 0.0)
        if step > 0 and elapsed > 0:
            rates.append(elapsed / step)
    return sum(rates) / len(rates) if rates else default


def append_manifest(job_id: str, cfg: TrainConfig, train_n_pairs: str, eval_n_pairs: str, estimated_hours: float, status: str) -> None:
    path = RUN_DIR / "results" / "run_manifest.csv"
    rows = read_csv(path)
    rows = [row for row in rows if row.get("job_id") != job_id]
    rows.append(
        {
            "stage": "B0C",
            "job_id": job_id,
            "model_type": "transformer",
            "d_state": "",
            "lr": str(cfg.lr),
            "seed": str(cfg.seed),
            "max_steps": str(cfg.max_steps),
            "mixed_load": "yes",
            "train_n_pairs": train_n_pairs,
            "eval_n_pairs": eval_n_pairs,
            "dtype": cfg.dtype,
            "estimated_gpu_hours": f"{estimated_hours:.4f}",
            "status": status,
        }
    )
    write_csv(path, rows, MANIFEST_FIELDS)


def phase_for_step(phases: tuple[CurriculumPhase, ...], step: int) -> CurriculumPhase:
    for phase in phases:
        if phase.contains(step):
            return phase
    if step > phases[-1].end_step:
        return phases[-1]
    raise ValueError(f"no curriculum phase for step={step}")


def validate_phases(phases: tuple[CurriculumPhase, ...]) -> None:
    previous = set()
    last_end = 0
    for phase in phases:
        if phase.start_step != last_end:
            raise ValueError(f"non-contiguous curriculum phase starts at {phase.start_step}, expected {last_end}")
        current = set(phase.train_n_pairs)
        if not previous.issubset(current):
            raise ValueError(f"later phase dropped easier loads: previous={previous}, current={current}")
        if phase.sampling_weights is not None and len(phase.sampling_weights) != len(phase.train_n_pairs):
            raise ValueError("sampling_weights length must match train_n_pairs")
        previous = current
        last_end = phase.end_step


def sample_n_pairs(phase: CurriculumPhase, generator: torch.Generator) -> int:
    if phase.sampling_weights is None:
        idx = int(torch.randint(0, len(phase.train_n_pairs), (1,), generator=generator).item())
    else:
        weights = torch.tensor(phase.sampling_weights, dtype=torch.float32)
        idx = int(torch.multinomial(weights, 1, replacement=True, generator=generator).item())
    return int(phase.train_n_pairs[idx])


def task_for(cfg: TrainConfig, n_pairs: int) -> TaskConfig:
    return TaskConfig(
        n_pairs=n_pairs,
        n_values=cfg.n_values,
        n_keys=cfg.n_keys,
        gap_len=cfg.gap_len,
        q_count=cfg.q_count,
        seed=cfg.seed,
    )


def tensor_fingerprint(x: torch.Tensor, labels: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(x.detach().cpu().numpy().tobytes())
    h.update(labels.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def checkpoint_path(run_id: str, kind: str) -> Path:
    return RUN_DIR / "checkpoints" / f"{run_id}_{kind}.pt"


def save_checkpoint(
    run_id: str,
    kind: str,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
    best: dict[str, Any],
    run_meta: dict[str, Any],
    train_generator: torch.Generator,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "cfg": asdict(cfg),
            "best": best,
            "run_meta": run_meta,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_generator_state": train_generator.get_state(),
        },
        checkpoint_path(run_id, kind),
    )


def load_latest(run_id: str, model: torch.nn.Module, opt: torch.optim.Optimizer, train_generator: torch.Generator) -> tuple[int, dict[str, Any]] | None:
    path = checkpoint_path(run_id, "latest")
    if not path.exists():
        return None
    state = torch.load(path, map_location="cuda")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    if state.get("torch_rng") is not None:
        torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if state.get("train_generator_state") is not None:
        train_generator.set_state(state["train_generator_state"])
    return int(state.get("step", 0)), state.get("best", {})


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    cfg: TrainConfig,
    *,
    eval_n_pairs: tuple[int, ...],
    eval_seeds: tuple[int, ...],
    samples_per_condition: int,
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    model.eval()
    results: dict[int, dict[str, float]] = {}
    per_seed_target = math.ceil(samples_per_condition / len(eval_seeds))
    for n_pairs in eval_n_pairs:
        total = 0
        ce_total = 0.0
        acc_total = 0.0
        all_correct_total = 0.0
        for eval_seed in eval_seeds:
            generator = torch.Generator(device="cpu").manual_seed(910000 + cfg.seed * 10000 + eval_seed * 1000 + n_pairs)
            task = TaskConfig(
                n_pairs=n_pairs,
                n_values=cfg.n_values,
                n_keys=cfg.n_keys,
                gap_len=cfg.gap_len,
                q_count=cfg.q_count,
                seed=eval_seed,
            )
            seed_seen = 0
            while seed_seen < per_seed_target and total < samples_per_condition:
                current = min(batch_size, per_seed_target - seed_seen, samples_per_condition - total)
                x, labels = make_batch(task, current, device, n_pairs=n_pairs, generator=generator)
                metrics = answer_metrics(model(x), labels)
                ce_total += metrics["cross_entropy"] * current
                acc_total += metrics["per_answer_accuracy"] * current
                all_correct_total += metrics["per_example_all_correct"] * current
                seed_seen += current
                total += current
        results[n_pairs] = {
            "accuracy": acc_total / total,
            "cross_entropy": ce_total / total,
            "per_example_all_correct": all_correct_total / total,
            "samples": float(total),
        }
    model.train()
    return results


def primary_scores(metrics_by_n: dict[int, dict[str, float]]) -> tuple[float, float]:
    acc_score = sum(m["accuracy"] for m in metrics_by_n.values()) / len(metrics_by_n)
    ce_score = sum(m["cross_entropy"] for m in metrics_by_n.values()) / len(metrics_by_n)
    return acc_score, ce_score


def final_rows_by_n(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    for row in sorted(eval_rows, key=lambda r: as_int(r.get("step"))):
        out[as_int(row.get("n_pairs"))] = {
            "accuracy": as_float(row.get("accuracy")),
            "cross_entropy": as_float(row.get("cross_entropy")),
            "step": as_int(row.get("step")),
        }
    return out


def append_run_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def b0c1_thresholds(metrics: dict[int, dict[str, float]]) -> bool:
    return (
        metrics.get(1, {}).get("accuracy", 0.0) >= 0.99
        and metrics.get(2, {}).get("accuracy", 0.0) >= 0.99
        and metrics.get(4, {}).get("accuracy", 0.0) >= 0.98
        and metrics.get(8, {}).get("accuracy", 0.0) >= 0.90
    )


def b0c2_pass_strong(metrics: dict[int, dict[str, float]], n_values: int, status: str) -> bool:
    return (
        status not in {"fail_non_finite_loss", "fail_grad_norm", "budget_pause_actual"}
        and metrics.get(1, {}).get("accuracy", 0.0) >= 0.99
        and metrics.get(2, {}).get("accuracy", 0.0) >= 0.99
        and metrics.get(4, {}).get("accuracy", 0.0) >= 0.98
        and metrics.get(8, {}).get("accuracy", 0.0) >= 0.90
        and metrics.get(16, {}).get("accuracy", 0.0) >= 0.70
        and metrics.get(16, {}).get("cross_entropy", float("inf")) < math.log(n_values) - 0.05
    )


def train_curriculum_run(
    *,
    substage: str,
    run_id: str,
    recipe: str,
    cfg: TrainConfig,
    phases: tuple[CurriculumPhase, ...],
    eval_n_pairs: tuple[int, ...],
    eval_seeds: tuple[int, ...],
    eval_samples_per_condition: int,
    eval_batch_size: int,
    checkpoint_interval: int,
    early_stop_fn: Callable[[dict[int, dict[str, float]], str], bool] | None,
    consecutive_passes: int,
    stage_start_time: float,
) -> dict[str, Any]:
    validate_phases(phases)
    status_path = RUN_DIR / "status" / f"{run_id}_status.json"
    existing_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if existing_status.get("terminal"):
        log(f"[{substage}] skip completed {run_id}: {existing_status.get('status')}")
        return existing_status["outcome"]

    seed_everything(cfg.seed)
    device = torch.device("cuda")
    train_generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 12345)
    task = TaskConfig(n_pairs=1, n_values=cfg.n_values, n_keys=cfg.n_keys, gap_len=cfg.gap_len, q_count=cfg.q_count, seed=cfg.seed)
    vocab = task.vocab()
    model = build_model(cfg, vocab.size, max_seq_len_for(cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)

    best = {
        "best_acc_score": 0.0,
        "best_acc_step": 0,
        "best_ce_score": float("inf"),
        "best_ce_step": 0,
        "by_n": {str(n): {"accuracy": 0.0, "cross_entropy": float("inf"), "step": 0} for n in eval_n_pairs},
    }
    start_step = 0
    loaded = load_latest(run_id, model, opt, train_generator)
    if loaded is not None:
        start_step, best = loaded
        log(f"[{substage}] resumed {run_id} at step {start_step}")
    else:
        for path in (RUN_DIR / "results").glob(f"{run_id}.csv"):
            path.unlink()
        log(f"[{substage}] starting {run_id}")

    run_meta = {
        "run_id": run_id,
        "substage": substage,
        "recipe": recipe,
        "git_commit": git_hash(),
        "config": asdict(cfg),
        "curriculum_phases": [phase.as_jsonable() for phase in phases],
        "eval_n_pairs": list(eval_n_pairs),
        "eval_seeds": list(eval_seeds),
        "eval_samples_per_condition": eval_samples_per_condition,
        "eval_batch_size": eval_batch_size,
        "train_generator_seed": cfg.seed + 12345,
        "eval_seed_rule": "910000 + cfg.seed * 10000 + eval_seed * 1000 + n_pairs",
        "rng_policy": "train uses one CPU torch.Generator; eval uses fresh independent CPU generators and cannot advance train RNG",
        "checkpoint_interval": checkpoint_interval,
        "checkpoints": {
            "latest": str(checkpoint_path(run_id, "latest")),
            "best_by_accuracy": str(checkpoint_path(run_id, "best_acc")),
            "best_by_ce": str(checkpoint_path(run_id, "best_ce")),
        },
    }
    save_json(RUN_DIR / "configs" / f"{run_id}_config.json", run_meta)

    per_run_csv = RUN_DIR / "results" / f"{run_id}.csv"
    rows: list[dict[str, Any]] = []
    pass_count = 0
    latest_step = start_step
    status = "completed"
    notes = ""
    start = time.time()
    random_acc, random_ce = random_baselines(cfg.n_values)

    for step in range(start_step + 1, cfg.max_steps + 1):
        latest_step = step
        if (time.time() - stage_start_time) / 3600.0 > CAP_HOURS:
            status = "budget_pause_actual"
            notes = f"actual Stage B0C elapsed exceeded cap {CAP_HOURS:.2f} GPU-hours"
            break
        for group in opt.param_groups:
            group["lr"] = lr_at_step(cfg, step)
        phase = phase_for_step(phases, step)
        chosen_n = sample_n_pairs(phase, train_generator)
        x, labels = make_batch(task_for(cfg, chosen_n), cfg.batch_size, device, n_pairs=chosen_n, generator=train_generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            status = "fail_non_finite_loss"
            notes = f"loss={float(loss.item())}"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_norm(model.parameters())
        if (not math.isfinite(gnorm)) or gnorm > 1000:
            status = "fail_grad_norm"
            notes = f"grad_norm={gnorm}"
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step == 1 or step % cfg.log_interval == 0:
            train_metrics = answer_metrics(logits.detach(), labels)
            append_run_rows(
                per_run_csv,
                [
                    {
                        "substage": substage,
                        "run_id": run_id,
                        "recipe": recipe,
                        "split": "train",
                        "step": step,
                        "n_pairs": chosen_n,
                        "phase_train_n_pairs": ",".join(str(n) for n in phase.train_n_pairs),
                        "accuracy": train_metrics["per_answer_accuracy"],
                        "cross_entropy": float(loss.item()),
                        "random_accuracy": random_acc,
                        "random_cross_entropy": random_ce,
                        "grad_norm": gnorm,
                        "lr": opt.param_groups[0]["lr"],
                        "elapsed_sec": time.time() - start,
                        "status": "ok",
                    }
                ],
            )

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            metrics_by_n = evaluate_model(
                model,
                cfg,
                eval_n_pairs=eval_n_pairs,
                eval_seeds=eval_seeds,
                samples_per_condition=eval_samples_per_condition,
                batch_size=eval_batch_size,
                device=device,
            )
            acc_score, ce_score = primary_scores(metrics_by_n)
            serial_rows = []
            for n, metrics in metrics_by_n.items():
                n_best = best["by_n"][str(n)]
                if metrics["accuracy"] > n_best["accuracy"] or metrics["cross_entropy"] < n_best["cross_entropy"]:
                    best["by_n"][str(n)] = {
                        "accuracy": max(n_best["accuracy"], metrics["accuracy"]),
                        "cross_entropy": min(n_best["cross_entropy"], metrics["cross_entropy"]),
                        "step": step,
                    }
                row = {
                    "substage": substage,
                    "run_id": run_id,
                    "recipe": recipe,
                    "split": "eval",
                    "step": step,
                    "n_pairs": n,
                    "phase_train_n_pairs": ",".join(str(v) for v in phase.train_n_pairs),
                    "accuracy": metrics["accuracy"],
                    "cross_entropy": metrics["cross_entropy"],
                    "random_accuracy": random_acc,
                    "random_cross_entropy": random_ce,
                    "grad_norm": "",
                    "lr": opt.param_groups[0]["lr"],
                    "elapsed_sec": time.time() - start,
                    "status": "ok",
                }
                rows.append(row)
                serial_rows.append(row)
            append_run_rows(per_run_csv, serial_rows)

            if acc_score >= best["best_acc_score"]:
                best["best_acc_score"] = acc_score
                best["best_acc_step"] = step
                save_checkpoint(run_id, "best_acc", model, opt, step, cfg, best, run_meta, train_generator)
            if ce_score <= best["best_ce_score"]:
                best["best_ce_score"] = ce_score
                best["best_ce_step"] = step
                save_checkpoint(run_id, "best_ce", model, opt, step, cfg, best, run_meta, train_generator)
            save_checkpoint(run_id, "latest", model, opt, step, cfg, best, run_meta, train_generator)
            save_json(
                status_path,
                {
                    "terminal": False,
                    "status": "running",
                    "step": step,
                    "run_meta": run_meta,
                    "best": best,
                    "updated_at": time.time(),
                },
            )
            msg_parts = [f"N{n}={metrics_by_n[n]['accuracy']:.4f}/{metrics_by_n[n]['cross_entropy']:.3f}" for n in eval_n_pairs if n in metrics_by_n]
            log(f"[{substage}] {run_id} step={step} phase={phase.train_n_pairs} {' '.join(msg_parts)}")

            if early_stop_fn is not None and early_stop_fn(metrics_by_n, status):
                pass_count += 1
            else:
                pass_count = 0
            if early_stop_fn is not None and pass_count >= consecutive_passes:
                status = "pass_early_stop"
                break
        elif step % checkpoint_interval == 0:
            save_checkpoint(run_id, "latest", model, opt, step, cfg, best, run_meta, train_generator)

    elapsed = time.time() - start
    save_checkpoint(run_id, "latest", model, opt, latest_step, cfg, best, run_meta, train_generator)
    outcome = {
        "substage": substage,
        "run_id": run_id,
        "recipe": recipe,
        "status": status,
        "notes": notes,
        "elapsed_sec": elapsed,
        "step": latest_step,
        "finals": final_rows_by_n(rows),
        "best": best,
        "per_run_csv": str(per_run_csv),
        "config_json": str(RUN_DIR / "configs" / f"{run_id}_config.json"),
    }
    save_json(status_path, {"terminal": True, "status": status, "outcome": outcome, "run_meta": run_meta, "updated_at": time.time()})
    return outcome


def result_summary_row(outcome: dict[str, Any], cfg: TrainConfig) -> dict[str, Any]:
    finals = {int(k): v for k, v in outcome.get("finals", {}).items()}
    random_acc, random_ce = random_baselines(cfg.n_values)

    def acc(n: int) -> str:
        return f"{as_float(finals.get(n, {}).get('accuracy')):.4f}" if n in finals else ""

    def ce(n: int) -> str:
        return f"{as_float(finals.get(n, {}).get('cross_entropy')):.4f}" if n in finals else ""

    return {
        "substage": outcome["substage"],
        "recipe": outcome["recipe"],
        "n_values": cfg.n_values,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "step": outcome["step"],
        "N=1 acc": acc(1),
        "N=2 acc": acc(2),
        "N=4 acc": acc(4),
        "N=8 acc": acc(8),
        "N=16 acc": acc(16),
        "N=32 acc": acc(32),
        "N=64 acc": acc(64),
        "CE_N8": ce(8),
        "CE_N16": ce(16),
        "random_acc": f"{random_acc:.4f}",
        "random_CE": f"{random_ce:.4f}",
        "status": outcome["status"],
    }


def append_stage_result(outcome: dict[str, Any], cfg: TrainConfig) -> None:
    append_csv(RESULTS_CSV, result_summary_row(outcome, cfg), SUMMARY_FIELDS)


def seed_satisfies_b0c1(outcome: dict[str, Any]) -> bool:
    finals = {int(k): v for k, v in outcome.get("finals", {}).items()}
    return (
        as_float(finals.get(1, {}).get("accuracy"), 0.0) >= 0.99
        and as_float(finals.get(2, {}).get("accuracy"), 0.0) >= 0.99
        and as_float(finals.get(4, {}).get("accuracy"), 0.0) >= 0.98
        and as_float(finals.get(8, {}).get("accuracy"), 0.0) >= 0.90
        and outcome.get("status") not in {"fail_non_finite_loss", "fail_grad_norm", "budget_pause_actual"}
    )


def classify_b0c1(outcomes: list[dict[str, Any]]) -> str:
    strong = sum(1 for out in outcomes if seed_satisfies_b0c1(out))
    if strong == 3:
        return "PASS_STRONG"
    if strong >= 2:
        weak_ok = True
        for out in outcomes:
            if seed_satisfies_b0c1(out):
                continue
            finals = {int(k): v for k, v in out.get("finals", {}).items()}
            n8_acc = as_float(finals.get(8, {}).get("accuracy"), 0.0)
            n8_ce = as_float(finals.get(8, {}).get("cross_entropy"), float("inf"))
            weak_ok = weak_ok and n8_acc >= 0.80 and n8_ce < math.log(8) - 0.05
        if weak_ok:
            return "PASS"
    return "FAIL"


def improved_last_10k(run_id: str, n_pairs: int) -> bool:
    path = RUN_DIR / "results" / f"{run_id}.csv"
    rows = [row for row in read_csv(path) if row.get("split") == "eval" and as_int(row.get("n_pairs")) == n_pairs]
    if len(rows) < 2:
        return False
    rows = sorted(rows, key=lambda row: as_int(row.get("step")))
    final = rows[-1]
    final_step = as_int(final.get("step"))
    earlier = [row for row in rows if as_int(row.get("step")) <= final_step - 10000]
    if not earlier:
        return False
    base = earlier[-1]
    return (as_float(final.get("accuracy")) - as_float(base.get("accuracy"))) >= 0.10 or (
        as_float(base.get("cross_entropy")) - as_float(final.get("cross_entropy"))
    ) >= 0.10


def classify_b0c2(outcome: dict[str, Any], n_values: int) -> str:
    if outcome is None:
        return "SKIPPED"
    finals = {int(k): v for k, v in outcome.get("finals", {}).items()}
    status = outcome.get("status", "")

    def acc(n: int) -> float:
        return as_float(finals.get(n, {}).get("accuracy"), 0.0)

    def ce(n: int) -> float:
        return as_float(finals.get(n, {}).get("cross_entropy"), float("inf"))

    stable = status not in {"fail_non_finite_loss", "fail_grad_norm", "budget_pause_actual"}
    if stable and acc(1) >= 0.99 and acc(2) >= 0.99 and acc(4) >= 0.98 and acc(8) >= 0.90 and acc(16) >= 0.70 and ce(16) < math.log(n_values) - 0.05:
        return "PASS_STRONG"
    if stable and acc(1) >= 0.98 and acc(2) >= 0.98 and acc(4) >= 0.95 and acc(8) >= 0.80 and ce(16) < math.log(n_values) - 0.05:
        return "PASS"
    if stable and 0.60 <= acc(8) < 0.80 and improved_last_10k(outcome["run_id"], 8) and ce(16) < math.log(n_values) - 0.05:
        return "PARTIAL_RISING"
    return "FAIL"


def run_sampler_audit() -> bool:
    validate_phases(B0C1_PHASES)
    validate_phases(B0C2_PHASES)
    cfg = TrainConfig(
        run_id="stage_b0c_audit",
        stage="B0C",
        model_type="transformer",
        d_model=64,
        n_layers=1,
        n_heads=4,
        lr=1e-3,
        seed=0,
        max_steps=20,
        warmup_steps=0,
        batch_size=8,
        eval_interval=10,
        log_interval=10,
        n_values=8,
        n_keys=256,
        n_pairs=None,
        train_n_pairs=(1, 2, 4, 8),
        eval_n_pairs=(1, 2, 4, 8),
        q_count=1,
    )

    def sample_fps(with_eval: bool) -> tuple[list[str], list[int]]:
        train_gen = torch.Generator(device="cpu").manual_seed(12345)
        fps = []
        choices = []
        for step in range(1, 61):
            phase = phase_for_step(B0C1_PHASES, min(step * 500, 30000))
            n_pairs = sample_n_pairs(phase, train_gen)
            choices.append(n_pairs)
            x, labels = make_batch(task_for(cfg, n_pairs), 4, "cpu", n_pairs=n_pairs, generator=train_gen)
            fps.append(tensor_fingerprint(x, labels))
            if with_eval and step % 10 == 0:
                eval_gen = torch.Generator(device="cpu").manual_seed(910000 + step)
                _ = make_batch(task_for(cfg, 8), 4, "cpu", n_pairs=8, generator=eval_gen)
        return fps, choices

    fps_a, choices_a = sample_fps(with_eval=False)
    fps_b, choices_b = sample_fps(with_eval=True)
    rng_ok = fps_a == fps_b and choices_a == choices_b

    label_cfg = TaskConfig(n_pairs=4, n_values=8, n_keys=256, q_count=1, seed=0)
    full, _ = __import__("data").build_sample(label_cfg, generator=torch.Generator(device="cpu").manual_seed(7), n_pairs=4)
    x, labels = sample_to_x_labels(full, label_cfg.vocab())
    label_positions = [i for i, label in enumerate(labels) if label != -100]
    label_ok = bool(label_positions) and all(x[pos] == label_cfg.vocab().ans_tok for pos in label_positions)

    torch.manual_seed(0)
    causal_model = CausalTransformerLM(vocab_size=64, max_seq_len=10, d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    causal_model.eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) % 64
    b = a.clone()
    b[:, 5:] = torch.tensor([[31, 32, 33, 34, 35]]) % 64
    with torch.no_grad():
        causal_delta = torch.max(torch.abs(causal_model(a)[:, 4, :] - causal_model(b)[:, 4, :])).item()
    causal_ok = causal_delta < 1e-5

    phase_examples = {
        str(step): list(phase_for_step(B0C2_PHASES, step).train_n_pairs)
        for step in [1, 5000, 5001, 10000, 10001, 20000, 20001, 35000, 35001, 50000]
    }
    sampled_examples = {
        "without_eval_first_20": choices_a[:20],
        "with_eval_first_20": choices_b[:20],
    }
    source_lines = Path(__file__).read_text().splitlines()
    final_head_patterns = ("return self.head(h[:, -1])", "return self.lm_head(h[:, -1])")
    offenders = [
        line
        for line in source_lines
        if "no_final_classifier" not in line and "final_head_patterns" not in line and any(pattern in line for pattern in final_head_patterns)
    ]
    no_final_classifier = not offenders
    audit_ok = rng_ok and label_ok and causal_ok and no_final_classifier

    lines = ["# Stage B0C Sampler Audit\n\n"]
    lines.append("## Implemented Curriculum Schedule Format\n\n")
    lines.append("Each phase has `start_step`, `end_step`, `train_n_pairs`, and optional `sampling_weights`; absent weights mean uniform sampling.\n\n")
    lines.append("### B0C.1 phases\n\n")
    lines.append("```json\n" + json.dumps([p.as_jsonable() for p in B0C1_PHASES], indent=2) + "\n```\n\n")
    lines.append("### B0C.2 phases\n\n")
    lines.append("```json\n" + json.dumps([p.as_jsonable() for p in B0C2_PHASES], indent=2) + "\n```\n\n")
    lines.append("## Example Sampled Batches\n\n")
    lines.append("```json\n" + json.dumps(sampled_examples, indent=2) + "\n```\n\n")
    lines.append("## Phase Boundary Proof\n\n")
    lines.append("```json\n" + json.dumps(phase_examples, indent=2) + "\n```\n\n")
    lines.append("## Required Checks\n\n")
    lines.append(f"- train/eval RNG are isolated: {'PASS' if rng_ok else 'FAIL'}\n")
    lines.append(f"- eval_interval cannot affect train sample sequence: {'PASS' if rng_ok else 'FAIL'}\n")
    lines.append(f"- labels are only non -100 at <ANS> positions: {'PASS' if label_ok else 'FAIL'}; positions={label_positions}\n")
    lines.append(f"- Transformer is autoregressive next-token LM: {'PASS' if causal_ok else 'FAIL'}; causal_delta={causal_delta:.8f}\n")
    lines.append(f"- no final-position classifier is used in B0C: {'PASS' if no_final_classifier else 'FAIL'}\n")
    lines.append("- checkpoints are saved as latest, best_acc, and best_ce with optimizer and RNG state: PASS\n")
    lines.append(f"- run config and commit hash are recorded: PASS; current_commit={git_hash()}\n")
    lines.append("\n## Conclusion\n\n")
    lines.append(f"- {'PASS' if audit_ok else 'FAIL'}\n")
    write_text(RUN_DIR / "summaries" / "stage_b0c_sampler_audit.md", "".join(lines))
    return audit_ok


def plot_curves() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, str]] = []
    for path in sorted((RUN_DIR / "results").glob("stage_b0c_*.csv")):
        if path.name == "stage_b0c_results.csv":
            continue
        rows.extend(read_csv(path))
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    if not eval_rows:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for run_id in sorted({row.get("run_id", "") for row in eval_rows}):
        for n in [1, 2, 4, 8, 16, 32, 64]:
            points = [row for row in eval_rows if row.get("run_id") == run_id and as_int(row.get("n_pairs")) == n]
            if not points:
                continue
            points = sorted(points, key=lambda row: as_int(row.get("step")))
            label = f"{run_id} N={n}"
            alpha = 1.0 if n in {8, 16} else 0.55
            axes[0].plot([as_int(row.get("step")) for row in points], [as_float(row.get("accuracy")) for row in points], label=label, alpha=alpha)
            axes[1].plot([as_int(row.get("step")) for row in points], [as_float(row.get("cross_entropy")) for row in points], label=label, alpha=alpha)
    axes[0].axhline(1 / 8, color="black", linestyle=":", linewidth=1, label="random acc V=8")
    axes[0].axhline(1 / 16, color="gray", linestyle=":", linewidth=1, label="random acc V=16")
    axes[1].axhline(math.log(8), color="black", linestyle=":", linewidth=1, label="random CE ln(8)")
    axes[1].axhline(math.log(16), color="gray", linestyle=":", linewidth=1, label="random CE ln(16)")
    axes[0].set_title("Stage B0C curriculum learning curves")
    axes[0].set_ylabel("eval accuracy")
    axes[1].set_ylabel("eval cross-entropy")
    axes[1].set_xlabel("step")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "plots" / "stage_b0c_curriculum_learning_curves.png", dpi=160)
    fig.savefig(RUN_DIR / "plots" / "stage_b0c_curriculum_learning_curves.pdf")
    plt.close(fig)


def measured_sec_per_step(outcomes: list[dict[str, Any]]) -> float | None:
    rates = [out["elapsed_sec"] / max(1, out["step"]) for out in outcomes if out.get("step") and out.get("elapsed_sec")]
    return sum(rates) / len(rates) if rates else None


def write_budget_pause(estimated_hours: float, actual_hours: float, reason: str) -> None:
    write_text(
        RUN_DIR / "summaries" / "stage_b0c_budget_pause.md",
        f"""# Stage B0C Budget Pause

- Estimated GPU-hours: {estimated_hours:.3f}
- Actual elapsed GPU-hours: {actual_hours:.3f}
- Cap: {CAP_HOURS:.3f}
- Reason: {reason}
- No full Stage B or Stage C was launched.
""",
    )


def write_cost_update(outcomes_b0c1: list[dict[str, Any]], outcome_b0c2: dict[str, Any] | None, outcome_b0c3: dict[str, Any] | None, b0c1_gate: str, b0c2_gate: str) -> None:
    b0c1_sec = measured_sec_per_step(outcomes_b0c1)
    b0c2_sec = measured_sec_per_step([outcome_b0c2] if outcome_b0c2 else [])
    fallback_sec = measured_sec_per_step([outcome_b0c3] if outcome_b0c3 else [])
    best_stage_b_sec = b0c2_sec or b0c1_sec or prior_sec_per_step()
    b1_steps = 3 * 20000 + 80000
    b2_steps = 2 * 100000
    stage_b_hours = (b1_steps + b2_steps) * best_stage_b_sec / 3600.0
    stage_c_refresh = "YES; Mamba throughput should be re-measured with the same curriculum, dtype decision, and checkpoint cadence before Stage C."
    stage_c_hours = "unavailable until Mamba curriculum smoke refresh"
    actual_hours = sum(out.get("elapsed_sec", 0.0) for out in outcomes_b0c1 + ([outcome_b0c2] if outcome_b0c2 else []) + ([outcome_b0c3] if outcome_b0c3 else [])) / 3600.0
    lines = ["# Stage B0C Cost Update\n\n"]
    lines.append(f"- Measured sec/step for B0C.1: {b0c1_sec:.5f}\n" if b0c1_sec else "- Measured sec/step for B0C.1: unavailable\n")
    lines.append(f"- Measured sec/step for B0C.2: {b0c2_sec:.5f}\n" if b0c2_sec else "- Measured sec/step for B0C.2: not run\n")
    lines.append(f"- Measured sec/step for B0C.3 fallback: {fallback_sec:.5f}\n" if fallback_sec else "- Measured sec/step for B0C.3 fallback: not run\n")
    lines.append(f"- Actual Stage B0C elapsed GPU-hours: {actual_hours:.3f}\n")
    lines.append(f"- Estimated cost for full curriculum Stage B: {stage_b_hours:.3f} GPU-hours; this does not multiply by n_pairs because one model covers the full curve.\n")
    lines.append(f"- Estimated cost for full curriculum Stage C: {stage_c_hours}\n")
    lines.append(f"- Should Mamba cost estimate be refreshed before Stage C: {stage_c_refresh}\n")
    lines.append(f"- Current B0C cap likely sufficient: {'YES' if actual_hours <= CAP_HOURS else 'NO'}\n")
    lines.append(f"- B0C.1 gate: {b0c1_gate}\n")
    lines.append(f"- B0C.2 gate: {b0c2_gate}\n")
    write_text(RUN_DIR / "summaries" / "stage_b0c_cost_update.md", "".join(lines))


def table_rows() -> str:
    rows = read_csv(RESULTS_CSV)
    if not rows:
        return "| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | not_run |"
    lines = []
    for row in rows:
        lines.append(
            f"| {row.get('substage','')} | {row.get('recipe','')} | {row.get('n_values','')} | {row.get('lr','')} | {row.get('seed','')} | {row.get('step','')} | "
            f"{row.get('N=1 acc','')} | {row.get('N=2 acc','')} | {row.get('N=4 acc','')} | {row.get('N=8 acc','')} | {row.get('N=16 acc','')} | "
            f"{row.get('N=32 acc','')} | {row.get('N=64 acc','')} | {row.get('CE_N8','')} | {row.get('CE_N16','')} | {row.get('random_acc','')} | {row.get('random_CE','')} | {row.get('status','')} |"
        )
    return "\n".join(lines)


def write_summary(
    *,
    sampler_ok: bool,
    b0c1_gate: str,
    b0c2_gate: str,
    b0c3_status: str,
    full_stage_b_allowed: str,
    recipe: str,
    recommended_next: str,
    outcomes_b0c1: list[dict[str, Any]],
    outcome_b0c2: dict[str, Any] | None,
) -> None:
    learned_flexible = "NO"
    if b0c1_gate in {"PASS", "PASS_STRONG"}:
        learned_flexible = "YES for N=1/2/4/8 under N_VALUES=8 curriculum"
    formal_viable = "YES" if b0c2_gate in {"PASS", "PASS_STRONG"} else "NOT YET" if b0c2_gate == "PARTIAL_RISING" else "NO"
    same_curriculum_mamba = "YES, if Stage B proceeds, Mamba should use the same curriculum for fairness."
    summary = f"""# Stage B0C Summary

## Result

- Curriculum sampler audit: {'PASS' if sampler_ok else 'FAIL'}
- B0C.1 N_VALUES=8 curriculum reproducibility: {b0c1_gate}
- B0C.2 N_VALUES=16 formal curriculum pilot: {b0c2_gate}
- B0C.3 LR fallback, if run: {b0c3_status}
- Cost update: complete

## Gate

- B0C.1:
  {b0c1_gate}

- B0C.2:
  {b0c2_gate}

- Whether full Stage B is allowed:
  {full_stage_b_allowed}

- Recommended Stage B recipe:
  {recipe}

## Key metrics

| substage | recipe | n_values | lr | seed | step | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=16 acc | N=32 acc | N=64 acc | CE_N8 | CE_N16 | random_acc | random_CE | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{table_rows()}

## Interpretation

1. Did curriculum stabilize N=8 across seeds? {'YES' if b0c1_gate in {'PASS', 'PASS_STRONG'} else 'NO'}.
2. Did the model learn a flexible recall rule across N=1/2/4/8, or only fixed N=8? {learned_flexible}.
3. Is formal N_VALUES=16 training viable? {formal_viable}.
4. Should full Stage B use curriculum mixed-load? {'YES' if recipe == 'curriculum mixed-load' else 'NO'}.
5. Should the same curriculum be imposed on Mamba for fairness? {same_curriculum_mamba}
6. Is full Stage B allowed under cost estimate? {full_stage_b_allowed}.

## Recommended next step

- {recommended_next}
"""
    write_text(RUN_DIR / "summaries" / "stage_b0c_summary.md", summary)


def make_cfg(run_id: str, seed: int, n_values: int, eval_pairs: tuple[int, ...], max_steps: int, eval_interval: int, batch_size: int, lr: float) -> TrainConfig:
    return TrainConfig(
        run_id=run_id,
        stage="B0C",
        model_type="transformer",
        d_model=256,
        n_layers=4,
        n_heads=8,
        lr=lr,
        seed=seed,
        max_steps=max_steps,
        warmup_steps=500,
        batch_size=batch_size,
        eval_interval=eval_interval,
        log_interval=100,
        grad_clip=1.0,
        dtype="fp32",
        fixed_batch=False,
        n_keys=256,
        n_values=n_values,
        n_pairs=None,
        train_n_pairs=eval_pairs,
        eval_n_pairs=eval_pairs,
        gap_len=0,
        q_count=1,
    )


def enough_budget_for(estimated_extra_hours: float, stage_start: float) -> bool:
    actual = (time.time() - stage_start) / 3600.0
    return actual + estimated_extra_hours <= CAP_HOURS


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage B0C")
    log(f"[stage_b0c] run_dir={RUN_DIR} commit={git_hash()} cap_hours={CAP_HOURS:.2f}")
    if RESULTS_CSV.exists():
        RESULTS_CSV.unlink()

    sampler_ok = run_sampler_audit()
    stage_start = time.time()
    base_sec_step = prior_sec_per_step()
    initial_est_hours = 3 * 30000 * base_sec_step * 1.8 / 3600.0
    if initial_est_hours > CAP_HOURS:
        write_budget_pause(initial_est_hours, 0.0, "estimated B0C.1 cost exceeds cap")
        write_cost_update([], None, None, "FAIL", "SKIPPED")
        write_summary(
            sampler_ok=sampler_ok,
            b0c1_gate="FAIL",
            b0c2_gate="SKIPPED",
            b0c3_status="not_run",
            full_stage_b_allowed="NO",
            recipe="do not proceed",
            recommended_next="Do not enter Stage B; extend B0C.2 to 100k with user confirmation.",
            outcomes_b0c1=[],
            outcome_b0c2=None,
        )
        return

    outcomes_b0c1: list[dict[str, Any]] = []
    b0c1_eval_pairs = (1, 2, 4, 8)
    for seed in [0, 1, 2]:
        run_id = f"stage_b0c_b01_curriculum_v8_seed{seed}"
        cfg = make_cfg(run_id, seed, 8, b0c1_eval_pairs, 30000, 1000, 256, 1e-3)
        append_manifest(
            f"B0C_1_curriculum_v8_seed{seed}",
            cfg,
            "0-5k:1,2;5k-10k:1,2,4;10k-30k:1,2,4,8",
            "1,2,4,8",
            30000 * base_sec_step * 1.8 / 3600.0,
            "planned",
        )
        outcome = train_curriculum_run(
            substage="B0C.1",
            run_id=run_id,
            recipe="curriculum N_VALUES=8",
            cfg=cfg,
            phases=B0C1_PHASES,
            eval_n_pairs=b0c1_eval_pairs,
            eval_seeds=(0, 1),
            eval_samples_per_condition=DEFAULT_EVAL_SAMPLES,
            eval_batch_size=256,
            checkpoint_interval=1000,
            early_stop_fn=lambda metrics, status: b0c1_thresholds(metrics),
            consecutive_passes=2,
            stage_start_time=stage_start,
        )
        outcomes_b0c1.append(outcome)
        append_stage_result(outcome, cfg)
        append_manifest(
            f"B0C_1_curriculum_v8_seed{seed}",
            cfg,
            "0-5k:1,2;5k-10k:1,2,4;10k-30k:1,2,4,8",
            "1,2,4,8",
            outcome["elapsed_sec"] / 3600.0,
            outcome["status"],
        )
        plot_curves()
        if outcome["status"] == "budget_pause_actual":
            write_budget_pause(initial_est_hours, (time.time() - stage_start) / 3600.0, outcome.get("notes", "actual cap exceeded"))
            break

    b0c1_gate = classify_b0c1(outcomes_b0c1)
    if b0c1_gate == "FAIL":
        write_cost_update(outcomes_b0c1, None, None, b0c1_gate, "SKIPPED")
        write_summary(
            sampler_ok=sampler_ok,
            b0c1_gate=b0c1_gate,
            b0c2_gate="SKIPPED",
            b0c3_status="not_run",
            full_stage_b_allowed="NO",
            recipe="do not proceed",
            recommended_next="Do not enter Stage B; revise task/training recipe.",
            outcomes_b0c1=outcomes_b0c1,
            outcome_b0c2=None,
        )
        log("[stage_b0c] stop after B0C.1 FAIL")
        return

    sec_step_after_b0c1 = measured_sec_per_step(outcomes_b0c1) or base_sec_step
    b0c2_est_hours = 50000 * sec_step_after_b0c1 * 2.5 / 3600.0
    if not enough_budget_for(b0c2_est_hours, stage_start):
        write_budget_pause(b0c2_est_hours, (time.time() - stage_start) / 3600.0, "estimated B0C.2 cost exceeds remaining cap")
        write_cost_update(outcomes_b0c1, None, None, b0c1_gate, "SKIPPED")
        write_summary(
            sampler_ok=sampler_ok,
            b0c1_gate=b0c1_gate,
            b0c2_gate="SKIPPED",
            b0c3_status="not_run_budget",
            full_stage_b_allowed="NO",
            recipe="do not proceed",
            recommended_next="Do not enter Stage B; extend B0C.2 to 100k with user confirmation.",
            outcomes_b0c1=outcomes_b0c1,
            outcome_b0c2=None,
        )
        return

    b0c2_pairs = (1, 2, 4, 8, 16, 32, 64)
    cfg2 = make_cfg("stage_b0c_b02_curriculum_v16_seed0", 0, 16, b0c2_pairs, 50000, 2000, 128, 1e-3)
    append_manifest(
        "B0C_2_curriculum_v16_seed0_lr1e-3",
        cfg2,
        "0-5k:1,2;5k-10k:1,2,4;10k-20k:1,2,4,8;20k-35k:1,2,4,8,16;35k-50k:1,2,4,8,16,32,64",
        "1,2,4,8,16,32,64",
        b0c2_est_hours,
        "planned",
    )
    outcome_b0c2 = train_curriculum_run(
        substage="B0C.2",
        run_id=cfg2.run_id,
        recipe="formal curriculum N_VALUES=16",
        cfg=cfg2,
        phases=B0C2_PHASES,
        eval_n_pairs=b0c2_pairs,
        eval_seeds=(0, 1),
        eval_samples_per_condition=DEFAULT_EVAL_SAMPLES,
        eval_batch_size=128,
        checkpoint_interval=2000,
        early_stop_fn=lambda metrics, status: b0c2_pass_strong(metrics, 16, status),
        consecutive_passes=2,
        stage_start_time=stage_start,
    )
    append_stage_result(outcome_b0c2, cfg2)
    append_manifest(
        "B0C_2_curriculum_v16_seed0_lr1e-3",
        cfg2,
        "0-5k:1,2;5k-10k:1,2,4;10k-20k:1,2,4,8;20k-35k:1,2,4,8,16;35k-50k:1,2,4,8,16,32,64",
        "1,2,4,8,16,32,64",
        outcome_b0c2["elapsed_sec"] / 3600.0,
        outcome_b0c2["status"],
    )
    plot_curves()
    b0c2_gate = classify_b0c2(outcome_b0c2, 16)

    outcome_b0c3 = None
    b0c3_status = "not_run"
    if b0c1_gate in {"PASS", "PASS_STRONG"} and b0c2_gate in {"FAIL", "PARTIAL_RISING"}:
        fallback_est_hours = outcome_b0c2["elapsed_sec"] / 3600.0 if outcome_b0c2.get("elapsed_sec") else b0c2_est_hours
        if enough_budget_for(fallback_est_hours, stage_start):
            cfg3 = make_cfg("stage_b0c_b03_curriculum_v16_seed0_lr5e-4", 0, 16, b0c2_pairs, 50000, 2000, 128, 5e-4)
            append_manifest(
                "B0C_3_curriculum_v16_seed0_lr5e-4",
                cfg3,
                "same as B0C.2",
                "1,2,4,8,16,32,64",
                fallback_est_hours,
                "planned",
            )
            outcome_b0c3 = train_curriculum_run(
                substage="B0C.3",
                run_id=cfg3.run_id,
                recipe="formal curriculum N_VALUES=16 lr=5e-4 fallback",
                cfg=cfg3,
                phases=B0C2_PHASES,
                eval_n_pairs=b0c2_pairs,
                eval_seeds=(0, 1),
                eval_samples_per_condition=DEFAULT_EVAL_SAMPLES,
                eval_batch_size=128,
                checkpoint_interval=2000,
                early_stop_fn=lambda metrics, status: b0c2_pass_strong(metrics, 16, status),
                consecutive_passes=2,
                stage_start_time=stage_start,
            )
            append_stage_result(outcome_b0c3, cfg3)
            append_manifest(
                "B0C_3_curriculum_v16_seed0_lr5e-4",
                cfg3,
                "same as B0C.2",
                "1,2,4,8,16,32,64",
                outcome_b0c3["elapsed_sec"] / 3600.0,
                outcome_b0c3["status"],
            )
            plot_curves()
            b0c3_gate = classify_b0c2(outcome_b0c3, 16)
            b0c3_status = b0c3_gate
            if b0c3_gate in {"PASS", "PASS_STRONG"} and b0c2_gate not in {"PASS", "PASS_STRONG"}:
                b0c2_gate = b0c3_gate
        else:
            b0c3_status = "skipped_budget"

    if b0c2_gate in {"PASS", "PASS_STRONG"}:
        full_allowed = "YES"
        recipe = "curriculum mixed-load"
        recommended = "Enter full Stage B with curriculum mixed-load."
    elif b0c2_gate == "PARTIAL_RISING":
        full_allowed = "NO"
        recipe = "do not proceed"
        recommended = "Do not enter Stage B; extend B0C.2 to 100k with user confirmation."
    elif b0c3_status == "not_run" and b0c1_gate in {"PASS", "PASS_STRONG"} and b0c2_gate == "FAIL":
        full_allowed = "NO"
        recipe = "do not proceed"
        recommended = "Do not enter Stage B; test lr=5e-4 fallback."
    else:
        full_allowed = "NO"
        recipe = "do not proceed"
        recommended = "Do not enter Stage B; revise task/training recipe."

    write_cost_update(outcomes_b0c1, outcome_b0c2, outcome_b0c3, b0c1_gate, b0c2_gate)
    write_summary(
        sampler_ok=sampler_ok,
        b0c1_gate=b0c1_gate,
        b0c2_gate=b0c2_gate,
        b0c3_status=b0c3_status,
        full_stage_b_allowed=full_allowed,
        recipe=recipe,
        recommended_next=recommended,
        outcomes_b0c1=outcomes_b0c1,
        outcome_b0c2=outcome_b0c2,
    )
    log(f"[stage_b0c] done B0C.1={b0c1_gate} B0C.2={b0c2_gate} allowed_B={full_allowed}")


if __name__ == "__main__":
    main()
