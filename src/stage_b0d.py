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

from data import TaskConfig, Vocab, answer_metrics, collate, random_baselines
from models_transformer import CausalTransformerLM, parameter_count
from train import TrainConfig, lr_at_step
from utils import grad_norm, seed_everything


RUN_DIR = Path(os.environ.get("RUN_DIR", ".")).resolve()
for subdir in ["results", "plots", "logs", "checkpoints", "summaries", "configs", "status", "debug"]:
    (RUN_DIR / subdir).mkdir(parents=True, exist_ok=True)

CAP_HOURS = float(os.environ.get("MAX_GPU_HOURS_STAGE_B0D", "4"))
DEFAULT_EVAL_SAMPLES = int(os.environ.get("B0D_EVAL_SAMPLES", "8192"))
RESULTS_CSV = RUN_DIR / "results" / "stage_b0d_results.csv"

SUMMARY_FIELDS = [
    "substage",
    "eval_mode",
    "n_values",
    "q_count_mode",
    "q_cap",
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
    "all_correct_N8",
    "all_correct_N16",
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
class DenseTaskConfig:
    n_pairs: int
    n_values: int = 8
    n_keys: int = 256
    n_noise: int = 256
    gap_len: int = 0
    q_count_mode: str = "capped"
    q_cap: int = 8
    seed: int = 0

    def vocab(self) -> Vocab:
        return Vocab(n_keys=self.n_keys, n_values=self.n_values, n_noise=self.n_noise)

    def q_count(self) -> int:
        if self.q_count_mode == "single":
            return 1
        if self.q_count_mode == "all":
            return self.n_pairs
        if self.q_count_mode == "capped":
            return min(self.n_pairs, self.q_cap)
        raise ValueError(f"unknown q_count_mode={self.q_count_mode}")


@dataclass(frozen=True)
class CurriculumPhase:
    start_step: int
    end_step: int
    train_n_pairs: tuple[int, ...]
    q_count_mode: str = "capped"
    q_cap: int = 8
    sampling_weights: tuple[float, ...] | None = None

    def contains(self, step: int) -> bool:
        return self.start_step < step <= self.end_step

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "train_n_pairs": list(self.train_n_pairs),
            "q_count_mode": self.q_count_mode,
            "q_cap": self.q_cap,
            "sampling_weights": list(self.sampling_weights) if self.sampling_weights is not None else "uniform",
        }


B0D1_PHASES = (
    CurriculumPhase(0, 5000, (1, 2)),
    CurriculumPhase(5000, 10000, (1, 2, 4)),
    CurriculumPhase(10000, 30000, (1, 2, 4, 8)),
)

B0D2_PHASES = (
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


def prior_sec_per_step(default: float = 0.036) -> float:
    rows = read_csv(RUN_DIR / "results" / "stage_b0c_results.csv")
    rates = []
    for row in rows:
        step = as_float(row.get("step"), 0.0)
        if step <= 0:
            continue
        run_id = f"stage_b0c_b01_curriculum_v8_seed{row.get('seed')}"
        status_path = RUN_DIR / "status" / f"{run_id}_status.json"
        if status_path.exists():
            try:
                elapsed = json.loads(status_path.read_text())["outcome"]["elapsed_sec"]
                rates.append(float(elapsed) / step)
            except Exception:
                pass
    return sum(rates) / len(rates) if rates else default


def append_manifest(job_id: str, cfg: TrainConfig, train_n_pairs: str, eval_n_pairs: str, estimated_hours: float, status: str) -> None:
    path = RUN_DIR / "results" / "run_manifest.csv"
    rows = [row for row in read_csv(path) if row.get("job_id") != job_id]
    rows.append(
        {
            "stage": "B0D",
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
            raise ValueError(f"non-contiguous phase starts at {phase.start_step}, expected {last_end}")
        current = set(phase.train_n_pairs)
        if not previous.issubset(current):
            raise ValueError(f"later phase dropped easier loads: previous={previous}, current={current}")
        if phase.sampling_weights is not None and len(phase.sampling_weights) != len(phase.train_n_pairs):
            raise ValueError("sampling weights length mismatch")
        previous = current
        last_end = phase.end_step


def sample_n_pairs(phase: CurriculumPhase, generator: torch.Generator) -> int:
    if phase.sampling_weights is None:
        idx = int(torch.randint(0, len(phase.train_n_pairs), (1,), generator=generator).item())
    else:
        weights = torch.tensor(phase.sampling_weights, dtype=torch.float32)
        idx = int(torch.multinomial(weights, 1, replacement=True, generator=generator).item())
    return int(phase.train_n_pairs[idx])


def build_dense_sample(cfg: DenseTaskConfig, *, generator: torch.Generator | None = None, corrupt: str | None = None) -> tuple[list[int], list[int], list[int], list[int]]:
    vocab = cfg.vocab()
    if cfg.n_pairs > cfg.n_keys:
        raise ValueError("n_pairs exceeds n_keys")
    q_count = cfg.q_count()
    if q_count > cfg.n_pairs:
        raise ValueError("q_count exceeds n_pairs")
    keys = torch.randperm(cfg.n_keys, generator=generator)[: cfg.n_pairs]
    vals = torch.randint(0, cfg.n_values, (cfg.n_pairs,), generator=generator)
    query_indices = torch.randperm(cfg.n_pairs, generator=generator)[:q_count]

    full = [vocab.bos]
    record_vals = vals.clone()
    if corrupt == "record_values_shuffled":
        record_vals = record_vals[torch.randperm(cfg.n_pairs, generator=generator)]
    if corrupt != "records_removed":
        for key, value in zip(keys.tolist(), record_vals.tolist()):
            full.extend([vocab.k_tok, vocab.key_id(int(key)), vocab.v_tok, vocab.value_id(int(value))])
    for _ in range(cfg.gap_len):
        noise = int(torch.randint(0, cfg.n_noise, (1,), generator=generator).item())
        full.append(vocab.noise_id(noise))

    query_keys = []
    answer_tokens = []
    for idx in query_indices.tolist():
        key = int(keys[idx].item())
        value = int(vals[idx].item())
        if corrupt == "query_key_randomized":
            key = int(torch.randint(0, cfg.n_keys, (1,), generator=generator).item())
        answer = vocab.value_id(value)
        if corrupt == "labels_shuffled":
            answer = vocab.value_id(int(torch.randint(0, cfg.n_values, (1,), generator=generator).item()))
        full.extend([vocab.q_tok, vocab.key_id(key), vocab.ans_tok, answer])
        query_keys.append(key)
        answer_tokens.append(answer)
    return full, answer_tokens, keys.tolist(), query_keys


def dense_sample_to_x_labels(full_tokens: list[int], vocab: Vocab) -> tuple[list[int], list[int]]:
    x = full_tokens[:-1]
    labels = [-100] * len(x)
    for idx, token in enumerate(x):
        if token == vocab.ans_tok:
            if idx + 1 >= len(full_tokens):
                raise ValueError("<ANS> cannot be last full token")
            labels[idx] = full_tokens[idx + 1]
    return x, labels


def make_dense_batch(
    cfg: DenseTaskConfig,
    batch_size: int,
    device: torch.device | str,
    *,
    generator: torch.Generator | None = None,
    corrupt: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    vocab = cfg.vocab()
    samples = []
    for _ in range(batch_size):
        full, _, _, _ = build_dense_sample(cfg, generator=generator, corrupt=corrupt)
        samples.append(dense_sample_to_x_labels(full, vocab))
    x, labels = collate(samples, vocab.pad)
    return x.to(device), labels.to(device)


def dense_decoded_example(cfg: DenseTaskConfig) -> str:
    vocab = cfg.vocab()
    full, answers, record_keys, query_keys = build_dense_sample(cfg, generator=torch.Generator(device="cpu").manual_seed(cfg.seed))
    x, labels = dense_sample_to_x_labels(full, vocab)
    lines = [
        f"n_pairs={cfg.n_pairs}",
        f"q_count_mode={cfg.q_count_mode}",
        f"q_cap={cfg.q_cap}",
        f"q_count={cfg.q_count()}",
        f"record_keys={record_keys}",
        f"query_keys={query_keys}",
        "full_tokens:",
        " ".join(vocab.decode_token(t) for t in full),
        "labels:",
        str(labels),
        "label_positions:",
        str([i for i, label in enumerate(labels) if label != -100]),
    ]
    return "\n".join(lines) + "\n"


def max_dense_seq_len(max_pairs: int, q_count_mode: str, q_cap: int, gap_len: int) -> int:
    if q_count_mode == "single":
        q_count = 1
    elif q_count_mode == "all":
        q_count = max_pairs
    else:
        q_count = min(max_pairs, q_cap)
    return 1 + 4 * max_pairs + gap_len + 4 * q_count - 1


def build_transformer(cfg: TrainConfig, max_pairs: int, q_count_mode: str, q_cap: int) -> torch.nn.Module:
    vocab = Vocab(n_keys=cfg.n_keys, n_values=cfg.n_values)
    return CausalTransformerLM(
        vocab_size=vocab.size,
        max_seq_len=max_dense_seq_len(max_pairs, q_count_mode, q_cap, cfg.gap_len),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
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
    eval_modes: tuple[tuple[str, str, int], ...],
    eval_seeds: tuple[int, ...],
    samples_per_condition: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[int, dict[str, float]]]:
    model.eval()
    outputs: dict[str, dict[int, dict[str, float]]] = {}
    per_seed_target = math.ceil(samples_per_condition / len(eval_seeds))
    for mode_name, q_count_mode, q_cap in eval_modes:
        outputs[mode_name] = {}
        for n_pairs in eval_n_pairs:
            total = 0
            ce_total = 0.0
            acc_total = 0.0
            all_correct_total = 0.0
            for eval_seed in eval_seeds:
                generator = torch.Generator(device="cpu").manual_seed(930000 + cfg.seed * 10000 + eval_seed * 1000 + n_pairs + q_cap)
                task = DenseTaskConfig(
                    n_pairs=n_pairs,
                    n_values=cfg.n_values,
                    n_keys=cfg.n_keys,
                    gap_len=cfg.gap_len,
                    q_count_mode=q_count_mode,
                    q_cap=q_cap,
                    seed=eval_seed,
                )
                seen = 0
                while seen < per_seed_target and total < samples_per_condition:
                    current = min(batch_size, per_seed_target - seen, samples_per_condition - total)
                    x, labels = make_dense_batch(task, current, device, generator=generator)
                    metrics = answer_metrics(model(x), labels)
                    ce_total += metrics["cross_entropy"] * current
                    acc_total += metrics["per_answer_accuracy"] * current
                    all_correct_total += metrics["per_example_all_correct"] * current
                    seen += current
                    total += current
            outputs[mode_name][n_pairs] = {
                "accuracy": acc_total / total,
                "cross_entropy": ce_total / total,
                "all_correct": all_correct_total / total,
                "samples": float(total),
            }
    model.train()
    return outputs


def primary_scores(metrics: dict[str, dict[int, dict[str, float]]], mode_name: str = "dense_capped") -> tuple[float, float]:
    selected = metrics[mode_name]
    return (
        sum(row["accuracy"] for row in selected.values()) / len(selected),
        sum(row["cross_entropy"] for row in selected.values()) / len(selected),
    )


def final_rows_by_mode(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, dict[str, float]]] = {}
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    for row in sorted(eval_rows, key=lambda item: as_int(item.get("step"))):
        mode = str(row.get("eval_mode"))
        out.setdefault(mode, {})[as_int(row.get("n_pairs"))] = {
            "accuracy": as_float(row.get("accuracy")),
            "cross_entropy": as_float(row.get("cross_entropy")),
            "all_correct": as_float(row.get("all_correct")),
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


def b0d1_early_stop(metrics: dict[str, dict[int, dict[str, float]]], status: str) -> bool:
    dense = metrics["dense_capped"]
    single = metrics["single_query"]
    return (
        status not in {"fail_non_finite_loss", "fail_grad_norm", "budget_pause_actual"}
        and dense.get(1, {}).get("accuracy", 0.0) >= 0.99
        and dense.get(2, {}).get("accuracy", 0.0) >= 0.99
        and dense.get(4, {}).get("accuracy", 0.0) >= 0.98
        and dense.get(8, {}).get("accuracy", 0.0) >= 0.95
        and single.get(1, {}).get("accuracy", 0.0) >= 0.99
        and single.get(2, {}).get("accuracy", 0.0) >= 0.99
        and single.get(4, {}).get("accuracy", 0.0) >= 0.95
        and single.get(8, {}).get("accuracy", 0.0) >= 0.85
    )


def b0d2_early_stop(metrics: dict[str, dict[int, dict[str, float]]], status: str, n_values: int) -> bool:
    dense = metrics["dense_capped"]
    return (
        status not in {"fail_non_finite_loss", "fail_grad_norm", "budget_pause_actual"}
        and dense.get(1, {}).get("accuracy", 0.0) >= 0.99
        and dense.get(2, {}).get("accuracy", 0.0) >= 0.99
        and dense.get(4, {}).get("accuracy", 0.0) >= 0.98
        and dense.get(8, {}).get("accuracy", 0.0) >= 0.95
        and dense.get(16, {}).get("accuracy", 0.0) >= 0.80
        and dense.get(16, {}).get("cross_entropy", float("inf")) < math.log(n_values) - 0.05
    )


def train_dense_run(
    *,
    substage: str,
    run_id: str,
    recipe: str,
    cfg: TrainConfig,
    phases: tuple[CurriculumPhase, ...],
    eval_n_pairs: tuple[int, ...],
    eval_modes: tuple[tuple[str, str, int], ...],
    eval_seeds: tuple[int, ...],
    eval_samples_per_condition: int,
    eval_batch_size: int,
    checkpoint_interval: int,
    early_stop_fn: Callable[[dict[str, dict[int, dict[str, float]]], str], bool] | None,
    consecutive_passes: int,
    stage_start_time: float,
) -> dict[str, Any]:
    validate_phases(phases)
    status_path = RUN_DIR / "status" / f"{run_id}_status.json"
    existing = json.loads(status_path.read_text()) if status_path.exists() else {}
    if existing.get("terminal"):
        log(f"[{substage}] skip completed {run_id}: {existing.get('status')}")
        return existing["outcome"]

    seed_everything(cfg.seed)
    device = torch.device("cuda")
    train_generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 22345)
    model = build_transformer(cfg, max(eval_n_pairs), "capped", 8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    best = {
        "best_acc_score": 0.0,
        "best_acc_step": 0,
        "best_ce_score": float("inf"),
        "best_ce_step": 0,
        "by_mode": {},
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
        "eval_modes": [{"name": name, "q_count_mode": mode, "q_cap": cap} for name, mode, cap in eval_modes],
        "eval_seeds": list(eval_seeds),
        "eval_samples_per_condition": eval_samples_per_condition,
        "eval_batch_size": eval_batch_size,
        "train_generator_seed": cfg.seed + 22345,
        "eval_seed_rule": "930000 + cfg.seed * 10000 + eval_seed * 1000 + n_pairs + q_cap",
        "rng_policy": "train uses one CPU torch.Generator; eval uses fresh independent CPU generators",
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
    start = time.time()
    latest_step = start_step
    status = "completed"
    notes = ""
    pass_count = 0
    random_acc, random_ce = random_baselines(cfg.n_values)

    for step in range(start_step + 1, cfg.max_steps + 1):
        latest_step = step
        if (time.time() - stage_start_time) / 3600.0 > CAP_HOURS:
            status = "budget_pause_actual"
            notes = f"actual Stage B0D elapsed exceeded cap {CAP_HOURS:.2f} GPU-hours"
            break
        for group in opt.param_groups:
            group["lr"] = lr_at_step(cfg, step)
        phase = phase_for_step(phases, step)
        n_pairs = sample_n_pairs(phase, train_generator)
        task = DenseTaskConfig(
            n_pairs=n_pairs,
            n_values=cfg.n_values,
            n_keys=cfg.n_keys,
            gap_len=cfg.gap_len,
            q_count_mode=phase.q_count_mode,
            q_cap=phase.q_cap,
            seed=cfg.seed,
        )
        x, labels = make_dense_batch(task, cfg.batch_size, device, generator=train_generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            status = "fail_non_finite_loss"
            notes = f"loss={float(loss.item())}"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_norm(model.parameters())
        if not math.isfinite(gnorm) or gnorm > 1000:
            status = "fail_grad_norm"
            notes = f"grad_norm={gnorm}"
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step == 1 or step % cfg.log_interval == 0:
            metrics = answer_metrics(logits.detach(), labels)
            append_run_rows(
                per_run_csv,
                [
                    {
                        "substage": substage,
                        "run_id": run_id,
                        "recipe": recipe,
                        "split": "train",
                        "eval_mode": "train_dense_capped",
                        "step": step,
                        "n_pairs": n_pairs,
                        "q_count_mode": phase.q_count_mode,
                        "q_cap": phase.q_cap,
                        "phase_train_n_pairs": ",".join(str(v) for v in phase.train_n_pairs),
                        "accuracy": metrics["per_answer_accuracy"],
                        "all_correct": metrics["per_example_all_correct"],
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
            metrics_by_mode = evaluate_model(
                model,
                cfg,
                eval_n_pairs=eval_n_pairs,
                eval_modes=eval_modes,
                eval_seeds=eval_seeds,
                samples_per_condition=eval_samples_per_condition,
                batch_size=eval_batch_size,
                device=device,
            )
            acc_score, ce_score = primary_scores(metrics_by_mode, "dense_capped")
            serial_rows = []
            for mode_name, mode_metrics in metrics_by_mode.items():
                mode_tuple = next(mode for mode in eval_modes if mode[0] == mode_name)
                for n, metrics in mode_metrics.items():
                    best["by_mode"].setdefault(mode_name, {}).setdefault(str(n), {"accuracy": 0.0, "cross_entropy": float("inf"), "step": 0})
                    existing_best = best["by_mode"][mode_name][str(n)]
                    if metrics["accuracy"] > existing_best["accuracy"] or metrics["cross_entropy"] < existing_best["cross_entropy"]:
                        best["by_mode"][mode_name][str(n)] = {
                            "accuracy": max(existing_best["accuracy"], metrics["accuracy"]),
                            "cross_entropy": min(existing_best["cross_entropy"], metrics["cross_entropy"]),
                            "all_correct": metrics["all_correct"],
                            "step": step,
                        }
                    row = {
                        "substage": substage,
                        "run_id": run_id,
                        "recipe": recipe,
                        "split": "eval",
                        "eval_mode": mode_name,
                        "step": step,
                        "n_pairs": n,
                        "q_count_mode": mode_tuple[1],
                        "q_cap": mode_tuple[2],
                        "phase_train_n_pairs": ",".join(str(v) for v in phase.train_n_pairs),
                        "accuracy": metrics["accuracy"],
                        "all_correct": metrics["all_correct"],
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
            save_json(status_path, {"terminal": False, "status": "running", "step": step, "run_meta": run_meta, "best": best, "updated_at": time.time()})
            dense = metrics_by_mode["dense_capped"]
            single = metrics_by_mode["single_query"]
            log(
                f"[{substage}] {run_id} step={step} phase={phase.train_n_pairs} "
                f"denseN8={dense.get(8, {}).get('accuracy', float('nan')):.4f}/{dense.get(8, {}).get('cross_entropy', float('nan')):.3f} "
                f"singleN8={single.get(8, {}).get('accuracy', float('nan')):.4f}/{single.get(8, {}).get('cross_entropy', float('nan')):.3f}"
            )

            if early_stop_fn is not None and early_stop_fn(metrics_by_mode, status):
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
        "finals": final_rows_by_mode(rows),
        "best": best,
        "per_run_csv": str(per_run_csv),
        "config_json": str(RUN_DIR / "configs" / f"{run_id}_config.json"),
    }
    save_json(status_path, {"terminal": True, "status": status, "outcome": outcome, "run_meta": run_meta, "updated_at": time.time()})
    return outcome


def result_summary_row(outcome: dict[str, Any], cfg: TrainConfig, eval_mode: str) -> dict[str, Any]:
    finals = outcome.get("finals", {}).get(eval_mode, {})
    random_acc, random_ce = random_baselines(cfg.n_values)

    def metric(n: int, name: str) -> str:
        return f"{as_float(finals.get(n, {}).get(name)):.4f}" if n in finals else ""

    return {
        "substage": outcome["substage"],
        "eval_mode": eval_mode,
        "n_values": cfg.n_values,
        "q_count_mode": "capped" if eval_mode == "dense_capped" else "single",
        "q_cap": 8 if eval_mode == "dense_capped" else 1,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "step": outcome["step"],
        "N=1 acc": metric(1, "accuracy"),
        "N=2 acc": metric(2, "accuracy"),
        "N=4 acc": metric(4, "accuracy"),
        "N=8 acc": metric(8, "accuracy"),
        "N=16 acc": metric(16, "accuracy"),
        "N=32 acc": metric(32, "accuracy"),
        "N=64 acc": metric(64, "accuracy"),
        "CE_N8": metric(8, "cross_entropy"),
        "CE_N16": metric(16, "cross_entropy"),
        "all_correct_N8": metric(8, "all_correct"),
        "all_correct_N16": metric(16, "all_correct"),
        "random_acc": f"{random_acc:.4f}",
        "random_CE": f"{random_ce:.4f}",
        "status": outcome["status"],
    }


def append_stage_results(outcome: dict[str, Any], cfg: TrainConfig) -> None:
    for mode in ["dense_capped", "single_query"]:
        append_csv(RESULTS_CSV, result_summary_row(outcome, cfg, mode), SUMMARY_FIELDS)


def seed_pass_strong(outcome: dict[str, Any]) -> bool:
    dense = outcome.get("finals", {}).get("dense_capped", {})
    single = outcome.get("finals", {}).get("single_query", {})
    return (
        as_float(dense.get(1, {}).get("accuracy"), 0.0) >= 0.99
        and as_float(dense.get(2, {}).get("accuracy"), 0.0) >= 0.99
        and as_float(dense.get(4, {}).get("accuracy"), 0.0) >= 0.98
        and as_float(dense.get(8, {}).get("accuracy"), 0.0) >= 0.95
        and as_float(single.get(8, {}).get("accuracy"), 0.0) >= 0.85
        and outcome.get("status") not in {"fail_non_finite_loss", "fail_grad_norm", "budget_pause_actual"}
    )


def seed_weak_pass(outcome: dict[str, Any]) -> bool:
    dense = outcome.get("finals", {}).get("dense_capped", {})
    single = outcome.get("finals", {}).get("single_query", {})
    return (
        as_float(dense.get(8, {}).get("accuracy"), 0.0) >= 0.85
        and as_float(single.get(8, {}).get("accuracy"), 0.0) >= 0.70
        and as_float(dense.get(8, {}).get("cross_entropy"), float("inf")) < math.log(8) - 0.05
    )


def classify_b0d1(outcomes: list[dict[str, Any]]) -> str:
    strong = sum(1 for outcome in outcomes if seed_pass_strong(outcome))
    if strong == 3:
        return "PASS_STRONG"
    if strong >= 2 and all(seed_pass_strong(outcome) or seed_weak_pass(outcome) for outcome in outcomes):
        return "PASS"
    dense_ok = []
    single_bad = []
    for outcome in outcomes:
        dense = outcome.get("finals", {}).get("dense_capped", {})
        single = outcome.get("finals", {}).get("single_query", {})
        dense_pass = (
            as_float(dense.get(1, {}).get("accuracy"), 0.0) >= 0.99
            and as_float(dense.get(2, {}).get("accuracy"), 0.0) >= 0.99
            and as_float(dense.get(4, {}).get("accuracy"), 0.0) >= 0.98
            and as_float(dense.get(8, {}).get("accuracy"), 0.0) >= 0.95
        )
        dense_ok.append(dense_pass)
        single_bad.append(as_float(single.get(8, {}).get("accuracy"), 0.0) < 0.70)
    if len(outcomes) == 3 and all(dense_ok) and any(single_bad):
        return "FORMAT_SPECIFIC"
    dense_n8_high = sum(1 for outcome in outcomes if as_float(outcome.get("finals", {}).get("dense_capped", {}).get(8, {}).get("accuracy"), 0.0) >= 0.85)
    if dense_n8_high < 2:
        return "FAIL"
    return "FAIL"


def improved_last_10k(run_id: str, n_pairs: int, mode: str) -> bool:
    rows = [
        row
        for row in read_csv(RUN_DIR / "results" / f"{run_id}.csv")
        if row.get("split") == "eval" and row.get("eval_mode") == mode and as_int(row.get("n_pairs")) == n_pairs
    ]
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


def classify_b0d2(outcome: dict[str, Any] | None, n_values: int) -> str:
    if outcome is None:
        return "SKIPPED"
    dense = outcome.get("finals", {}).get("dense_capped", {})
    status = outcome.get("status", "")

    def acc(n: int) -> float:
        return as_float(dense.get(n, {}).get("accuracy"), 0.0)

    def ce(n: int) -> float:
        return as_float(dense.get(n, {}).get("cross_entropy"), float("inf"))

    stable = status not in {"fail_non_finite_loss", "fail_grad_norm", "budget_pause_actual"}
    if stable and acc(1) >= 0.99 and acc(2) >= 0.99 and acc(4) >= 0.98 and acc(8) >= 0.95 and acc(16) >= 0.80 and ce(16) < math.log(n_values) - 0.05:
        return "PASS_STRONG"
    if stable and acc(1) >= 0.98 and acc(2) >= 0.98 and acc(4) >= 0.95 and acc(8) >= 0.90 and acc(16) >= 0.60 and ce(16) < math.log(n_values) - 0.05:
        return "PASS"
    if stable and 0.75 <= acc(8) < 0.90 and improved_last_10k(outcome["run_id"], 8, "dense_capped") and ce(16) < math.log(n_values) - 0.05:
        return "PARTIAL_RISING"
    return "FAIL"


def unit_tests() -> dict[str, Any]:
    q_counts = {}
    label_positions_by_n = {}
    for n_pairs in [1, 2, 4, 8, 16]:
        cfg = DenseTaskConfig(n_pairs=n_pairs, q_count_mode="capped", q_cap=8, n_values=8, seed=n_pairs)
        full, _, record_keys, query_keys = build_dense_sample(cfg, generator=torch.Generator(device="cpu").manual_seed(n_pairs))
        x, labels = dense_sample_to_x_labels(full, cfg.vocab())
        q_counts[n_pairs] = cfg.q_count()
        assert cfg.q_count() == min(n_pairs, 8)
        assert len(query_keys) == len(set(query_keys))
        assert set(query_keys).issubset(set(record_keys))
        label_positions = [idx for idx, label in enumerate(labels) if label != -100]
        label_positions_by_n[n_pairs] = label_positions
        assert len(label_positions) == cfg.q_count()
        for pos in label_positions:
            assert x[pos] == cfg.vocab().ans_tok
            assert cfg.vocab().value_base <= labels[pos] < cfg.vocab().noise_base

    source_files = [Path(__file__), RUN_DIR / "src" / "models_transformer.py"]
    final_head_patterns = ("return self.head(h[:, -1])", "return self.lm_head(h[:, -1])", "hidden[:, -1]")
    offenders = []
    for path in source_files:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if "final_head_patterns" in line or "offenders" in line:
                continue
            if any(pattern in line for pattern in final_head_patterns):
                offenders.append(f"{path}:{line_no}:{line.strip()}")
    assert not offenders

    torch.manual_seed(0)
    model = CausalTransformerLM(vocab_size=64, max_seq_len=10, d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    model.eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) % 64
    b = a.clone()
    b[:, 5:] = torch.tensor([[31, 32, 33, 34, 35]]) % 64
    with torch.no_grad():
        causal_delta = torch.max(torch.abs(model(a)[:, 4, :] - model(b)[:, 4, :])).item()
    assert causal_delta < 1e-5

    corruptions = ["query_key_randomized", "record_values_shuffled", "records_removed", "labels_shuffled"]
    for corruption in corruptions:
        cfg = DenseTaskConfig(n_pairs=8, q_count_mode="capped", q_cap=8, n_values=8)
        full, _, _, _ = build_dense_sample(cfg, generator=torch.Generator(device="cpu").manual_seed(123), corrupt=corruption)
        x, labels = dense_sample_to_x_labels(full, cfg.vocab())
        assert int(sum(1 for label in labels if label != -100)) == cfg.q_count()

    return {
        "q_counts": q_counts,
        "label_positions": label_positions_by_n,
        "causal_delta": causal_delta,
        "corrupted_eval_tests_available": corruptions,
    }


def run_sampler_audit() -> bool:
    validate_phases(B0D1_PHASES)
    validate_phases(B0D2_PHASES)
    test_info = unit_tests()

    def sample_fps(with_eval: bool) -> tuple[list[str], list[int]]:
        train_gen = torch.Generator(device="cpu").manual_seed(22345)
        fps = []
        choices = []
        for step in range(1, 41):
            phase = phase_for_step(B0D1_PHASES, min(step * 750, 30000))
            n_pairs = sample_n_pairs(phase, train_gen)
            choices.append(n_pairs)
            task = DenseTaskConfig(n_pairs=n_pairs, n_values=8, q_count_mode=phase.q_count_mode, q_cap=phase.q_cap)
            x, labels = make_dense_batch(task, 4, "cpu", generator=train_gen)
            fps.append(tensor_fingerprint(x, labels))
            if with_eval and step % 10 == 0:
                eval_gen = torch.Generator(device="cpu").manual_seed(930000 + step)
                _ = make_dense_batch(DenseTaskConfig(n_pairs=8, n_values=8, q_count_mode="single", q_cap=1), 4, "cpu", generator=eval_gen)
        return fps, choices

    fp_a, choices_a = sample_fps(False)
    fp_b, choices_b = sample_fps(True)
    rng_ok = fp_a == fp_b and choices_a == choices_b
    examples = {
        "n_pairs_4": dense_decoded_example(DenseTaskConfig(n_pairs=4, q_count_mode="capped", q_cap=8, n_values=8, seed=4)),
        "n_pairs_8": dense_decoded_example(DenseTaskConfig(n_pairs=8, q_count_mode="capped", q_cap=8, n_values=8, seed=8)),
    }
    phase_examples = {
        str(step): list(phase_for_step(B0D2_PHASES, step).train_n_pairs)
        for step in [1, 5000, 5001, 10000, 10001, 20000, 20001, 35000, 35001, 50000]
    }
    lines = ["# Stage B0D Sampler Audit\n\n"]
    lines.append("## Dense Multi-Query Format\n\n")
    lines.append("- `single`: q_count = 1\n")
    lines.append("- `all`: q_count = n_pairs\n")
    lines.append("- `capped`: q_count = min(n_pairs, q_cap)\n")
    lines.append("- B0D default: q_count_mode=capped, q_cap=8\n\n")
    lines.append("## Curriculum Phases\n\n")
    lines.append("### B0D.1\n\n```json\n" + json.dumps([phase.as_jsonable() for phase in B0D1_PHASES], indent=2) + "\n```\n\n")
    lines.append("### B0D.2\n\n```json\n" + json.dumps([phase.as_jsonable() for phase in B0D2_PHASES], indent=2) + "\n```\n\n")
    lines.append("## Decoded Examples\n\n")
    for name, text in examples.items():
        lines.append(f"### {name}\n\n```text\n{text}```\n\n")
    lines.append("## q_count Counts And Label Positions\n\n")
    lines.append("```json\n" + json.dumps({"q_counts": test_info["q_counts"], "label_positions": test_info["label_positions"]}, indent=2) + "\n```\n\n")
    lines.append("## Phase Boundary Proof\n\n")
    lines.append("```json\n" + json.dumps(phase_examples, indent=2) + "\n```\n\n")
    lines.append("## Required Checks\n\n")
    lines.append("- q_count test: PASS\n")
    lines.append("- unique query test: PASS\n")
    lines.append("- label alignment test: PASS\n")
    lines.append("- answer-loss count test: PASS\n")
    lines.append("- no final-position classifier test: PASS\n")
    lines.append(f"- causal invariance test: PASS; causal_delta={test_info['causal_delta']:.8f}\n")
    lines.append(f"- corrupted eval tests available: {test_info['corrupted_eval_tests_available']}\n")
    lines.append(f"- train/eval RNG remain isolated: {'PASS' if rng_ok else 'FAIL'}\n")
    lines.append(f"- run config and commit hash are recorded: PASS; current_commit={git_hash()}\n\n")
    lines.append("## Conclusion\n\n")
    lines.append(f"- {'PASS' if rng_ok else 'FAIL'}\n")
    write_text(RUN_DIR / "summaries" / "stage_b0d_sampler_audit.md", "".join(lines))
    return rng_ok


def plot_curves() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, str]] = []
    for path in sorted((RUN_DIR / "results").glob("stage_b0d_*.csv")):
        if path.name == "stage_b0d_results.csv":
            continue
        rows.extend(read_csv(path))
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    if not eval_rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    mode_to_col = {"dense_capped": 0, "single_query": 1}
    for mode, col in mode_to_col.items():
        mode_rows = [row for row in eval_rows if row.get("eval_mode") == mode]
        for run_id in sorted({row.get("run_id", "") for row in mode_rows}):
            for n in [1, 2, 4, 8, 16, 32, 64]:
                points = [row for row in mode_rows if row.get("run_id") == run_id and as_int(row.get("n_pairs")) == n]
                if not points:
                    continue
                points = sorted(points, key=lambda row: as_int(row.get("step")))
                label = f"{run_id} N={n}"
                alpha = 1.0 if n in {8, 16} else 0.55
                axes[0][col].plot([as_int(row.get("step")) for row in points], [as_float(row.get("accuracy")) for row in points], label=label, alpha=alpha)
                axes[1][col].plot([as_int(row.get("step")) for row in points], [as_float(row.get("cross_entropy")) for row in points], label=label, alpha=alpha)
        axes[0][col].set_title(f"{mode}: accuracy")
        axes[1][col].set_title(f"{mode}: cross-entropy")
        axes[0][col].axhline(1 / 8, color="black", linestyle=":", linewidth=1, label="random acc V=8")
        axes[0][col].axhline(1 / 16, color="gray", linestyle=":", linewidth=1, label="random acc V=16")
        axes[1][col].axhline(math.log(8), color="black", linestyle=":", linewidth=1, label="random CE ln(8)")
        axes[1][col].axhline(math.log(16), color="gray", linestyle=":", linewidth=1, label="random CE ln(16)")
    for row in axes:
        for axis in row:
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=5, ncol=2)
    axes[1][0].set_xlabel("step")
    axes[1][1].set_xlabel("step")
    axes[0][0].set_ylabel("per-answer accuracy")
    axes[1][0].set_ylabel("cross-entropy")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "plots" / "stage_b0d_dense_query_learning_curves.png", dpi=160)
    fig.savefig(RUN_DIR / "plots" / "stage_b0d_dense_query_learning_curves.pdf")
    plt.close(fig)


def measured_sec_per_step(outcomes: list[dict[str, Any]]) -> float | None:
    rates = [out["elapsed_sec"] / max(1, out["step"]) for out in outcomes if out.get("step") and out.get("elapsed_sec")]
    return sum(rates) / len(rates) if rates else None


def write_budget_pause(estimated_hours: float, actual_hours: float, reason: str) -> None:
    write_text(
        RUN_DIR / "summaries" / "stage_b0d_budget_pause.md",
        f"""# Stage B0D Budget Pause

- Estimated GPU-hours: {estimated_hours:.3f}
- Actual elapsed GPU-hours: {actual_hours:.3f}
- Cap: {CAP_HOURS:.3f}
- Reason: {reason}
- No full Stage B or Stage C was launched.
""",
    )


def write_cost_update(outcomes_b0d1: list[dict[str, Any]], outcome_b0d2: dict[str, Any] | None, outcome_b0d3: dict[str, Any] | None, b0d1_gate: str, b0d2_gate: str) -> None:
    b0d1_sec = measured_sec_per_step(outcomes_b0d1)
    b0d2_sec = measured_sec_per_step([outcome_b0d2] if outcome_b0d2 else [])
    fallback_sec = measured_sec_per_step([outcome_b0d3] if outcome_b0d3 else [])
    best_sec = b0d2_sec or b0d1_sec or prior_sec_per_step()
    b1_steps = 3 * 20000 + 80000
    b2_steps = 2 * 100000
    stage_b_hours = (b1_steps + b2_steps) * best_sec / 3600.0
    actual_hours = sum(out.get("elapsed_sec", 0.0) for out in outcomes_b0d1 + ([outcome_b0d2] if outcome_b0d2 else []) + ([outcome_b0d3] if outcome_b0d3 else [])) / 3600.0
    n64_seq = max_dense_seq_len(64, "capped", 8, 0)
    lines = ["# Stage B0D Cost Update\n\n"]
    lines.append(f"- Measured sec/step for B0D.1: {b0d1_sec:.5f}\n" if b0d1_sec else "- Measured sec/step for B0D.1: unavailable\n")
    lines.append(f"- Measured sec/step for B0D.2: {b0d2_sec:.5f}\n" if b0d2_sec else "- Measured sec/step for B0D.2: not run\n")
    lines.append(f"- Measured sec/step for B0D.3 fallback: {fallback_sec:.5f}\n" if fallback_sec else "- Measured sec/step for B0D.3 fallback: not run\n")
    lines.append(f"- Actual Stage B0D elapsed GPU-hours: {actual_hours:.3f}\n")
    lines.append(f"- Estimated full Stage B cost under dense capped multi-query curriculum: {stage_b_hours:.3f} GPU-hours; one model covers all eval_n_pairs.\n")
    lines.append("- Estimated full Stage C cost under same recipe: unavailable until Mamba dense sequence throughput probe.\n")
    lines.append("- Mamba throughput probe should be refreshed: YES, because q_cap=8 increases sequence length and answer positions per sample.\n")
    lines.append(f"- q_cap=8 N=64 x-sequence length: {n64_seq}; acceptable for 24GB 4090 with conservative batch sizing, but Mamba/Transformer throughput should be remeasured before Stage B/C.\n")
    lines.append(f"- Current B0D cap likely sufficient: {'YES' if actual_hours <= CAP_HOURS else 'NO'}\n")
    lines.append(f"- B0D.1 gate: {b0d1_gate}\n")
    lines.append(f"- B0D.2 gate: {b0d2_gate}\n")
    write_text(RUN_DIR / "summaries" / "stage_b0d_cost_update.md", "".join(lines))


def table_rows() -> str:
    rows = read_csv(RESULTS_CSV)
    if not rows:
        return "| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | not_run |"
    lines = []
    for row in rows:
        lines.append(
            f"| {row.get('substage','')} | {row.get('eval_mode','')} | {row.get('n_values','')} | {row.get('q_count_mode','')} | {row.get('q_cap','')} | {row.get('lr','')} | {row.get('seed','')} | {row.get('step','')} | "
            f"{row.get('N=1 acc','')} | {row.get('N=2 acc','')} | {row.get('N=4 acc','')} | {row.get('N=8 acc','')} | {row.get('N=16 acc','')} | {row.get('N=32 acc','')} | {row.get('N=64 acc','')} | "
            f"{row.get('CE_N8','')} | {row.get('CE_N16','')} | {row.get('all_correct_N8','')} | {row.get('all_correct_N16','')} | {row.get('random_acc','')} | {row.get('random_CE','')} | {row.get('status','')} |"
        )
    return "\n".join(lines)


def write_summary(
    *,
    sampler_ok: bool,
    b0d1_gate: str,
    b0d2_gate: str,
    b0d3_status: str,
    formal_task: str,
    stage_b_recipe: str,
    full_stage_b_allowed: str,
    recommended_next: str,
) -> None:
    stabilized = "YES" if b0d1_gate in {"PASS", "PASS_STRONG"} else "FORMAT-SPECIFIC" if b0d1_gate == "FORMAT_SPECIFIC" else "NO"
    flexible = "YES" if b0d1_gate in {"PASS", "PASS_STRONG", "FORMAT_SPECIFIC"} else "NO"
    transfer = "YES" if b0d1_gate in {"PASS", "PASS_STRONG"} else "WEAK" if b0d1_gate == "FORMAT_SPECIFIC" else "NO"
    formal_viable = "YES" if b0d2_gate in {"PASS", "PASS_STRONG"} else "NOT YET" if b0d2_gate == "PARTIAL_RISING" else "NO"
    summary = f"""# Stage B0D Summary

## Result

- Dense multi-query sampler audit: {'PASS' if sampler_ok else 'FAIL'}
- B0D.1 N_VALUES=8 dense reproducibility: {b0d1_gate}
- B0D.2 N_VALUES=16 formal dense pilot: {b0d2_gate}
- B0D.3 LR fallback, if run: {b0d3_status}
- Cost update: complete

## Gate

- B0D.1:
  {b0d1_gate}

- B0D.2:
  {b0d2_gate}

- Whether full Stage B is allowed:
  {full_stage_b_allowed}

- Recommended formal task:
  {formal_task}

- Recommended Stage B recipe:
  {stage_b_recipe}

## Key metrics

| substage | eval_mode | n_values | q_count_mode | q_cap | lr | seed | step | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=16 acc | N=32 acc | N=64 acc | CE_N8 | CE_N16 | all_correct_N8 | all_correct_N16 | random_acc | random_CE | status |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{table_rows()}

## Interpretation

1. Did dense multi-query supervision stabilize N=8 across seeds? {stabilized}.
2. Did the model learn a flexible recall rule across N=1/2/4/8? {flexible}.
3. Does the dense-trained model transfer to single-query eval? {transfer}.
4. Is formal N_VALUES=16 training viable? {formal_viable}.
5. Should the formal task be redefined as dense capped multi-query recall? {'YES' if formal_task == 'dense capped multi-query' else 'NO'}.
6. Should the same recipe be imposed on Mamba for fairness? {'YES' if full_stage_b_allowed == 'YES' else 'NO, because Stage B is still blocked'}.
7. Is full Stage B allowed under cost estimate? {full_stage_b_allowed}.

## Recommended next step

- {recommended_next}
"""
    write_text(RUN_DIR / "summaries" / "stage_b0d_summary.md", summary)


def make_cfg(run_id: str, seed: int, n_values: int, eval_pairs: tuple[int, ...], max_steps: int, eval_interval: int, batch_size: int, lr: float) -> TrainConfig:
    return TrainConfig(
        run_id=run_id,
        stage="B0D",
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
        raise RuntimeError("CUDA is required for Stage B0D")
    log(f"[stage_b0d] run_dir={RUN_DIR} commit={git_hash()} cap_hours={CAP_HOURS:.2f}")
    if RESULTS_CSV.exists():
        RESULTS_CSV.unlink()

    sampler_ok = run_sampler_audit()
    if not sampler_ok:
        write_summary(
            sampler_ok=False,
            b0d1_gate="FAIL",
            b0d2_gate="SKIPPED",
            b0d3_status="not_run",
            formal_task="do not proceed",
            stage_b_recipe="do not proceed",
            full_stage_b_allowed="NO",
            recommended_next="Do not enter Stage B; revise architecture/training recipe.",
        )
        return

    stage_start = time.time()
    base_sec = prior_sec_per_step()
    initial_est = 3 * 30000 * base_sec * 2.4 / 3600.0
    if initial_est > CAP_HOURS:
        write_budget_pause(initial_est, 0.0, "estimated B0D.1 cost exceeds cap")
        write_cost_update([], None, None, "FAIL", "SKIPPED")
        write_summary(
            sampler_ok=True,
            b0d1_gate="FAIL",
            b0d2_gate="SKIPPED",
            b0d3_status="not_run",
            formal_task="do not proceed",
            stage_b_recipe="do not proceed",
            full_stage_b_allowed="NO",
            recommended_next="Do not enter Stage B; extend B0D.2 to 100k with user confirmation.",
        )
        return

    eval_modes = (("dense_capped", "capped", 8), ("single_query", "single", 1))
    outcomes_b0d1: list[dict[str, Any]] = []
    b0d1_pairs = (1, 2, 4, 8)
    for seed in [0, 1, 2]:
        run_id = f"stage_b0d_b01_dense_v8_seed{seed}"
        cfg = make_cfg(run_id, seed, 8, b0d1_pairs, 30000, 1000, 192, 1e-3)
        append_manifest(
            f"B0D_1_dense_v8_seed{seed}",
            cfg,
            "0-5k:1,2;5k-10k:1,2,4;10k-30k:1,2,4,8;q_count_mode=capped;q_cap=8",
            "1,2,4,8;eval_modes=dense_capped,single_query",
            30000 * base_sec * 2.4 / 3600.0,
            "planned",
        )
        outcome = train_dense_run(
            substage="B0D.1",
            run_id=run_id,
            recipe="dense capped curriculum N_VALUES=8",
            cfg=cfg,
            phases=B0D1_PHASES,
            eval_n_pairs=b0d1_pairs,
            eval_modes=eval_modes,
            eval_seeds=(0, 1),
            eval_samples_per_condition=DEFAULT_EVAL_SAMPLES,
            eval_batch_size=192,
            checkpoint_interval=1000,
            early_stop_fn=b0d1_early_stop,
            consecutive_passes=2,
            stage_start_time=stage_start,
        )
        outcomes_b0d1.append(outcome)
        append_stage_results(outcome, cfg)
        append_manifest(
            f"B0D_1_dense_v8_seed{seed}",
            cfg,
            "0-5k:1,2;5k-10k:1,2,4;10k-30k:1,2,4,8;q_count_mode=capped;q_cap=8",
            "1,2,4,8;eval_modes=dense_capped,single_query",
            outcome["elapsed_sec"] / 3600.0,
            outcome["status"],
        )
        plot_curves()
        if outcome["status"] == "budget_pause_actual":
            write_budget_pause(initial_est, (time.time() - stage_start) / 3600.0, outcome.get("notes", "actual cap exceeded"))
            break

    b0d1_gate = classify_b0d1(outcomes_b0d1)
    if b0d1_gate == "FAIL":
        write_cost_update(outcomes_b0d1, None, None, b0d1_gate, "SKIPPED")
        write_summary(
            sampler_ok=True,
            b0d1_gate=b0d1_gate,
            b0d2_gate="SKIPPED",
            b0d3_status="not_run",
            formal_task="do not proceed",
            stage_b_recipe="do not proceed",
            full_stage_b_allowed="NO",
            recommended_next="Do not enter Stage B; revise architecture/training recipe.",
        )
        log("[stage_b0d] stop after B0D.1 FAIL")
        return
    if b0d1_gate == "FORMAT_SPECIFIC":
        write_cost_update(outcomes_b0d1, None, None, b0d1_gate, "SKIPPED")
        write_summary(
            sampler_ok=True,
            b0d1_gate=b0d1_gate,
            b0d2_gate="SKIPPED",
            b0d3_status="not_run",
            formal_task="dense capped multi-query",
            stage_b_recipe="do not proceed",
            full_stage_b_allowed="NO",
            recommended_next="Do not enter Stage B; revise architecture/training recipe.",
        )
        log("[stage_b0d] stop after B0D.1 FORMAT_SPECIFIC")
        return

    sec_after_b0d1 = measured_sec_per_step(outcomes_b0d1) or base_sec
    b0d2_est = 50000 * sec_after_b0d1 * 2.6 / 3600.0
    if not enough_budget_for(b0d2_est, stage_start):
        write_budget_pause(b0d2_est, (time.time() - stage_start) / 3600.0, "estimated B0D.2 cost exceeds remaining cap")
        write_cost_update(outcomes_b0d1, None, None, b0d1_gate, "SKIPPED")
        write_summary(
            sampler_ok=True,
            b0d1_gate=b0d1_gate,
            b0d2_gate="SKIPPED",
            b0d3_status="not_run_budget",
            formal_task="do not proceed",
            stage_b_recipe="do not proceed",
            full_stage_b_allowed="NO",
            recommended_next="Do not enter Stage B; extend B0D.2 to 100k with user confirmation.",
        )
        return

    b0d2_pairs = (1, 2, 4, 8, 16, 32, 64)
    cfg2 = make_cfg("stage_b0d_b02_dense_v16_seed0", 0, 16, b0d2_pairs, 50000, 2000, 96, 1e-3)
    append_manifest(
        "B0D_2_dense_v16_seed0_lr1e-3",
        cfg2,
        "curriculum to 64;q_count_mode=capped;q_cap=8",
        "1,2,4,8,16,32,64;eval_modes=dense_capped,single_query",
        b0d2_est,
        "planned",
    )
    outcome_b0d2 = train_dense_run(
        substage="B0D.2",
        run_id=cfg2.run_id,
        recipe="formal dense capped curriculum N_VALUES=16",
        cfg=cfg2,
        phases=B0D2_PHASES,
        eval_n_pairs=b0d2_pairs,
        eval_modes=eval_modes,
        eval_seeds=(0, 1),
        eval_samples_per_condition=DEFAULT_EVAL_SAMPLES,
        eval_batch_size=96,
        checkpoint_interval=2000,
        early_stop_fn=lambda metrics, status: b0d2_early_stop(metrics, status, 16),
        consecutive_passes=2,
        stage_start_time=stage_start,
    )
    append_stage_results(outcome_b0d2, cfg2)
    append_manifest(
        "B0D_2_dense_v16_seed0_lr1e-3",
        cfg2,
        "curriculum to 64;q_count_mode=capped;q_cap=8",
        "1,2,4,8,16,32,64;eval_modes=dense_capped,single_query",
        outcome_b0d2["elapsed_sec"] / 3600.0,
        outcome_b0d2["status"],
    )
    plot_curves()
    b0d2_gate = classify_b0d2(outcome_b0d2, 16)

    outcome_b0d3 = None
    b0d3_status = "not_run"
    if b0d1_gate in {"PASS", "PASS_STRONG"} and b0d2_gate in {"FAIL", "PARTIAL_RISING"}:
        fallback_est = outcome_b0d2["elapsed_sec"] / 3600.0 if outcome_b0d2.get("elapsed_sec") else b0d2_est
        if enough_budget_for(fallback_est, stage_start):
            cfg3 = make_cfg("stage_b0d_b03_dense_v16_seed0_lr5e-4", 0, 16, b0d2_pairs, 50000, 2000, 96, 5e-4)
            append_manifest(
                "B0D_3_dense_v16_seed0_lr5e-4",
                cfg3,
                "same as B0D.2",
                "1,2,4,8,16,32,64;eval_modes=dense_capped,single_query",
                fallback_est,
                "planned",
            )
            outcome_b0d3 = train_dense_run(
                substage="B0D.3",
                run_id=cfg3.run_id,
                recipe="formal dense capped curriculum N_VALUES=16 lr=5e-4 fallback",
                cfg=cfg3,
                phases=B0D2_PHASES,
                eval_n_pairs=b0d2_pairs,
                eval_modes=eval_modes,
                eval_seeds=(0, 1),
                eval_samples_per_condition=DEFAULT_EVAL_SAMPLES,
                eval_batch_size=96,
                checkpoint_interval=2000,
                early_stop_fn=lambda metrics, status: b0d2_early_stop(metrics, status, 16),
                consecutive_passes=2,
                stage_start_time=stage_start,
            )
            append_stage_results(outcome_b0d3, cfg3)
            append_manifest(
                "B0D_3_dense_v16_seed0_lr5e-4",
                cfg3,
                "same as B0D.2",
                "1,2,4,8,16,32,64;eval_modes=dense_capped,single_query",
                outcome_b0d3["elapsed_sec"] / 3600.0,
                outcome_b0d3["status"],
            )
            plot_curves()
            b0d3_status = classify_b0d2(outcome_b0d3, 16)
            if b0d3_status in {"PASS", "PASS_STRONG"} and b0d2_gate not in {"PASS", "PASS_STRONG"}:
                b0d2_gate = b0d3_status
        else:
            b0d3_status = "skipped_budget"

    if b0d2_gate in {"PASS", "PASS_STRONG"}:
        full_allowed = "YES"
        formal_task = "dense capped multi-query"
        stage_b_recipe = "dense capped multi-query curriculum"
        recommended = "Enter full Stage B with dense capped multi-query curriculum."
    elif b0d2_gate == "PARTIAL_RISING":
        full_allowed = "NO"
        formal_task = "do not proceed"
        stage_b_recipe = "do not proceed"
        recommended = "Do not enter Stage B; extend B0D.2 to 100k with user confirmation."
    elif b0d3_status == "not_run" and b0d2_gate == "FAIL":
        full_allowed = "NO"
        formal_task = "do not proceed"
        stage_b_recipe = "do not proceed"
        recommended = "Do not enter Stage B; test lr=5e-4 fallback."
    else:
        full_allowed = "NO"
        formal_task = "do not proceed"
        stage_b_recipe = "do not proceed"
        recommended = "Do not enter Stage B; revise architecture/training recipe."

    write_cost_update(outcomes_b0d1, outcome_b0d2, outcome_b0d3, b0d1_gate, b0d2_gate)
    write_summary(
        sampler_ok=True,
        b0d1_gate=b0d1_gate,
        b0d2_gate=b0d2_gate,
        b0d3_status=b0d3_status,
        formal_task=formal_task,
        stage_b_recipe=stage_b_recipe,
        full_stage_b_allowed=full_allowed,
        recommended_next=recommended,
    )
    log(f"[stage_b0d] done B0D.1={b0d1_gate} B0D.2={b0d2_gate} allowed_B={full_allowed}")


if __name__ == "__main__":
    main()
