from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from data import TaskConfig, answer_metrics, make_batch, make_fixed_batch, random_baselines
from models_mamba import MambaLM
from models_transformer import CausalTransformerLM, parameter_count
from utils import RateMeter, append_jsonl, grad_norm, seed_everything, write_json


@dataclass
class TrainConfig:
    run_id: str
    stage: str
    model_type: str
    d_model: int
    n_layers: int
    n_heads: int = 8
    d_state: int = 16
    lr: float = 5e-4
    seed: int = 0
    max_steps: int = 2000
    warmup_steps: int = 1000
    batch_size: int = 256
    eval_interval: int = 1000
    log_interval: int = 100
    grad_clip: float = 1.0
    dtype: str = "fp32"
    fixed_batch: bool = False
    n_keys: int = 256
    n_values: int = 8
    n_pairs: int | None = 1
    train_n_pairs: tuple[int, ...] | None = None
    eval_n_pairs: tuple[int, ...] = (1,)
    gap_len: int = 0
    q_count: int = 1
    early_acc: float | None = None
    early_patience: int = 3


def max_seq_len_for(cfg: TrainConfig) -> int:
    max_pairs = max(cfg.eval_n_pairs + (cfg.n_pairs or 0,) if cfg.train_n_pairs is None else tuple(cfg.train_n_pairs) + cfg.eval_n_pairs)
    # full length: BOS + 4 tokens per record + gap + q_count * 4; x removes final answer token.
    return 1 + 4 * max_pairs + cfg.gap_len + 4 * cfg.q_count - 1


def build_model(cfg: TrainConfig, vocab_size: int, max_seq_len: int) -> torch.nn.Module:
    if cfg.model_type == "transformer":
        return CausalTransformerLM(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
        )
    if cfg.model_type == "mamba":
        return MambaLM(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            d_state=cfg.d_state,
        )
    raise ValueError(f"unknown model_type={cfg.model_type}")


def lr_at_step(cfg: TrainConfig, step: int) -> float:
    if cfg.warmup_steps <= 0:
        return cfg.lr
    return cfg.lr * min(1.0, step / cfg.warmup_steps)


@torch.no_grad()
def evaluate(model: torch.nn.Module, cfg: TrainConfig, run_dir: Path, step: int, device: torch.device) -> list[dict]:
    model.eval()
    rows = []
    rand_acc, rand_ce = random_baselines(cfg.n_values)
    for n_pairs in cfg.eval_n_pairs:
        task = TaskConfig(
            n_pairs=n_pairs,
            n_values=cfg.n_values,
            n_keys=cfg.n_keys,
            gap_len=cfg.gap_len,
            q_count=cfg.q_count,
            seed=cfg.seed + 17,
        )
        agg = {"cross_entropy": 0.0, "per_answer_accuracy": 0.0, "per_example_all_correct": 0.0}
        batches = 32
        for _ in range(batches):
            x, labels = make_batch(task, cfg.batch_size, device, n_pairs=n_pairs)
            metrics = answer_metrics(model(x), labels)
            for key in agg:
                agg[key] += metrics[key]
        row = {
            "stage": cfg.stage,
            "run_id": cfg.run_id,
            "model_type": cfg.model_type,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_state": cfg.d_state,
            "lr": cfg.lr,
            "seed": cfg.seed,
            "step": step,
            "n_pairs": n_pairs,
            "n_values": cfg.n_values,
            "gap_len": cfg.gap_len,
            "q_count": cfg.q_count,
            "split": "eval",
            "accuracy": agg["per_answer_accuracy"] / batches,
            "cross_entropy": agg["cross_entropy"] / batches,
            "random_accuracy": rand_acc,
            "random_cross_entropy": rand_ce,
            "per_example_all_correct": agg["per_example_all_correct"] / batches,
            "params": parameter_count(model),
            "dtype": cfg.dtype,
            "wall_time_sec": time.time(),
            "tokens_seen": 0,
            "status": "ok",
        }
        rows.append(row)
    model.train()
    return rows


def train_run(cfg: TrainConfig, run_dir: Path) -> dict:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = TaskConfig(
        n_pairs=cfg.n_pairs,
        train_n_pairs=cfg.train_n_pairs,
        n_values=cfg.n_values,
        n_keys=cfg.n_keys,
        gap_len=cfg.gap_len,
        q_count=cfg.q_count,
        seed=cfg.seed,
    )
    vocab = task.vocab()
    model = build_model(cfg, vocab.size, max_seq_len_for(cfg)).to(device)
    params = parameter_count(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    result_csv = run_dir / "results" / f"{cfg.run_id}.csv"
    log_jsonl = run_dir / "results" / f"{cfg.run_id}.jsonl"
    status_path = run_dir / "status.json"
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    fixed = make_fixed_batch(task, cfg.batch_size, device) if cfg.fixed_batch else None
    meter = RateMeter()
    start = time.time()
    best = {n: 0.0 for n in cfg.eval_n_pairs}
    pass_counts = {n: 0 for n in cfg.eval_n_pairs}
    final_rows = []

    last_step = 0
    with result_csv.open("w", newline="") as handle:
        fieldnames = [
            "stage", "run_id", "model_type", "d_model", "n_layers", "n_heads", "d_state", "lr", "seed", "step",
            "n_pairs", "n_values", "gap_len", "q_count", "split", "accuracy", "cross_entropy", "random_accuracy",
            "random_cross_entropy", "per_example_all_correct", "params", "dtype", "wall_time_sec", "tokens_seen", "status"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(1, cfg.max_steps + 1):
            last_step = step
            for group in opt.param_groups:
                group["lr"] = lr_at_step(cfg, step)
            if fixed is None:
                x, labels = make_batch(task, cfg.batch_size, device)
            else:
                x, labels = fixed
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            meter.update(x.shape[0], x.numel())

            if step % cfg.log_interval == 0 or step == 1:
                metrics = answer_metrics(logits.detach(), labels)
                exps, toks, elapsed = meter.rates()
                mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                log_row = {
                    "event": "train",
                    "step": step,
                    "loss": float(loss.item()),
                    "accuracy": metrics["per_answer_accuracy"],
                    "lr": opt.param_groups[0]["lr"],
                    "grad_norm": gnorm,
                    "examples_sec": exps,
                    "tokens_sec": toks,
                    "gpu_mem_gb": mem,
                    "elapsed_sec": elapsed,
                }
                append_jsonl(log_jsonl, log_row)
                write_json(status_path, {"stage": cfg.stage, "run_id": cfg.run_id, "step": step, "last_train": log_row})

            if step % cfg.eval_interval == 0 or step == cfg.max_steps:
                rows = evaluate(model, cfg, run_dir, step, device)
                final_rows = rows
                for row in rows:
                    row["wall_time_sec"] = time.time() - start
                    row["tokens_seen"] = meter.tokens
                    writer.writerow(row)
                    append_jsonl(log_jsonl, {"event": "eval", **row})
                    n = int(row["n_pairs"])
                    acc = float(row["accuracy"])
                    best[n] = max(best[n], acc)
                    if cfg.early_acc is not None and acc >= cfg.early_acc:
                        pass_counts[n] += 1
                    else:
                        pass_counts[n] = 0
                handle.flush()
                if cfg.early_acc is not None and all(pass_counts[n] >= cfg.early_patience for n in cfg.eval_n_pairs):
                    break

    return {
        "config": asdict(cfg),
        "params": params,
        "seconds": time.time() - start,
        "steps": last_step,
        "best_accuracy": best,
        "final_rows": final_rows,
        "result_csv": str(result_csv),
        "log_jsonl": str(log_jsonl),
    }
