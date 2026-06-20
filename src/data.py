from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class Vocab:
    n_keys: int = 256
    n_values: int = 16
    n_noise: int = 256

    @property
    def pad(self) -> int:
        return 0

    @property
    def bos(self) -> int:
        return 1

    @property
    def k_tok(self) -> int:
        return 2

    @property
    def v_tok(self) -> int:
        return 3

    @property
    def q_tok(self) -> int:
        return 4

    @property
    def ans_tok(self) -> int:
        return 5

    @property
    def noise_tok(self) -> int:
        return 6

    @property
    def key_base(self) -> int:
        return 7

    @property
    def value_base(self) -> int:
        return self.key_base + self.n_keys

    @property
    def noise_base(self) -> int:
        return self.value_base + self.n_values

    @property
    def size(self) -> int:
        return self.noise_base + self.n_noise

    def key_id(self, key: int) -> int:
        return self.key_base + key

    def value_id(self, value: int) -> int:
        return self.value_base + value

    def noise_id(self, noise: int) -> int:
        return self.noise_base + noise

    def decode_token(self, token: int) -> str:
        special = {
            self.pad: "<PAD>",
            self.bos: "<BOS>",
            self.k_tok: "<K>",
            self.v_tok: "<V>",
            self.q_tok: "<Q>",
            self.ans_tok: "<ANS>",
            self.noise_tok: "<NOISE>",
        }
        if token in special:
            return special[token]
        if self.key_base <= token < self.value_base:
            return f"key_{token - self.key_base:03d}"
        if self.value_base <= token < self.noise_base:
            return f"val_{token - self.value_base:02d}"
        if self.noise_base <= token < self.size:
            return f"noise_{token - self.noise_base:03d}"
        return f"<UNK:{token}>"


@dataclass(frozen=True)
class TaskConfig:
    n_pairs: int | None = 8
    train_n_pairs: tuple[int, ...] | None = None
    n_values: int = 16
    n_keys: int = 256
    n_noise: int = 256
    gap_len: int = 0
    q_count: int = 1
    seed: int = 0
    fixed_batch: bool = False

    def vocab(self) -> Vocab:
        return Vocab(n_keys=self.n_keys, n_values=self.n_values, n_noise=self.n_noise)

    def choose_n_pairs(self) -> int:
        if self.train_n_pairs:
            return int(random.choice(self.train_n_pairs))
        if self.n_pairs is None:
            raise ValueError("n_pairs is required when train_n_pairs is not set")
        return int(self.n_pairs)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_sample(cfg: TaskConfig, *, n_pairs: int | None = None, generator: torch.Generator | None = None) -> tuple[list[int], list[int]]:
    vocab = cfg.vocab()
    actual_pairs = cfg.choose_n_pairs() if n_pairs is None else n_pairs
    if actual_pairs > cfg.n_keys:
        raise ValueError(f"n_pairs={actual_pairs} exceeds n_keys={cfg.n_keys}")
    if cfg.q_count > actual_pairs:
        raise ValueError(f"q_count={cfg.q_count} exceeds n_pairs={actual_pairs}")

    keys = torch.randperm(cfg.n_keys, generator=generator)[:actual_pairs]
    values = torch.randint(0, cfg.n_values, (actual_pairs,), generator=generator)
    query_indices = torch.randperm(actual_pairs, generator=generator)[: cfg.q_count]

    full = [vocab.bos]
    for key, value in zip(keys.tolist(), values.tolist()):
        full.extend([vocab.k_tok, vocab.key_id(key), vocab.v_tok, vocab.value_id(value)])
    for _ in range(cfg.gap_len):
        noise = int(torch.randint(0, cfg.n_noise, (1,), generator=generator).item())
        full.append(vocab.noise_id(noise))
    answer_tokens = []
    for idx in query_indices.tolist():
        key = int(keys[idx].item())
        value = int(values[idx].item())
        answer = vocab.value_id(value)
        full.extend([vocab.q_tok, vocab.key_id(key), vocab.ans_tok, answer])
        answer_tokens.append(answer)
    return full, answer_tokens


def sample_to_x_labels(full_tokens: list[int], vocab: Vocab) -> tuple[list[int], list[int]]:
    x = full_tokens[:-1]
    labels = [-100] * len(x)
    for i, tok in enumerate(x):
        if tok == vocab.ans_tok:
            if i + 1 >= len(full_tokens):
                raise ValueError("<ANS> cannot be last full token")
            labels[i] = full_tokens[i + 1]
    return x, labels


def collate(samples: Iterable[tuple[list[int], list[int]]], pad_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*samples)
    max_len = max(len(x) for x in xs)
    batch_x = torch.full((len(xs), max_len), pad_token, dtype=torch.long)
    batch_y = torch.full((len(xs), max_len), -100, dtype=torch.long)
    for row, (x, y) in enumerate(zip(xs, ys)):
        batch_x[row, : len(x)] = torch.tensor(x, dtype=torch.long)
        batch_y[row, : len(y)] = torch.tensor(y, dtype=torch.long)
    return batch_x, batch_y


def make_batch(
    cfg: TaskConfig,
    batch_size: int,
    device: torch.device | str,
    *,
    generator: torch.Generator | None = None,
    n_pairs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    vocab = cfg.vocab()
    samples = []
    for _ in range(batch_size):
        full, _ = build_sample(cfg, n_pairs=n_pairs, generator=generator)
        samples.append(sample_to_x_labels(full, vocab))
    x, labels = collate(samples, vocab.pad)
    return x.to(device), labels.to(device)


def make_fixed_batch(cfg: TaskConfig, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    return make_batch(cfg, batch_size, device, generator=g)


def answer_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    mask = labels != -100
    if int(mask.sum().item()) == 0:
        raise ValueError("no answer labels found")
    vocab = logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100)
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == labels[mask]).float()
    per_answer_acc = float(correct.mean().item())

    per_example = []
    for row in range(labels.shape[0]):
        row_mask = mask[row]
        if int(row_mask.sum().item()) == 0:
            continue
        per_example.append(bool((pred[row, row_mask] == labels[row, row_mask]).all().item()))
    all_correct = float(sum(per_example) / max(1, len(per_example)))
    return {
        "cross_entropy": float(loss.item()),
        "per_answer_accuracy": per_answer_acc,
        "per_example_all_correct": all_correct,
        "answer_count": int(mask.sum().item()),
    }


def decoded_sample(cfg: TaskConfig, *, n_pairs: int | None = None) -> str:
    vocab = cfg.vocab()
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    full, answers = build_sample(cfg, n_pairs=n_pairs, generator=g)
    x, labels = sample_to_x_labels(full, vocab)
    lines = [
        f"vocab_size={vocab.size}",
        f"n_pairs={n_pairs or cfg.n_pairs}",
        f"gap_len={cfg.gap_len}",
        f"q_count={cfg.q_count}",
        "full_tokens:",
        " ".join(vocab.decode_token(t) for t in full),
        "x_ids:",
        str(x),
        "labels:",
        str(labels),
        "non_ignore_labels:",
    ]
    for i, label in enumerate(labels):
        if label != -100:
            lines.append(f"pos={i} input={vocab.decode_token(x[i])} label={vocab.decode_token(label)}")
    return "\n".join(lines) + "\n"


def random_baselines(n_values: int) -> tuple[float, float]:
    return 1.0 / n_values, math.log(n_values)
