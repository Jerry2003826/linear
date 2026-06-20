from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = float(p.grad.detach().data.norm(2).item())
        total += param_norm * param_norm
    return math.sqrt(total)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")


class RateMeter:
    def __init__(self) -> None:
        self.start = time.time()
        self.examples = 0
        self.tokens = 0

    def update(self, examples: int, tokens: int) -> None:
        self.examples += examples
        self.tokens += tokens

    def rates(self) -> tuple[float, float, float]:
        elapsed = max(1e-9, time.time() - self.start)
        return self.examples / elapsed, self.tokens / elapsed, elapsed
