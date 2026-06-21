from __future__ import annotations

import csv
import os
import time
from pathlib import Path

DEFAULT_PRICE = float(os.environ.get("GPU_PRICE_PER_HOUR", "1.0"))
MAX_GPU_HOURS_TOTAL = float(os.environ.get("MAX_GPU_HOURS_TOTAL", "10"))


class CostTracker:
    """Tracks cumulative GPU-hours and per-stage runtime; writes runtime_profile.csv."""

    def __init__(self, results_dir: str | Path, price_per_hour: float | None = None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path = self.results_dir / "runtime_profile.csv"
        self.price = price_per_hour if price_per_hour is not None else DEFAULT_PRICE
        self.total_gpu_hours = 0.0
        self._load_existing()

    def _load_existing(self) -> None:
        if self.profile_path.exists():
            with self.profile_path.open() as f:
                for row in csv.DictReader(f):
                    try:
                        if row.get("uses_gpu", "1") in ("1", "True", "true"):
                            self.total_gpu_hours += float(row["elapsed_hours"])
                    except Exception:
                        pass

    def record(
        self,
        stage: str,
        elapsed_seconds: float,
        uses_gpu: bool = True,
        notes: str = "",
    ) -> dict:
        hours = elapsed_seconds / 3600.0
        if uses_gpu:
            self.total_gpu_hours += hours
        cost = hours * self.price if uses_gpu else 0.0
        row = {
            "stage": stage,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "elapsed_hours": round(hours, 5),
            "uses_gpu": int(uses_gpu),
            "gpu_price_per_hour": self.price,
            "stage_cost": round(cost, 4),
            "cumulative_gpu_hours": round(self.total_gpu_hours, 5),
            "cumulative_cost": round(self.total_gpu_hours * self.price, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": notes,
        }
        header = list(row.keys())
        exists = self.profile_path.exists()
        with self.profile_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if not exists:
                w.writeheader()
            w.writerow(row)
        return row

    def remaining_gpu_hours(self) -> float:
        return MAX_GPU_HOURS_TOTAL - self.total_gpu_hours

    def over_budget(self) -> bool:
        return self.total_gpu_hours >= MAX_GPU_HOURS_TOTAL
