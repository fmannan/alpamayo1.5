"""Latency statistics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

DEADLINE_MS = 100.0


def summarize(values_ms: Iterable[float], deadline_ms: float = DEADLINE_MS) -> dict:
    """Return p50/p95/p99/max and deadline miss rate for a list of latencies in ms."""
    v = np.asarray([x for x in values_ms if x is not None and not math.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
        "pct_over_deadline": float((v > deadline_ms).mean() * 100.0),
        "deadline_ms": deadline_ms,
    }


def is_stable(values_ms: list[float], window: int = 5, rel_tol: float = 0.05) -> bool:
    """Warm-up stopping rule: last two windows have medians within rel_tol."""
    if len(values_ms) < 2 * window:
        return False
    a = float(np.median(values_ms[-2 * window : -window]))
    b = float(np.median(values_ms[-window:]))
    return abs(a - b) <= rel_tol * max(a, b)
