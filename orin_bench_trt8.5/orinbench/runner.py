"""Measurement protocol for one run configuration.

  1. ``cold``     -- first call; reported separately, never in the statistics.
  2. ``warmup``   -- repeat until the medians of the last two windows agree within 5 % (bounded by warmup_min/max).
  3. ``varied``   -- n_varied calls cycling the scenes with seed = seed_base + i (e2e mode). The headline numbers.
  4. ``repeated`` -- n_repeated calls on scene 0 with one fixed seed (identical inputs; e2e mode). Kept apart so any
                     input or cache reuse cannot flatter the headline numbers.
  5. ``staged``   -- n_staged calls with a synchronize at every stage boundary (per-stage breakdown).

e2e mode synchronizes only at the start and at the final device->host copy, so its wall time is the deployable
input-ready -> trajectory-ready latency. Telemetry is summarized per phase.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .model import to_jsonable
from .stats import is_stable, summarize
from .telemetry import Telemetry


@dataclass
class Plan:
    warmup_min: int = 3
    warmup_max: int = 15
    n_varied: int = 100
    n_repeated: int = 20
    n_staged: int = 10
    seed_base: int = 1000
    repeated_seed: int = 42
    max_minutes: float = 60.0  # soft cap per phase; the number of completed calls is always recorded

    @classmethod
    def from_dict(cls, d: dict | None) -> "Plan":
        return cls(**(d or {}))


def _phase_stats(recs: list[dict]) -> dict:
    if not recs:
        return {}
    out: dict = {"e2e_wall_ms": summarize([r["e2e_wall_ms"] for r in recs])}
    if all("e2e_gpu_ms" in r for r in recs):
        out["e2e_gpu_ms"] = summarize([r["e2e_gpu_ms"] for r in recs])
    stages, gstages = defaultdict(list), defaultdict(list)
    for r in recs:
        for k, v in r["wall_ms"].items():
            stages[k].append(v)
        for k, v in r.get("gpu_ms", {}).items():
            gstages[k].append(v)
    out["stage_wall_ms_mean"] = {k: float(np.mean(v)) for k, v in stages.items()}
    out["stage_wall_ms_p50"] = {k: float(np.median(v)) for k, v in stages.items()}
    out["stage_gpu_ms_mean"] = {k: float(np.mean(v)) for k, v in gstages.items()}
    for key in ("n_prompt_tokens", "n_image_tokens", "n_generated_tokens", "ttft_wall_ms"):
        vals = [r[key] for r in recs if r.get(key) is not None]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    steps = [s for r in recs for s in r.get("decode_step_ms", [])]
    if steps:
        out["decode_step_ms"] = summarize(steps)
        out["decode_tok_per_s_from_steps"] = 1e3 / float(np.median(steps))
    for flag in ("reasoning_complete", "traj_start_emitted", "finite"):
        out[f"rate_{flag}"] = float(np.mean([r[flag] for r in recs]))
    return out


def run_config(bench, rc, scenes, plan: Plan, telem: Telemetry | None, out_dir: str) -> dict:
    """Run the protocol for one configuration; writes calls.jsonl and summary.json into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    summary: dict = {"config": rc.name, "plan": plan.__dict__}
    windows: list[dict] = []
    done: dict = defaultdict(int)
    call_log = open(os.path.join(out_dir, "calls.jsonl"), "w", buffering=1)  # written outside the timed interval
    failure = None

    def phase(name, n, pick, sync):
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        recs = []
        for i in range(n):
            scene, seed = pick(i)
            r = bench.call(scene, rc, seed=seed, sync_each_stage=sync)
            r["phase"], r["i"] = name, i
            recs.append(r)
            done[name] += 1
            call_log.write(json.dumps(to_jsonable({k: v for k, v in r.items() if k != "gen_token_ids"}), default=str) + "\n")
            if (i + 1) % 10 == 0 or i + 1 == n:
                print(f"[run] {rc.name} {name}: {i + 1}/{n}", flush=True)
            if (time.time() - t0) / 60 > plan.max_minutes and i + 1 >= min(n, 10):
                summary.setdefault("truncated_phases", {})[name] = {"planned": n, "done": i + 1}
                break
        w = {"phase": name, "t0": t0, "t1": time.time(), "n_calls": len(recs)}
        if torch.cuda.is_available():
            w["torch_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        if telem is not None:
            w["telemetry"] = telem.summarize_window(w["t0"], w["t1"], len(recs))
        windows.append(w)
        return recs

    try:
        summary["cold_call_ms"] = phase("cold", 1, lambda i: (scenes[0], plan.seed_base), False)[0]["e2e_wall_ms"]
        window = min(5, max(2, plan.warmup_min))
        lat: list[float] = []
        t0 = time.time()
        while len(lat) < plan.warmup_max:
            i = len(lat)
            r = bench.call(scenes[i % len(scenes)], rc, seed=plan.seed_base + i, sync_each_stage=False)
            # e2e varies with the number of generated tokens; time to first token is the fixed-work signal
            lat.append(r.get("ttft_wall_ms") or r["e2e_wall_ms"])
            if len(lat) >= plan.warmup_min and is_stable(lat, window=window):
                break
        summary["warmup"] = {"n": len(lat), "latencies_ms": lat, "stable": is_stable(lat, window=window), "s": time.time() - t0}
        summary["varied"] = _phase_stats(phase("varied", plan.n_varied, lambda i: (scenes[i % len(scenes)], plan.seed_base + i), False))
        summary["repeated"] = _phase_stats(phase("repeated", plan.n_repeated, lambda i: (scenes[0], plan.repeated_seed), False))
        summary["staged"] = _phase_stats(phase("staged", plan.n_staged, lambda i: (scenes[i % len(scenes)], plan.seed_base + i), True))
    except Exception as e:  # noqa: BLE001
        failure = e
        summary["error"], summary["traceback"] = repr(e), traceback.format_exc()
        print(f"[run] {rc.name} FAILED: {e!r}", flush=True)
    finally:
        call_log.close()
    summary["windows"], summary["completed_calls"] = windows, dict(done)
    write_json(summary, os.path.join(out_dir, "summary.json"))
    if failure is not None:
        raise RuntimeError(f"benchmark {rc.name} failed; see {out_dir}/summary.json") from failure
    return summary


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
