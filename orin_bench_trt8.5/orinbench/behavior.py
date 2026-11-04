"""Outputs on fixed scenes and seeds, and their comparison with a reference run.

Differences from the reference measure how well a configuration preserves the reference model's outputs, not driving
quality. Errors against the logged path are reported separately (ego frame at t0: x forward, y left, metres; 10 Hz;
horizons 3.0 s and 6.4 s) and are open-loop only.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .model import ade_fde, to_jsonable


def behavior_pass(bench, rc, scenes, seeds: list[int], out_path: str) -> list[dict]:
    """Run every scene with every seed, writing each record to out_path as soon as it completes."""
    out = []
    total = len(scenes) * len(seeds)
    with open(out_path, "w", buffering=1) as log:
        for s in scenes:
            for seed in seeds:
                r = bench.call(s, rc, seed=seed, sync_each_stage=False)
                rec = {k: r[k] for k in ("scene_id", "seed", "pred_xyz", "cot", "gen_token_ids", "n_generated_tokens",
                                         "reasoning_complete", "traj_start_emitted", "finite", "e2e_wall_ms")}
                rec["config"] = rc.name
                if s.ego_future_xyz is not None and r["finite"]:
                    rec["vs_logged_path"] = ade_fde(r["pred_xyz"], s.ego_future_xyz[0, 0].numpy())
                out.append(rec)
                log.write(json.dumps(to_jsonable(rec), default=str) + "\n")
                if len(out) % 10 == 0 or len(out) == total:
                    print(f"[behavior] {rc.name}: {len(out)}/{total}", flush=True)
    return out


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compare(ref: list[dict], test: list[dict]) -> dict:
    """Pair records by (scene_id, seed). Unpaired and non-finite records are counted, not silently dropped."""

    def index(records, label):
        out = {}
        for r in records:
            key = (r["scene_id"], r["seed"])
            if key in out:
                raise ValueError(f"duplicate {label} scene/seed: {key}")
            out[key] = r
        return out

    refs, tests = index(ref, "reference"), index(test, "test")
    rows = []
    for key in sorted(refs.keys() & tests.keys()):
        r, t = refs[key], tests[key]
        if not (r["finite"] and t["finite"]):
            continue
        a, b = np.asarray(t["pred_xyz"]), np.asarray(r["pred_xyz"])
        if a.shape != b.shape:
            raise ValueError(f"trajectory shape mismatch for {key}: {a.shape} vs {b.shape}")
        rows.append({"scene_id": key[0], "seed": key[1],
                     "identical_reasoning_tokens": t["gen_token_ids"] == r["gen_token_ids"], "vs_ref": ade_fde(a, b)})
    agg: dict = {
        "n_test": len(test),
        "n_reference": len(ref),
        "n_paired_finite": len(rows),
        "n_test_only": len(tests.keys() - refs.keys()),
        "n_reference_only": len(refs.keys() - tests.keys()),
        "finite_rate": float(np.mean([t["finite"] for t in test])) if test else None,
        "reasoning_complete_rate": float(np.mean([t["reasoning_complete"] for t in test])) if test else None,
    }
    if rows:
        agg["identical_reasoning_rate"] = float(np.mean([r["identical_reasoning_tokens"] for r in rows]))
        for k in rows[0]["vs_ref"]:
            vals = [r["vs_ref"][k] for r in rows]
            agg[f"vs_ref_{k}_mean"] = float(np.mean(vals))
            agg[f"vs_ref_{k}_p90"] = float(np.percentile(vals, 90))
            agg[f"vs_ref_{k}_max"] = float(np.max(vals))
    for label, records in (("test", test), ("reference", ref)):
        logged = [r["vs_logged_path"] for r in records if r.get("vs_logged_path")]
        if logged:
            for k in logged[0]:
                agg[f"{label}_vs_logged_path_{k}_mean"] = float(np.mean([g[k] for g in logged]))
    return {"aggregate": agg, "rows": rows}


def seed_spread(recs: list[dict]) -> dict:
    """Natural variability: displacement between different seeds on the same scene. A scale for compare()."""
    by_scene = defaultdict(list)
    for r in recs:
        if r["finite"]:
            by_scene[r["scene_id"]].append(np.asarray(r["pred_xyz"]))
    ades, fdes = [], []
    for trajs in by_scene.values():
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                m = ade_fde(trajs[i], trajs[j])
                ades.append(m["ade_6.4s"])
                fdes.append(m["fde_6.4s"])
    if not ades:
        return {}
    return {"pairs": len(ades), "seed_pair_ade_6.4s_mean": float(np.mean(ades)),
            "seed_pair_ade_6.4s_p90": float(np.percentile(ades, 90)), "seed_pair_fde_6.4s_mean": float(np.mean(fdes))}
