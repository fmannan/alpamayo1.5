"""Report for a pytorch run and a hybrid run (scripts/run.sh), plus engine logs (scripts/build_engines.sh).

    python -m orinbench.report --pytorch RUN_DIR --hybrid RUN_DIR [--engines ENGINES_DIR...] [--scenes DIR] --out DIR

Writes DIR/REPORT.md and DIR/figures/*.png:
  latency.png       distribution of end-to-end latency, every timed call, per config
  stages.png        where one call spends its time (staged mode), pytorch vs hybrid
  engines.png       the expert engine's per-step time by layer type (trtexec per-layer profile)
  trajectories.png  pytorch vs hybrid trajectories and the logged path, every scene (seed 0)
  agreement.png     per-scene trajectory distance between the two backends, against the seed-to-seed spread
Needs matplotlib. Runs anywhere the result directories are readable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from .behavior import compare, load_jsonl, seed_spread

STAGES = ["preprocess_cpu", "h2d", "pre_vision", "vision", "llm_prefill", "decode", "expert_prep", "flow", "postprocess"]
STAGE_LABELS = {
    "preprocess_cpu": "preprocess (CPU)", "h2d": "host to device", "pre_vision": "token fusion + embed",
    "vision": "vision tower", "llm_prefill": "LLM prefill", "decode": "LLM decode (reasoning)",
    "expert_prep": "expert inputs", "flow": "flow (10 expert steps)", "postprocess": "dynamics + copy back",
}
# Reference palette, light mode (validated: CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BACKEND_COLOR = {"pytorch": "#2a78d6", "hybrid": "#eb6834"}
LOGGED = "#52514e"
# Expert engine layer types, first match wins (names come from the ONNX export of the Qwen3-VL text layers).
LAYER_TYPES = [
    ("mask input conversion", r"Network Input attention_mask"),
    ("MLP", r"/mlp/"),
    ("attention projections", r"self_attn/[qkvo]_proj"),
    ("KV expand + layout (GQA)", r"self_attn/(Expand|Reshape|Transpose|Slice|Concat|Unsqueeze)"),
    ("attention scores + softmax", r"self_attn/(MatMul|Softmax|Add)"),
    ("norms", r"norm"),
    ("rotary", r"rotary|rotate|self_attn/(Mul|Neg)"),
]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _configs(run_dir: str) -> dict[str, str]:
    return {os.path.basename(os.path.dirname(p)): os.path.dirname(p)
            for p in sorted(glob.glob(os.path.join(run_dir, "*", "summary.json")))}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _calls(cdir: str, phase: str = "varied") -> list[dict]:
    return [r for r in load_jsonl(os.path.join(cdir, "calls.jsonl")) if r.get("phase") == phase]


def engine_stats(engines_dirs: list[str] | None) -> dict:
    """GPU compute medians from trtexec logs and the expert's per-layer profile grouped by layer type.

    Several build_engines.sh log directories may be given (e.g. vision and expert built separately); for each engine
    the last directory that has its timing log wins.
    """
    out: dict = {}
    for part in ("vision", "expert"):
        dirs = [d for d in engines_dirs or [] if os.path.exists(os.path.join(d, f"{part}_time.log"))]
        if not dirs:
            continue
        engines_dir = dirs[-1]
        info: dict = {}
        for key, log in (("median_ms", f"{part}_time.log"), ("graph_median_ms", f"{part}_time_graph.log")):
            path = os.path.join(engines_dir, log)
            if os.path.exists(path):
                m = re.search(r"GPU Compute Time: .*?median = ([0-9.]+) ms", open(path).read())
                info[key] = float(m.group(1)) if m else None
        gate = os.path.join(engines_dir, f"{part}_gate.log")
        if os.path.exists(gate):
            info["gate"] = [ln.strip() for ln in open(gate) if re.match(r"s\d+_", ln)]
        prof = os.path.join(engines_dir, f"{part}_profile.json")
        if os.path.exists(prof):
            rows = [r for r in _load(prof) if "averageMs" in r]
            total = sum(r["averageMs"] for r in rows)
            groups: dict = defaultdict(float)
            for r in rows:
                kind = next((k for k, pat in LAYER_TYPES if re.search(pat, r["name"])), "other")
                groups[kind] += r["averageMs"]
            top = sorted(rows, key=lambda r: -r["averageMs"])[:5]
            info["profile"] = {"n_layers": len(rows), "total_ms": total, "by_type_ms": dict(groups),
                               "top": [(r["name"][:80], r["averageMs"]) for r in top]}
        out[part] = info
    return out


def fig_latency(runs: dict, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    for backend, configs in runs.items():
        for name, cdir in configs.items():
            v = np.sort([r["e2e_wall_ms"] for r in _calls(cdir)]) / 1e3
            if not len(v):
                continue
            y = np.arange(1, len(v) + 1) / len(v)
            dashed = name == "force_traj"
            ax.step(v, y, where="post", color=BACKEND_COLOR[backend], linewidth=2, linestyle="--" if dashed else "-",
                    label=f"{backend} · {name}")
    ax.set_xlabel("end-to-end latency per call (s), input ready to trajectory ready", color=INK_2, fontsize=9)
    ax.set_ylabel("share of calls", color=INK_2, fontsize=9)
    ax.set_title("Latency distribution, 100 varied calls per config", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_stages(stage_means: dict, path: str) -> None:
    import matplotlib.pyplot as plt

    stages = [s for s in STAGES if any(s in m for m in stage_means.values())]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(stages))
    h = 0.38
    for i, (backend, means) in enumerate(stage_means.items()):
        vals = [means.get(s, 0.0) / 1e3 for s in stages]
        bars = ax.barh(y + (i - 0.5) * (h + 0.02), vals, height=h, color=BACKEND_COLOR[backend], label=backend)
        for b, v in zip(bars, vals):
            if v >= 0.3:
                ax.text(b.get_width() + 0.03, b.get_y() + h / 2, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [STAGE_LABELS[s] for s in stages], color=INK_2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean seconds per call (staged mode: synchronized at every boundary)", color=INK_2, fontsize=9)
    ax.set_title("Where one call spends its time", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_engines(engines: dict, path: str) -> bool:
    import matplotlib.pyplot as plt

    prof = engines.get("expert", {}).get("profile")
    if not prof:
        return False
    items = sorted(prof["by_type_ms"].items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    vals = [v for _, v in items]
    bars = ax.barh(range(len(items)), vals, height=0.6, color=BACKEND_COLOR["hybrid"])
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.4, b.get_y() + 0.3, f"{v:.1f} ms  ({100 * v / prof['total_ms']:.0f} %)", va="center",
                fontsize=8, color=INK_2)
    ax.set_yticks(range(len(items)), [k for k, _ in items], color=INK_2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("ms per flow step, summed over 36 layers (trtexec per-layer profile, includes profiling overhead)",
                  color=INK_2, fontsize=8)
    ax.set_title("Expert engine: one flow step by layer type", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def _scene_paths(ref: list[dict], test: list[dict], seed: int):
    tests = {(r["scene_id"], r["seed"]): r for r in test}
    for r in ref:
        t = tests.get((r["scene_id"], r["seed"]))
        if r["seed"] == seed and t is not None and r["finite"] and t["finite"]:
            yield r["scene_id"], np.asarray(r["pred_xyz"]), np.asarray(t["pred_xyz"])


def fig_trajectories(ref: list[dict], test: list[dict], logged: dict, path: str) -> None:
    import matplotlib.pyplot as plt

    pairs = list(_scene_paths(ref, test, seed=0))
    n = len(pairs)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.5 * rows), facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    for ax, (sid, a, b) in zip(np.atleast_1d(axes).ravel(), pairs):
        _style(ax)
        if sid in logged:
            g = logged[sid]
            ax.plot(-g[:, 1], g[:, 0], color=LOGGED, linewidth=1.5, linestyle=(0, (3, 2)))
        ax.plot(-a[:, 1], a[:, 0], color=BACKEND_COLOR["pytorch"], linewidth=2)
        ax.plot(-b[:, 1], b[:, 0], color=BACKEND_COLOR["hybrid"], linewidth=2, linestyle=(0, (1, 1.5)))
        ax.set_title(sid.split("_")[0], fontsize=8, color=INK_2, loc="left")
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=6)
    handles = [plt.Line2D([], [], color=BACKEND_COLOR["pytorch"], linewidth=2, label="pytorch"),
               plt.Line2D([], [], color=BACKEND_COLOR["hybrid"], linewidth=2, linestyle=(0, (1, 1.5)), label="hybrid"),
               plt.Line2D([], [], color=LOGGED, linewidth=1.5, linestyle=(0, (3, 2)), label="logged path")]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=3, labelcolor=INK_2)
    fig.suptitle("Predicted 6.4 s trajectories, seed 0 (ego frame: up = forward, right = right; metres)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_agreement(rows: list[dict], spread: dict, path: str) -> None:
    import matplotlib.pyplot as plt

    by_scene: dict = defaultdict(list)
    for r in rows:
        by_scene[r["scene_id"]].append(r["vs_ref"]["ade_6.4s"])
    scenes = sorted(by_scene)
    vals = [100 * float(np.mean(by_scene[s])) for s in scenes]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.bar(range(len(scenes)), vals, width=0.7, color=BACKEND_COLOR["hybrid"])
    if spread.get("seed_pair_ade_6.4s_mean"):
        s = 100 * spread["seed_pair_ade_6.4s_mean"]
        ax.axhline(s, color=INK_2, linewidth=1, linestyle=(0, (3, 2)))
        ax.text(len(scenes) - 0.5, s, f"seed-to-seed spread of pytorch: {s:.0f} cm", ha="right", va="bottom",
                fontsize=8, color=INK_2)
        ax.set_ylim(0, s * 1.15)
    ax.set_xticks(range(len(scenes)), [s.split("_")[0] for s in scenes], rotation=90, fontsize=7, color=INK_2)
    ax.set_ylabel("hybrid vs pytorch, same seed\nmean ADE over 6.4 s (cm)", color=INK_2, fontsize=9)
    ax.set_title("How far the hybrid's trajectories are from PyTorch's, per scene", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _fmt(v, nd=0):
    return "-" if v is None else f"{v:,.{nd}f}"


def build(a) -> None:
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    runs = {"pytorch": _configs(a.pytorch), "hybrid": _configs(a.hybrid)}
    summaries = {b: {n: _load(os.path.join(d, "summary.json")) for n, d in c.items()} for b, c in runs.items()}
    meta = {b: _load(os.path.join(r, "meta.json")) for b, r in (("pytorch", a.pytorch), ("hybrid", a.hybrid))}
    env = _load(os.path.join(a.hybrid, "env.json"))
    engines = engine_stats(a.engines)
    main_cfg = {"pytorch": "official", "hybrid": "reasoning"}
    lines = ["# Alpamayo 1.5 on Jetson AGX Orin: PyTorch vs TensorRT 8.5 hybrid", ""]
    l4t = re.search(r"R(\d+) \(release\), REVISION: ([\d.]+)", str(env.get("l4t_release", "")))
    mode = (env.get("nvpmodel") or "").splitlines()[0].replace("NV Power Mode: ", "") if env.get("nvpmodel") else "?"
    lines += [f"Device: {env.get('device_tree_model')}, L4T {('R' + l4t.group(1) + '.' + l4t.group(2)) if l4t else '?'}, "
              f"power mode {mode}, "
              f"GPU {int(env.get('gpu_devfreq', {}).get('cur_freq') or 0) / 1e6:.1f} MHz. "
              f"torch {env.get('torch')}, transformers {env.get('pkg_transformers')}.", ""]
    lines += ["Runs: `" + os.path.basename(a.pytorch) + "` (pytorch) and `" + os.path.basename(a.hybrid) + "` (hybrid). "
              "Latency is from input ready (decoded frames and ego history in host memory) to trajectory ready "
              "(copied back to the host).", ""]

    # latency
    lines += ["## Latency", "", "| backend | config | n | p50 ms | p95 ms | p99 ms | max ms | tokens generated (mean) | "
              "reasoning complete | trajectory emitted |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b, cs in summaries.items():
        for n, s in cs.items():
            v = s.get("varied", {})
            e = v.get("e2e_wall_ms", {})
            lines.append(f"| {b} | {n} | {e.get('n', 0)} | {_fmt(e.get('p50'))} | {_fmt(e.get('p95'))} | {_fmt(e.get('p99'))} "
                         f"| {_fmt(e.get('max'))} | {_fmt(v.get('n_generated_tokens', {}).get('mean'), 1)} "
                         f"| {_fmt(100 * v.get('rate_reasoning_complete', 0))} % | {_fmt(100 * v.get('rate_traj_start_emitted', 0))} % |")
    fig_latency(runs, os.path.join(a.out, "figures", "latency.png"))
    lines += ["", "![latency](figures/latency.png)", ""]

    # stages
    stage_means = {b: summaries[b][main_cfg[b]]["staged"]["stage_wall_ms_mean"] for b in summaries if main_cfg[b] in summaries[b]}
    fig_stages(stage_means, os.path.join(a.out, "figures", "stages.png"))
    lines += ["## Where the time goes", "", "Staged calls synchronize the GPU at every stage boundary, so the stages add up "
              "to the call.", "", "| stage | " + " | ".join(f"{b} ms" for b in stage_means) + " | share of hybrid call |",
              "|---|" + "---|" * (len(stage_means) + 1)]
    total_h = sum(stage_means.get("hybrid", {}).values()) or 1.0
    for s in STAGES:
        lines.append(f"| {STAGE_LABELS[s]} | " + " | ".join(_fmt(m.get(s)) for m in stage_means.values())
                     + f" | {_fmt(100 * stage_means.get('hybrid', {}).get(s, 0) / total_h)} % |")
    dec = {b: summaries[b][main_cfg[b]]["varied"].get("decode_step_ms", {}).get("p50") for b in stage_means}
    ttft = {b: summaries[b][main_cfg[b]]["varied"].get("ttft_wall_ms", {}).get("mean") for b in stage_means}
    lines += ["", "Decode, per generated token (p50): " + ", ".join(f"{b} {_fmt(v, 1)} ms" for b, v in dec.items())
              + ". Time to first token (mean): " + ", ".join(f"{b} {_fmt(v)} ms" for b, v in ttft.items()) + ".",
              "", "![stages](figures/stages.png)", ""]

    # engines
    if engines:
        lines += ["## TensorRT engines", "", "| engine | GPU compute median ms | with CUDA graph ms | parity gate |", "|---|---|---|---|"]
        for part, info in engines.items():
            gate = "passed" if info.get("gate") else "-"
            lines.append(f"| {part} | {_fmt(info.get('median_ms'), 1)} | {_fmt(info.get('graph_median_ms'), 1)} | {gate} |")
        if fig_engines(engines, os.path.join(a.out, "figures", "engines.png")):
            lines += ["", "![expert engine by layer type](figures/engines.png)"]
        vp = engines.get("vision", {}).get("profile")
        if vp:
            name, ms = vp["top"][0]
            lines += ["", f"The vision engine is {vp['n_layers']} profiled nodes; TensorRT fuses the 27 transformer blocks "
                      f"into one node ({ms:.0f} of {vp['total_ms']:.0f} ms), so it cannot be split by layer type."]
        lines.append("")

    # outputs
    lines += ["## Outputs: hybrid vs PyTorch", ""]
    comp_rows = []
    for pt_cfg, hy_cfg in (("official", "reasoning"), ("force_traj", "force_traj")):
        p, h = os.path.join(runs["pytorch"].get(pt_cfg, ""), "behavior.jsonl"), os.path.join(runs["hybrid"].get(hy_cfg, ""), "behavior.jsonl")
        if not (os.path.exists(p) and os.path.exists(h)):
            continue
        ref, test = load_jsonl(p), load_jsonl(h)
        c = compare(ref, test)
        g, spread = c["aggregate"], seed_spread(ref)
        comp_rows.append((pt_cfg, hy_cfg, g, spread))
        if pt_cfg == "official":
            logged = {}
            for scene_pt in sorted(glob.glob(os.path.join(a.scenes, "*.pt"))) if a.scenes else []:
                import torch

                d = torch.load(scene_pt, weights_only=False)
                if d.get("ego_future_xyz") is not None:
                    logged[d["scene_id"]] = d["ego_future_xyz"][0, 0].numpy()
            fig_trajectories(ref, test, logged, os.path.join(a.out, "figures", "trajectories.png"))
            fig_agreement(c["rows"], spread, os.path.join(a.out, "figures", "agreement.png"))
    if comp_rows:
        lines += ["| pytorch config | hybrid config | scene/seed pairs | identical reasoning | trajectory distance, mean ADE 6.4 s "
                  "| max | seed-to-seed spread (pytorch) | error vs logged path, pytorch -> hybrid |", "|---|---|---|---|---|---|---|---|"]
        for pt_cfg, hy_cfg, g, spread in comp_rows:
            lines.append(
                f"| {pt_cfg} | {hy_cfg} | {g['n_paired_finite']} | {_fmt(100 * g.get('identical_reasoning_rate', 0))} % "
                f"| {_fmt(100 * g.get('vs_ref_ade_6.4s_mean', 0), 1)} cm | {_fmt(100 * g.get('vs_ref_ade_6.4s_max', 0), 1)} cm "
                f"| {_fmt(100 * spread.get('seed_pair_ade_6.4s_mean', 0))} cm "
                f"| {_fmt(g.get('reference_vs_logged_path_ade_6.4s_mean'), 3)} -> {_fmt(g.get('test_vs_logged_path_ade_6.4s_mean'), 3)} m |")
        lines += ["", "![agreement](figures/agreement.png)", "", "![trajectories](figures/trajectories.png)", ""]

    # memory
    lines += ["## Memory", "", "| backend | system RAM used, peak (incl. /dev/shm) | of which /dev/shm at load | PyTorch allocator peak |",
              "|---|---|---|---|"]
    for b, run_dir in (("pytorch", a.pytorch), ("hybrid", a.hybrid)):
        peak = [w.get("telemetry", {}).get("sys_used_mb_max") for s in summaries[b].values() for w in s.get("windows", [])]
        alloc = [w.get("torch_max_allocated_mb") for s in summaries[b].values() for w in s.get("windows", [])]
        peak, alloc = [p for p in peak if p], [x for x in alloc if x]
        shm = meta[b].get("mem_after_load", {}).get("Shmem")
        lines.append(f"| {b} | {_fmt(max(peak) / 1024 if peak else None, 1)} GiB | {_fmt(shm / 1024 if shm else None, 1)} GiB "
                     f"| {_fmt(max(alloc) / 1024 if alloc else None, 1)} GiB |")
    lines.append("")
    with open(os.path.join(a.out, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "report_data.json"), "w") as f:
        json.dump({"stage_means_ms": stage_means, "engines": engines,
                   "comparisons": [{"pytorch": p, "hybrid": h, "aggregate": g, "seed_spread": s} for p, h, g, s in comp_rows]},
                  f, indent=2, default=str)
    print(f"report -> {os.path.join(a.out, 'REPORT.md')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pytorch", required=True, help="run directory of scripts/run.sh pytorch")
    p.add_argument("--hybrid", required=True, help="run directory of scripts/run.sh hybrid")
    p.add_argument("--engines", nargs="*", default=None, help="engines_* directories of scripts/build_engines.sh")
    p.add_argument("--scenes", default=None, help="scene packs, to draw the logged path")
    p.add_argument("--out", required=True)
    build(p.parse_args())


if __name__ == "__main__":
    main()
