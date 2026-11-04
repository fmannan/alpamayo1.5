"""orinbench command line. ``python -m orinbench.cli <command> -h`` for details.

    env          platform fingerprint
    prep-scenes  freeze dataset scenes into scene packs (needs HuggingFace dataset access)
    run          benchmark + outputs of one config file (configs/*.yaml)
    compare      pair two behavior.jsonl files by scene and seed
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time

import yaml


def load_yaml(path: str) -> dict:
    """YAML with ${VAR} environment references expanded; unset variables are an error."""
    with open(path) as f:
        text = os.path.expandvars(f.read())
    left = sorted(set(re.findall(r"\$\{(\w+)\}", text)))
    if left:
        raise SystemExit(f"{path}: environment variables not set: {left} (source env.sh)")
    return yaml.safe_load(text)


def _mem_snapshot() -> dict:
    import torch

    from .telemetry import meminfo

    out = dict(meminfo())
    if torch.cuda.is_available():
        out["torch_allocated_mb"] = torch.cuda.memory_allocated() / 2**20
    return out


def cmd_env(a) -> None:
    from .envinfo import collect

    print(json.dumps(collect(), indent=2, default=str))


def cmd_prep_scenes(a) -> None:
    from .inputs import prepare_scenes

    prepare_scenes(load_yaml(a.specs)["scenes"], a.out)


def cmd_run(a) -> None:
    import torch

    from .behavior import behavior_pass, seed_spread
    from .envinfo import collect
    from .inputs import load_scenes
    from .model import InstrumentedAlpamayo, RunConfig
    from .runner import Plan, run_config, write_json
    from .telemetry import Telemetry

    m = load_yaml(a.config)
    if a.drop_weights_after_load and not os.path.realpath(a.model).startswith("/dev/shm/"):
        raise SystemExit("--drop-weights-after-load only deletes a checkpoint copy in /dev/shm, never one on disk")
    out = os.path.join(a.out, f"{dt.datetime.now():%Y%m%d-%H%M%S}_{m['experiment']}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(m, f)
    write_json(collect(), os.path.join(out, "env.json"))
    scenes = load_scenes(a.scenes, limit=m.get("n_scenes"))
    print(f"[run] {len(scenes)} scenes -> {out}", flush=True)
    telem = Telemetry(os.path.join(out, "telemetry"), interval_ms=a.telemetry_ms).start()
    meta: dict = {"argv": sys.argv, "mem_before_load": _mem_snapshot(), "t_load_start": time.time()}
    try:
        kwargs = dict(vlm_config_path=a.vlm_config, dtype=m.get("load_dtype", "bfloat16"),
                      attn_implementation=m.get("attn_implementation", "sdpa"))
        if m.get("trt"):
            from .hybrid import HybridAlpamayo

            bench = HybridAlpamayo(a.model, **kwargs, **m["trt"])
        else:
            bench = InstrumentedAlpamayo(a.model, **kwargs)
        meta.update(t_load_end=time.time(), load=bench.load_info, mem_after_load=_mem_snapshot())
        if a.drop_weights_after_load:
            # The checkpoint is resident in the model now. On a 64 GB unified-memory device a checkpoint kept in
            # /dev/shm competes with the running model for the same RAM; deleting the files does not change the math.
            for root, _, files in os.walk(a.model):
                for fn in files:
                    if fn.endswith(".safetensors"):
                        os.remove(os.path.join(root, fn))
            meta["mem_after_drop_weights"] = _mem_snapshot()
        meta["optimizations"] = bench.enable_optimizations(m.get("optimizations"))
    except Exception as e:  # noqa: BLE001
        import traceback

        meta.update(load_error=repr(e), traceback=traceback.format_exc())
        write_json(meta, os.path.join(out, "meta.json"))
        telem.stop()
        raise
    write_json(meta, os.path.join(out, "meta.json"))
    beh = m.get("behavior", {})
    try:
        for c in m["configs"]:
            c = dict(c)
            plan = Plan.from_dict({**m.get("plan", {}), **c.pop("plan", {})})
            rc = RunConfig(**c)
            cdir = os.path.join(out, rc.name)
            write_json(rc.as_dict(), os.path.join(cdir, "config.json"))
            print(f"[run] config {rc.name}: {rc.as_dict()}", flush=True)
            if not a.skip_timing:
                lat = run_config(bench, rc, scenes, plan, telem, cdir).get("varied", {}).get("e2e_wall_ms", {})
                print(f"[run]   {rc.name} e2e ms: p50={lat.get('p50')} p99={lat.get('p99')} max={lat.get('max')}", flush=True)
            if beh.get("seeds"):
                torch.cuda.empty_cache()
                recs = behavior_pass(bench, rc, scenes[: beh.get("n_scenes", len(scenes))], beh["seeds"],
                                     out_path=os.path.join(cdir, "behavior.jsonl"))
                write_json(seed_spread(recs), os.path.join(cdir, "seed_spread.json"))
    finally:
        telem.stop()
    print(f"[run] done -> {out}", flush=True)


def cmd_compare(a) -> None:
    from .behavior import compare, load_jsonl

    result = compare(load_jsonl(a.reference), load_jsonl(a.test))
    print(json.dumps(result["aggregate"], indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"reference": a.reference, "test": a.test, **result}, f, indent=2)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="orinbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("env", help="platform fingerprint")
    s.set_defaults(fn=cmd_env)

    s = sub.add_parser("prep-scenes", help="freeze dataset scenes into packs (needs HF dataset access)")
    s.add_argument("--specs", required=True, help="YAML with a 'scenes' list of {clip_id, t0_us} (data/scenes.yaml)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_prep_scenes)

    s = sub.add_parser("run", help="benchmark + outputs for one config file")
    s.add_argument("--model", required=True, help="Alpamayo-1.5-10B checkpoint directory")
    s.add_argument("--scenes", required=True, help="scene pack directory")
    s.add_argument("--config", required=True, help="configs/*.yaml")
    s.add_argument("--out", required=True, help="results directory; each run gets a timestamped subdirectory")
    s.add_argument("--vlm-config", default=None, help="local Qwen3-VL config dir (default: the checkpoint's)")
    s.add_argument("--telemetry-ms", type=int, default=100)
    s.add_argument("--drop-weights-after-load", action="store_true", help="delete the checkpoint's .safetensors after load")
    s.add_argument("--skip-timing", action="store_true", help="outputs (behavior pass) only")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("compare", help="pair two behavior.jsonl files by scene and seed")
    s.add_argument("--reference", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--out", default=None, help="also write the full comparison (per-pair rows) as JSON")
    s.set_defaults(fn=cmd_compare)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
