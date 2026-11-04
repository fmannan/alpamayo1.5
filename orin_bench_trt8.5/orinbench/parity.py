"""Parity references on real scenes, used by the engine gates in orinbench.trt_stage.

vision: official 1.5 preprocessing -> pixel_values.bin (float32) and a float32 PyTorch reference of the vision tower.
        CPU only; run on the export host.
expert: runs the official model on the GPU (the Orin) and captures the inputs of the first flow step, padded to the
        engine's fixed KV length, with float32 (gate) and BF16 (deployed PyTorch) reference velocities. Padding is
        checked: padded and unpadded float32 must agree.

The differences measure numerical fidelity of the FP16 engines, not driving quality.

    python -m orinbench.parity vision --model CKPT --scenes DIR --ids S [S ...] --out DIR [--vlm-config DIR]
    python -m orinbench.parity expert --model CKPT --scenes DIR --ids S [S ...] --out DIR [--vlm-config DIR]
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from .export import CACHE_LEN, load_prefix, vlm_config
from .hybrid import MASKED
from .inputs import load_scenes


def make_vision(a) -> None:
    from alpamayo1_5 import helper
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    from .patches import enable_batched_vision_attention

    cfg = vlm_config(a.model, a.vlm_config).vision_config
    cfg._attn_implementation = "sdpa"  # the path PyTorch deploys
    model = Qwen3VLVisionModel(cfg).float().eval()
    model.load_state_dict(load_prefix(a.model, "vlm.model.visual.", torch.float32))
    enable_batched_vision_attention(model)
    # the image processor the model uses at run time (helper.get_processor)
    proc = AutoProcessor.from_pretrained(helper.BASE_PROCESSOR_NAME, min_pixels=helper.MIN_PIXELS, max_pixels=helper.MAX_PIXELS)
    os.makedirs(a.out, exist_ok=True)
    for s in load_scenes(a.scenes, ids=a.ids):
        msg = helper.create_message(s.image_frames.flatten(0, 1), camera_indices=s.camera_indices,
                                    num_frames_per_camera=s.image_frames.shape[1])
        inp = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                       return_dict=True, return_tensors="pt")
        px, grid = inp["pixel_values"].float(), inp["image_grid_thw"]
        print(s.scene_id, tuple(px.shape), grid[0].tolist(), flush=True)
        with torch.inference_mode():
            merged, deep = model(px, grid_thw=grid)
        px.numpy().astype(np.float32).tofile(os.path.join(a.out, f"{s.scene_id}.pixel_values.bin"))
        np.savez(os.path.join(a.out, f"{s.scene_id}.ref.npz"), merged=merged.numpy(),
                 **{f"deepstack{i}": d.numpy() for i, d in enumerate(deep)})
    print("VISION REFERENCES DONE", flush=True)


def capture_expert(a) -> None:
    from transformers import DynamicCache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    from .model import InstrumentedAlpamayo, RunConfig

    bench = InstrumentedAlpamayo(a.model, vlm_config_path=a.vlm_config)
    m = bench.model
    rotary = Qwen3VLTextRotaryEmbedding(m.vlm.config.text_config).float().cuda()
    rc = RunConfig(name="capture", max_generation_length=256)
    captured: dict = {}
    expert_forward, aip_forward = m.expert.forward, m.action_in_proj.forward

    def expert_spy(*args, **k):
        if "kv" not in captured:
            captured.update(
                position_ids=k["position_ids"].clone(), attention_mask=k["attention_mask"].clone(),
                n_tokens=k["inputs_embeds"].shape[1],
                kv=[(layer.keys.clone(), layer.values.clone()) for layer in k["past_key_values"].layers],
            )
        return expert_forward(*args, **k)

    def aip_spy(x, t):
        if "x" not in captured:
            captured.update(x=x.clone(), t=t.clone())
        return aip_forward(x, t)

    def step(expert, aip, aop, x, t, pos, mask, kvs, dtype, n):
        cache = DynamicCache()
        for i, (k, v) in enumerate(kvs):
            cache.update(k.to(dtype), v.to(dtype), i)
        out = type(expert).forward(expert, inputs_embeds=aip(x.to(dtype), t.to(dtype)), position_ids=pos,
                                   past_key_values=cache, attention_mask=mask.to(dtype), use_cache=True, is_causal=False)
        return aop(out.last_hidden_state[:, -n:]).view(1, *x.shape[1:]).float()

    def rel(u, w):
        return float((u - w).norm() / w.norm())

    float_modules = None
    os.makedirs(a.out, exist_ok=True)
    for s in load_scenes(a.scenes, ids=a.ids):
        captured.clear()
        m.expert.forward, m.action_in_proj.forward = expert_spy, aip_spy
        bench.call(s, rc, seed=1000, sync_each_stage=False)
        m.expert.forward, m.action_in_proj.forward = expert_forward, aip_forward
        length, n = captured["kv"][0][0].shape[2], captured["n_tokens"]
        if length > CACHE_LEN:
            raise SystemExit(f"{s.scene_id}: KV cache {length} > engine length {CACHE_LEN} (reasoning too long)")
        pad = CACHE_LEN - length
        kv = [(torch.nn.functional.pad(k, (0, 0, 0, pad)).float(), torch.nn.functional.pad(v, (0, 0, 0, pad)).float())
              for k, v in captured["kv"]]
        # the official mask extended over the padding, as orinbench.hybrid writes it at run time
        raw = captured["attention_mask"].float()
        mask = torch.zeros(1, 1, n, CACHE_LEN + n, device=raw.device)
        mask[..., :length][raw[..., :length] < 0] = MASKED
        mask[..., length:CACHE_LEN] = MASKED
        mask[..., CACHE_LEN:] = raw[..., length:]
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            cos, sin = rotary(torch.zeros(1, dtype=torch.float32, device="cuda"), captured["position_ids"])
        inputs = {"x": captured["x"].float(), "t": captured["t"].float(), "attention_mask": mask, "cos": cos, "sin": sin}
        for i, (k, v) in enumerate(kv):
            inputs[f"kv{2 * i}"], inputs[f"kv{2 * i + 1}"] = k, v
        d = os.path.join(a.out, s.scene_id)
        os.makedirs(d, exist_ok=True)
        for name, tensor in inputs.items():
            tensor.detach().float().cpu().numpy().astype(np.float32).tofile(os.path.join(d, name + ".bin"))
        with open(os.path.join(d, "inputs.json"), "w") as f:
            json.dump({k: list(v.shape) for k, v in inputs.items()} | {"prompt_cache_len": length}, f)

        pos = captured["position_ids"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16 = step(m.expert, aip_forward, m.action_out_proj, captured["x"], captured["t"], pos, raw,
                        captured["kv"], torch.bfloat16, n)
        if float_modules is None:
            float_modules = tuple(copy.deepcopy(mod).float() for mod in (m.expert, m.action_in_proj, m.action_out_proj))
        fe, fi, fo = float_modules
        with torch.inference_mode():
            unpadded = step(fe, fi, fo, captured["x"].float(), captured["t"].float(), pos, raw,
                            [(k.float(), v.float()) for k, v in captured["kv"]], torch.float32, n)
            padded = step(fe, fi, fo, inputs["x"], inputs["t"], pos, mask, kv, torch.float32, n)
        check = {"prompt_cache_len": length, "padded_vs_unpadded_fp32": rel(padded, unpadded), "bf16_vs_fp32": rel(bf16, unpadded)}
        print(s.scene_id, json.dumps(check), flush=True)
        if not check["padded_vs_unpadded_fp32"] < 1e-4:
            raise SystemExit(f"{s.scene_id}: padding changes the float32 result")
        np.savez(os.path.join(d, "ref.npz"), velocity=padded.cpu().numpy(), bf16=bf16.cpu().numpy())
    print("EXPERT REFERENCES DONE", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("vision", make_vision), ("expert", capture_expert)):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True, help="Alpamayo-1.5-10B checkpoint directory")
        s.add_argument("--scenes", required=True, help="scene pack directory (orinbench.cli prep-scenes)")
        s.add_argument("--ids", nargs="+", required=True, help="scene ids from the pack's manifest.json")
        s.add_argument("--out", required=True)
        s.add_argument("--vlm-config", default=None, help="local Qwen3-VL config dir (default: the checkpoint's)")
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
