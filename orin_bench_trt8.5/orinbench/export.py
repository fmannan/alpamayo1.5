"""Export the two fixed-shape subgraphs that TensorRT 8.5 runs: the vision tower and one flow step of the expert.

TensorRT 8.5 needs static shapes and has no BF16, so the engines are FP16 although the checkpoint is BF16. Export runs
in float32 on the CPU (any host with ~64 GB of RAM; no GPU needed) and trtexec converts with --fp16. Two changes keep
the FP16 engines valid, and neither changes the math (orinbench/patches.py):
  * ScaledLayerNorm / ScaledRMSNorm: per-token rescaling so squared activations cannot overflow FP16;
  * batched vision attention: one attention call over the 16 equal-sized images instead of a Python loop.

    python -m orinbench.export vision --model CKPT --out DIR/vision.onnx [--vlm-config DIR]
    python -m orinbench.export expert --model CKPT --out DIR/expert.onnx [--vlm-config DIR]
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import torch
from safetensors import safe_open

from .patches import enable_batched_vision_attention, enable_scaled_layernorm, enable_scaled_rmsnorm

# Official 1.5 prompt: 4 cameras x 4 frames, each 20 x 36 patches after the official min/max pixel bounds.
N_IMAGES, GRID_H, GRID_W = 16, 20, 36
# Expert KV length: 3,086 prompt tokens + 50 generated. Longer reasoning than 50 tokens does not fit the engine.
CACHE_LEN = 3136


def vlm_config(model_dir: str, override: str | None):
    """Qwen3-VL config named by the checkpoint, or a local stand-in directory (see README, 'VLM config')."""
    from transformers import Qwen3VLConfig

    if override is None:
        with open(os.path.join(model_dir, "config.json")) as f:
            override = json.load(f)["vlm_name_or_path"]
    return Qwen3VLConfig.from_pretrained(override)


def load_prefix(model_dir: str, prefix: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Read only the tensors under ``prefix`` from a sharded safetensors checkpoint."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    wanted = {k for k in weight_map if k.startswith(prefix)}
    sd: dict[str, torch.Tensor] = {}
    for shard in sorted({weight_map[k] for k in wanted}):
        with safe_open(os.path.join(model_dir, shard), "pt") as f:
            for key in f.keys():
                if key in wanted:
                    sd[key[len(prefix):]] = f.get_tensor(key).to(dtype)
    if not sd:
        raise SystemExit(f"no weights under {prefix!r} in {model_dir}")
    return sd


class VisionWrapper(torch.nn.Module):
    """Vision tower at one fixed image grid: pixels -> merged embeddings + deepstack features."""

    def __init__(self, visual: torch.nn.Module, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual
        self.register_buffer("grid_thw", grid_thw, persistent=False)

    def forward(self, pixel_values: torch.Tensor):
        merged, deepstack = self.visual(pixel_values, grid_thw=self.grid_thw)
        return (merged, *deepstack)


def export_vision(a) -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = vlm_config(a.model, a.vlm_config).vision_config
    cfg._attn_implementation = "eager"  # same math as SDPA; the ONNX exporter cannot convert this SDPA call
    model = Qwen3VLVisionModel(cfg).float().eval()
    # Weights rounded to FP16 first (the engine precision), then traced in float32: CPU half ops are slow/incomplete.
    sd = load_prefix(a.model, "vlm.model.visual.", torch.float16)
    result = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    print("missing", result.missing_keys[:5], "unexpected", result.unexpected_keys[:5])
    if result.unexpected_keys:
        raise SystemExit("checkpoint/config mismatch")
    print("batched attention blocks:", enable_batched_vision_attention(model))
    print("scaled layernorms:", enable_scaled_layernorm(model))
    grid = torch.tensor([[1, GRID_H, GRID_W]] * N_IMAGES, dtype=torch.long)
    channels = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2
    dummy = torch.randn(int(grid.prod(dim=1).sum()), channels)
    wrapper = VisionWrapper(model, grid).eval()
    names = ["merged"] + [f"deepstack{i}" for i in range(len(cfg.deepstack_visual_indexes))]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with torch.inference_mode():
        outputs = wrapper(dummy)
        # opset 16 decomposes LayerNormalization, which TensorRT 8.5 cannot import
        torch.onnx.export(wrapper, (dummy,), a.out, input_names=["pixel_values"], output_names=names,
                          opset_version=16, do_constant_folding=True, dynamo=False)
    print(f"exported {a.out}: pixel_values={tuple(dummy.shape)} outputs={[tuple(t.shape) for t in outputs]}")


class RotaryInput(torch.nn.Module):
    """Stands in for the rotary module: cos/sin are engine inputs computed in float32 outside the engine."""

    def forward(self, x, position_ids):
        return self.cos, self.sin


class ExpertStep(torch.nn.Module):
    """One flow step at a fixed cache length: (x, t, attention_mask, cos, sin, kv...) -> velocity.

    Rotary angles are inputs because positions here are about 500-600 after rope deltas, where FP16 angle spacing is
    0.5 rad: too coarse for cos/sin at high frequency. Upstream also computes them in float32.
    """

    def __init__(self, expert, action_in_proj, action_out_proj, dims, position_ids):
        super().__init__()
        self.expert, self.action_in_proj, self.action_out_proj = expert, action_in_proj, action_out_proj
        self.dims = tuple(dims)
        self.expert.rotary_emb = RotaryInput()
        self.register_buffer("position_ids", position_ids, persistent=False)

    def forward(self, x, t, attention_mask, cos, sin, *kv):
        from transformers import DynamicCache

        cache = DynamicCache()
        for i in range(len(kv) // 2):
            cache.update(kv[2 * i], kv[2 * i + 1], i)
        self.expert.rotary_emb.cos, self.expert.rotary_emb.sin = cos, sin
        out = self.expert(
            inputs_embeds=self.action_in_proj(x, t), position_ids=self.position_ids, past_key_values=cache,
            attention_mask=attention_mask, use_cache=True, is_causal=False,
        )
        return self.action_out_proj(out.last_hidden_state[:, -self.dims[0]:]).view(-1, *self.dims)


def _sdpa_math_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Same math as SDPA in the input dtype; transformers' eager path adds a float32 softmax the deployed path lacks."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : key.shape[-2]]
    weights = torch.nn.functional.softmax(weights, dim=-1)
    return torch.matmul(weights, value).transpose(1, 2), weights


def export_expert(a) -> None:
    import hydra.utils as hyu
    from transformers import AutoModel
    from transformers.models.qwen3_vl import modeling_qwen3_vl as mq

    with open(os.path.join(a.model, "config.json")) as f:
        raw = json.load(f)
    text = copy.deepcopy(vlm_config(a.model, a.vlm_config).text_config)
    for k, v in raw["expert_cfg"].items():
        setattr(text, k, v)
    text._attn_implementation = "eager"
    mq.eager_attention_forward = _sdpa_math_eager
    expert = AutoModel.from_config(text, dtype=torch.float32).float().eval()
    del expert.embed_tokens
    dims = list(hyu.instantiate(raw["action_space_cfg"]).get_action_space_dims())
    aip = hyu.instantiate(raw["action_in_proj_cfg"], in_dims=dims, out_dim=text.hidden_size).float().eval()
    aop = hyu.instantiate(raw["action_out_proj_cfg"], in_features=text.hidden_size, out_features=dims[-1]).float().eval()
    for prefix, mod in (("expert.", expert), ("action_in_proj.", aip), ("action_out_proj.", aop)):
        r = mod.load_state_dict(load_prefix(a.model, prefix, torch.float32), strict=False)
        if r.unexpected_keys or [k for k in r.missing_keys if "rotary" not in k]:
            raise SystemExit(f"{prefix} checkpoint/config mismatch: {r}")
    print("scaled rmsnorms:", enable_scaled_rmsnorm(expert), "scaled layernorms:", enable_scaled_layernorm(aip), flush=True)
    kv = [torch.zeros(1, text.num_key_value_heads, CACHE_LEN, text.head_dim) for _ in range(2 * text.num_hidden_layers)]
    x, t = torch.randn(1, *dims), torch.tensor([[[0.5]]])
    pos = torch.arange(dims[0]).view(1, 1, -1).expand(3, 1, -1).clone() + CACHE_LEN
    cos, sin = expert.rotary_emb(x, pos)
    step = ExpertStep(expert, aip, aop, dims, pos).eval()
    mask = torch.zeros(1, 1, dims[0], CACHE_LEN + dims[0])
    names = ["x", "t", "attention_mask", "cos", "sin"] + [f"kv{i}" for i in range(len(kv))]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with torch.inference_mode():
        out = step(x, t, mask, cos, sin, *kv)
        torch.onnx.export(step, (x, t, mask, cos, sin, *kv), a.out, input_names=names, output_names=["velocity"],
                          opset_version=16, do_constant_folding=True, dynamo=False)
    print(f"exported {a.out}: cache_len={CACHE_LEN} layers={text.num_hidden_layers} output={tuple(out.shape)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("vision", export_vision), ("expert", export_expert)):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True, help="Alpamayo-1.5-10B checkpoint directory")
        s.add_argument("--out", required=True)
        s.add_argument("--vlm-config", default=None, help="local Qwen3-VL config dir (default: the checkpoint's)")
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
