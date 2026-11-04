"""Official rollout with TensorRT FP16 vision and expert engines; the LLM stays in PyTorch BF16.

The engines run through trt_bridge/libtrt_stage.so on PyTorch tensors in this process. The official
``sample_trajectories_from_data_with_vlm_rollout`` is unchanged: only the vision forward and the flow step function are
replaced, so tokenization, generation, position ids, flow integration and dynamics stay the official code.

NOT numerically identical to the PyTorch path. The engines are FP16 and the vision output feeds the LLM, so sampled
reasoning can differ. Compare outputs with the PyTorch run (``orinbench.cli compare``) before relying on latencies.
"""

from __future__ import annotations

import torch

from .model import InstrumentedAlpamayo
from .trt_stage import TrtEngine

MASKED = -1e4  # finfo(float32).min overflows FP16; exp(-1e4) is exactly 0 after softmax
GRID_THW = (1, 20, 36)  # per-image patch grid the vision engine is exported for (official 1.5 pixel bounds)


class HybridAlpamayo(InstrumentedAlpamayo):
    def __init__(self, *args, vision_engine: str, expert_engine: str, expert_graph: bool = True, **kwargs):
        self._engine_paths = (vision_engine, expert_engine)
        self._expert_graph = expert_graph
        super().__init__(*args, **kwargs)

    def _install_hooks(self) -> None:
        """Swap in the engines, then let the base class add its stage marks around them."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

        m = self.model
        self.rotary = Qwen3VLTextRotaryEmbedding(m.vlm.config.text_config).float().to(self.device)

        # These weights now live in the engines. visual keeps patch_embed because callers read visual.dtype from it.
        m.vlm.model.visual.blocks = torch.nn.ModuleList()
        m.vlm.model.visual.deepstack_merger_list = torch.nn.ModuleList()
        m.expert.layers = torch.nn.ModuleList()
        torch.cuda.empty_cache()

        self.vision_engine, self.expert_engine = (TrtEngine(p) for p in self._engine_paths)
        if self._expert_graph:
            self.expert_engine.enable_graph()
        self.load_info["hybrid"] = {
            "vision_engine": self._engine_paths[0],
            "expert_engine": self._engine_paths[1],
            "expert_graph": self._expert_graph,
            "expert_cache_len": self.expert_engine.inputs["kv0"][2],
        }
        m.vlm.model.visual.forward = self._vision_forward

        generate = m.vlm.generate

        def keep_cache(*a, **k):
            out = generate(*a, **k)
            self._prompt_cache = out.past_key_values
            return out

        m.vlm.generate = keep_cache

        build = m._build_expert_pos_ids_and_attn_mask

        def keep_expert_inputs(*a, **k):
            self._expert_inputs = build(*a, **k)
            return self._expert_inputs

        m._build_expert_pos_ids_and_attn_mask = keep_expert_inputs

        sample = m.diffusion.sample

        def sample_with_engine(*a, **k):
            self._load_expert_inputs()
            k["step_fn"] = self._flow_step
            return sample(*a, **k)

        m.diffusion.sample = sample_with_engine
        super()._install_hooks()

    def enable_optimizations(self, opts: dict | None) -> dict:
        if opts and (opts.get("vision_batched_attention") or opts.get("expert_cuda_graph")):
            raise ValueError("vision and expert run in TensorRT here; only prefill_last_logit_only applies")
        return super().enable_optimizations(opts)

    def _vision_forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None, **_):
        engine = self.vision_engine
        shape = engine.inputs["pixel_values"]
        if pixel_values.shape != torch.Size(shape):
            raise RuntimeError(f"vision engine is fixed at {shape}, got {tuple(pixel_values.shape)}")
        if grid_thw is not None and (grid_thw != torch.tensor(GRID_THW, device=grid_thw.device)).any():
            raise RuntimeError(f"vision engine is fixed at a {GRID_THW} grid per image, got {grid_thw.tolist()}")
        engine.buffers["pixel_values"].copy_(pixel_values)
        engine.run()
        return engine.buffers["merged"].to(self.dtype), [engine.buffers[f"deepstack{i}"].to(self.dtype) for i in range(3)]

    def _load_expert_inputs(self) -> None:
        """Once per call: write the prompt KV cache, mask and rotary angles into the engine's fixed-length buffers."""
        buffers = self.expert_engine.buffers
        position_ids, official_mask = self._expert_inputs
        cache_len = self.expert_engine.inputs["kv0"][2]
        layers = self._prompt_cache.layers
        length = layers[0].keys.shape[2]
        if length > cache_len:
            raise RuntimeError(f"KV cache has {length} tokens but the expert engine is fixed at {cache_len}")
        for i, layer in enumerate(layers):
            for name, tensor in ((f"kv{2 * i}", layer.keys), (f"kv{2 * i + 1}", layer.values)):
                buffers[name][:, :, :length].copy_(tensor)
                buffers[name][:, :, length:].zero_()
        mask = buffers["attention_mask"]
        mask.zero_()
        mask[..., :length][official_mask[..., :length] < 0] = MASKED
        mask[..., length:cache_len] = MASKED
        mask[..., cache_len:].copy_(official_mask[..., length:])
        with torch.autocast(self.device, enabled=False):
            cos, sin = self.rotary(torch.zeros(1, dtype=torch.float32, device=self.device), position_ids)
        if cos.shape != buffers["cos"].shape:
            raise RuntimeError(f"rotary shape {tuple(cos.shape)} != engine {tuple(buffers['cos'].shape)}")
        buffers["cos"].copy_(cos)
        buffers["sin"].copy_(sin)
        self._prompt_cache = None  # released here; the engine reads only its own buffers from now on

    def _flow_step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        buffers = self.expert_engine.buffers
        buffers["x"].copy_(x)
        buffers["t"].copy_(t)
        self.expert_engine.run()
        return buffers["velocity"]
