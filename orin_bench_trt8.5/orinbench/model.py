"""Instrumented official Alpamayo 1.5 rollout.

``Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout`` runs unmodified unless a config enables
``optimizations``. Stage boundaries are taken by wrapping submodule entry points:

    start ......... input ready: decoded uint8 frames + ego history in host memory
    preprocess_cpu  official chat template, Qwen3-VL image resize/normalize/patchify, tokenize
    h2d ........... host->device copies of pixel values, ids, ego history
    pre_vision .... history-trajectory token fusion + generate() setup + text embedding
    vision ........ Qwen3-VL vision tower (vlm.model.visual)
    llm_prefill ... remainder of the first LM forward (deepstack merge, 36 layers, lm_head)
    decode ........ autoregressive steps until <traj_future_start> + 1 token, or the budget
    expert_prep ... cache length, position ids (rope deltas), attention mask
    flow .......... all flow-matching steps (action_in_proj -> expert over the prompt KV cache -> out_proj)
    postprocess ... unicycle integration, reshape, text extraction, device->host copy ==> trajectory ready
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from transformers import LogitsProcessor

from .inputs import Scene
from .patches import apply_platform_patches, enable_batched_vision_attention
from .timing import StageClock


@dataclass
class RunConfig:
    name: str
    max_generation_length: int = 256  # official example value
    top_p: float = 0.98
    temperature: float = 0.6
    force_tokens: list[str] | None = None  # NOT equivalent: these tokens replace the first sampled ones (skips reasoning)
    common_flow_noise: bool = False  # reseed before flow so configs sharing a seed start from identical noise

    def deviations(self) -> list[str]:
        d = []
        if self.force_tokens:
            d.append(f"forced_tokens={self.force_tokens}(not_equivalent)")
        if self.max_generation_length < 256:
            d.append(f"reasoning_budget={self.max_generation_length}(truncation_possible)")
        return d

    def as_dict(self) -> dict:
        return {**asdict(self), "deviations_from_official": self.deviations()}


class ForceTokens(LogitsProcessor):
    """Force the first generated tokens, then leave sampling untouched."""

    def __init__(self, token_ids: list[int], prompt_len: int):
        self.token_ids, self.prompt_len = token_ids, prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        i = input_ids.shape[1] - self.prompt_len
        if i < len(self.token_ids):
            scores[:] = float("-inf")
            scores[:, self.token_ids[i]] = 0.0
        return scores


def load_config(model_path: str, vlm_config_path: str | None, attn_implementation: str):
    """Checkpoint config, optionally with a local stand-in for the (gated) VLM config named in it.

    The config class builds the tokenizer from ``vlm_name_or_path`` while constructing, so the path is injected
    into the raw dict first. ``check_trajectory_tokens`` then verifies the resulting token ids against the checkpoint.
    """
    from alpamayo1_5.config import Alpamayo1_5Config

    with open(os.path.join(model_path, "config.json")) as f:
        raw = json.load(f)
    if vlm_config_path:
        raw["vlm_name_or_path"] = vlm_config_path
    # The checkpoint requests flash_attention_2 (flash-attn package). SDPA is the upstream-documented fallback.
    raw["attn_implementation"] = attn_implementation
    cfg = Alpamayo1_5Config.from_dict(raw)
    cfg._attn_implementation = attn_implementation
    cfg.attn_implementation = attn_implementation
    return cfg


class InstrumentedAlpamayo:
    def __init__(self, model_path: str, *, vlm_config_path: str | None = None, dtype: str = "bfloat16",
                 attn_implementation: str = "sdpa", device: str = "cuda"):
        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        self.helper, self.device = helper, device
        self.dtype = getattr(torch, dtype)
        self.load_info: dict = {"model_path": model_path, "dtype": dtype, "vlm_config_path": vlm_config_path}
        t0 = time.perf_counter()
        cfg = load_config(model_path, vlm_config_path, attn_implementation)
        model, loading = Alpamayo1_5.from_pretrained(model_path, config=cfg, dtype=self.dtype, output_loading_info=True)
        if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
            raise RuntimeError(f"checkpoint/config mismatch: { {k: list(v)[:20] for k, v in loading.items()} }")
        model = model.to(device).eval()
        self.load_info["token_checks"] = check_trajectory_tokens(model, cfg)
        if device == "cuda":
            torch.cuda.synchronize()
        self.load_info["load_s"] = time.perf_counter() - t0
        self.model = model
        self.processor = helper.get_processor(model.tokenizer)
        self.load_info["platform_patches"] = apply_platform_patches(model.vlm.model.visual)
        self.load_info["attn_impl_vlm"] = getattr(model.vlm.config, "_attn_implementation", None)
        self.load_info["attn_impl_expert"] = getattr(model.expert.config, "_attn_implementation", None)
        self._install_hooks()

    # ------------------------------------------------------------------ instrumentation
    def _install_hooks(self) -> None:
        m = self.model
        self.clock: StageClock | None = None
        self._state: dict = {}
        self.logits_to_keep: int | None = None

        vis_fwd = m.vlm.model.visual.forward

        def visual_forward(*a, **k):
            if self.clock:
                self.clock.mark("pre_vision")
            out = vis_fwd(*a, **k)
            if self.clock:
                self.clock.mark("vision")
            return out

        m.vlm.model.visual.forward = visual_forward

        vlm_fwd = m.vlm.forward

        def vlm_forward(*a, **k):
            st = self._state
            st["n_fwd"] = st.get("n_fwd", 0) + 1
            if self.logits_to_keep is not None and st["n_fwd"] == 1:
                k["logits_to_keep"] = self.logits_to_keep
            if st["n_fwd"] == 2 and self.clock:
                st["ttft_wall"] = time.perf_counter()  # the first token is sampled before the 2nd forward
            out = vlm_fwd(*a, **k)
            if st["n_fwd"] == 1 and self.clock:
                self.clock.mark("llm_prefill")
            elif self.clock:
                st.setdefault("step_wall", []).append(time.perf_counter())
            return out

        m.vlm.forward = vlm_forward

        gen = m.vlm.generate

        def generate(*a, **k):
            if self._state.get("force_ids"):
                ids = k.get("input_ids") if k.get("input_ids") is not None else a[0]
                k["logits_processor"].append(ForceTokens(self._state["force_ids"], ids.shape[1]))
            out = gen(*a, **k)
            if self.clock:
                self.clock.mark("decode")
            self._state["sequences"] = out.sequences
            self._state["kv_seq_len"] = int(out.past_key_values.get_seq_length())
            return out

        m.vlm.generate = generate

        sample = m.diffusion.sample

        def diffusion_sample(*a, **k):
            if self.clock:
                self.clock.mark("expert_prep")
            if self._state.get("flow_seed") is not None:
                torch.manual_seed(self._state["flow_seed"])
                if self.device == "cuda":
                    torch.cuda.manual_seed_all(self._state["flow_seed"])
            out = sample(*a, **k)
            if self.clock:
                self.clock.mark("flow")
            return out

        m.diffusion.sample = diffusion_sample

    # ------------------------------------------------------------------ optimizations
    def enable_optimizations(self, opts: dict | None) -> dict:
        """Behavior-preserving options. Check each against the baseline's outputs before trusting its latency."""
        opts = opts or {}
        info: dict = {}
        if opts.get("prefill_last_logit_only"):
            # generate() samples from the last position only, but the default forward projects all ~3,000
            # prefill positions through lm_head (4096 x ~156k). Same values at the last position.
            self.logits_to_keep = 1
            info["prefill_last_logit_only"] = True
        if opts.get("vision_batched_attention"):
            info["vision_batched_attention_blocks"] = enable_batched_vision_attention(self.model.vlm.model.visual)
        if opts.get("expert_cuda_graph"):
            g = opts["expert_cuda_graph"]
            self.model.enable_diffusion_expert_cuda_graph(
                max_batch_size=g.get("max_batch_size", 1), max_graphs=g.get("max_graphs", 4)
            )
            info["expert_cuda_graph"] = g
        return info

    # ------------------------------------------------------------------ one policy call
    def build_inputs(self, scene: Scene):
        frames = scene.image_frames.flatten(0, 1)
        # official 1.5 prompt: every image is labeled with its camera name and frame index
        messages = self.helper.create_message(
            frames, camera_indices=scene.camera_indices,
            num_frames_per_camera=frames.shape[0] // len(scene.camera_indices),
        )
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
            return_dict=True, return_tensors="pt",
        )

    @torch.inference_mode()
    def call(self, scene: Scene, rc: RunConfig, seed: int, sync_each_stage: bool) -> dict:
        tok = self.model.tokenizer
        self._state = {
            "force_ids": tok.convert_tokens_to_ids(rc.force_tokens) if rc.force_tokens else None,
            "flow_seed": seed if rc.common_flow_noise else None,
        }
        if self.device == "cuda":
            torch.cuda.synchronize()
        clock = self.clock = StageClock(sync_each_stage=sync_each_stage, use_cuda=self.device == "cuda")
        clock.mark("start")
        inputs = self.build_inputs(scene)
        n_prompt = int(inputs["input_ids"].shape[1])
        n_image = int((inputs["input_ids"] == self.model.vlm.config.image_token_id).sum())
        clock.mark("preprocess_cpu")
        model_inputs = self.helper.to_device(
            {"tokenized_data": inputs, "ego_history_xyz": scene.ego_history_xyz, "ego_history_rot": scene.ego_history_rot},
            self.device,
        )
        clock.mark("h2d")
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.autocast(self.device, dtype=self.dtype):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=rc.top_p, temperature=rc.temperature, num_traj_samples=1,
                max_generation_length=rc.max_generation_length, return_extra=True,
            )
        pred_xyz, pred_rot = pred_xyz.float().cpu(), pred_rot.float().cpu()
        clock.mark("postprocess")
        self.clock = None
        res = clock.finalize()
        st = self._state
        gen_ids = st["sequences"][0, n_prompt:].tolist()
        steps = st.get("step_wall", [])
        res.update(
            {
                "seed": seed,
                "scene_id": scene.scene_id,
                "n_prompt_tokens": n_prompt,
                "n_image_tokens": n_image,
                "n_generated_tokens": len(gen_ids),
                "kv_seq_len": st.get("kv_seq_len"),
                "reasoning_complete": tok.convert_tokens_to_ids("<|cot_end|>") in gen_ids,
                "traj_start_emitted": tok.convert_tokens_to_ids("<|traj_future_start|>") in gen_ids,
                "ttft_wall_ms": (st["ttft_wall"] - clock.marks[0].wall) * 1e3 if "ttft_wall" in st else None,
                "decode_step_ms": (np.diff(np.array(steps)) * 1e3).tolist() if len(steps) > 1 else [],
                "cot": str(extra["cot"].reshape(-1)[0]) if "cot" in extra else None,
                "pred_xyz": pred_xyz[0, 0, 0].numpy(),
                "gen_token_ids": gen_ids,
                "finite": bool(torch.isfinite(pred_xyz).all() and torch.isfinite(pred_rot).all()),
            }
        )
        return res


def check_trajectory_tokens(model, cfg) -> dict:
    """The tokenizer must reproduce the checkpoint's trajectory token ids and vocabulary size exactly."""
    names = {"future": "traj_future", "future_end": "traj_future_end", "future_start": "traj_future_start",
             "history": "traj_history", "history_end": "traj_history_end", "history_start": "traj_history_start"}
    got = {k: model.tokenizer.convert_tokens_to_ids(f"<|{v}|>") for k, v in names.items()}
    want = dict(cfg.traj_token_ids)
    if got != want or len(model.tokenizer) != cfg.vocab_size:
        raise RuntimeError(f"tokenizer mismatch: got {got} len={len(model.tokenizer)}, want {want} len={cfg.vocab_size}")
    return {"traj_token_ids_match": True, "vocab_size": cfg.vocab_size}


def ade_fde(pred: np.ndarray, ref: np.ndarray, horizons_steps=(30, 64)) -> dict:
    """xy displacement errors in the ego frame at t0 (x forward, y left), 10 Hz steps."""
    d = np.linalg.norm(np.asarray(pred)[:, :2] - np.asarray(ref)[:, :2], axis=1)
    out = {}
    for h in horizons_steps:
        h = min(h, len(d))
        out[f"ade_{h / 10:.1f}s"] = float(d[:h].mean())
        out[f"fde_{h / 10:.1f}s"] = float(d[h - 1])
    return out


def to_jsonable(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out
