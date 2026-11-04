# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Inference code for **Alpamayo 1.5** (NVIDIA), a ~10B Vision-Language-Action model for autonomous driving. A Qwen3-VL backbone produces chain-of-causation (CoC) reasoning text. A diffusion "expert" transformer (flow matching) then produces 6.4 s future ego trajectories (64 waypoints at 10 Hz), conditioned on the VLM's KV cache. Weights live on HuggingFace (`nvidia/Alpamayo-1.5-10B`, gated, ~22 GB). Data streams from the gated `nvidia/PhysicalAI-Autonomous-Vehicles` dataset via the `physical_ai_av` package. Training and post-training code is **not** in this repo (it lives in NVlabs/alpamayo-recipes).

## Environment & commands

Python is pinned to 3.12, and `torch==2.8.0` and `transformers==4.57.1` are pinned exactly. Internal code relies on transformers cache internals, so don't bump these casually. The project uses `uv`.

```bash
uv venv a1_5_venv && source a1_5_venv/bin/activate
uv sync --active                                     # needs CUDA 12.x nvcc to build flash-attn
uv sync --active --no-install-package flash-attn     # fallback; then load with attn_implementation="sdpa"

pytest tests/                                         # unit tests (GPU-only; skipped without CUDA)
pytest tests/test_diffusion_expert_cuda_graph.py::<test_name>   # single test
python src/alpamayo1_5/test_inference.py             # end-to-end smoke test: downloads weights + a clip, prints CoC and minADE
```

- `hf auth login` is required for the end-to-end script and the notebooks.
- Inference needs a GPU with at least 24 GB of VRAM.
- Formatting: ruff with `line-length = 100` (configured in `pyproject.toml`). CONTRIBUTING asks for `pre-commit` formatting, but the repo has no `.pre-commit-config.yaml`.
- There is no CI, so tests and the inference script are run manually.

## Architecture

**Model class hierarchy** (`src/alpamayo1_5/models/`):
- `base_model.py`: `ReasoningVLAConfig` / `ReasoningVLA` (a HF `PreTrainedModel`).
  - Building the config instantiates the Qwen3-VL processor. It also adds `traj_vocab_size` (768) discrete `<iN>` trajectory tokens plus trajectory special tokens to the tokenizer.
  - `TrajectoryFusionMixin.fuse_traj_tokens` tokenizes ego history (xyz + rotation) and substitutes the tokens into placeholder pad positions in `input_ids`.
  - `generate_text` is the VQA (text-only) path.
- `alpamayo1_5.py`: `Alpamayo1_5` adds the diffusion expert. The expert is a copy of the VLM's text config, built with `AutoModel.from_config`, without `embed_tokens`. It is forced to `sdpa` because the expert does not support FlashAttention 2.
  - `action_space`, `diffusion`, `action_in_proj`, and `action_out_proj` are all built from config dicts via `hydra.utils.instantiate`. Their class paths live in the HF checkpoint's config, not in this repo.
- `config.py`: `Alpamayo1_5Config` (HF `model_type = "alpamayo1_5"`).

**Inference flow**, from `sample_trajectories_from_data_with_vlm_rollout`:
1. Fuse the history trajectory tokens into `input_ids`.
2. Run `vlm.generate` with sampling.
   - `ExpertLogitsProcessor` masks out the discrete trajectory tokens.
   - `StopAfterEOS` stops one token *after* `<traj_future_start>`, so that the KV cache contains it.
3. Build the expert inputs on top of the VLM KV cache.
   - Use Qwen 3-component RoPE position ids (offset by `rope_deltas` and the EOS position).
   - Build a 4D float attention mask that hides the gap between the EOS and the diffusion tokens, plus any left-padding.
4. `step_fn(x, t)`: `action_in_proj` → expert forward with `past_key_values=prompt_cache` → `prompt_cache.crop(prefill_len)` → `action_out_proj`.
5. `FlowMatching.sample` (Euler integration) calls `step_fn` repeatedly.
6. `action_space.action_to_traj` (unicycle acceleration/curvature model) converts the actions to xyz and rotation.
7. Outputs have shape `[B, num_traj_sets, num_traj_samples, T, ...]`. With `return_extra=True`, the decoded text fields (e.g. `extra["cot"]`) are also returned.

`sample_trajectories_from_data_with_vlm_rollout_cfg_nav` is the navigation classifier-free-guidance variant. It builds a second, "unguided" prompt by stripping the `<|route_start|>…<|route_end|>` span (`nav_utils.remove_nav_text`). It prefills that prompt separately and passes both guided and unguided step functions to the diffusion sampler.

**CUDA graph acceleration** (`models/diffusion_expert_cuda_graph.py`):
- This is an opt-in wrapper around `self.expert.forward`, enabled with `model.enable_diffusion_expert_cuda_graph(max_batch_size, max_graphs)`.
- It captures exact-shape graphs lazily, keeps an LRU-bounded set of them, and falls back to eager execution for shapes it hasn't captured.
- It uses a read-only prompt-cache layer, so `crop` doesn't mutate static buffers. It is active only inside the `sampling()` context.
- Changes to `step_fn`'s expert call signature or to the cache handling must stay compatible with this wrapper. `tests/test_diffusion_expert_cuda_graph.py` covers it with a tiny fake expert.

**Input preparation**:
- `load_physical_aiavdataset.py`: given a clip id and `t0_us`, returns a dict with `image_frames`, `camera_indices`, and `ego_history_xyz/rot`.
- `ego_future_xyz/rot` holds the ground truth.
- `helper.py`: `create_message` / `create_vqa_message` (chat messages), `get_processor`, and `to_device`.
- Model input dict: `{"tokenized_data": processor.apply_chat_template(..., return_dict=True), "ego_history_xyz", "ego_history_rot"}`. Exactly one trajectory group is supported at inference time.

The notebooks in `notebooks/` demonstrate the four capabilities: standard inference, navigation conditioning, variable camera counts, and VQA.

## Contribution conventions (from CONTRIBUTING.md)

- Commit format: `#<Issue Number> - <Imperative title>`, signed off with `git commit -s` (DCO is required).
- Every source file starts with the NVIDIA SPDX Apache-2.0 header.
- Keep each PR to a single concern. New components need a test.

## Jetson AGX Orin kit (`orin_bench_trt8.5/`)

A self-contained TensorRT 8.5 benchmark for Jetson AGX Orin 64 GB / JetPack 5.1; its `README.md` is the guide. It uses
`src/alpamayo1_5` unmodified (added to `sys.path` by `orinbench/__init__.py`, not installed).

- Two backends: `pytorch` (official rollout, BF16) and `hybrid` (vision tower + one expert flow step as TensorRT FP16 engines,
  LLM still in PyTorch). `orinbench/hybrid.py` swaps only `visual.forward` and the diffusion `step_fn`, so tokenization,
  generation and flow integration stay the official code.
- JetPack 5.1's TensorRT Python bindings are Python 3.8 only, so `trt_bridge/` exposes the C++ runtime through a C ABI that
  `orinbench/trt_stage.py` loads with ctypes (RTLD_LAZY) on PyTorch CUDA buffers.
- FP16 engines are only valid with the export-time patches in `orinbench/patches.py` (per-token scaled LayerNorm/RMSNorm,
  rotary angles as float32 inputs). Every engine must pass its parity gate (`build_engines.sh` writes `<engine>.gated`;
  `run.sh` refuses unmarked engines).
- The expert engine has a fixed 3,136-token KV cache (3,086 prompt + 50), so hybrid reasoning is capped at 50 tokens; the
  runtime pads the prompt cache with zeros and masks the padding with -1e4 (`tests/test_expert_step.py` checks this).
- CPU tests: `PYTHONPATH=orin_bench_trt8.5 python -m pytest orin_bench_trt8.5/tests`.
- `checksums/*.sha256` pin the checkpoint, ONNX, parity references and scene packs; `scripts/verify_artifacts.sh` checks them.
- A `pkill -f`/`pgrep -f` whose pattern appears in an ssh command line matches that ssh session itself: kill by PID instead.
