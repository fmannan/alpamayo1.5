# Alpamayo 1.5 on Jetson AGX Orin with TensorRT 8.5

Runs Alpamayo 1.5 on a Jetson AGX Orin 64 GB (JetPack 5.1, TensorRT 8.5) in two ways and measures both:

* **pytorch**: the official rollout in PyTorch BF16, with two optimizations that leave its outputs unchanged.
* **hybrid**: the vision tower and the diffusion expert as TensorRT FP16 engines, and the LLM still in PyTorch BF16, all
  in one process. This is not numerically identical to PyTorch, so its outputs are compared against the pytorch run.

The model code is this repository's `src/alpamayo1_5`, unmodified. `orinbench` wraps it with stage timing and swaps in
the engines. It does not install the package, whose dependencies include flash-attn.

## What to expect

These are measurements from one Orin (MAXN, GPU at 1,300.5 MHz) and are not guaranteed for another device.

| | Value |
|---|---|
| pytorch, end to end p50 / p99 | 6,595 / 7,952 ms |
| hybrid, end to end p50 / p99 | 6,559 / 7,689 ms (0.55 % faster: not an established speedup) |
| hybrid, forced trajectory (no reasoning) p50 | 4,924 ms |
| vision engine / expert engine (trtexec GPU compute median) | 866.5 ms / 97.2 ms (88.9 ms with a CUDA graph) |
| hybrid vs pytorch outputs, 100 scene/seed pairs | reasoning identical on 100/100; trajectories 1.5 cm apart on average (max 8.7 cm), against a ~95 cm spread between seeds; error vs the logged path +2.5 mm |

Most of the time is in the LLM, which stays in PyTorch. Prefill takes about 2.9 s and decode about 1.7 s. Stock TensorRT
8.5 is slower than PyTorch for this LLM, so it is not part of this kit.

## Requirements

| Requirement | Why |
|---|---|
| Jetson AGX Orin, **64 GB** module | The BF16 model needs ~22 GB, and the hybrid process peaks at ~35 GB of the shared RAM. |
| JetPack 5.1.x (L4T R35), TensorRT 8.5, `trtexec` | The engines are built and run on the device. They are specific to the device and the TensorRT version. |
| `cuda-toolkit-12-2` and `cuda-compat-12-2` (one-time, root) | PyTorch is built for CUDA 12.2. The compat package supplies a 12.2 driver on top of JetPack 5.1's 11.4 driver. |
| `g++` | Builds the TensorRT bridge (`trt_bridge/`). |
| MAXN power mode (`sudo nvpmodel -m 0`) | Timings are only comparable in MAXN. |
| Hugging Face access to `nvidia/Alpamayo-1.5-10B` (and the dataset, to make scene packs) | Both are gated. |
| `Qwen/Qwen3-VL-2B-Instruct` processor files in the HF cache | The model loads its image processor from Hugging Face. On an offline Orin, download it once and set `HF_HOME` / `HF_HUB_OFFLINE=1`. |

Nothing else needs root. `scripts/preflight.sh` checks all of this.

**Why the TensorRT bridge exists.** JetPack 5.1's TensorRT Python bindings are for Python 3.8. transformers 4.57 needs
Python 3.9 or newer, so `trt_bridge/libtrt_stage.so` exposes the TensorRT C++ runtime through a C ABI. `orinbench.trt_stage`
loads it with ctypes and runs engines on PyTorch CUDA buffers, with no copies and no second process.

## Setup

Copy this repository to the Orin, then:

```bash
export ORINBENCH_HOME=/path/with/space          # default ~/orinbench; NVMe recommended
export A15_WEIGHTS=/path/to/Alpamayo-1.5-10B     # checkpoint directory (22 GB)
export A15_VLM_CONFIG=/path/to/qwen3-vl-config   # optional, see "VLM config" below
. orin_bench_trt8.5/env.sh
orin_bench_trt8.5/scripts/preflight.sh
orin_bench_trt8.5/scripts/setup_venv.sh          # PyTorch 2.8.0 for sm_87 from source: hours, once
orin_bench_trt8.5/trt_bridge/build.sh
```

`setup_venv.sh` builds PyTorch and torchvision because no wheel exists for this combination: Python 3.12, sm_87, CUDA 12.2
on JetPack 5.1. If you already have such a wheel, from another Orin for example, pass it as
`TORCH_WHEEL=/path/torch-2.8.0-*.whl`. That PyTorch has no cuDNN, so `orinbench/patches.py` replaces the vision patch
Conv3d with the equivalent linear layer.

**VLM config.** The checkpoint names `nvidia/Cosmos-Reason2-8B` as its VLM config, which is gated. If you lack access,
download the config, tokenizer and processor files of `Qwen/Qwen3-VL-8B-Instruct` (revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`) into a directory and set `A15_VLM_CONFIG` to it. At load, the model checks
that the tokenizer reproduces the checkpoint's trajectory token ids and vocabulary size, and refuses to run otherwise.

## Steps

All paths come from `env.sh`: `$ORINBENCH_SCENES`, `$ORINBENCH_WORK` (ONNX and parity references), `$ORINBENCH_ENGINES`
and `$ORINBENCH_RESULTS`.

**1. Scene packs** need dataset access and can be made on any host. They are 25 fixed scenes (`data/scenes.yaml`), each with
16 decoded images plus ego history and the logged future. Every run reads these files, so every device sees identical inputs.

```bash
python -m orinbench.cli prep-scenes --specs orin_bench_trt8.5/data/scenes.yaml --out $ORINBENCH_SCENES
```

**2. ONNX and vision references** run on CPU, on any host with ~64 GB of RAM. They need `requirements-export.txt`.

```bash
orin_bench_trt8.5/scripts/export_onnx.sh   # -> $ORINBENCH_WORK/onnx/{vision,expert}/, $ORINBENCH_WORK/parity/vision/
```

**3. Expert references** run on the Orin. These are real flow-step inputs and float32 outputs, captured from the PyTorch model.

```bash
orin_bench_trt8.5/scripts/capture_expert_refs.sh   # -> $ORINBENCH_WORK/parity/expert/
```

**Reusing validated artifacts instead of steps 1-3.** If you have the validated ONNX, parity references and scene packs, put
them at `$ORINBENCH_WORK/onnx/{vision,expert}/`, `$ORINBENCH_WORK/parity/{vision,expert}/` and `$ORINBENCH_SCENES/`, then run
`scripts/verify_artifacts.sh`. It checks every file against `checksums/{onnx,parity,scenes}.sha256`. `orinbench.export vision` reproduces the
validated vision ONNX byte for byte, all 371 files.

**4. Engines.** Build, gate, time and profile them on the Orin. This takes 30-40 minutes.

```bash
orin_bench_trt8.5/scripts/build_engines.sh   # -> $ORINBENCH_ENGINES/{vision,expert}.engine; logs in $ORINBENCH_RESULTS/engines_*/
```

The gates use the parity references from real scenes:

* vision: relative L2 at most 0.06 per output (validated worst 0.045);
* expert: relative L2 at most 0.005 against float32 (validated 0.0018-0.0027), both eagerly and through a CUDA graph with
  rewritten inputs.

An engine that fails its gate is deleted. Engines rebuilt on another device can choose different kernels, so expect
millimetre-level output differences from the validated numbers, not bit-identical results.

**5. Runs.**

```bash
orin_bench_trt8.5/scripts/run.sh smoke          # minutes: the whole PyTorch path on 2 scenes
orin_bench_trt8.5/scripts/run.sh smoke_hybrid   # minutes: the TensorRT path on the same 2 scenes
orin_bench_trt8.5/scripts/run.sh pytorch   # ~1.5 h: 100 timed calls + 25 scenes x 4 seeds of outputs
orin_bench_trt8.5/scripts/run.sh hybrid    # ~1.5 h, then compared automatically with the latest pytorch run
```

Each run writes `$ORINBENCH_RESULTS/<time>_<mode>/` containing:

* `meta.json`, `env.json` and `telemetry/`;
* per config: `summary.json` (latency percentiles and per-stage means), `calls.jsonl` (every timed call), `behavior.jsonl`
  (reasoning text, tokens and the 64-waypoint trajectory per scene and seed) and `seed_spread.json`;
* after a hybrid run, `vs_pytorch.json`.

Timing protocol (`orinbench/runner.py`):

1. one cold call;
2. warm-up until stable;
3. 100 *varied* calls, cycling scenes with a new seed each: the headline numbers;
4. *repeated* calls on one scene and seed;
5. *staged* calls, synchronized at every stage boundary to break down per-stage time.

Latency is measured from input ready to trajectory ready: decoded uint8 frames and ego history in host memory, through to
the trajectory copied back to the host.

## Verified on

One Jetson AGX Orin 64 GB (JetPack 5.1, L4T R35.2.1, TensorRT 8.5.2, MAXN, GPU at 1,300.5 MHz), 2026-11-03, with the
validated ONNX, parity references and scene packs (all matching `checksums/`) and the engines built from them:

| Step | Result |
|---|---|
| `preflight.sh` | 0 FAIL |
| `trt_bridge/build.sh` | builds |
| `verify_artifacts.sh parity scenes` | all 190 files match |
| `build_engines.sh --reuse` (gates, timing, profile) | vision passed: deepstack relative L2 0.009-0.023; expert passed: 0.0027 / 0.0018 vs float32, identical after CUDA-graph replay. GPU compute 865 ms / 95.7 ms (88.7 ms with a graph) |
| `run.sh smoke` | 3 calls, p50 6,949 ms, every call emitted a trajectory |
| `run.sh smoke_hybrid` | 3 calls, p50 6,890 ms; same reasoning as `smoke` on both scenes, trajectories 4.3 cm apart (ADE 6.4 s); peak RAM 57.8 of 61 GB with a 26 GB checkpoint + engines in `/dev/shm` |

Not run in this verification: the full 100-call `pytorch` and `hybrid` runs (their reference numbers are in "What to
expect"), the PyTorch build in `setup_venv.sh` (an existing venv with the same versions was used), the ONNX export of the
expert, and `capture_expert_refs.sh`.

## Things that will bite

* **Memory.** CPU and GPU share 64 GB. A checkpoint copy in `/dev/shm` counts against that pool, and with it the hybrid run
  can be OOM-killed. Keep the checkpoint on disk, or pass `--drop-weights-after-load` to `run.sh`. That deletes the
  `/dev/shm` copy once the model is resident; it refuses any path outside `/dev/shm`.
* **`/dev/shm` is wiped at logout** when systemd-logind's `RemoveIPC=yes` (the default) and your last session ends. Keep a
  session open, or keep files on disk.
* **Reasoning is capped at 50 tokens in hybrid mode.** The expert engine has a fixed 3,136-token KV cache (3,086 prompt +
  50). Reasoning on these scenes has stayed well below that. `rate_traj_start_emitted` in `summary.json` shows whether any
  call hit the cap.
* **FP16 overflow is silent.** Without the per-token scaled LayerNorm, the vision engine is finite but wrong (143-161 %
  error). With rotary angles computed inside the engine, the expert loses 15-34 %. The export applies both fixes. Never
  trust an engine that has not passed its gate.
* **Library loading.** The bridge is loaded with `RTLD_LAZY`, because the compat `libnvcudla` lacks DLA symbols that
  libnvinfer references. PyTorch must create the CUDA context first. PyTorch and TensorRT run on separate streams and are
  synchronized around each engine call.
* GPU jobs from these scripts take `$ORINBENCH_LOCK` (`flock`), so they never overlap. Other GPU work distorts timings.

## Layout

```
env.sh               paths and CUDA/TensorRT environment (source it)
trt_bridge/          C ABI over the TensorRT 8.5 runtime + build.sh
orinbench/
  export.py          ONNX: vision (16 images, 20x36 patches each) and one expert flow step (KV 3,136)
  parity.py          float32 references: vision (CPU) and captured expert inputs (GPU)
  trt_stage.py       engines on PyTorch buffers; vision-check / expert-check gates
  model.py           instrumented official rollout (stage clock, seeds, optimizations)
  hybrid.py          the same rollout with the vision forward and flow step routed to the engines
  patches.py         FP16-safe norms, batched vision attention, Conv3d without cuDNN (none change the math)
  runner.py, behavior.py, telemetry.py, envinfo.py, stats.py, timing.py, inputs.py, cli.py
configs/             smoke, smoke_hybrid, pytorch, hybrid
scripts/             preflight, setup_venv, export_onnx, capture_expert_refs, verify_artifacts, build_engines, run
checksums/           SHA256 of the validated ONNX, parity references and scene packs
data/scenes.yaml     the 25 evaluation scenes
tests/               CPU tests: patches, padded expert step vs the official call, output pairing
```

CPU tests: `PYTHONPATH=orin_bench_trt8.5 python -m pytest orin_bench_trt8.5/tests` (needs torch, transformers and pytest).
