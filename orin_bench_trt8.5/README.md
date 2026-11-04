# Alpamayo 1.5 on Jetson AGX Orin with TensorRT 8.5

Runs Alpamayo 1.5 on a Jetson AGX Orin 64 GB (JetPack 5.1, TensorRT 8.5) in two ways and measures both:

* **pytorch**: the official rollout in PyTorch BF16, with two optimizations that leave its outputs unchanged.
* **hybrid**: the vision tower and the diffusion expert as TensorRT FP16 engines, and the LLM still in PyTorch BF16, all
  in one process. This is not numerically identical to PyTorch, so its outputs are compared against the pytorch run.

The model code is this repository's `src/alpamayo1_5`, unmodified. `orinbench` wraps it with stage timing and swaps in
the engines. It does not install the package, whose dependencies include flash-attn.

## What to expect

From a clean run of this kit on one Jetson AGX Orin 64 GB (MAXN, GPU at 1,300.5 MHz), 2026-11-04. The full report, with
figures, profile and bottleneck analysis, is [`reports/2026-11-04_agx-orin/REPORT.md`](reports/2026-11-04_agx-orin/REPORT.md).
Another device can differ.

| 100 calls per config | p50 | p99 |
|---|---|---|
| pytorch, with reasoning | 6,563 ms | 7,959 ms |
| hybrid, with reasoning | 6,553 ms | 7,683 ms |
| pytorch, forced trajectory (no reasoning; not behavior-preserving) | 5,139 ms | 5,159 ms |
| hybrid, forced trajectory | 4,875 ms | 4,888 ms |

| | Value |
|---|---|
| Engines (trtexec GPU compute median) | vision 865.6 ms; expert 96.3 ms per flow step (88.9 ms with a CUDA graph) |
| Hybrid vs pytorch outputs, 100 scene/seed pairs | reasoning identical on 100/100; trajectories 1.5 cm apart on average (max 9.2 cm), against a 95 cm spread between seeds; error vs the logged path 1.534 -> 1.536 m |
| Where a call goes (hybrid) | LLM prefill 44 %, LLM decode 26 % (112 ms per token), flow 14 %, vision 13 % |

The hybrid is not faster with reasoning. The LLM is 70 % of a call and stays in PyTorch in both backends: stock TensorRT
8.5 is slower than PyTorch for this LLM, so it is not part of this kit. Prefill, decode and vision already run at 60-70 % of
the device's compute or bandwidth limits. The TensorRT expert is the one stage far from its floor, mostly because of
per-step KV expansion for grouped-query attention. See the report's "Bottlenecks".

## Requirements

| Requirement | Why |
|---|---|
| Jetson AGX Orin, **64 GB** module | The BF16 model needs ~22 GB. A run peaks at 31-35 GB of the shared RAM, plus whatever sits in `/dev/shm`. |
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
`scripts/verify_artifacts.sh onnx parity scenes`. It checks every file against `checksums/*.sha256`. `orinbench.export vision`
reproduces the validated vision ONNX byte for byte, all 371 files. `verify_artifacts.sh weights` checks a copied checkpoint.

**4. Engines.** Build, gate, time and profile them on the Orin. This takes 30-40 minutes.

```bash
orin_bench_trt8.5/scripts/build_engines.sh   # -> $ORINBENCH_ENGINES/{vision,expert}.engine; logs in $ORINBENCH_RESULTS/engines_*/
```

The gates use the parity references from real scenes:

* vision: relative L2 at most 0.06 per output (validated worst 0.045);
* expert: relative L2 at most 0.005 against float32 (validated 0.0018-0.0027), both eagerly and through a CUDA graph with
  rewritten inputs.

An engine that fails its gate is deleted; one that passes gets a `<engine>.gated` marker, and `run.sh` uses only marked
engines. Engines rebuilt on another device can choose different kernels, so expect
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

**6. Report.** On any machine that can read both run directories (needs matplotlib):

```bash
python -m orinbench.report --pytorch $ORINBENCH_RESULTS/<time>_pytorch --hybrid $ORINBENCH_RESULTS/<time>_hybrid \
    --engines $ORINBENCH_RESULTS/engines_* --scenes $ORINBENCH_SCENES --out report/
```

It writes `report/REPORT.md` with latency, the per-stage breakdown, engine timings, the expert engine's time by layer type,
output agreement and memory, and figures in `report/figures/`.

### Orins without spare disk

If the only space is `/dev/shm` (a stock 64 GB eMMC with ~1 GB free), RAM holds everything. A 22 GB checkpoint plus 5.8 GB
of engines leaves too little for a 100-call run (pytorch peaks at ~33 GB, hybrid at ~35 GB). This order fits:

1. Copy the ONNX and vision references to `$ORINBENCH_WORK`, run `build_engines.sh`, then delete `$ORINBENCH_WORK/onnx`.
   The vision engine passes its gate. The expert engine is kept but left ungated, because its references need the model.
2. Copy the checkpoint, run `capture_expert_refs.sh` then `build_engines.sh --reuse expert`, then delete
   `$ORINBENCH_WORK/parity/expert`.
3. `run.sh pytorch` fits with the checkpoint in `/dev/shm` as long as the ONNX and expert references are gone (peak 52 GB
   in the clean run). `run.sh hybrid` also holds the engines, so use `run.sh hybrid --drop-weights-after-load`: it deletes
   the RAM copy of the checkpoint once loaded (peak 40 GB). Copy the checkpoint again before the next run.

Downloading the checkpoint on the Orin itself (`huggingface-cli download nvidia/Alpamayo-1.5-10B --revision
7aba8293c09993f2e125c6819df05d7fa3e873ea --local-dir ...`, with a token that has access) took about 7 minutes in the clean
run. Check it with `verify_artifacts.sh weights`.

## Verified on

The clean run behind "What to expect" used every step of this kit from an empty `/dev/shm`, on a stock Orin whose eMMC
had 1 GB free (so everything lived in RAM, in the order of "Orins without spare disk"):

* The ONNX exports reproduced the validated files byte for byte (`checksums/onnx.sha256`).
* The checkpoint was downloaded on the Orin and checked against `checksums/weights.sha256`.
* Both engines were built fresh and passed their gates: vision 0.009-0.023, expert 0.0025 / 0.0018.
* `run.sh pytorch` and `run.sh hybrid` each did 2 configs x (100 timed calls + 25 scenes x 4 seeds).

Details are in the report's "How this run was made". Not exercised: the from-source PyTorch build in `setup_venv.sh` (a
venv with the same wheel was used, for lack of disk) and `prep-scenes` (the validated scene packs were used and checked
against `checksums/scenes.sha256`).

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
  report.py          REPORT.md + figures from a pytorch run, a hybrid run and the engine logs
  runner.py, behavior.py, telemetry.py, envinfo.py, stats.py, timing.py, inputs.py, cli.py
configs/             smoke, smoke_hybrid, pytorch, hybrid
scripts/             preflight, setup_venv, export_onnx, capture_expert_refs, verify_artifacts, build_engines, run
checksums/           SHA256 of the checkpoint, the validated ONNX, parity references and scene packs
reports/             reports of runs (REPORT.md generated by orinbench.report, plus a hand-written analysis)
data/scenes.yaml     the 25 evaluation scenes
tests/               CPU tests: patches, padded expert step vs the official call, output pairing
```

CPU tests: `PYTHONPATH=orin_bench_trt8.5 python -m pytest orin_bench_trt8.5/tests` (needs torch, transformers and pytest).
