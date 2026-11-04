# Source this on the Orin (and on the export host): `. orin_bench_trt8.5/env.sh`. Every path can be overridden
# from the environment before sourcing. Nothing here needs root or touches the system Python.
ORINBENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ORINBENCH_DIR
export ORINBENCH_HOME=${ORINBENCH_HOME:-$HOME/orinbench}               # venv, builds, results, logs (~6 GB + PyTorch build)
export ORINBENCH_VENV=${ORINBENCH_VENV:-$ORINBENCH_HOME/venv}
export ORINBENCH_RESULTS=${ORINBENCH_RESULTS:-$ORINBENCH_HOME/results}
export ORINBENCH_SCENES=${ORINBENCH_SCENES:-$ORINBENCH_HOME/scenes}     # scene packs (2.4 GB), orinbench.cli prep-scenes
export ORINBENCH_WORK=${ORINBENCH_WORK:-$ORINBENCH_HOME/work}           # ONNX (~11 GB) + parity references (~3 GB)
export ORINBENCH_ENGINES=${ORINBENCH_ENGINES:-$ORINBENCH_HOME/engines}  # TensorRT engines (~5.8 GB)
export A15_WEIGHTS=${A15_WEIGHTS:-}        # Alpamayo-1.5-10B checkpoint directory (22 GB); required by most steps
export A15_VLM_CONFIG=${A15_VLM_CONFIG:-}  # optional local Qwen3-VL config dir standing in for the gated one (README)
export ORINBENCH_LOCK=${ORINBENCH_LOCK:-/tmp/orinbench_gpu.lock}       # GPU jobs of this kit never overlap

# JetPack 5.1 ships a CUDA 11.4 driver; cuda-compat-12-2 provides the 12.2 user-mode driver PyTorch is built against.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.2}
export TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
if [ -d "$CUDA_HOME/compat" ]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/compat:$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ORINBENCH_HOME/.uvcache} UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$ORINBENCH_HOME/python}
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$ORINBENCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
[ -x "$ORINBENCH_VENV/bin/python" ] && export PATH="$ORINBENCH_VENV/bin:$PATH"

ob_log() { echo "[$(date '+%F %T')] $*"; }
ob_die() { echo "ERROR: $*" >&2; exit 1; }
ob_ts() { date '+%Y%m%d-%H%M%S'; }
ob_need_weights() { [ -n "$A15_WEIGHTS" ] && [ -f "$A15_WEIGHTS/model.safetensors.index.json" ] || ob_die "set A15_WEIGHTS to the Alpamayo-1.5-10B checkpoint directory"; }
ob_vlm_args() { [ -n "$A15_VLM_CONFIG" ] && echo "--vlm-config $A15_VLM_CONFIG"; }
ob_gpu_freq() { cat /sys/devices/platform/*.ga10b/devfreq/*/cur_freq 2>/dev/null | head -1; }
