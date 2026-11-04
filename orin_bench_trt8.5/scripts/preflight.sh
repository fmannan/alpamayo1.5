#!/bin/bash
# Run on the target Orin (no root). Checks that it can run this kit and says what to fix. Exits 1 on any FAIL.
. "$(dirname "$0")/../env.sh"
fails=0; warns=0
check() {  # check FAIL|WARN NAME OK(0/1) DETAIL FIX
  local st=PASS
  if [ "$3" != 1 ]; then st=$1; [ "$1" = FAIL ] && fails=$((fails + 1)) || warns=$((warns + 1)); fi
  printf '%-5s %-14s %s%s\n' "$st" "$2" "$4" "$([ "$st" != PASS ] && [ -n "${5:-}" ] && echo "  -> $5")"
}
ok() { "$@" > /dev/null 2>&1 && echo 1 || echo 0; }

model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
check FAIL device "$([[ $model == *Orin* ]] && echo 1 || echo 0)" "${model:-unknown}" "a Jetson Orin (sm_87) is required"
rel=$(head -1 /etc/nv_tegra_release 2>/dev/null)
check FAIL l4t "$([[ $rel == *"R35 (release)"* ]] && echo 1 || echo 0)" "${rel:0:60}" "JetPack 5.1.x (L4T R35) is the supported system"
trt=$(dpkg-query -W -f='${Version}' libnvinfer8 2>/dev/null)
check FAIL tensorrt "$([[ $trt == 8.5* ]] && echo 1 || echo 0)" "libnvinfer8 ${trt:-missing}" "TensorRT 8.5 (JetPack 5.1)"
check FAIL trtexec "$([ -x "$TRTEXEC" ] && echo 1 || echo 0)" "$TRTEXEC" "install the JetPack TensorRT samples/dev package, or set TRTEXEC"
c12=0; [ -e "$CUDA_HOME/lib64/libcudart.so.12" ] && [ -e "$CUDA_HOME/compat/libcuda.so.1" ] && c12=1
check FAIL cuda-12.2 "$c12" "$CUDA_HOME runtime + compat driver" \
  "one-time, needs root: sudo apt install cuda-toolkit-12-2 cuda-compat-12-2 (NVIDIA CUDA repository for L4T)"
check FAIL nvcc "$([ -x "$CUDA_HOME/bin/nvcc" ] && echo 1 || echo 0)" "$CUDA_HOME/bin/nvcc" "cuda-toolkit-12-2 provides it (PyTorch build)"
check FAIL g++ "$(ok command -v g++)" "$(g++ --version 2>/dev/null | head -1)" "sudo apt install g++"
mem=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
check FAIL ram "$([ "$mem" -ge 58 ] && echo 1 || echo 0)" "${mem} GiB" \
  "the 64 GB module is required: the BF16 model alone needs ~22 GB plus ~10 GB of activations and caches"
mkdir -p "$ORINBENCH_HOME"; disk=$(df -BG --output=avail "$ORINBENCH_HOME" | tail -1 | tr -dc 0-9)
check WARN disk "$([ "$disk" -ge 30 ] && echo 1 || echo 0)" "$ORINBENCH_HOME: ${disk} GiB free" \
  "~30 GB for venv, PyTorch build, scenes and engines; set ORINBENCH_HOME / ORINBENCH_ENGINES to an NVMe path"
mode=$(nvpmodel -q 2>/dev/null | head -1)
check WARN power-mode "$([[ $mode == *MAXN* ]] && echo 1 || echo 0)" "${mode:-nvpmodel unavailable}" \
  "timings are only comparable in MAXN: sudo nvpmodel -m 0"
check WARN python "$([ -x "$ORINBENCH_VENV/bin/python" ] && echo 1 || echo 0)" "$ORINBENCH_VENV" "scripts/setup_venv.sh"
check WARN bridge "$([ -s "$ORINBENCH_DIR/trt_bridge/libtrt_stage.so" ] && echo 1 || echo 0)" "trt_bridge/libtrt_stage.so" "trt_bridge/build.sh"
check WARN weights "$([ -f "${A15_WEIGHTS:-/nonexistent}/model.safetensors.index.json" ] && echo 1 || echo 0)" "A15_WEIGHTS=${A15_WEIGHTS:-unset}" \
  "download nvidia/Alpamayo-1.5-10B and set A15_WEIGHTS"
others=$(pgrep -fc "trtexec|python.* -m orinbench")
check WARN gpu-idle "$([ "$others" = 0 ] && echo 1 || echo 0)" "$others other GPU jobs" "they would distort timings"
echo "preflight: $fails FAIL, $warns WARN"
[ $fails = 0 ]
