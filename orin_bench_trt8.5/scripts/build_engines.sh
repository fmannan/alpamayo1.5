#!/bin/bash
# Run on the Orin. Builds the two TensorRT 8.5 FP16 engines from the ONNX files, gates each on the parity references
# from real scenes, then times and profiles it. An engine that fails its gate is deleted.
#   vision.engine  16 images x 20x36 patches -> merged embeddings + 3 deepstack features   gate: relative L2 <= 0.06
#   expert.engine  one flow step over a 3,136-token KV cache                                 gate: relative L2 <= 0.005,
#                  eager and through a CUDA graph with rewritten inputs
# Inputs: $ORINBENCH_WORK/onnx/{vision,expert}/ and parity/{vision,expert}/ (scripts/export_onnx.sh, capture_expert_refs.sh).
# Each build takes 10-20 minutes and up to 32 GB of builder workspace. usage: build_engines.sh [--reuse] [vision|expert]...
set -uo pipefail
. "$(dirname "$0")/../env.sh"
REUSE=0; [ "${1:-}" = --reuse ] && { REUSE=1; shift; }
PARTS=("$@"); [ ${#PARTS[@]} = 0 ] && PARTS=(vision expert)
[ -x "$TRTEXEC" ] || ob_die "trtexec not found at $TRTEXEC"
[ -s "$ORINBENCH_DIR/trt_bridge/libtrt_stage.so" ] || "$ORINBENCH_DIR/trt_bridge/build.sh" > /dev/null || ob_die "trt_bridge/build.sh failed"
OUT=$ORINBENCH_RESULTS/engines_$(ob_ts); mkdir -p "$OUT" "$ORINBENCH_ENGINES"
cd "$ORINBENCH_DIR"
exec 9> "$ORINBENCH_LOCK"; flock 9
{ date; tr -d '\0' < /proc/device-tree/model; echo; head -1 /etc/nv_tegra_release; nvpmodel -q 2>/dev/null | head -1
  dpkg-query -W libnvinfer8 2>/dev/null; echo "gpu_cur_freq $(ob_gpu_freq)"; } > "$OUT/env.txt" 2>&1
rc=0

median() { grep -m1 "GPU Compute Time" "$1" | sed -E 's/.*median = ([0-9.]+) ms.*/\1/'; }

build_one() {  # build_one PART GATE
  local part=$1 gate=$2 onnx=$ORINBENCH_WORK/onnx/$1/$1.onnx engine=$ORINBENCH_ENGINES/$1.engine t0=$SECONDS
  if [ $REUSE = 1 ] && [ -s "$engine" ]; then
    ob_log "$part: reusing $engine"
  else
    [ -s "$onnx" ] || { ob_log "$part: missing $onnx"; return 1; }
    ob_log "$part: building (10-20 min)"
    # run from the ONNX directory: the weights are external data files next to it
    (cd "$(dirname "$onnx")" && "$TRTEXEC" --onnx="$(basename "$onnx")" --fp16 --memPoolSize=workspace:32768 --buildOnly \
       --saveEngine="$engine" --verbose > "$OUT/${part}_build.log" 2>&1) \
      || { ob_log "$part: build FAILED"; grep -E "\[E\]|ERROR" "$OUT/${part}_build.log" | head -5; rm -f "$engine"; return 1; }
    ob_log "$part: built in $((SECONDS - t0)) s"
  fi
  ob_log "$part: parity gate"
  if ! python -u -m orinbench.trt_stage "$gate" --engine "$engine" --parity "$ORINBENCH_WORK/parity/$part" > "$OUT/${part}_gate.log" 2>&1; then
    ob_log "$part: parity gate FAILED, engine deleted"; grep -iv warn "$OUT/${part}_gate.log" | tail -4; rm -f "$engine"; return 1
  fi
  grep -E "^s[0-9]+_" "$OUT/${part}_gate.log" | cut -c1-220
  ob_log "$part: timing and per-layer profile"
  local common=(--loadEngine="$engine" --noDataTransfers --useSpinWait --duration=0)
  "$TRTEXEC" "${common[@]}" --avgRuns=100 --iterations=100 --warmUp=3000 --exportTimes="$OUT/${part}_times.json" > "$OUT/${part}_time.log" 2>&1
  "$TRTEXEC" "${common[@]}" --useCudaGraph --avgRuns=100 --iterations=100 --warmUp=3000 > "$OUT/${part}_time_graph.log" 2>&1
  "$TRTEXEC" "${common[@]}" --avgRuns=10 --separateProfileRun --exportProfile="$OUT/${part}_profile.json" > "$OUT/${part}_profile.log" 2>&1
  ob_log "$part: GPU compute median $(median "$OUT/${part}_time.log") ms, $(median "$OUT/${part}_time_graph.log") ms with a CUDA graph"
  sha256sum "$engine" >> "$OUT/engines.sha256"
}

for part in "${PARTS[@]}"; do
  case $part in
    vision) build_one vision vision-check || rc=1 ;;
    expert) build_one expert expert-check || rc=1 ;;
    *) ob_die "unknown part $part" ;;
  esac
done
ob_log "engines $([ $rc = 0 ] && echo OK || echo FAILED): logs in $OUT"
exit $rc
