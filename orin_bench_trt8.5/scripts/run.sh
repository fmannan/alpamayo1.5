#!/bin/bash
# Run on the Orin. One benchmark + outputs run: scripts/run.sh MODE [extra orinbench.cli run args], MODE = configs/MODE.yaml
#   smoke         best PyTorch config, 2 scenes x 1 seed (minutes)
#   smoke_hybrid  the TensorRT path, 2 scenes x 1 seed (minutes)
#   pytorch  best behavior-preserving PyTorch BF16 config: 100 timed calls + 25 scenes x 4 seeds of outputs (~1.5 h)
#   hybrid   TensorRT vision + expert engines and the PyTorch LLM, same protocol (~1.5 h)
# Results: $ORINBENCH_RESULTS/<time>_<mode>/<config>/{summary.json,calls.jsonl,behavior.jsonl}, plus meta.json, env.json,
# telemetry/. After a hybrid run, the outputs are compared with the latest pytorch run if there is one.
#
# Memory: CPU and GPU share the 64 GB. If the checkpoint sits in /dev/shm it counts against that pool too; pass
# --drop-weights-after-load to delete that RAM copy once the model has loaded (the computation is unchanged).
set -uo pipefail
. "$(dirname "$0")/../env.sh"
MODE=${1:?usage: run.sh smoke|smoke_hybrid|pytorch|hybrid [extra args]}; shift
CFG=$ORINBENCH_DIR/configs/$MODE.yaml
[ -f "$CFG" ] || ob_die "no config $CFG"
ob_need_weights
[ -f "$ORINBENCH_SCENES/manifest.json" ] || ob_die "no scene packs in $ORINBENCH_SCENES"
if grep -q "^trt:" "$CFG"; then
  for e in vision expert; do [ -s "$ORINBENCH_ENGINES/$e.engine" ] || ob_die "missing $ORINBENCH_ENGINES/$e.engine (scripts/build_engines.sh)"; done
  [ -s "$ORINBENCH_DIR/trt_bridge/libtrt_stage.so" ] || "$ORINBENCH_DIR/trt_bridge/build.sh" > /dev/null || ob_die "trt_bridge/build.sh failed"
fi
mkdir -p "$ORINBENCH_HOME/logs" "$ORINBENCH_RESULTS"; cd "$ORINBENCH_DIR"
LOG=$ORINBENCH_HOME/logs/run_${MODE}_$(ob_ts).log
ob_log "run $MODE (log $LOG); GPU clock $(ob_gpu_freq) Hz"
flock "$ORINBENCH_LOCK" python -u -m orinbench.cli run --model "$A15_WEIGHTS" --scenes "$ORINBENCH_SCENES" --config "$CFG" \
  --out "$ORINBENCH_RESULTS" $(ob_vlm_args) "$@" > "$LOG" 2>&1
rc=$?
dir=$(grep -o "done -> .*" "$LOG" | tail -1 | sed 's/done -> //')
[ $rc = 0 ] && [ -n "$dir" ] || { grep -iv warn "$LOG" | tail -15; ob_die "run $MODE failed (log $LOG)"; }
python - "$dir" <<'PY'
import glob, json, os, sys
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*", "summary.json"))):
    s = json.load(open(path)).get("varied", {})
    lat = {k: round(v, 1) for k, v in s.get("e2e_wall_ms", {}).items() if k in ("n", "p50", "p95", "p99", "max")}
    print(os.path.basename(os.path.dirname(path)), "end-to-end ms", lat, "traj_start_rate", s.get("rate_traj_start_emitted"))
PY
if [ "$MODE" = hybrid ]; then
  ref=$(ls -d "$ORINBENCH_RESULTS"/*_pytorch 2>/dev/null | tail -1)
  for pair in "official reasoning" "force_traj force_traj"; do
    set -- $pair
    [ -n "$ref" ] && [ -s "$ref/$1/behavior.jsonl" ] && [ -s "$dir/$2/behavior.jsonl" ] || continue
    ob_log "compare $2 with $(basename "$ref")/$1"
    python -m orinbench.cli compare --reference "$ref/$1/behavior.jsonl" --test "$dir/$2/behavior.jsonl" \
      --out "$dir/$2/vs_pytorch.json" | grep -E "identical_reasoning_rate|vs_ref_ade_6.4s_(mean|max)|n_paired|finite_rate"
  done
fi
ob_log "run $MODE OK: $dir"
