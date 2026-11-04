#!/bin/bash
# Run on the Orin. Runs the official PyTorch model on a few scenes and saves the real inputs of one flow step (padded to
# the engine's KV length) with float32 and BF16 reference outputs: $ORINBENCH_WORK/parity/expert/<scene>/.
# The expert engine gate (scripts/build_engines.sh) needs these. Takes ~5 minutes after the model has loaded.
set -euo pipefail
. "$(dirname "$0")/../env.sh"
ob_need_weights
EXPERT_PARITY_SCENES=${EXPERT_PARITY_SCENES:-"s00_030c760c_5100ms s02_a8dd5ceb_5100ms"}
mkdir -p "$ORINBENCH_HOME/logs"; cd "$ORINBENCH_DIR"
LOG=$ORINBENCH_HOME/logs/capture_expert_$(ob_ts).log
ob_log "capturing expert references on $EXPERT_PARITY_SCENES (log $LOG)"
flock "$ORINBENCH_LOCK" python -u -m orinbench.parity expert --model "$A15_WEIGHTS" --scenes "$ORINBENCH_SCENES" \
  --ids $EXPERT_PARITY_SCENES --out "$ORINBENCH_WORK/parity/expert" $(ob_vlm_args) > "$LOG" 2>&1 \
  || { grep -iv warn "$LOG" | tail -15; ob_die "capture failed"; }
grep -E "^s[0-9]+_" "$LOG"
