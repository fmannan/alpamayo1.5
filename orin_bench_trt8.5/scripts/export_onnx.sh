#!/bin/bash
# Run on the export host: any Linux machine with ~64 GB of free RAM, the checkpoint and the scene packs; no GPU needed.
# The Orin itself also works if nothing else is loaded. Needs requirements-export.txt.
# Writes $ORINBENCH_WORK/onnx/{vision,expert}/ (ONNX + external weight files, ~11 GB) and the float32 vision references
# $ORINBENCH_WORK/parity/vision/. Copy both directories to the same paths on the Orin.
set -euo pipefail
. "$(dirname "$0")/../env.sh"
ob_need_weights
VISION_PARITY_SCENES=${VISION_PARITY_SCENES:-"s00_030c760c_5100ms s02_a8dd5ceb_5100ms s14_4b11c90e_5100ms"}
[ -f "$ORINBENCH_SCENES/manifest.json" ] || ob_die "no scene packs in $ORINBENCH_SCENES (python -m orinbench.cli prep-scenes)"
cd "$ORINBENCH_DIR"
for part in vision expert; do
  out=$ORINBENCH_WORK/onnx/$part/$part.onnx
  if [ -s "$out" ]; then ob_log "$part: $out exists, kept"; continue; fi
  ob_log "$part: exporting (float32 on CPU, several minutes)"
  python -u -m orinbench.export $part --model "$A15_WEIGHTS" --out "$out" $(ob_vlm_args)
done
ob_log "vision parity references on $VISION_PARITY_SCENES"
python -u -m orinbench.parity vision --model "$A15_WEIGHTS" --scenes "$ORINBENCH_SCENES" --ids $VISION_PARITY_SCENES \
  --out "$ORINBENCH_WORK/parity/vision" $(ob_vlm_args)
du -sh "$ORINBENCH_WORK/onnx" "$ORINBENCH_WORK/parity/vision"
