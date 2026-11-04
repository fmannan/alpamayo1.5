#!/bin/bash
# Check that the ONNX files, parity references and scene packs are byte-identical to the validated ones.
#   checksums/onnx.sha256    $ORINBENCH_WORK/onnx/{vision,expert}/   (validated ONNX + external weights)
#   checksums/parity.sha256  $ORINBENCH_WORK/parity/{vision,expert}/ (validated parity references)
#   checksums/scenes.sha256  $ORINBENCH_SCENES/                       (the 25 validated scene packs)
# The validated vision ONNX is reproduced byte for byte by `orinbench.export vision`. Scene packs and expert
# references made fresh (prep-scenes, capture_expert_refs.sh) are valid but not byte-identical: check those with the
# engine gates instead. usage: verify_artifacts.sh [onnx|parity|scenes]...  (default: all three)
set -uo pipefail
. "$(dirname "$0")/../env.sh"
PARTS=("$@"); [ ${#PARTS[@]} = 0 ] && PARTS=(onnx parity scenes)
rc=0
for part in "${PARTS[@]}"; do
  case $part in
    onnx|parity) root=$ORINBENCH_WORK ;;
    scenes) root=$ORINBENCH_SCENES ;;
    *) ob_die "unknown part $part" ;;
  esac
  list=$ORINBENCH_DIR/checksums/$part.sha256
  ob_log "$part: checking $(wc -l < "$list") files in $root"
  if (cd "$root" && sha256sum --quiet -c "$list"); then ob_log "$part: all files match"; else ob_log "$part: MISMATCH (lines above)"; rc=1; fi
done
exit $rc
