#!/bin/bash
# Run on the Orin (no root). Creates $ORINBENCH_VENV: CPython 3.12 + PyTorch 2.8.0 and torchvision 0.23.0 built from source
# for sm_87 + the inference dependencies at the versions this repository pins.
#
# Why from source: transformers 4.57 needs Python >= 3.9, NVIDIA's JetPack 5 PyTorch wheels are for Python 3.8, and the
# PyPI aarch64 wheels carry no sm_87 kernels ("no kernel image is available"). There is no cuDNN for CUDA 12.2 on
# JetPack 5.1, so PyTorch is built without it (orinbench/patches.py replaces the one Conv3d the model uses).
# The PyTorch build takes several hours. TORCH_WHEEL=/path/to/torch-2.8.0-*.whl skips it (e.g. a wheel from another Orin).
set -euo pipefail
. "$(dirname "$0")/../env.sh"
SRC=$ORINBENCH_HOME/src; mkdir -p "$SRC"
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
[ -x "$ORINBENCH_VENV/bin/python" ] || uv venv -p 3.12 "$ORINBENCH_VENV"
PY=$ORINBENCH_VENV/bin/python
export PATH=$CUDA_HOME/bin:$ORINBENCH_VENV/bin:$PATH

uv pip install -p "$PY" numpy pyyaml typing-extensions setuptools wheel cmake ninja packaging
if [ -z "${TORCH_WHEEL:-}" ]; then
  [ -d "$SRC/pytorch" ] || git clone --recursive --branch v2.8.0 https://github.com/pytorch/pytorch.git "$SRC/pytorch"
  uv pip install -p "$PY" -r "$SRC/pytorch/requirements.txt"
  (
    cd "$SRC/pytorch"
    export TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=${MAX_JOBS:-8} BUILD_TEST=0 USE_CUDA=1
    export USE_CUDNN=0 USE_CUSPARSELT=0 USE_CUDSS=0 USE_CUFILE=0 USE_NCCL=0 USE_MPI=0 USE_DISTRIBUTED=1 USE_GLOO=1
    export USE_FBGEMM=0 USE_MKLDNN=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 USE_PYTORCH_QNNPACK=0
    export USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1 USE_KINETO=1
    export PYTORCH_BUILD_VERSION=2.8.0 PYTORCH_BUILD_NUMBER=1
    "$PY" setup.py bdist_wheel
  )
  TORCH_WHEEL=$(ls "$SRC"/pytorch/dist/torch-2.8.0*.whl | head -1)
fi
uv pip install -p "$PY" "$TORCH_WHEEL"

# torchvision from source against this torch (the PyPI aarch64 binary lacks torchvision::nms for it). No image codecs needed.
[ -d "$SRC/vision" ] || git clone -q --depth 1 --branch v0.23.0 https://github.com/pytorch/vision.git "$SRC/vision"
uv pip install -p "$PY" "setuptools<81"  # torchvision's setup.py needs pkg_resources
TORCHVISION_USE_PNG=0 TORCHVISION_USE_JPEG=0 TORCHVISION_USE_WEBP=0 TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_FFMPEG=0 \
  TORCHVISION_USE_VIDEO_CODEC=0 uv pip install -p "$PY" --no-build-isolation --no-deps --reinstall "$SRC/vision"

uv pip install -p "$PY" -r "$ORINBENCH_DIR/requirements.txt"
"$PY" -c "import torch, torchvision, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_arch_list(), 'transformers', transformers.__version__)"
