"""Alpamayo 1.5 on Jetson AGX Orin with TensorRT 8.5: ONNX export, FP16 engines, parity gates and benchmark."""

import os
import sys

# The model code is this repository's src/alpamayo1_5; make it importable without installing the package
# (its pyproject pulls flash-attn, which is not needed here).
_REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if os.path.isdir(os.path.join(_REPO_SRC, "alpamayo1_5")) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)
