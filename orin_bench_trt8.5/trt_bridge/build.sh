#!/bin/bash
# Run on the Orin. Builds libtrt_stage.so, which orinbench.trt_stage loads with ctypes.
# Links the same CUDA runtime as PyTorch (CUDA_HOME, default 12.2), so both share one runtime in the process.
set -euo pipefail
cd "$(dirname "$0")"
CUDA=${CUDA_HOME:-/usr/local/cuda-12.2}
[ -e "$CUDA/include/cuda_runtime_api.h" ] || { echo "no CUDA toolkit at $CUDA (set CUDA_HOME)" >&2; exit 1; }
g++ -std=c++17 -O2 -Wall -pthread -Wno-deprecated-declarations -I"$CUDA/include" -shared -fPIC trt_stage.cpp \
    -o libtrt_stage.so -L"$CUDA/lib64" -lcudart -lnvinfer -Wl,-rpath,"$CUDA/lib64"
ls -la libtrt_stage.so
