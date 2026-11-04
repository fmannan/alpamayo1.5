"""TensorRT engines on PyTorch tensors, in the PyTorch process, through trt_bridge/libtrt_stage.so.

Also the parity gates that decide whether a freshly built engine may be used:

    python -m orinbench.trt_stage vision-check --engine E --parity DIR   (references: orinbench.parity vision)
    python -m orinbench.trt_stage expert-check --engine E --parity DIR   (references: orinbench.parity expert)
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import time

import numpy as np
import torch

_LIB_PATH = os.environ.get("ORINBENCH_TRT_BRIDGE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trt_bridge", "libtrt_stage.so"
)
_lib: ctypes.CDLL | None = None
_DTYPES = {0: torch.float32, 1: torch.float16, 2: torch.int8, 3: torch.int32, 4: torch.bool, 5: torch.uint8}


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(f"{_LIB_PATH} not found: run trt_bridge/build.sh on the Orin")
        # RTLD_LAZY: with cuda-compat, libnvinfer pulls in a libnvcudla whose DLA symbols JetPack 5.1 may lack. DLA is unused.
        lib = ctypes.CDLL(_LIB_PATH, mode=os.RTLD_LAZY | os.RTLD_LOCAL)
        lib.trts_last_error.restype = ctypes.c_char_p
        lib.trts_load.restype = ctypes.c_void_p
        lib.trts_load.argtypes = [ctypes.c_char_p]
        lib.trts_free.argtypes = [ctypes.c_void_p]
        lib.trts_n_io.argtypes = [ctypes.c_void_p]
        for fn in ("trts_io_is_input", "trts_io_dtype", "trts_io_rank"):
            getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.trts_io_name.restype = ctypes.c_char_p
        lib.trts_io_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.trts_io_dim.restype = ctypes.c_int64
        lib.trts_io_dim.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.trts_bind.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        lib.trts_enable_graph.argtypes = [ctypes.c_void_p]
        lib.trts_run.argtypes = [ctypes.c_void_p]
        _lib = lib
    return _lib


class TrtEngine:
    """Fixed-shape engine with one PyTorch CUDA buffer per input and output."""

    def __init__(self, path: str):
        self.handle = None
        torch.cuda.init()
        torch.zeros(1, device="cuda")  # TensorRT must find PyTorch's CUDA context already current on this thread
        self.lib = _load_lib()
        self.handle = self.lib.trts_load(path.encode())
        if not self.handle:
            raise RuntimeError(f"{path}: {self.lib.trts_last_error().decode()}")
        self.inputs: dict[str, tuple[int, ...]] = {}
        self.outputs: dict[str, tuple[int, ...]] = {}
        self.buffers: dict[str, torch.Tensor] = {}
        try:
            for i in range(self.lib.trts_n_io(self.handle)):
                name = self.lib.trts_io_name(self.handle, i).decode()
                rank = self.lib.trts_io_rank(self.handle, i)
                shape = tuple(self.lib.trts_io_dim(self.handle, i, d) for d in range(rank))
                if rank < 0 or any(d < 0 for d in shape):
                    raise RuntimeError(f"{name}: engine has an unresolved dynamic shape {shape}")
                code = self.lib.trts_io_dtype(self.handle, i)
                if code not in _DTYPES:
                    raise RuntimeError(f"{name}: unsupported TensorRT I/O dtype code {code}")
                (self.inputs if self.lib.trts_io_is_input(self.handle, i) else self.outputs)[name] = shape
                buffer = torch.zeros(shape, dtype=_DTYPES[code], device="cuda")
                self.buffers[name] = buffer
                self._check(self.lib.trts_bind(self.handle, name.encode(), ctypes.c_void_p(buffer.data_ptr())))
        except BaseException:
            self.close()
            raise

    def _check(self, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(self.lib.trts_last_error().decode())

    def enable_graph(self) -> None:
        """Capture a CUDA graph. Buffers stay in place, so keep writing inputs into ``self.buffers``."""
        torch.cuda.synchronize()
        self._check(self.lib.trts_enable_graph(self.handle))

    def run(self) -> None:
        torch.cuda.synchronize()  # PyTorch and the engine use separate streams
        self._check(self.lib.trts_run(self.handle))

    def close(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.handle = None
            self.lib.trts_free(handle)

    def __del__(self):
        self.close()


def _relative_l2(got: np.ndarray, want: np.ndarray) -> float:
    got, want = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _timed_runs(engine: TrtEngine, n: int) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        engine.run()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times[1:] or times))


def vision_check(a) -> None:
    engine = TrtEngine(a.engine)
    print("inputs", engine.inputs, "outputs", engine.outputs, flush=True)
    references = sorted(glob.glob(os.path.join(a.parity, "*.ref.npz")))
    if not references:
        raise SystemExit(f"no vision references in {a.parity}")
    for ref_path in references:
        sid = os.path.basename(ref_path)[: -len(".ref.npz")]
        pixels = np.fromfile(os.path.join(a.parity, sid + ".pixel_values.bin"), dtype=np.float32)
        engine.buffers["pixel_values"].copy_(torch.from_numpy(pixels).reshape(engine.inputs["pixel_values"]).cuda())
        ms = _timed_runs(engine, a.iterations)
        ref, rows = np.load(ref_path), {}
        for name in engine.outputs:
            got = engine.buffers[name].float().cpu().numpy()
            rows[name] = {"finite": bool(np.isfinite(got).all()), "relative_l2": _relative_l2(got, ref[name])}
        print(sid, json.dumps({"ms_p50": ms, "outputs": rows}), flush=True)
        if any(not r["finite"] or not r["relative_l2"] <= a.max_relative_l2 for r in rows.values()):
            raise SystemExit(f"{sid}: vision parity exceeds {a.max_relative_l2}")
    print("VISION GATE PASSED", flush=True)


def expert_check(a) -> None:
    """Real captured flow-step inputs, eagerly and then through one CUDA graph with inputs rewritten in place."""
    engine = TrtEngine(a.engine)
    print("inputs", len(engine.inputs), "outputs", engine.outputs, flush=True)
    dirs = sorted(d for d in glob.glob(os.path.join(a.parity, "*")) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no expert references in {a.parity}")
    eager: dict = {}

    def one(d: str, label: str) -> None:
        for name, shape in engine.inputs.items():
            data = np.fromfile(os.path.join(d, name + ".bin"), dtype=np.float32)
            engine.buffers[name].copy_(torch.from_numpy(data).reshape(shape).cuda())
        ms = _timed_runs(engine, a.iterations)
        got = engine.buffers["velocity"].float().cpu().numpy().ravel()
        ref = np.load(os.path.join(d, "ref.npz"))
        rows = {"vs_float32": _relative_l2(got, ref["velocity"]), "vs_bf16_pytorch": _relative_l2(got, ref["bf16"])}
        finite = bool(np.isfinite(got).all())
        print(os.path.basename(d), label, json.dumps({"ms_p50": ms, "finite": finite, "relative_l2": rows}), flush=True)
        if not finite or not rows["vs_float32"] <= a.max_relative_l2:
            raise SystemExit(f"{d}: expert parity exceeds {a.max_relative_l2}")
        if label == "eager":
            eager[d] = got.copy()
        elif not np.allclose(got, eager[d], rtol=1e-5, atol=1e-5):
            raise SystemExit(f"{d}: graph output differs from eager with the same inputs")

    for d in dirs:
        one(d, "eager")
    engine.enable_graph()
    for d in dirs:  # a graph that ignored rewritten inputs would match only the scene loaded at capture
        one(d, "graph")
    print("EXPERT GATE PASSED", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, limit in (("vision-check", vision_check, 0.06), ("expert-check", expert_check, 0.005)):
        s = sub.add_parser(name)
        s.add_argument("--engine", required=True)
        s.add_argument("--parity", required=True)
        s.add_argument("--iterations", type=int, default=6)
        s.add_argument("--max-relative-l2", type=float, default=limit)
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
