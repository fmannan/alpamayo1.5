"""Platform and software fingerprint, recorded with every run. Works unprivileged."""

from __future__ import annotations

import glob
import importlib
import os
import platform
import subprocess
import sys

from .telemetry import gpu_devfreq_dir


def _sh(cmd: str, timeout: float = 10) -> str | None:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip() or None
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def _read(p: str) -> str | None:
    try:
        with open(p, "rb") as f:
            return f.read().replace(b"\0", b" ").decode().strip()
    except OSError:
        return None


def _ver(mod: str) -> str | None:
    try:
        return getattr(importlib.import_module(mod), "__version__", "present")
    except Exception:  # noqa: BLE001
        return None


def collect() -> dict:
    info: dict = {"kernel": platform.release(), "machine": platform.machine(), "python": sys.version.split()[0]}
    if os.path.exists("/etc/nv_tegra_release"):
        info["device_tree_model"] = _read("/proc/device-tree/model")
        info["l4t_release"] = _read("/etc/nv_tegra_release")
        info["nvpmodel"] = _sh("nvpmodel -q 2>&1 | head -2")
        d = gpu_devfreq_dir()
        if d:
            info["gpu_devfreq"] = {k: _read(os.path.join(d, k)) for k in ("governor", "min_freq", "max_freq", "cur_freq")}
        info["cpu_freq_khz"] = {
            "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "cur": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
            "max": _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),
        }
        info["online_cpus"] = _read("/sys/devices/system/cpu/online")
        info["thermal_zones_c"] = {
            _read(z + "/type"): int(_read(z + "/temp") or -256000) / 1000.0
            for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*"))
        }
        info["packages"] = _sh(
            "dpkg-query -W -f='${Package} ${Version}\\n' nvidia-l4t-core libnvinfer8 cuda-compat-12-2 cuda-toolkit-12-2 2>/dev/null"
        )
    info["mem"] = (_read("/proc/meminfo") or "").splitlines()[:3]
    info["disk_shm"] = _sh("df -h /dev/shm | tail -1")
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["torch_cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["cuda_device"] = {"name": p.name, "cc": f"{p.major}.{p.minor}", "sm_count": p.multi_processor_count,
                                   "total_mem_mb": p.total_memory // 2**20}
            info["torch_arch_list"] = torch.cuda.get_arch_list()
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = repr(e)
    for m in ("transformers", "accelerate", "safetensors", "numpy"):
        info[f"pkg_{m}"] = _ver(m)
    return info
