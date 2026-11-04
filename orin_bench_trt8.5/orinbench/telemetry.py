"""Background telemetry on Jetson: tegrastats, sysfs GPU/CPU clocks, /proc/meminfo.

Each sample is stamped with host ``time.time()`` when read, because tegrastats prints 1 s timestamps. Benchmark phases
record ``(t0, t1)`` windows on the same clock and ``summarize_window`` aggregates over them.

Power: on JetPack 5.1 the INA3221 rails are readable by root only, so unprivileged tegrastats has no ``VDD_*`` fields
and no power is reported. Clocks, GPU load, temperatures and memory are always recorded.
"""

from __future__ import annotations

import csv
import glob
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np

_RAIL_RE = re.compile(r"\b((?:VDD|VIN)_[A-Z0-9_]+) (\d+)mW/(\d+)mW")
_TEMP_RE = re.compile(r"\b([A-Za-z0-9_]+)@(-?[\d.]+)C\b")
_RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
_GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")
_EMC_RE = re.compile(r"EMC_FREQ (\d+)%")


def parse_tegrastats_line(line: str) -> dict:
    d: dict = {}
    if m := _RAM_RE.search(line):
        d["ram_used_mb"], d["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    if m := _GR3D_RE.search(line):
        d["gr3d_pct"] = int(m.group(1))
    if m := _EMC_RE.search(line):
        d["emc_pct"] = int(m.group(1))
    for name, inst, _avg in _RAIL_RE.findall(line):
        d[f"p_{name}_mw"] = int(inst)
    for name, t in _TEMP_RE.findall(line):
        if float(t) > -200:  # -256C = sensor absent
            d[f"t_{name}_c"] = float(t)
    return d


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def gpu_devfreq_dir() -> str | None:
    for p in glob.glob("/sys/devices/platform/*.ga10b/devfreq/*") + glob.glob("/sys/devices/gpu.0/devfreq/*"):
        if os.path.exists(os.path.join(p, "cur_freq")):
            return p
    return None


def meminfo() -> dict:
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable", "Shmem"):
            out[k] = int(v.split()[0]) // 1024  # MB
    if "MemTotal" in out and "MemAvailable" in out:
        out["sys_used_mb"] = out["MemTotal"] - out["MemAvailable"]
        # files parked in /dev/shm (weights, engines) are not part of the running model's footprint
        out["sys_used_excl_shm_mb"] = out["sys_used_mb"] - out.get("Shmem", 0)
    return out


@dataclass
class Telemetry:
    out_dir: str
    interval_ms: int = 100
    samples: list[dict] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _threads: list = field(default_factory=list)
    _proc: subprocess.Popen | None = None

    def start(self) -> "Telemetry":
        os.makedirs(self.out_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._teg_latest: dict = {}
        self._raw_teg = open(os.path.join(self.out_dir, "tegrastats_raw.log"), "w")
        if shutil.which("tegrastats"):
            self._proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)], stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
            self._threads.append(threading.Thread(target=self._read_tegrastats, daemon=True))
        self._devfreq = gpu_devfreq_dir()
        self._threads.append(threading.Thread(target=self._poll, daemon=True))
        for t in self._threads:
            t.start()
        return self

    def _read_tegrastats(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            now = time.time()
            self._raw_teg.write(f"{now:.3f} {line}")
            with self._lock:
                self._teg_latest = {"teg_t": now, **parse_tegrastats_line(line)}
            if self._stop.is_set():
                break

    def _poll(self) -> None:
        period = self.interval_ms / 1000.0
        while not self._stop.is_set():
            t = time.time()
            s: dict = {"t": t, **meminfo()}
            if self._devfreq:
                f = _read(os.path.join(self._devfreq, "cur_freq"))
                s["gpu_freq_mhz"] = int(f) / 1e6 if f else None
                load = _read("/sys/devices/platform/gpu.0/load") or _read("/sys/devices/gpu.0/load")
                s["gpu_load_pct"] = int(load) / 10 if load else None
                c = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
                s["cpu0_freq_mhz"] = int(c) / 1e3 if c else None
            with self._lock:
                s.update(self._teg_latest)
            self.samples.append(s)
            time.sleep(max(0.0, period - (time.time() - t)))

    def stop(self) -> str:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        for t in self._threads:
            t.join(timeout=2)
        self._raw_teg.close()
        path = os.path.join(self.out_dir, "telemetry.csv")
        keys = sorted({k for s in self.samples for k in s})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.samples)
        return path

    def summarize_window(self, t0: float, t1: float, n_calls: int | None = None) -> dict:
        win = [s for s in self.samples if t0 <= s["t"] <= t1]
        out: dict = {"window_s": t1 - t0, "n_samples": len(win)}
        if not win:
            return out
        ts = np.array([s["t"] for s in win])
        numeric = sorted({k for s in win for k, v in s.items() if isinstance(v, (int, float)) and k != "t"})
        for k in numeric:
            vals = np.array([s.get(k, np.nan) for s in win], dtype=float)
            if np.all(np.isnan(vals)):
                continue
            if k.startswith("p_"):
                out[f"{k}_avg"], out[f"{k}_peak"] = float(np.nanmean(vals)), float(np.nanmax(vals))
                ok = ~np.isnan(vals)
                if ok.sum() > 1:
                    energy = float(np.trapezoid(vals[ok], ts[ok]) / 1e3)
                    out[f"{k}_energy_j"] = energy
                    if n_calls:
                        out[f"{k}_energy_per_call_j"] = energy / n_calls
            elif k.startswith("t_") or k in ("sys_used_mb", "sys_used_excl_shm_mb", "ram_used_mb"):
                out[f"{k}_max"], out[f"{k}_mean"] = float(np.nanmax(vals)), float(np.nanmean(vals))
            elif k in ("gpu_freq_mhz", "cpu0_freq_mhz"):
                out[f"{k}_min"], out[f"{k}_mean"] = float(np.nanmin(vals)), float(np.nanmean(vals))
            elif k in ("gr3d_pct", "gpu_load_pct", "emc_pct"):
                out[f"{k}_mean"] = float(np.nanmean(vals))
        return out
