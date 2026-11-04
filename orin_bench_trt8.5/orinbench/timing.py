"""Stage timing without double counting.

Two measurements are taken for every policy call:

* ``wall`` -- host ``perf_counter`` timestamps at stage boundaries. In ``staged`` mode a
  ``torch.cuda.synchronize()`` is inserted at each boundary, so the wall time of a stage is
  exactly the time to finish its GPU work and the stages sum to the end-to-end latency.
  In ``e2e`` mode only the start and the final device->host copy are synchronized, which gives
  the deployed latency (inner boundaries are then only host launch timestamps).
* ``gpu`` -- CUDA events recorded on the current stream at the same boundaries. Event deltas
  are stream-timeline intervals (they include GPU idle gaps while the host launches work).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class Mark:
    name: str
    wall: float
    event: torch.cuda.Event | None


@dataclass
class StageClock:
    """Ordered boundary marks for one policy call. Interval i is named by the mark that ends it."""

    sync_each_stage: bool
    use_cuda: bool = True
    marks: list[Mark] = field(default_factory=list)

    def mark(self, name: str) -> None:
        ev = None
        if self.use_cuda:
            if self.sync_each_stage:
                torch.cuda.synchronize()
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
        self.marks.append(Mark(name, time.perf_counter(), ev))

    def finalize(self) -> dict:
        if self.use_cuda:
            torch.cuda.synchronize()
        out: dict = {"wall_ms": {}, "gpu_ms": {}}
        for prev, cur in zip(self.marks[:-1], self.marks[1:]):
            out["wall_ms"][cur.name] = out["wall_ms"].get(cur.name, 0.0) + (cur.wall - prev.wall) * 1e3
            if prev.event is not None and cur.event is not None:
                out["gpu_ms"][cur.name] = out["gpu_ms"].get(cur.name, 0.0) + prev.event.elapsed_time(cur.event)
        first, last = self.marks[0], self.marks[-1]
        out["e2e_wall_ms"] = (last.wall - first.wall) * 1e3
        if first.event is not None:
            out["e2e_gpu_ms"] = first.event.elapsed_time(last.event)
        return out
