"""Fixed scene packs: decoded camera frames + ego history (+ logged future) frozen to disk.

Packs are made once on a host with dataset access (``orinbench.cli prep-scenes``) with the official
``alpamayo1_5.load_physical_aiavdataset`` loader, then copied to the Orin. Freezing the decoded frames gives every
device byte-identical inputs and keeps video decoding and streaming out of the timed call (a vehicle delivers
decoded frames).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import torch


@dataclass
class Scene:
    scene_id: str
    clip_id: str
    t0_us: int
    image_frames: torch.Tensor  # (n_cam, n_frames, 3, H, W) uint8
    camera_indices: torch.Tensor
    ego_history_xyz: torch.Tensor  # (1, 1, 16, 3) float32, ego frame at t0
    ego_history_rot: torch.Tensor  # (1, 1, 16, 3, 3)
    ego_future_xyz: torch.Tensor | None  # (1, 1, 64, 3)
    ego_future_rot: torch.Tensor | None


_KEYS = ("image_frames", "camera_indices", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_scenes(specs: list[dict], out_dir: str) -> dict:
    """specs: [{"clip_id": ..., "t0_us": ...}, ...]. Needs HuggingFace access to the dataset."""
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    os.makedirs(out_dir, exist_ok=True)
    manifest: dict = {"scenes": [], "loader": "alpamayo1_5.load_physical_aiavdataset (official defaults)"}
    for i, spec in enumerate(specs):
        sid = f"s{i:02d}_{spec['clip_id'][:8]}_{spec['t0_us'] // 1000}ms"
        path = os.path.join(out_dir, f"{sid}.pt")
        if not os.path.exists(path):
            try:
                data = load_physical_aiavdataset(spec["clip_id"], t0_us=spec["t0_us"])
            except Exception as e:  # noqa: BLE001
                manifest["scenes"].append({**spec, "scene_id": sid, "error": repr(e)})
                print(f"[prep] {sid} FAILED: {e!r}", flush=True)
                continue
            payload = {"scene_id": sid, "clip_id": spec["clip_id"], "t0_us": int(spec["t0_us"])}
            payload.update({k: data[k] for k in _KEYS if k in data})
            torch.save(payload, path)
        d = torch.load(path, weights_only=False)
        manifest["scenes"].append(
            {**spec, "scene_id": sid, "file": os.path.basename(path), "sha256": _sha256(path),
             "image_frames_shape": list(d["image_frames"].shape)}
        )
        print(f"[prep] {sid} frames={tuple(d['image_frames'].shape)}", flush=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_scenes(scene_dir: str, limit: int | None = None, ids: list[str] | None = None) -> list[Scene]:
    with open(os.path.join(scene_dir, "manifest.json")) as f:
        manifest = json.load(f)
    out = []
    for s in manifest["scenes"]:
        if "file" not in s or (ids and s["scene_id"] not in ids):
            continue
        d = torch.load(os.path.join(scene_dir, s["file"]), weights_only=False)
        out.append(
            Scene(
                scene_id=d["scene_id"],
                clip_id=d["clip_id"],
                t0_us=d["t0_us"],
                image_frames=d["image_frames"],
                camera_indices=d["camera_indices"],
                ego_history_xyz=d["ego_history_xyz"].float(),
                ego_history_rot=d["ego_history_rot"].float(),
                ego_future_xyz=d.get("ego_future_xyz"),
                ego_future_rot=d.get("ego_future_rot"),
            )
        )
        if limit and len(out) >= limit:
            break
    if ids and len(out) != len(ids):
        raise RuntimeError(f"scenes not found in {scene_dir}: {sorted(set(ids) - {s.scene_id for s in out})}")
    return out
