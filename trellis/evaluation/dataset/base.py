"""Generic, path-templated evaluation dataset.

No hard-coded directory layout — you point it at your own GT via format-string
path templates, so the same eval code works for ActionBench, Motion80,
Consist4D, or any custom dataset. Configure with a JSON file (or dict); see
``config/eval_gt_dataset.example.json``.

Template fields (Python ``str.format``):
    {scene}   scene id
    {az}      GT view-dir name for the current azimuth (from ``az_order``)
    {frame}   per-frame index (format it, e.g. ``{frame:04d}``)

Config keys:
    scenes           : explicit list of scene ids.                  (one of
    scenes_glob      : glob whose matched basenames are scene ids.    these
                       two is required.)
    az_order         : 4 GT view-dir names aligned to gen azimuths
                       (0, 90, 180, 270), in that order. Default
                       ["0","90","180","270"]. This is the gen<->GT view map.
    search_azimuths_deg : list[float], per-scene calibration candidates
                       (only used by inference; eval ignores it). Default
                       [0,90,180,270].
    gt_4view         : template for appearance GT frames, e.g.
                       "/gt/{scene}/{az}/frame_{frame:04d}.png". Optional;
                       without it, appearance metrics cannot run.
    gt_pcd           : template for GT point clouds. Either per-frame files
                       (template contains {frame}, each file [N,3] or [N,>=3])
                       or a single file per scene (no {frame}; array
                       [T,N,>=3]). Optional; without it, geometry metrics
                       (CD / F-score / P2S) cannot run.
    gt_face_glb      : template for a per-scene .glb providing FACE topology,
                       e.g. "/gt/{scene}.glb". Combined with the per-frame
                       gt_pcd vertices to form GT meshes for P2S. Optional —
                       if absent, P2S is skipped (CD / F-score still run).

Only paths are configured here; nothing about a specific benchmark is baked in.

Writing your own dataset
------------------------
If path templates can't express your layout, the metric suites only duck-type
the following interface — implement these on any class and pass an instance
(skip ``make_dataset``):

    len(ds)                      -> int
    ds.scene_names               -> list[str]            (geometry enumeration)
    ds.geometry_supported        -> bool
    ds[i]                        -> {'scene_id', 'gt_4view_video' (or None),
                                     'num_frames', 'search_azimuths_deg'}
    ds.load_gt_pcd_sequence(i)   -> Tensor[T, N, 3]       (CD / F-score)
    ds.load_gt_mesh_sequence(i)  -> list[trimesh.Trimesh] (P2S; raise
                                    NotImplementedError to skip)
"""
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ._io import load_gt_view_frame_white_bg


def _frame_glob(template: str, scene: str, az: Optional[str] = None) -> List[str]:
    """Fill {scene}/{az}, turn the {frame...} field into '*', glob, sort."""
    pat = re.sub(r"\{frame[^}]*\}", "*", template)
    fmt: Dict[str, Any] = {"scene": scene}
    if az is not None:
        fmt["az"] = az
    return sorted(glob.glob(pat.format(**fmt)))


def _has_frame_field(template: str) -> bool:
    return re.search(r"\{frame[^}]*\}", template) is not None


class PlaceholderInferenceDataset:
    """Path-templated GT dataset for the eval suites. See module docstring."""

    # True iff a point-cloud template is configured (set in __init__).
    geometry_supported: bool = False

    def __init__(self, config: Dict[str, Any]):
        self.root = ""  # no single root; everything is template-driven
        self.gt_4view = config.get("gt_4view")
        self.gt_pcd = config.get("gt_pcd")
        self.gt_face_glb = config.get("gt_face_glb")
        self.az_order = list(config.get("az_order", ["0", "90", "180", "270"]))
        if len(self.az_order) != 4:
            raise ValueError(f"az_order must have 4 entries, got {self.az_order}")
        self.search_azimuths_deg = list(
            config.get("search_azimuths_deg", [0.0, 90.0, 180.0, 270.0])
        )
        # Geometry is available iff a point-cloud template is configured.
        self.geometry_supported = self.gt_pcd is not None

        # Scene enumeration: explicit list, or basenames from a glob.
        if config.get("scenes"):
            self.scene_names = list(config["scenes"])
        elif config.get("scenes_glob"):
            matched = sorted(glob.glob(config["scenes_glob"]))
            self.scene_names = [os.path.basename(p.rstrip("/")) for p in matched]
        else:
            raise ValueError("config must provide 'scenes' or 'scenes_glob'")
        if not self.scene_names:
            print("[Placeholder] WARNING: 0 scenes resolved from config")
        else:
            print(f"[Placeholder] {len(self.scene_names)} scenes")

    def __len__(self) -> int:
        return len(self.scene_names)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ------------------------------------------------------------------
    # Appearance GT
    # ------------------------------------------------------------------
    def _load_gt_4view(self, scene: str) -> torch.Tensor:
        """[4, T, 3, H, W] in [0,1], gen-azimuth ordered (RGBA composited on
        white). T is the min frame count across the four views."""
        per_view_paths = [_frame_glob(self.gt_4view, scene, az) for az in self.az_order]
        counts = [len(p) for p in per_view_paths]
        if min(counts) == 0:
            raise FileNotFoundError(
                f"[{scene}] missing GT 4-view frames for az_order={self.az_order} "
                f"(counts={counts})"
            )
        T = min(counts)
        views = []
        for paths in per_view_paths:
            frames = [
                torch.from_numpy(load_gt_view_frame_white_bg(p)).permute(2, 0, 1).float() / 255.0
                for p in paths[:T]
            ]
            views.append(torch.stack(frames, dim=0))
        return torch.stack(views, dim=0)  # [4, T, 3, H, W]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one scene. Note: NO cond_video — evaluation does not need the
        conditioning frames (only GT views / pcds / meshes)."""
        scene = self.scene_names[idx]
        gt_4view = self._load_gt_4view(scene) if self.gt_4view else None
        if gt_4view is not None:
            T = int(gt_4view.shape[1])
        elif self.geometry_supported:
            T = int(self.load_gt_pcd_sequence(idx).shape[0])
        else:
            T = 0
        return {
            "scene_id": scene,
            "gt_4view_video": gt_4view,
            "num_frames": T,
            "search_azimuths_deg": list(self.search_azimuths_deg),
        }

    # ------------------------------------------------------------------
    # Geometry GT
    # ------------------------------------------------------------------
    def load_gt_pcd_sequence(self, idx: int) -> torch.Tensor:
        """[T, N, 3] float xyz. Per-frame files (template has {frame}) are
        stacked; a single per-scene file (no {frame}) is loaded as [T,N,>=3]."""
        if not self.gt_pcd:
            raise NotImplementedError("no gt_pcd template configured")
        scene = self.scene_names[idx]
        if _has_frame_field(self.gt_pcd):
            files = _frame_glob(self.gt_pcd, scene)
            if not files:
                raise FileNotFoundError(f"[{scene}] no GT pcd files for template {self.gt_pcd}")
            arr = np.stack([np.load(f)[..., :3] for f in files], axis=0)  # (T,N,3)
        else:
            arr = np.load(self.gt_pcd.format(scene=scene))[..., :3]        # (T,N,3)
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    def load_gt_mesh_sequence(self, idx: int):
        """Per-frame GT meshes = shared FACE topology (from gt_face_glb) +
        per-frame vertices (from gt_pcd). Raises NotImplementedError when no
        face template is configured, so P2S is cleanly skipped."""
        if not self.gt_face_glb:
            raise NotImplementedError(
                "no gt_face_glb configured; P2S skipped (CD / F-score still run)"
            )
        import trimesh

        scene = self.scene_names[idx]
        glb_path = self.gt_face_glb.format(scene=scene)
        if not os.path.isfile(glb_path):
            raise FileNotFoundError(f"[{scene}] gt_face_glb missing: {glb_path}")
        src = trimesh.load(glb_path, force="mesh", process=False)
        faces = np.asarray(src.faces)
        n_v = src.vertices.shape[0]

        pcd = self.load_gt_pcd_sequence(idx).numpy()  # (T, N, 3)
        meshes = []
        for verts in pcd:
            if verts.shape[0] != n_v:
                raise ValueError(
                    f"[{scene}] per-frame vertex count {verts.shape[0]} != "
                    f"gt_face_glb vertex count {n_v}; cannot reuse face topology."
                )
            meshes.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
        return meshes


def make_dataset(config) -> PlaceholderInferenceDataset:
    """Build the dataset from a JSON config path or an already-parsed dict."""
    if isinstance(config, str):
        with open(config, "r") as f:
            config = json.load(f)
    if not isinstance(config, dict):
        raise TypeError(f"config must be a path or dict, got {type(config)}")
    return PlaceholderInferenceDataset(config)
