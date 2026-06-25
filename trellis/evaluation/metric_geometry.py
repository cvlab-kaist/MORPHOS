"""Geometry metrics over per-frame .glb predictions.

Metrics (all share a single unified-ICP alignment built from frame 0, so the
numbers are directly comparable):

    CD       : Chamfer distance — single ICP from frame 0, then mean
               Chamfer distance across frames.
    F-score  : Motion324-style KD-tree F-score with default threshold 0.01,
               evaluated on the unified-ICP-aligned pred point cloud against
               GT pcds; per-frame, mean across frames.
    P2S      : Symmetric Point-to-Surface, evaluated on unified-ICP-aligned
               pred meshes against GT meshes. NaN for datasets that do not
               expose ``load_gt_mesh_sequence`` (currently only motion80
               does).

Pred meshes are pre-normalized to a shared [-1, 1] bbox across all frames
before ICP / sampling (mirrors ``evaluate_motion80.py``). GT remains in
original scale; the ICP scale parameter absorbs the rescale. This is a
pure ICP-stability transform — does not change CD / F-score semantics,
since all of them apply the learned ICP transform back.

ICP config (mirrors ``trellis/evaluation/benchmark/icp.py:gradient_icp``):
    - similarity transform with ANISOTROPIC scale (3 per-axis factors)
    - 6D continuous rotation parameterization, composed on top of one of
      24 canonical axis-aligned rotations used as multi-start inits
    - free 3-vector translation
    - Adam, lr=0.01, n_iter=200; loss = pytorch3d.loss.chamfer_distance
      (batch_reduction=None) with backward over loss.mean() across all 24
      starts; best-of-K tracking by min single-init loss
    - returned as ``Transform3d = Scale(s) ∘ Rotate(R) ∘ Translate(t)``

GT loading (via the generic path-templated dataset; see
``config/eval_gt_dataset.example.json``):
    - ``gt_pcd``      -> per-frame point clouds for CD / F-score (required).
    - ``gt_face_glb`` -> face topology; combined with the gt_pcd vertices to
      build per-frame GT meshes for P2S (optional — absent => P2S skipped).

Pred structure:
    {pred_path}/{scene_id}/frame_NNNN.glb

Usage:
    python -m trellis.evaluation.metric_geometry \\
        --gt_config /path/to/eval_gt_dataset.json \\
        --pred_path /path/to/output/geometry \\
        [--output_csv metrics_geometry.csv] [--device cuda] \\
        [--n_pts_icp 10000] [--n_pts_chamfer 100000] \\
        [--fscore_threshold 0.01] [--p2s_n_samples 50000] \\
        [--seed 44] [--recompute]
"""
import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

from .benchmark import (
    compute_chamfer_score,
    compute_fscore_pcd,
    compute_p2s,
    gradient_icp,
    sample_meshes,
    sample_point_cloud,
)
from .dataset import make_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Result containers
# =============================================================================

@dataclass
class SampleResult:
    """Per-scene metric row.

    Unavailable metrics use ``-1.0`` as a sentinel (both in memory and in the
    saved CSV) — e.g. ``p2s`` when the dataset has no GT mesh (actionbench).
    Aggregation in :meth:`DatasetResults.summary` skips ``-1`` values.
    """
    uid: str
    chamfer_distance: float = -1.0
    f_score: float = -1.0
    p2s: float = -1.0
    n_frames: int = 0
    status: str = "pending"
    error_message: str = ""


@dataclass
class DatasetResults:
    """Aggregates per-scene rows + per-frame breakdowns.

    The per-frame table is long-format with columns
    ``[uid, frame_idx, chamfer_distance, f_score, p2s]``.
    """
    samples: List[SampleResult] = field(default_factory=list)
    per_frame_rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        r: SampleResult,
        per_frame: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.samples.append(r)
        if per_frame:
            self.per_frame_rows.extend(per_frame)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self.samples])

    def per_frame_dataframe(self) -> pd.DataFrame:
        cols = ["uid", "frame_idx", "chamfer_distance", "f_score", "p2s"]
        if not self.per_frame_rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self.per_frame_rows, columns=cols)

    def summary(self) -> Dict[str, float]:
        df = self.to_dataframe()
        ok = df[df["status"] == "success"] if not df.empty else df
        out = {
            "n_total": int(len(df)),
            "n_success": int(len(ok)),
            "n_failed": int(len(df) - len(ok)),
            "success_rate": float(len(ok) / len(df)) if len(df) else 0.0,
        }
        for k in ("chamfer_distance", "f_score", "p2s"):
            # Skip both NaN (legacy CSVs) and -1 sentinels (current "unavailable").
            if len(ok) and k in ok.columns:
                vals = ok[k].astype(float).values
                valid = vals[np.isfinite(vals) & (vals != -1.0)]
                out[f"{k}_mean"] = float(np.mean(valid)) if len(valid) else float("nan")
            else:
                out[f"{k}_mean"] = float("nan")
        return out


def _per_frame_csv_path(csv_path: str) -> str:
    return os.path.splitext(csv_path)[0] + "_per_frame.csv"


def _summary_json_path(csv_path: str) -> str:
    return os.path.splitext(csv_path)[0] + ".summary.json"


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(data: Dict[str, Any], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_existing(csv_path: str) -> Tuple[Dict[str, SampleResult], List[Dict[str, Any]]]:
    """Return ``(samples_by_uid, per_frame_rows)``. Either may be empty.

    Tolerant of legacy CSVs that used different column names (e.g. ``cd_4d``):
    any missing column reads back as ``-1.0`` and the row is treated as
    incomplete, so a recompute is needed.
    """
    samples: Dict[str, SampleResult] = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            samples[row["uid"]] = SampleResult(
                uid=row["uid"],
                chamfer_distance=float(row["chamfer_distance"])
                    if "chamfer_distance" in row else -1.0,
                f_score=float(row["f_score"]) if "f_score" in row else -1.0,
                p2s=float(row["p2s"]) if "p2s" in row else -1.0,
                n_frames=int(row["n_frames"]),
                status=str(row["status"]),
                error_message=(
                    str(row.get("error_message", ""))
                    if not pd.isna(row.get("error_message", "")) else ""
                ),
            )
    per_frame_rows: List[Dict[str, Any]] = []
    pf_path = _per_frame_csv_path(csv_path)
    if os.path.exists(pf_path):
        df_pf = pd.read_csv(pf_path)
        per_frame_rows = df_pf.to_dict("records")
    return samples, per_frame_rows


def save_results(results: DatasetResults, csv_path: str) -> None:
    """Atomically write per-scene CSV + per-frame CSV + summary JSON.

    Each file is written via a ``<path>.tmp`` then ``os.replace`` so the
    on-disk state is consistent at every flush. Called after every scene in
    :func:`evaluate_dataset` so progress survives a crash.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    _atomic_write_csv(results.to_dataframe(), csv_path)
    _atomic_write_csv(results.per_frame_dataframe(), _per_frame_csv_path(csv_path))
    _atomic_write_json(results.summary(), _summary_json_path(csv_path))


# =============================================================================
# Per-frame mesh helpers
# =============================================================================

def _icp_transform_mesh(
    mesh: trimesh.Trimesh, transform, device: torch.device,
) -> trimesh.Trimesh:
    """Apply a pytorch3d Transform3d to mesh vertices, returning a new
    trimesh with the same face topology."""
    v = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    v_aligned = transform.transform_points(v.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    return trimesh.Trimesh(
        vertices=v_aligned,
        faces=np.asarray(mesh.faces),
        process=False,
    )


def _normalize_pred_meshes(meshes: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
    """Shared-bbox global rescale of pred meshes to a [-1, 1] bbox across
    ALL frames. Stabilizes ICP convergence; GT stays in original scale and
    the ICP scale parameter absorbs the rescale.

    Single shared center / scale (NOT per-frame) so inter-frame geometry is
    preserved. Mirrors ``_normalize_meshes`` in ``evaluate_motion80.py``.
    """
    all_verts = np.concatenate([np.asarray(m.vertices) for m in meshes], axis=0)
    lo = all_verts.min(axis=0)
    hi = all_verts.max(axis=0)
    center = (hi + lo) / 2.0
    extent = float((hi - lo).max())
    scale = extent / 2.0 if extent > 0 else 1.0
    return [
        trimesh.Trimesh(
            vertices=(np.asarray(m.vertices) - center) / scale,
            faces=np.asarray(m.faces),
            process=False,
        )
        for m in meshes
    ]


# =============================================================================
# Per-scene driver (single ICP, all metrics)
# =============================================================================

def evaluate_scene(
    scene_id: str,
    gt_pc: torch.Tensor,
    pred_dir: str,
    *,
    device: str,
    n_pts_icp: int,
    n_pts_chamfer: int,
    seed: int,
    fscore_threshold: float,
    p2s_n_samples: int,
    gt_meshes_loader=None,        # callable[[], List[trimesh.Trimesh]] | None
    mesh_pattern: str = "frame_*.glb",
) -> Tuple[SampleResult, List[Dict[str, Any]]]:
    """Compute Chamfer distance (CD), F-score, and (when GT meshes are
    available) P2S for one scene. A single unified ICP transform is fitted
    on frame 0 and reused for all three metrics so the numbers are directly
    comparable.

    Returns
    -------
    SampleResult
        Per-scene scalar row with means over frames.
    list[dict]
        Per-frame rows for the per-frame CSV. One dict per (uid, frame_idx)
        with columns ``{uid, frame_idx, chamfer_distance, f_score, p2s}``.
        Empty list if the scene errored before the metrics ran. Frames with
        no P2S source (no GT mesh) carry ``p2s = -1.0``.
    """
    result = SampleResult(uid=scene_id)
    per_frame_rows: List[Dict[str, Any]] = []
    try:
        if not os.path.isdir(pred_dir):
            result.status = "error"
            result.error_message = f"Pred dir missing: {pred_dir}"
            return result, per_frame_rows

        pred_files = sorted(glob(os.path.join(pred_dir, mesh_pattern)))
        if not pred_files:
            result.status = "error"
            result.error_message = f"No pred meshes in {pred_dir}"
            return result, per_frame_rows
        n_frames = min(int(gt_pc.shape[0]), len(pred_files))
        result.n_frames = n_frames
        gt_pc_clip_cpu = gt_pc[:n_frames]
        pred_meshes = [
            trimesh.load(p, force="mesh", process=False)
            for p in pred_files[:n_frames]
        ]
        pred_meshes = _normalize_pred_meshes(pred_meshes)

        # ----------------------------------------------------------------
        # 1) Sample point clouds + unified ICP (from frame 0).
        # ----------------------------------------------------------------
        pred_pc = sample_meshes(
            pred_meshes, n_pts=n_pts_chamfer, synchronized=False, seed=seed,
        )
        pred_pc_icp = sample_point_cloud(pred_pc, n_pts=n_pts_icp, seed=seed)
        gt_pc_icp = sample_point_cloud(gt_pc_clip_cpu, n_pts=n_pts_icp, seed=seed)

        dev = torch.device(device)
        pred_pc = pred_pc.to(dev)
        gt_pc_clip = gt_pc_clip_cpu.to(dev)
        pred_pc_icp = pred_pc_icp.to(dev)
        gt_pc_icp = gt_pc_icp.to(dev)

        icp_unified = gradient_icp(
            pc_pred=pred_pc_icp[0], pc_gt=gt_pc_icp[0], lr=0.01, n_iter=200,
        )
        pred_pc_aligned = icp_unified.transform_points(pred_pc)

        # ----------------------------------------------------------------
        # 2) Chamfer distance — per-frame on unified-ICP-aligned pcds.
        # ----------------------------------------------------------------
        cd_pf = [
            float(compute_chamfer_score(
                gt=gt_pc_clip[k].cpu(), pred=pred_pc_aligned[k].cpu(),
            )) for k in range(n_frames)
        ]
        result.chamfer_distance = float(np.mean(cd_pf))

        # ----------------------------------------------------------------
        # 3) F-score on the same unified-ICP-aligned pcds.
        # ----------------------------------------------------------------
        f_score_pf = [
            float(compute_fscore_pcd(
                pred_points=pred_pc_aligned[k].cpu().numpy(),
                gt_points=gt_pc_clip[k].cpu().numpy(),
                threshold=fscore_threshold,
            )) for k in range(n_frames)
        ]
        result.f_score = float(np.mean(f_score_pf))

        # ----------------------------------------------------------------
        # 4) P2S on unified-ICP-aligned pred meshes vs GT meshes (optional)
        # ----------------------------------------------------------------
        p2s_pf: List[float] = [-1.0] * n_frames
        gt_meshes = None
        if gt_meshes_loader is not None:
            try:
                gt_meshes = gt_meshes_loader()
            except NotImplementedError as e:
                logger.info(f"[{scene_id}] P2S skipped (no GT meshes): {e}")
                gt_meshes = None
        if gt_meshes is not None:
            if len(gt_meshes) < n_frames:
                logger.warning(
                    f"[{scene_id}] GT meshes shorter ({len(gt_meshes)}) than "
                    f"pred ({n_frames}); truncating P2S to GT length."
                )
                n_p2s = min(len(gt_meshes), n_frames)
            else:
                n_p2s = n_frames
            pred_meshes_aligned = [
                _icp_transform_mesh(pred_meshes[k], icp_unified, device=dev)
                for k in range(n_p2s)
            ]
            p2s_mean, p2s_pf_partial = compute_p2s(
                pred_meshes_aligned=pred_meshes_aligned,
                gt_meshes=gt_meshes[:n_p2s],
                symmetric=True,
                n_samples=p2s_n_samples,
                seed=seed,
                device=device,
                return_per_frame=True,
            )
            result.p2s = float(p2s_mean)
            for k in range(n_p2s):
                p2s_pf[k] = float(p2s_pf_partial[k])

        # Build per-frame rows once all metrics are populated.
        for k in range(n_frames):
            per_frame_rows.append({
                "uid": scene_id,
                "frame_idx": k,
                "chamfer_distance": cd_pf[k],
                "f_score": f_score_pf[k],
                "p2s": p2s_pf[k],
            })

        result.status = "success"
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)
        per_frame_rows = []  # don't keep a partial frame list on error
        logger.error(f"[{scene_id}] {e}")
    return result, per_frame_rows


# =============================================================================
# Driver / CLI
# =============================================================================

def evaluate_dataset(
    gt_config: str,
    pred_path: str,
    output_csv: Optional[str],
    *,
    device: str = "cuda",
    n_pts_icp: int = 10_000,
    n_pts_chamfer: int = 100_000,
    seed: int = 44,
    mesh_pattern: str = "frame_*.glb",
    fscore_threshold: float = 0.01,
    p2s_n_samples: int = 50_000,
    recompute: bool = False,
) -> DatasetResults:
    dataset = make_dataset(gt_config)
    if len(dataset) == 0:
        logger.warning("0 scenes resolved from gt_config; nothing to evaluate.")
        return DatasetResults()
    if not getattr(dataset, "geometry_supported", False):
        raise ValueError(
            "Geometry GT not available: set 'gt_pcd' in the gt_config to enable "
            "CD / F-score (and 'gt_face_glb' for P2S)."
        )

    existing_samples: Dict[str, SampleResult] = {}
    existing_per_frame: List[Dict[str, Any]] = []
    if output_csv and not recompute:
        existing_samples, existing_per_frame = load_existing(output_csv)
    if existing_samples:
        n_done = sum(1 for r in existing_samples.values() if r.status == "success")
        logger.info(f"loaded {len(existing_samples)} prior results ({n_done} successful). "
                    "Use --recompute to redo.")
    # Drop per-frame rows for any uid that isn't a successful prior sample —
    # avoids mixing stale frame rows from earlier runs with the new state.
    successful_uids = {
        uid for uid, r in existing_samples.items() if r.status == "success"
    }
    existing_per_frame = [r for r in existing_per_frame if r["uid"] in successful_uids]

    results = DatasetResults(per_frame_rows=list(existing_per_frame))
    for i in tqdm(range(len(dataset)), desc="geometry"):
        # Resolve scene_id without a full __getitem__ (avoid loading cond_video).
        if hasattr(dataset, "scene_names"):
            scene_id = dataset.scene_names[i]
        elif hasattr(dataset, "scenes"):
            scene_id = dataset.scenes[i][0]
        else:
            scene_id = dataset[i]["scene_id"]

        if (
            scene_id in existing_samples
            and not recompute
            and existing_samples[scene_id].status == "success"
        ):
            results.samples.append(existing_samples[scene_id])
            continue

        try:
            gt_pc = dataset.load_gt_pcd_sequence(i)
        except Exception as e:
            r = SampleResult(uid=scene_id, status="error",
                             error_message=f"GT load failed: {e}")
            results.add(r)
            if output_csv:
                save_results(results, output_csv)
            logger.error(f"[{scene_id}] GT load failed: {e}")
            continue

        # P2S needs per-frame GT meshes; lazy-loaded inside evaluate_scene.
        def _gt_mesh_loader(idx=i):
            return dataset.load_gt_mesh_sequence(idx)

        pred_dir = os.path.join(pred_path, scene_id)
        r, pf_rows = evaluate_scene(
            scene_id, gt_pc, pred_dir,
            device=device,
            n_pts_icp=n_pts_icp, n_pts_chamfer=n_pts_chamfer,
            seed=seed, mesh_pattern=mesh_pattern,
            fscore_threshold=fscore_threshold,
            p2s_n_samples=p2s_n_samples,
            gt_meshes_loader=_gt_mesh_loader,
        )
        results.add(r, per_frame=pf_rows)
        if r.status == "success":
            msg = (
                f"[{scene_id}] CD={r.chamfer_distance:.4f} "
                f"F={r.f_score:.4f}"
            )
            if r.p2s != -1.0:
                msg += f" P2S={r.p2s:.4f}"
            logger.info(msg)
        if output_csv:
            save_results(results, output_csv)

    if output_csv:
        save_results(results, output_csv)
        logger.info(f"saved CSV to {output_csv}")
    return results


def print_summary(results: DatasetResults):
    s = results.summary()
    print("\n" + "=" * 60)
    print("GEOMETRY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  total   : {s['n_total']}")
    print(f"  success : {s['n_success']}")
    print(f"  failed  : {s['n_failed']}")
    print(f"  rate    : {s['success_rate']:.1%}")
    if s["n_success"] > 0:
        print("\nMean metrics (NaNs skipped):")
        print(f"  CD      : {s['chamfer_distance_mean']:.4f}")
        print(f"  F-score : {s['f_score_mean']:.4f}")
        print(f"  P2S     : {s['p2s_mean']:.4f}")
    df = results.to_dataframe()
    failed = df[df["status"] != "success"] if not df.empty else df
    if len(failed):
        print(f"\nFailed ({len(failed)}):")
        for _, row in failed.iterrows():
            print(f"  [{row['uid']}] {row['status']}: {row['error_message']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Chamfer distance / F-score / P2S over per-frame "
                    ".glb predictions.",
    )
    parser.add_argument("--gt_config", required=True,
                        help="JSON describing GT paths (see "
                             "config/eval_gt_dataset.example.json). Needs "
                             "'gt_pcd' for CD/F-score; 'gt_face_glb' for P2S.")
    parser.add_argument("--pred_path", required=True,
                        help="<pred_path>/<scene>/frame_NNNN.glb")
    parser.add_argument("--output_csv", default=None,
                        help="Default: <pred_path>/../metrics_geometry.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_pts_icp", type=int, default=10_000)
    parser.add_argument("--n_pts_chamfer", type=int, default=100_000)
    parser.add_argument("--fscore_threshold", type=float, default=0.01,
                        help="Distance threshold for F-score (Motion324 default 0.02; "
                             "this script defaults to 0.01).")
    parser.add_argument("--p2s_n_samples", type=int, default=50_000,
                        help="Surface-sample budget per direction for P2S.")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--mesh_pattern", default="frame_*.glb",
                        help="Glob inside <pred_path>/<scene>/.")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute even if --output_csv already has the scene.")
    args = parser.parse_args()

    output_csv = args.output_csv or os.path.join(
        os.path.dirname(args.pred_path.rstrip("/")) or ".",
        "metrics_geometry.csv",
    )

    results = evaluate_dataset(
        gt_config=args.gt_config,
        pred_path=args.pred_path,
        output_csv=output_csv,
        device=args.device,
        n_pts_icp=args.n_pts_icp,
        n_pts_chamfer=args.n_pts_chamfer,
        seed=args.seed,
        mesh_pattern=args.mesh_pattern,
        fscore_threshold=args.fscore_threshold,
        p2s_n_samples=args.p2s_n_samples,
        recompute=args.recompute,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
