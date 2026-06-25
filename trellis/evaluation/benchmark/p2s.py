# Point-to-Surface (P2S) metric: mean per-pred-point distance to GT surface,
# averaged over time. The caller is expected to ICP-align pred meshes using
# the SAME unified-ICP transform used for the Chamfer distance, so P2S
# numbers compare directly.
#
# Implementation uses pytorch3d's _PointFaceDistance (per-point squared face
# distance on the GPU) — avoids the rtree dependency that
# trimesh.proximity.closest_point pulls in.

import numpy as np
import torch
import trimesh


def compute_p2s(
    pred_meshes_aligned: list,  # list[trimesh.Trimesh]
    gt_meshes: list,            # list[trimesh.Trimesh]
    symmetric: bool = True,
    n_samples: int = 50_000,
    seed: int = 44,
    device: str = "cuda",
    return_per_frame: bool = False,
):
    """Point-to-surface distance using PyTorch3D's
    :func:`point_mesh_face_distance` (avoids the rtree dependency that
    trimesh.proximity.closest_point pulls in).

    For each frame:
        forward  = mean over sampled pred_points of dist(p, gt_surface_triangles)
        backward = mean over sampled gt_points   of dist(g, pred_surface_triangles)
        frame_p2s = (forward + backward) / 2    (if symmetric; else just forward)

    Final metric = mean over frames.

    If `return_per_frame=True`, returns `(mean, per_frame_list)` instead of
    just the mean — callers that persist per-frame breakdowns use this.
    """
    from pytorch3d.structures import Meshes, Pointclouds

    assert len(pred_meshes_aligned) == len(gt_meshes)
    dev = torch.device(device)
    frame_scores: list = []
    rng = np.random.default_rng(seed)

    def _to_p3d_mesh(tm: trimesh.Trimesh) -> Meshes:
        v = torch.tensor(np.asarray(tm.vertices), dtype=torch.float32, device=dev)
        f = torch.tensor(np.asarray(tm.faces), dtype=torch.int64, device=dev)
        return Meshes(verts=[v], faces=[f])

    def _one_dir(pm: trimesh.Trimesh, target: trimesh.Trimesh) -> float:
        pts, _ = trimesh.sample.sample_surface(
            pm, n_samples, seed=int(rng.integers(0, 1 << 30)),
        )
        pts_t = torch.tensor(np.asarray(pts), dtype=torch.float32, device=dev).unsqueeze(0)
        p3d_pcl = Pointclouds(points=pts_t)
        p3d_mesh = _to_p3d_mesh(target)
        # The pytorch3d helper returns *squared* face-distance per point,
        # summed; we want mean L2, so sqrt and average over the cloud.
        with torch.no_grad():
            # Custom flow: use point_face_distance directly for per-point distances.
            from pytorch3d.loss.point_mesh_distance import _PointFaceDistance  # noqa: E402
            pts_packed = p3d_pcl.points_packed()
            first_idx = p3d_pcl.cloud_to_packed_first_idx()
            max_pts = p3d_pcl.num_points_per_cloud().max().item()
            tris = p3d_mesh.verts_packed()[p3d_mesh.faces_packed()]
            tris_first = p3d_mesh.mesh_to_faces_packed_first_idx()
            d2 = _PointFaceDistance.apply(
                pts_packed, first_idx, tris, tris_first, max_pts, 5e-3,
            )
            d = torch.sqrt(d2.clamp_min(0.0))
        return float(d.mean().item())

    for pm, gm in zip(pred_meshes_aligned, gt_meshes):
        fwd = _one_dir(pm, gm)  # sample on pred, distance to gt-surface
        if symmetric:
            bwd = _one_dir(gm, pm)
            frame_scores.append((fwd + bwd) / 2)
        else:
            frame_scores.append(fwd)
    mean = float(np.mean(frame_scores))
    if return_per_frame:
        return mean, [float(v) for v in frame_scores]
    return mean
