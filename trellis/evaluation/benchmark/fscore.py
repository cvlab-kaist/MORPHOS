# F-score (point-cloud based) for sequence evaluation.
# Logic mirrors Motion324/evaluation/evaluation_pcd.py:compute_fscore — KD-tree
# nearest-neighbor distances on both clouds, precision = mean(d < tau),
# recall = mean(d_other < tau), F = 2 P R / (P + R). The metric is symmetric
# in P/R so the labeling of pred vs gt doesn't change the result.

import numpy as np
import torch
from scipy.spatial import cKDTree


def compute_fscore_pcd(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """F-score between two point clouds at the given distance threshold.

    Args:
        pred_points: (N, 3) predicted points.
        gt_points:   (M, 3) ground-truth points.
        threshold:   Distance threshold tau (same units as the point clouds).

    Returns:
        F-score in [0, 1]. 0 if both precision and recall are 0.
    """
    pred = np.asarray(pred_points)
    gt = np.asarray(gt_points)
    if pred.size == 0 or gt.size == 0:
        return 0.0

    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)

    d_p2g, _ = tree_gt.query(pred, k=1)   # for each pred -> nearest gt
    d_g2p, _ = tree_pred.query(gt, k=1)   # for each gt   -> nearest pred

    precision = float(np.mean(d_p2g < threshold))
    recall = float(np.mean(d_g2p < threshold))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_fscore_sequence(
    gt_pcs: torch.Tensor,
    pred_pcs_aligned: torch.Tensor,
    threshold: float = 0.01,
) -> float:
    """Per-frame F-score, averaged across frames.

    Args:
        gt_pcs: (T, M, 3) GT point clouds (any device; moved to CPU).
        pred_pcs_aligned: (T, N, 3) predicted point clouds AFTER ICP
            alignment (any device; moved to CPU).
        threshold: Distance threshold for F-score.
    """
    assert gt_pcs.shape[0] == pred_pcs_aligned.shape[0], "Mismatched frame counts"
    gt_np = gt_pcs.detach().cpu().numpy()
    pr_np = pred_pcs_aligned.detach().cpu().numpy()
    scores = [
        compute_fscore_pcd(pr_np[k], gt_np[k], threshold=threshold)
        for k in range(gt_pcs.shape[0])
    ]
    return float(np.mean(scores))
