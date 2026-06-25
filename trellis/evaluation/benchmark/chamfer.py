# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Vendored verbatim from ActionMesh actionbench/chamfer.py.

import numpy as np
import torch
from scipy.spatial import KDTree


def compute_chamfer_score(
    pred: torch.Tensor,
    gt: torch.Tensor,
    n: int = 10_000,
    seed: int = 44,
) -> float:
    """
    Compute symmetric Chamfer distance between two point clouds.

    Args:
        pred: (N, 3) predicted points.
        gt: (M, 3) ground truth points.
        n: Maximum number of points to sample from each cloud. If <= 0,
            uses all points.
        seed: Random seed for reproducible subsampling.

    Returns:
        Symmetric Chamfer distance (sum of both directions).
    """
    rng_pred = np.random.RandomState(seed=seed)
    rng_gt = np.random.RandomState(seed=seed + 1)

    if 0 < n < len(pred):
        indices_pred = rng_pred.permutation(len(pred))[:n]
    else:
        indices_pred = np.arange(len(pred))

    if 0 < n < len(gt):
        indices_gt = rng_gt.permutation(len(gt))[:n]
    else:
        indices_gt = np.arange(len(gt))

    tree_pred = KDTree(pred)
    d1, _ = tree_pred.query(gt[indices_gt])
    gt_to_pred = np.mean(d1)

    tree_gt = KDTree(gt)
    d2, _ = tree_gt.query(pred[indices_pred])
    pred_to_gt = np.mean(d2)

    return float(gt_to_pred + pred_to_gt)


