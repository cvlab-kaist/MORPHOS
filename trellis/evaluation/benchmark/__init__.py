# Vendored from ActionMesh / actionbench
# (https://github.com/facebookresearch/actionbench, files under
#  actionmesh/actionbench/{chamfer,icp,sample_mesh,sample_point_cloud}.py).
#
# Original copyright:
#   Copyright (c) Meta Platforms, Inc. and affiliates.
#   All rights reserved.
#   Licensed under the LICENSE in the upstream repository.
#
# Why vendored: lets `from trellis.evaluation.benchmark import
# compute_chamfer_score` work directly after a fresh checkout, without
# cloning the actionmesh repo or appending its directory to PYTHONPATH.
# Sibling-relative imports inside the original files have been rewritten
# to package-relative (`from .chamfer import ...`) so the modules form a
# self-contained package. No other functional changes.

from .chamfer import compute_chamfer_score
from .icp import gradient_icp
from .sample_mesh import sample_meshes
from .sample_point_cloud import sample_point_cloud

# F-score and P2S are NOT vendored from actionmesh — they are added here for
# use by `metric_geometry`. F-score follows Motion324's KD-tree convention;
# P2S follows the spec in trellis/evaluation/metric_geometry.py docstring.
from .fscore import compute_fscore_pcd, compute_fscore_sequence
from .p2s import compute_p2s

__all__ = [
    'compute_chamfer_score',
    'gradient_icp',
    'sample_meshes',
    'sample_point_cloud',
    'compute_fscore_pcd',
    'compute_fscore_sequence',
    'compute_p2s',
]
