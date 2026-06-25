"""Evaluation dataset layer for the combined SS+SLat metrics.

A single generic, path-templated dataset (:class:`PlaceholderInferenceDataset`)
serves every benchmark. Point it at your own GT with format-string path
templates — no benchmark layout is hard-coded — so the same eval code works for
ActionBench / Motion80 / Consist4D or any custom data.

Each scene exposes:

    {
        'scene_id'           : str,
        'gt_4view_video'     : Tensor[4, T, 3, H, W] | None,   # white-bg, gen-az ordered (appearance)
        'num_frames'         : int,
        'search_azimuths_deg': List[float],                    # inference-only; eval ignores
    }
    load_gt_pcd_sequence(idx)  -> Tensor[T, N, 3]              # geometry CD / F-score
    load_gt_mesh_sequence(idx) -> List[trimesh.Trimesh]       # P2S (raises if no faces)

Evaluation does NOT use conditioning frames, so no ``cond_video`` key is
provided. Use ``make_dataset(config)`` where ``config`` is a JSON path or dict;
see ``config/eval_gt_dataset.example.json``. To write a custom dataset, just
duck-type the interface documented in :mod:`.base`.
"""
from .base import PlaceholderInferenceDataset, make_dataset

__all__ = [
    "PlaceholderInferenceDataset",
    "make_dataset",
]
