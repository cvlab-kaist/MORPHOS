"""Evaluation package for the combined SS+SLat inference pipeline.

Modules:
- `dataset/`: external-evaluation dataset wrappers (motion80 / actionbench /
  consist4d). Use `from trellis.evaluation.dataset import make_dataset`.
- `metric_appearance`: LPIPS / CLIP / DreamSim / FVD over saved per-frame,
  per-azimuth gaussian PNGs. Run as
  `python -m trellis.evaluation.metric_appearance --help`.
- `metric_geometry`: Chamfer distance / F-score / P2S over saved per-frame
  .glb predictions. Run as
  `python -m trellis.evaluation.metric_geometry --help`.
"""
