"""Shared GT image I/O for evaluation datasets.

GT view frames: load PNG -> if RGBA, composite onto white -> RGB uint8.
"""
import numpy as np
from PIL import Image


def load_gt_view_frame_white_bg(path: str) -> np.ndarray:
    """Load one PNG, return RGB [H, W, 3] uint8 composited onto white if RGBA.
    RGB and grayscale inputs are normalized to RGB. Result is always a
    writable numpy array (PIL hands back read-only buffers; we copy)."""
    arr = np.array(Image.open(path))  # np.array (not asarray) -> writable copy
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        rgb = arr[..., :3].astype(np.float32) / 255.0
        a = arr[..., 3:4].astype(np.float32) / 255.0
        out = (rgb * a + (1.0 - a)) * 255.0
        return out.clip(0, 255).astype(np.uint8)
    return arr  # already RGB
