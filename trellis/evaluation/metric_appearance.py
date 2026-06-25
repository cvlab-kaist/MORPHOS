"""Appearance metrics over saved per-frame, per-azimuth gaussian PNGs.

Output mirrors `metric_geometry.py` with two extra long-format breakdowns:
    - `metrics_appearance.csv`            — per-scene scalar means.
    - `metrics_appearance_per_azimuth.csv` — per (scene, azimuth) means.
    - `metrics_appearance_per_frame.csv`   — per (scene, azimuth, frame).
    - `metrics_appearance.summary.json`    — dataset-level aggregate +
      FVD (FVD is dataset-level only, never per-scene).
    - `<pred_path>/<scene>/comparison.mp4` — per-scene grid video; top row =
      GT 4 views, bottom row = pred 4 views at the matching gen azimuths
      (one column per metric-compare pair). Disable with --no_grid_video.

All four are written next to the gen output (default:
`<pred_path>/../metrics_appearance.csv` and friends), atomically flushed
after every scene so progress survives a crash. Resume on start reads the
per-scene CSV and the per-frame CSV; per-azimuth is always recomputed
from the per-frame table at save time. `--recompute` forces redo.

Unavailable per-frame metrics are written as `-1.0` (e.g. a metric not
requested via `--metrics`); aggregation skips -1.

GT loading uses the generic path-templated dataset in
`trellis.evaluation.dataset` (configured via a JSON `gt_config`; see
`config/eval_gt_dataset.example.json`). It returns `gt_4view_video`
pre-permuted into gen-azimuth order — `gt_4view_video[k]` is the GT view
aligned with gen az AZIMUTHS_DEG[k]. The GT-view dir aligned with each gen az
is the config's `az_order` tuple; edit it to remap which GT view feeds which
gen az.

Predictions are loaded directly from
    `{pred_path}/{scene_id}/{az}/frame_NNNN.png`
with az in {0, 90, 180, 270}.

Metrics (one model each, lazily loaded only if requested):
    LPIPS    : lpips.LPIPS(net='vgg', spatial=True). Per-frame value =
              mean over the spatial map; per-azimuth + per-scene means
              are aggregated downstream.
    CLIP     : open_clip 'ViT-bigG-14' / 'laion2B_s39B_b160k'. Per-frame
              cosine similarity of L2-normalized image features.
    DreamSim : dreamsim(pretrained=True). Per-frame distance.
    FVD      : trellis.evaluation.fvd.styleganv.fvd (vendored from
              Motion324). Dataset-level Frechet distance over I3D
              features. Each (scene, azimuth) clip is reverse-flip-padded
              to TARGET_T frames (default 32) and split into TARGET_T-frame
              subvideos a la Motion324's `process_single_video`.

Frame-count handling: per-frame metrics use min(T_gen, T_gt) real frames
(no padding). Only FVD applies the Motion324 pad/split policy.

Usage:
    python -m trellis.evaluation.metric_appearance \\
        --gt_config /path/to/eval_gt_dataset.json \\
        --pred_path /path/to/output/appearance \\
        [--metrics lpips clip dreamsim fvd] [--output_csv ...] [--recompute]
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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .dataset import make_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Gen-side render azimuths (also drives `<pred>/<scene>/<az>/` lookup).
AZIMUTHS_DEG: Tuple[int, int, int, int] = (0, 90, 180, 270)
# Per-scene metric columns; FVD is dataset-level only.
METRIC_KEYS: Tuple[str, ...] = ("lpips", "clip", "dreamsim")


# =============================================================================
# I/O helpers
# =============================================================================

def _load_pred_view(view_dir: str) -> np.ndarray:
    """Read all `frame_NNNN.png` in lexical order. Returns [T, H, W, 3] uint8."""
    files = sorted(glob(os.path.join(view_dir, 'frame_*.png')))
    if not files:
        return np.zeros((0,), dtype=np.uint8)
    return np.stack([np.array(Image.open(f).convert('RGB')) for f in files], axis=0)


def _gt_view_to_uint8(gt_view: torch.Tensor) -> np.ndarray:
    """gt_view: [T, 3, H, W] float in [0, 1] -> [T, H, W, 3] uint8."""
    return (gt_view.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def _resize_video_u8(video_u8: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize [T, H, W, 3] uint8 to [T, H_new, W_new, 3] uint8 via PIL bilinear."""
    H, W = hw
    if video_u8.shape[1:3] == (H, W):
        return video_u8
    out = np.empty((len(video_u8), H, W, 3), dtype=np.uint8)
    for i, f in enumerate(video_u8):
        out[i] = np.array(Image.fromarray(f).resize((W, H), Image.BILINEAR))
    return out


def _compose_and_save_grid_video(
    out_path: str,
    pred_per_az_u8: List[Optional[np.ndarray]],   # 4 entries; missing az -> None
    gt_per_az_u8: List[np.ndarray],               # 4 entries; same indexing
    *,
    fps: int = 4,
) -> Optional[str]:
    """Write an MP4 at `out_path` with per-frame layout:

        +-----+-----+-----+-----+
        | GT0 | GT1 | GT2 | GT3 |   row 0 — ground truth at gen az k
        +-----+-----+-----+-----+
        | PR0 | PR1 | PR2 | PR3 |   row 1 — pred at the same gen az k
        +-----+-----+-----+-----+

    Each column k is one metric-compare pair (gt[k], pred[k]). Both rows
    are resized to the pred's per-az resolution (matches the metric
    convention in `PerFrameMetrics.compute`); missing pred azimuths are
    replaced by a black tile of the same size.

    Returns the saved path, or None if T == 0 (nothing to write).
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # type: ignore
        except ImportError:
            logger.warning(f'imageio not available; skipping grid video {out_path}')
            return None

    # Pick target H, W: prefer the first non-None pred (mirrors metric pairing);
    # fall back to GT[0] if no pred azimuths were rendered.
    target_hw = None
    for a in pred_per_az_u8:
        if a is not None and len(a) > 0:
            target_hw = a.shape[1:3]
            break
    if target_hw is None:
        target_hw = gt_per_az_u8[0].shape[1:3]
    H, W = int(target_hw[0]), int(target_hw[1])
    black_frame = np.zeros((H, W, 3), dtype=np.uint8)

    # T = min over non-None arrays. (FVD pad/split is irrelevant here — the
    # video is for human inspection, not metric scoring.)
    candidate_lens: List[int] = []
    for a in pred_per_az_u8:
        if a is not None:
            candidate_lens.append(len(a))
    for a in gt_per_az_u8:
        if a is not None:
            candidate_lens.append(len(a))
    if not candidate_lens:
        return None
    T = min(candidate_lens)
    if T == 0:
        return None

    # Pre-resize each az's pred + gt video to (H, W) once.
    pred_resized: List[np.ndarray] = []
    gt_resized: List[np.ndarray] = []
    for k in range(4):
        if pred_per_az_u8[k] is None or len(pred_per_az_u8[k]) == 0:
            pred_resized.append(None)
        else:
            pred_resized.append(_resize_video_u8(pred_per_az_u8[k][:T], (H, W)))
        gt_resized.append(_resize_video_u8(gt_per_az_u8[k][:T], (H, W)))

    frames: List[np.ndarray] = []
    for t in range(T):
        row_gt = np.concatenate([gt_resized[k][t] for k in range(4)], axis=1)
        row_pred = np.concatenate(
            [(pred_resized[k][t] if pred_resized[k] is not None else black_frame)
             for k in range(4)], axis=1,
        )
        frames.append(np.concatenate([row_gt, row_pred], axis=0))

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    # imageio.v3.imwrite supports mp4 via the ffmpeg backend; the older API
    # uses mimsave. Try the v3 path first, fall back to mimsave for older
    # imageio installs.
    if hasattr(iio, 'imwrite'):
        iio.imwrite(out_path, np.stack(frames, axis=0), fps=int(fps))
    else:
        iio.mimsave(out_path, frames, fps=int(fps))
    return out_path


# =============================================================================
# Per-frame metrics: LPIPS, CLIP, DreamSim
# =============================================================================

class PerFrameMetrics:
    """Lazy-loaded LPIPS / CLIP / DreamSim. Only metrics in `metrics` are loaded.
    `compute(...)` returns per-frame value lists (one float per frame).

    `lpips_batch_size` chunks LPIPS along the time axis on GPU to bound
    activation memory for long videos; the per-frame value is independent
    across `T` (`d.mean(dim=(1,2,3))`), so chunking is bit-equivalent to
    a single forward.
    """

    def __init__(self, metrics: List[str], device: str = 'cuda',
                 lpips_batch_size: int = 16):
        self.metrics = set(metrics)
        self.device = device
        self.lpips_batch_size = max(1, int(lpips_batch_size))
        self.lpips_fn = None
        self.clip_model = None
        self.clip_preproc = None
        self.dreamsim_model = None
        self.dreamsim_preproc = None

        if 'lpips' in self.metrics:
            import lpips as lpips_pkg
            self.lpips_fn = lpips_pkg.LPIPS(net='vgg', spatial=True).to(device).eval()
        if 'clip' in self.metrics:
            import open_clip
            self.clip_model, _, self.clip_preproc = open_clip.create_model_and_transforms(
                'ViT-bigG-14', pretrained='laion2B_s39B_b160k')
            self.clip_model = self.clip_model.to(device).eval()
        if 'dreamsim' in self.metrics:
            from dreamsim import dreamsim
            self.dreamsim_model, self.dreamsim_preproc = dreamsim(
                pretrained=True, device=device)

    @torch.no_grad()
    def _to_lpips(self, frames_u8: np.ndarray) -> torch.Tensor:
        # [T, H, W, 3] uint8 -> [T, 3, H, W] in [-1, 1]
        x = torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2).to(self.device)
        return x / 127.5 - 1.0

    @torch.no_grad()
    def _clip_features(self, frames_u8: np.ndarray) -> torch.Tensor:
        feats = []
        for f in frames_u8:
            x = self.clip_preproc(Image.fromarray(f)).unsqueeze(0).to(self.device)
            feats.append(self.clip_model.encode_image(x))
        return F.normalize(torch.cat(feats, dim=0), dim=-1)

    @torch.no_grad()
    def compute(
        self, gen_u8: np.ndarray, gt_u8: np.ndarray,
    ) -> Dict[str, List[float]]:
        """gen_u8, gt_u8: [T, H, W, 3] uint8. Returns per-frame values per
        metric (each value is one scalar for one frame), with length =
        min(T_gen, T_gt) after spatial alignment."""
        T = min(len(gen_u8), len(gt_u8))
        gen_u8, gt_u8 = gen_u8[:T], gt_u8[:T]
        if gen_u8.shape[1:3] != gt_u8.shape[1:3]:
            gt_u8 = _resize_video_u8(gt_u8, gen_u8.shape[1:3])

        out: Dict[str, List[float]] = {}
        if self.lpips_fn is not None:
            # Chunk along T to bound GPU memory on long videos. Per-frame
            # value is `d.mean(dim=(1,2,3))` — independent across T, so
            # chunking is bit-equivalent to a single full forward.
            lpips_vals: List[float] = []
            B = self.lpips_batch_size
            for i in range(0, T, B):
                gen_chunk = self._to_lpips(gen_u8[i:i + B])
                gt_chunk = self._to_lpips(gt_u8[i:i + B])
                d = self.lpips_fn(gen_chunk, gt_chunk)
                # spatial=True returns [b, 1, H, W]; mean over (1, 2, 3) -> [b].
                lpips_vals.extend(d.mean(dim=(1, 2, 3)).detach().cpu().tolist())
                del gen_chunk, gt_chunk, d
            out['lpips'] = lpips_vals
        if self.clip_model is not None:
            gf = self._clip_features(gen_u8)
            tf = self._clip_features(gt_u8)
            out['clip'] = (gf * tf).sum(dim=-1).detach().cpu().tolist()
        if self.dreamsim_model is not None:
            ds_vals: List[float] = []
            for g, t in zip(gen_u8, gt_u8):
                g_t = self.dreamsim_preproc(Image.fromarray(g)).to(self.device)
                t_t = self.dreamsim_preproc(Image.fromarray(t)).to(self.device)
                ds_vals.append(float(self.dreamsim_model(g_t, t_t).item()))
            out['dreamsim'] = ds_vals
        return out


# =============================================================================
# Motion324-style temporal pad / split (FVD only)
# =============================================================================

def _pad_split_motion324(video_t: torch.Tensor, target_T: int = 32) -> List[torch.Tensor]:
    """Mirror `Motion324/evaluation/evaluation.py:process_single_video`.

    video_t: [T, C, H, W] tensor in [0, 1].
      1. If T < target_T: pad by appending the last (target_T - T) frames
         reversed in time (Motion324's behaviour). For very short clips
         (T < target_T - T, i.e. one flip can't fill the gap), repeat the
         tail-flip until target_T is reached.
      2. Split into target_T-frame chunks (stride = target_T, no overlap).

    Returns a list of [target_T, C, H, W] tensors."""
    T = video_t.shape[0]
    if T < target_T:
        take_n = min(target_T - T, T)
        video_t = torch.cat([video_t, video_t[-take_n:].flip(0)], dim=0)
        while video_t.shape[0] < target_T:
            need = target_T - video_t.shape[0]
            tail = video_t[-min(need, video_t.shape[0]):].flip(0)
            video_t = torch.cat([video_t, tail], dim=0)
        video_t = video_t[:target_T]

    T = video_t.shape[0]
    return [video_t[i:i + target_T] for i in range(0, T - target_T + 1, target_T)]


# =============================================================================
# Dataset-level FVD (StyleGAN-V I3D, vendored)
# =============================================================================

class FVDAggregator:
    """Collects (pred, gt) clip pairs, then computes a single dataset-level FVD
    using the StyleGAN-V I3D backbone (vendored under
    `trellis.evaluation.fvd.styleganv.fvd`).

    Each (scene, azimuth) clip is padded to TARGET_T frames using Motion324's
    reverse-flip pad and split into TARGET_T-frame subvideos.
    """

    TARGET_T = 32  # Motion324's TIMESTAMP_LIMIT

    def __init__(self, device: str = 'cuda', batch_size: int = 8):
        """`batch_size` chunks the I3D forward over (scene × az) clips so
        we never materialize the full [N_clips, T, 3, H, W] tensor on GPU
        at once. Frechet distance is computed on the aggregated feature
        mean + cov, so batching is bit-equivalent."""
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._pred_clips: List[torch.Tensor] = []
        self._gt_clips: List[torch.Tensor] = []
        try:
            from .fvd.styleganv.fvd import (
                load_i3d_pretrained, get_fvd_feats, frechet_distance,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import vendored fvd module at "
                "trellis.evaluation.fvd.styleganv.fvd. Ensure the package "
                "directory and __init__.py files exist."
            ) from e
        self._get_fvd_feats = get_fvd_feats
        self._frechet_distance = frechet_distance
        self.i3d = load_i3d_pretrained(device=torch.device(device))

    @staticmethod
    def _u8_video_to_torch(vid_u8: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(vid_u8).permute(0, 3, 1, 2).float() / 255.0

    def add(self, pred_u8: np.ndarray, gt_u8: np.ndarray):
        T = min(len(pred_u8), len(gt_u8))
        if T == 0:
            return
        pred_t = self._u8_video_to_torch(pred_u8[:T])
        gt_t = self._u8_video_to_torch(gt_u8[:T])
        pred_subs = _pad_split_motion324(pred_t, target_T=self.TARGET_T)
        gt_subs = _pad_split_motion324(gt_t, target_T=self.TARGET_T)
        for sp, sg in zip(pred_subs, gt_subs):
            self._pred_clips.append(sp)
            self._gt_clips.append(sg)

    def compute(self) -> float:
        if not self._pred_clips:
            return float('nan')
        n = len(self._pred_clips)
        pred_chunks: list = []
        gt_chunks: list = []
        for i in range(0, n, self.batch_size):
            # Stack only this chunk's clips → (b, T, 3, H, W) → BCTHW.
            pred_btchw = torch.stack(
                self._pred_clips[i:i + self.batch_size], dim=0,
            ).permute(0, 2, 1, 3, 4)
            gt_btchw = torch.stack(
                self._gt_clips[i:i + self.batch_size], dim=0,
            ).permute(0, 2, 1, 3, 4)
            pred_chunks.append(self._get_fvd_feats(pred_btchw, self.i3d, self.device))
            gt_chunks.append(self._get_fvd_feats(gt_btchw, self.i3d, self.device))
            del pred_btchw, gt_btchw

        def _concat(chunks):
            # `get_fvd_feats` may return either np.ndarray or torch.Tensor
            # depending on the vendored fvd backend; concatenate accordingly.
            if isinstance(chunks[0], torch.Tensor):
                return torch.cat(chunks, dim=0)
            return np.concatenate(chunks, axis=0)

        feats_pred = _concat(pred_chunks)
        feats_gt = _concat(gt_chunks)
        return float(self._frechet_distance(feats_pred, feats_gt))


# =============================================================================
# Result containers (mirror metric_geometry's pattern)
# =============================================================================

@dataclass
class SampleResult:
    """Per-scene appearance row.

    Each metric is the mean over (per-azimuth means of per-frame values).
    Unavailable metrics use `-1.0` as a sentinel (in memory and in the
    saved CSV) — e.g. a metric not requested via `--metrics`. Aggregation
    in `summary()` skips -1 values.
    """
    uid: str
    n_frames: int = 0
    lpips: float = -1.0
    clip: float = -1.0
    dreamsim: float = -1.0
    status: str = "pending"
    error_message: str = ""


@dataclass
class DatasetResults:
    """Aggregates per-scene rows + per (scene, azimuth, frame) breakdowns.

    The per-frame table is long-format with columns
    `[uid, az, frame_idx, lpips, clip, dreamsim]`. Per-azimuth means are
    derived from this table at save time (no separate in-memory store).
    """
    samples: List[SampleResult] = field(default_factory=list)
    per_frame_rows: List[Dict[str, Any]] = field(default_factory=list)
    fvd: float = -1.0  # dataset-level; updated once at the end if requested.

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
        cols = ["uid", "az", "frame_idx", *METRIC_KEYS]
        if not self.per_frame_rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self.per_frame_rows, columns=cols)

    def per_azimuth_dataframe(self) -> pd.DataFrame:
        """Group per-frame rows by (uid, az); for each metric, mean ignores
        -1 sentinels. Adds an `n_frames` column = count of rows in the
        group."""
        cols = ["uid", "az", "n_frames", *METRIC_KEYS]
        df = self.per_frame_dataframe()
        if df.empty:
            return pd.DataFrame(columns=cols)
        rows: List[Dict[str, Any]] = []
        for (uid, az), sub in df.groupby(["uid", "az"], sort=False):
            row: Dict[str, Any] = {"uid": uid, "az": az, "n_frames": int(len(sub))}
            for k in METRIC_KEYS:
                vals = sub[k].astype(float).values
                valid = vals[np.isfinite(vals) & (vals != -1.0)]
                row[k] = float(np.mean(valid)) if len(valid) else -1.0
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    def summary(self) -> Dict[str, float]:
        df = self.to_dataframe()
        ok = df[df["status"] == "success"] if not df.empty else df
        out: Dict[str, float] = {
            "n_total": int(len(df)),
            "n_success": int(len(ok)),
            "n_failed": int(len(df) - len(ok)),
            "success_rate": float(len(ok) / len(df)) if len(df) else 0.0,
        }
        for k in METRIC_KEYS:
            if len(ok) and k in ok.columns:
                vals = ok[k].astype(float).values
                valid = vals[np.isfinite(vals) & (vals != -1.0)]
                out[f"{k}_mean"] = float(np.mean(valid)) if len(valid) else float("nan")
            else:
                out[f"{k}_mean"] = float("nan")
        out["fvd"] = float(self.fvd) if self.fvd != -1.0 else float("nan")
        return out


def _per_frame_csv_path(csv_path: str) -> str:
    return os.path.splitext(csv_path)[0] + "_per_frame.csv"


def _per_azimuth_csv_path(csv_path: str) -> str:
    return os.path.splitext(csv_path)[0] + "_per_azimuth.csv"


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


def load_existing(
    csv_path: str,
) -> Tuple[Dict[str, SampleResult], List[Dict[str, Any]]]:
    """Return (samples_by_uid, per_frame_rows). Per-azimuth is recomputed
    from per-frame on save, so it doesn't need a load step."""
    samples: Dict[str, SampleResult] = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            samples[row["uid"]] = SampleResult(
                uid=row["uid"],
                n_frames=int(row["n_frames"]) if "n_frames" in row else 0,
                lpips=float(row["lpips"]) if "lpips" in row else -1.0,
                clip=float(row["clip"]) if "clip" in row else -1.0,
                dreamsim=float(row["dreamsim"]) if "dreamsim" in row else -1.0,
                status=str(row["status"]) if "status" in row else "pending",
                error_message=(
                    str(row["error_message"])
                    if "error_message" in row and not pd.isna(row.get("error_message", ""))
                    else ""
                ),
            )
    per_frame_rows: List[Dict[str, Any]] = []
    pf_path = _per_frame_csv_path(csv_path)
    if os.path.exists(pf_path):
        df_pf = pd.read_csv(pf_path)
        per_frame_rows = df_pf.to_dict("records")
    return samples, per_frame_rows


def save_results(results: DatasetResults, csv_path: str) -> None:
    """Atomically write per-scene CSV + per-azimuth CSV + per-frame CSV +
    summary JSON. Each file is written via a `<path>.tmp` then `os.replace`
    so the on-disk state is consistent at every flush."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    _atomic_write_csv(results.to_dataframe(), csv_path)
    _atomic_write_csv(results.per_azimuth_dataframe(), _per_azimuth_csv_path(csv_path))
    _atomic_write_csv(results.per_frame_dataframe(), _per_frame_csv_path(csv_path))
    _atomic_write_json(results.summary(), _summary_json_path(csv_path))


# =============================================================================
# Per-scene driver
# =============================================================================

def _mean_or_sentinel(xs: List[float]) -> float:
    """Mean if non-empty, else -1.0 sentinel (consistent with the CSV)."""
    return float(np.mean(xs)) if xs else -1.0


def evaluate_scene(
    scene_id: str,
    pred_path: str,
    gt_4view: torch.Tensor,
    metrics: PerFrameMetrics,
    fvd: Optional[FVDAggregator],
    *,
    save_grid_video: bool = True,
    grid_video_fps: int = 4,
) -> Tuple[SampleResult, List[Dict[str, Any]]]:
    """Per-scene LPIPS / CLIP / DreamSim, averaged over azimuths and frames.

    `gt_4view` is expected to be pre-permuted into gen-azimuth order
    (gt_4view[k] is the GT view aligned with gen AZIMUTHS_DEG[k]) — the
    generic dataset does this via the config's `az_order` tuple.

    Returns
    -------
    SampleResult
        Per-scene scalar row (means over azimuths and frames).
    list[dict]
        Per-frame rows for the per-frame CSV. One dict per (uid, az,
        frame_idx) with columns {uid, az, frame_idx, lpips, clip, dreamsim}.
        Metrics not requested via `--metrics` are recorded as -1.0.

    Side effect: the (pred, gt) clip pair for each azimuth is added to
    `fvd` if provided; FVD itself is computed dataset-level by the caller.
    """
    result = SampleResult(uid=scene_id)
    per_frame_rows: List[Dict[str, Any]] = []
    # Retain pred frames per az so the grid video can compose them after
    # the loop. None entries = az was missing or empty.
    pred_per_az_u8: List[Optional[np.ndarray]] = [None] * 4
    try:
        gt_views_u8 = [_gt_view_to_uint8(gt_4view[k]) for k in range(4)]
        # Per-azimuth scalar means (used to derive scene mean).
        per_az_means: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
        n_frames_used = 0
        any_az_evaluated = False
        for k, az in enumerate(AZIMUTHS_DEG):
            view_dir = os.path.join(pred_path, scene_id, str(az))
            if not os.path.isdir(view_dir):
                logger.info(f"[{scene_id}] az={az}: missing pred dir, skipping")
                continue
            pred_u8 = _load_pred_view(view_dir)
            gt_u8 = gt_views_u8[k]
            if len(pred_u8) == 0:
                logger.info(f"[{scene_id}] az={az}: 0 pred frames, skipping")
                continue
            pred_per_az_u8[k] = pred_u8
            m = metrics.compute(pred_u8, gt_u8)  # {metric: [v_per_frame]}
            T = min(len(pred_u8), len(gt_u8))
            for fi in range(T):
                row: Dict[str, Any] = {
                    "uid": scene_id, "az": int(az), "frame_idx": fi,
                }
                for name in METRIC_KEYS:
                    vals = m.get(name)
                    row[name] = float(vals[fi]) if vals is not None else -1.0
                per_frame_rows.append(row)
            for name, values in m.items():
                if values:
                    per_az_means[name].append(float(np.mean(values)))
            any_az_evaluated = True
            n_frames_used = max(n_frames_used, T)
            if fvd is not None:
                fvd.add(pred_u8, gt_u8)

        if not any_az_evaluated:
            result.status = "error"
            result.error_message = "no azimuth had usable pred + gt"
            return result, []

        result.n_frames = n_frames_used
        for name in METRIC_KEYS:
            setattr(result, name, _mean_or_sentinel(per_az_means[name]))
        result.status = "success"

        # Grid video: top row GT, bottom row pred, 4 cols = the 4 az pairs
        # the metrics compared. Saved alongside the per-az PNGs.
        if save_grid_video:
            try:
                out_path = os.path.join(pred_path, scene_id, 'comparison.mp4')
                _compose_and_save_grid_video(
                    out_path, pred_per_az_u8, gt_views_u8, fps=grid_video_fps,
                )
            except Exception as e:
                logger.warning(f"[{scene_id}] grid video failed: {e}")
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)
        per_frame_rows = []  # don't keep partial frame rows on error
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
    metrics_set: List[str],
    device: str = "cuda",
    recompute: bool = False,
    save_grid_video: bool = True,
    grid_video_fps: int = 4,
    lpips_batch_size: int = 16,
    fvd_batch_size: int = 8,
) -> DatasetResults:
    dataset = make_dataset(gt_config)
    logger.info(f"{len(dataset)} scenes resolved from gt_config")
    if len(dataset) == 0:
        return DatasetResults()

    metrics_set = list(metrics_set)
    logger.info(f"loading metric models ({sorted(metrics_set)}) on {device}")
    per_frame = PerFrameMetrics(
        [m for m in METRIC_KEYS if m in metrics_set],
        device=device,
        lpips_batch_size=lpips_batch_size,
    )
    fvd: Optional[FVDAggregator] = None
    if 'fvd' in metrics_set:
        fvd = FVDAggregator(device=device, batch_size=fvd_batch_size)

    existing_samples: Dict[str, SampleResult] = {}
    existing_per_frame: List[Dict[str, Any]] = []
    if output_csv and not recompute:
        existing_samples, existing_per_frame = load_existing(output_csv)
    if existing_samples:
        n_done = sum(1 for r in existing_samples.values() if r.status == "success")
        logger.info(
            f"loaded {len(existing_samples)} prior results ({n_done} successful). "
            "Use --recompute to redo."
        )
    successful_uids = {
        uid for uid, r in existing_samples.items() if r.status == "success"
    }
    existing_per_frame = [r for r in existing_per_frame if r["uid"] in successful_uids]

    results = DatasetResults(per_frame_rows=list(existing_per_frame))
    for i in tqdm(range(len(dataset)), desc="appearance"):
        item = dataset[i]
        scene_id = item['scene_id']

        if (
            scene_id in existing_samples
            and not recompute
            and existing_samples[scene_id].status == "success"
        ):
            results.samples.append(existing_samples[scene_id])
            continue

        scene_pred_dir = os.path.join(pred_path, scene_id)
        if not os.path.isdir(scene_pred_dir):
            r = SampleResult(
                uid=scene_id, status="error",
                error_message=f"pred dir missing: {scene_pred_dir}",
            )
            results.add(r)
            if output_csv:
                save_results(results, output_csv)
            logger.info(f"[{scene_id}] pred dir missing, skipping")
            continue

        gt_4view = item.get('gt_4view_video')  # [4, T, 3, H, W] in [0, 1] | None
        if gt_4view is None:
            r = SampleResult(
                uid=scene_id, status="error",
                error_message="no gt_4view_video (set 'gt_4view' in gt_config)",
            )
            results.add(r)
            if output_csv:
                save_results(results, output_csv)
            logger.info(f"[{scene_id}] no GT 4-view configured, skipping")
            continue
        r, pf_rows = evaluate_scene(
            scene_id, pred_path, gt_4view, per_frame, fvd,
            save_grid_video=save_grid_video,
            grid_video_fps=grid_video_fps,
        )
        results.add(r, per_frame=pf_rows)
        if r.status == "success":
            parts: List[str] = [f"[{scene_id}]"]
            for name in METRIC_KEYS:
                v = getattr(r, name)
                if v != -1.0:
                    parts.append(f"{name.upper()}={v:.4f}")
            logger.info(" ".join(parts))
        if output_csv:
            save_results(results, output_csv)

    # Dataset-level FVD (computed once over all collected clip pairs).
    if fvd is not None:
        results.fvd = fvd.compute()
        logger.info(f"FVD={results.fvd:.4f}")

    if output_csv:
        save_results(results, output_csv)
        logger.info(f"saved CSV to {output_csv}")
    return results


def print_summary(results: DatasetResults):
    s = results.summary()
    print("\n" + "=" * 60)
    print("APPEARANCE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  total   : {s['n_total']}")
    print(f"  success : {s['n_success']}")
    print(f"  failed  : {s['n_failed']}")
    print(f"  rate    : {s['success_rate']:.1%}")
    if s["n_success"] > 0:
        print("\nMean metrics (-1 / NaN skipped):")
        for k in METRIC_KEYS:
            print(f"  {k.upper():<8}: {s[f'{k}_mean']:.4f}")
        print(f"  FVD     : {s['fvd']:.4f}")
    df = results.to_dataframe()
    failed = df[df["status"] != "success"] if not df.empty else df
    if len(failed):
        print(f"\nFailed ({len(failed)}):")
        for _, row in failed.iterrows():
            print(f"  [{row['uid']}] {row['status']}: {row['error_message']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="LPIPS / CLIP / DreamSim / FVD over per-frame, per-azimuth "
                    "gen PNGs vs dataset-wrapper GT 4-view (gen-azimuth ordered).",
    )
    parser.add_argument('--gt_config', required=True,
                        help='JSON describing GT paths (see '
                             'config/eval_gt_dataset.example.json). Needs '
                             "'gt_4view' for appearance metrics.")
    parser.add_argument('--pred_path', required=True,
                        help='Pred appearance root: <pred_path>/<scene>/<az>/frame_NNNN.png')
    parser.add_argument('--metrics', nargs='+',
                        default=['lpips', 'clip', 'dreamsim', 'fvd'],
                        choices=['lpips', 'clip', 'dreamsim', 'fvd'])
    parser.add_argument("--output_csv", default=None,
                        help="Default: <pred_path>/../metrics_appearance.csv")
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute even if --output_csv already has the scene.")
    parser.add_argument("--no_grid_video", action="store_true",
                        help="Skip the per-scene comparison.mp4 (top=GT, "
                             "bottom=pred, 4 cols = the metric pairs). "
                             "Default: write the video into "
                             "<pred_path>/<scene>/comparison.mp4.")
    parser.add_argument("--grid_video_fps", type=int, default=10,
                        help="FPS for the per-scene comparison.mp4.")
    parser.add_argument("--lpips_batch_size", type=int, default=16,
                        help="Chunk LPIPS along T to bound GPU activation "
                             "memory on long videos. Per-frame value is "
                             "independent across T, so batching is bit-"
                             "equivalent to a single full forward.")
    parser.add_argument("--fvd_batch_size", type=int, default=8,
                        help="Chunk the I3D forward over (scene × az) clips "
                             "in FVD. Frechet distance is computed on the "
                             "aggregated feature mean + cov, so batching is "
                             "bit-equivalent.")
    args = parser.parse_args()

    output_csv = args.output_csv or os.path.join(
        os.path.dirname(args.pred_path.rstrip("/")) or ".",
        "metrics_appearance.csv",
    )

    results = evaluate_dataset(
        gt_config=args.gt_config,
        pred_path=args.pred_path,
        output_csv=output_csv,
        metrics_set=args.metrics,
        device=args.device,
        recompute=args.recompute,
        save_grid_video=not args.no_grid_video,
        grid_video_fps=args.grid_video_fps,
        lpips_batch_size=args.lpips_batch_size,
        fvd_batch_size=args.fvd_batch_size,
    )
    print_summary(results)


if __name__ == '__main__':
    main()
