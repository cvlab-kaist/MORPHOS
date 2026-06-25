"""Detect bad renders in renders_cond/ by inspecting per-frame alpha channels.

Two failure modes are flagged:
  1. CROPPED — silhouette overflows the image border (auto-framing was done
     against the rest-pose aabb, but animated frames extend past it).
  2. BLANK  — nothing was rendered (alpha is zero across every frame of
     every view).

For every frame we compute:
    border_ratio   = fraction of border pixels with alpha > BORDER_ALPHA_THR
    coverage_ratio = fraction of pixels with alpha > 0

Per-id we keep the worst (max) of each across all 6 views * 12 frames.

Usage:
    python dataset_toolkits/detect_failure.py --data_dir <DATASET_ROOT>

Reads <DATASET_ROOT>/renders_cond/ and writes both lists back under
<DATASET_ROOT>/:
    blank_ids.txt    — max coverage < BLANK_COVERAGE_THR
    cropped_ids.txt  — max border_ratio above the candidate threshold
                       (0.02–0.50) that flags the most ids
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from PIL import Image

BORDER_ALPHA_THR = 16          # alpha value considered "foreground" on border
BORDER_TOUCH_THRS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]  # candidates; keep the one flagging the most ids
BLANK_COVERAGE_THR = 0.001     # < 0.1% coverage anywhere => blank render
N_WORKERS = max(1, mp.cpu_count() - 4)


def frame_metrics(path):
    try:
        im = Image.open(path)
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        a = np.array(im.split()[-1], dtype=np.uint8)
    except Exception:
        return None
    top = a[0, :]
    bot = a[-1, :]
    left = a[1:-1, 0]
    right = a[1:-1, -1]
    border = np.concatenate([top, bot, left, right])
    border_ratio = float((border > BORDER_ALPHA_THR).sum()) / border.size
    coverage_ratio = float((a > 0).sum()) / a.size
    return border_ratio, coverage_ratio


def process_id(sub):
    sub = Path(sub)
    max_border = 0.0
    max_cov = 0.0
    worst_view = -1
    worst_frame = ""
    n_frames = 0
    for view_dir in sorted(sub.glob("view_*")):
        try:
            vi = int(view_dir.name.split("_")[1])
        except ValueError:
            continue
        for png in sorted(view_dir.glob("*.png")):
            m = frame_metrics(png)
            if m is None:
                continue
            br, cov = m
            n_frames += 1
            if br > max_border:
                max_border = br
                worst_view = vi
                worst_frame = png.name
            if cov > max_cov:
                max_cov = cov
    return sub.name, max_border, max_cov, worst_view, worst_frame, n_frames


def parse_args():
    p = argparse.ArgumentParser(description="Flag cropped / blank renders in a dataset's renders_cond/.")
    p.add_argument("--data_dir", type=Path, required=True,
                   help="Dataset root containing renders_cond/. Outputs are written here.")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    renders_dir = data_dir / "renders_cond"
    out_blank = data_dir / "blank_ids.txt"
    out_cropped = data_dir / "cropped_ids.txt"
    if not renders_dir.is_dir():
        raise SystemExit(f"renders_cond/ not found under {data_dir}")

    subdirs = sorted(p for p in renders_dir.iterdir() if p.is_dir())
    print(f"Scanning {len(subdirs)} ids with {N_WORKERS} workers...")
    t0 = time.time()
    results = []
    with mp.Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(process_id, [str(s) for s in subdirs], chunksize=4), 1):
            results.append(r)
            if i % 200 == 0 or i == len(subdirs):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(subdirs) - i) / rate if rate > 0 else 0
                print(f"  {i}/{len(subdirs)}  ({rate:.1f} ids/s  ETA {eta:.0f}s)")
    results.sort(key=lambda r: -r[1])

    blank = [r for r in results if r[2] < BLANK_COVERAGE_THR]
    with open(out_blank, "w") as f:
        for row in blank:
            f.write(row[0] + "\n")
    print(f"wrote {len(blank)} blank ids (coverage < {BLANK_COVERAGE_THR}) to {out_blank}")

    # Among the candidate thresholds, keep the one flagging the most ids and
    # write that single set (border_ratio > thr, so the smallest threshold wins).
    cropped_by_thr = {thr: [r for r in results if r[1] > thr] for thr in BORDER_TOUCH_THRS}
    best_thr = max(cropped_by_thr, key=lambda t: len(cropped_by_thr[t]))
    cropped = cropped_by_thr[best_thr]
    with open(out_cropped, "w") as f:
        for row in cropped:
            f.write(row[0] + "\n")
    print(f"wrote {len(cropped)} cropped ids (border_ratio > {best_thr:.2f}) to {out_cropped}")

    borders = np.array([r[1] for r in results])
    covs = np.array([r[2] for r in results])

    def stats(label, arr):
        print(f"\n[{label}] n={len(arr)}")
        print(f"  min/median/mean/max: {arr.min():.4f} / {np.median(arr):.4f} / {arr.mean():.4f} / {arr.max():.4f}")
        for p in (50, 75, 90, 95, 99):
            print(f"  p{p:02d}: {np.percentile(arr, p):.4f}")
        for t in (0.01, 0.02, 0.05, 0.1, 0.2):
            c = (arr > t).sum()
            print(f"  > {t:.2f}: {c:>5} ({100*c/len(arr):5.2f}%)")

    stats("max_border_ratio (per id)", borders)
    stats("max_coverage    (per id)", covs)

    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
