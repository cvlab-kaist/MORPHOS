"""Compute per-channel SLat normalization stats over a processed dataset tree.

Walks <DATA_DIR>/slat_latents/<obj_id>/frame_*.npz, takes the per-channel
mean / std of the `feats` arrays, and writes them to
<DATA_DIR>/slat_norm_stats.json. Paste the result into the
`dataset.args.normalization` block of
config/temporal_slat_flow_dit_w3_diffforcing.json.

Usage:
    python dataset_toolkits/compute_norm_stats.py --data_dir <DATA_DIR>
"""
import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Dataset root containing slat_latents/<obj_id>/frame_*.npz')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='Max number of per-frame .npz files to sample for stats.')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for the random subsample when there are more files than --num_samples.')
    opt = parser.parse_args()

    slat_dir = os.path.join(opt.data_dir, 'slat_latents')
    if not os.path.isdir(slat_dir):
        raise SystemExit(f'slat_latents/ not found under {opt.data_dir}')

    files = sorted(glob.glob(os.path.join(slat_dir, '*', 'frame_*.npz')))
    if len(files) == 0:
        raise SystemExit(f'no frame_*.npz under {slat_dir}')
    if len(files) > opt.num_samples:
        rng = np.random.RandomState(opt.seed)
        files = sorted(rng.choice(files, opt.num_samples, replace=False).tolist())
    print(f'Computing stats over {len(files)} frame latents')

    # Each frame contributes one per-channel mean and second moment (over its
    # voxels); frames are weighted equally.
    means = []
    mean2s = []
    with ThreadPoolExecutor(max_workers=16) as executor, \
        tqdm(total=len(files), desc="Reading feats") as pbar:
        def worker(path):
            try:
                feats = np.load(path)['feats']
                means.append(feats.mean(axis=0))
                mean2s.append((feats ** 2).mean(axis=0))
            except Exception as e:
                print(f"Error reading {path}: {e}")
            finally:
                pbar.update()

        list(executor.map(worker, files))

    mean = np.array(means).mean(axis=0)
    mean2 = np.array(mean2s).mean(axis=0)
    std = np.sqrt(mean2 - mean ** 2)

    print('mean:', mean.tolist())
    print('std:', std.tolist())

    out_path = os.path.join(opt.data_dir, 'slat_norm_stats.json')
    with open(out_path, 'w') as f:
        json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f, indent=4)
    print(f'wrote {out_path}')
