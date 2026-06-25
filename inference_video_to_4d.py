"""Standalone video-only SS+SLat Diffusion-Forcing inference.

Runs the pipeline on **plain videos / frame folders**. Rendering always uses
the pipeline's default camera pose (yaw/pitch/distance) — see
``VideoTo4DPipeline.decode_and_save``. It depends only on the ``trellis``
library and the shared JSON config for ckpt paths / sampling knobs.

Input resolution (``--input``):
    * a single ``*.mp4`` file              -> one scene (the file)
    * a directory containing sub-dirs      -> one scene PER sub-dir, each read
                                              as its ``*.png`` frames
    * a directory containing ``*.mp4``     -> one scene per video
    * a directory containing ``*.png``     -> a single scene (the directory)
Sub-dirs take priority over loose ``*.mp4``, which take priority over loose
``*.png`` when a directory contains a mix.

Conditioning matches the training manifold:
    * ``*.png`` frames: resize (LANCZOS) -> alpha-premultiply if RGBA (else RGB)
      -> [0,1] CHW. (Same as the benchmark frame loader.)
    * ``*.mp4`` frames: resize (LANCZOS) -> RGB -> [0,1] CHW. (mp4s made from
      RGBA frames are expected to already be alpha-premultiplied, i.e.
      transparent -> black, so the RGB load round-trips to the same manifold.)

Output layout (ar_kv mode only, so no per-mode sub-dir), T = num frames:
    {output_dir}/appearance/{scene_id}/{0|90|180|270}/frame_NNNN.png
    {output_dir}/appearance/{scene_id}/grid_video.mp4   # 2x2 azimuth grid
    {output_dir}/geometry/{scene_id}/frame_NNNN.glb
    {output_dir}/slat/{scene_id}/frame_NNNN.npz
"""
import os
os.environ.setdefault('SPCONV_ALGO', 'native')

import argparse
import glob
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from trellis.pipelines import VideoTo4DPipeline


def _list_pngs(d: str) -> List[str]:
    return sorted(glob.glob(os.path.join(d, '*.png')))


def _resolve_scenes(path: str) -> List[Dict]:
    """Resolve ``--input`` into a list of scenes.

    Each scene is ``{'scene_id': str, 'kind': 'mp4'|'pngs', 'source': ...}``
    where ``source`` is an mp4 path ('mp4') or a list of png paths ('pngs').
    """
    if os.path.isfile(path):
        if not path.lower().endswith('.mp4'):
            raise ValueError(f"Input file is not a .mp4: {path}")
        return [{'scene_id': os.path.splitext(os.path.basename(path))[0],
                 'kind': 'mp4', 'source': path}]

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input path does not exist: {path}")

    # Priority 1: sub-directories -> one scene per sub-dir (png frames).
    subdirs = sorted(
        d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)
    )
    if subdirs:
        scenes = []
        for d in subdirs:
            pngs = _list_pngs(d)
            if not pngs:
                print(f"  [skip] sub-dir has no *.png frames: {d}")
                continue
            scenes.append({'scene_id': os.path.basename(d.rstrip('/')),
                           'kind': 'pngs', 'source': pngs})
        if not scenes:
            raise FileNotFoundError(
                f"Directory has sub-dirs but none contain *.png frames: {path}")
        return scenes

    # Priority 2: loose *.mp4 files -> one scene per video.
    mp4s = sorted(glob.glob(os.path.join(path, '*.mp4')))
    if mp4s:
        return [{'scene_id': os.path.splitext(os.path.basename(m))[0],
                 'kind': 'mp4', 'source': m} for m in mp4s]

    # Priority 3: loose *.png files -> a single scene (this directory).
    pngs = _list_pngs(path)
    if pngs:
        return [{'scene_id': os.path.basename(path.rstrip('/')),
                 'kind': 'pngs', 'source': pngs}]

    raise FileNotFoundError(
        f"No sub-dirs, *.mp4, or *.png found under directory: {path}")


def _png_to_cond_frame(path: str, image_size: int) -> torch.Tensor:
    """Load one PNG -> [3,S,S] in [0,1], alpha-premultiplied if RGBA (resize
    first, then premultiply — mirrors the training frame loader)."""
    image = Image.open(path).resize(
        (image_size, image_size), Image.Resampling.LANCZOS)
    if image.mode == 'RGBA':
        alpha = image.getchannel(3)
        rgb = image.convert('RGB')
        rgb_t = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        a_t = torch.from_numpy(np.array(alpha)).float() / 255.0
        return rgb_t * a_t.unsqueeze(0)
    rgb = image.convert('RGB')
    return torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0


def _load_scene_cond(scene: Dict, image_size: int) -> torch.Tensor:
    """Decode a scene -> cond_video tensor [T, 3, image_size, image_size]."""
    if scene['kind'] == 'pngs':
        frames = [_png_to_cond_frame(p, image_size) for p in scene['source']]
    else:  # 'mp4'
        import imageio.v3 as iio
        frames = []
        for frame in iio.imiter(scene['source'], plugin='FFMPEG'):  # HxWx3 uint8
            img = Image.fromarray(frame).convert('RGB').resize(
                (image_size, image_size), Image.Resampling.LANCZOS)
            frames.append(
                torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)
    if not frames:
        raise RuntimeError(f"No frames decoded for scene {scene['scene_id']}")
    return torch.stack(frames, dim=0)


def _save_appearance_grid_video(appearance_scene_dir: str, fps: int = 10) -> str:
    """Build a 2x2 azimuth grid video from the saved appearance PNGs.

    Reads ``{appearance_scene_dir}/{0,90,180,270}/frame_NNNN.png`` and writes
    ``{appearance_scene_dir}/grid_video.mp4``. Per output frame the four views
    are tiled as:

        +-----+-----+
        |  0  |  90 |
        +-----+-----+
        | 180 | 270 |
        +-----+-----+

    Returns the output path, or '' if no frames were found.
    """
    import imageio.v2 as imageio

    az_quadrants = [0, 90, 180, 270]  # TL, TR, BL, BR
    az_frames = {
        az: sorted(glob.glob(os.path.join(appearance_scene_dir, str(az), 'frame_*.png')))
        for az in az_quadrants
    }
    counts = [len(v) for v in az_frames.values()]
    if not counts or min(counts) == 0:
        print(f'  [grid] no appearance frames under {appearance_scene_dir}; skipping')
        return ''
    T = min(counts)
    if len(set(counts)) != 1:
        print(f'  [grid] azimuth frame counts differ {counts}; using first {T}')

    grid_frames = []
    for i in range(T):
        imgs = [np.array(Image.open(az_frames[az][i]).convert('RGB')) for az in az_quadrants]
        top = np.concatenate([imgs[0], imgs[1]], axis=1)   # 0  | 90
        bottom = np.concatenate([imgs[2], imgs[3]], axis=1)  # 180 | 270
        grid_frames.append(np.concatenate([top, bottom], axis=0))

    out_path = os.path.join(appearance_scene_dir, 'grid_video.mp4')
    imageio.mimwrite(out_path, grid_frames, fps=fps, quality=9, macro_block_size=1)
    print(f'  [grid] wrote {out_path}  ({T} frames @ {fps} fps)')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input', required=True,
        help='A *.mp4 file, or a directory of sub-dirs / *.mp4 / *.png.',
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Outputs go directly under here: {output_dir}/{appearance,'
             'geometry,slat}/{scene_id}/...',
    )
    parser.add_argument(
        '--config', default='config/inference_video_to_4d.json',
        help='JSON config for ckpts/sampling/render/save. Default: %(default)s.',
    )
    parser.add_argument(
        '--image_size', type=int, default=518,
        help='Conditioning resize (DINOv2 expects a multiple of 14). '
             'Default: %(default)s (matches training).',
    )
    parser.add_argument(
        '--num_frames', type=int, default=None,
        help='Optional cap on frames per scene. Default: use all frames.',
    )
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Frame rate for the 2x2 azimuth grid video. Default: %(default)s.',
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    cks = cfg['ckpts']
    smp = cfg['sampling']
    rdr = cfg['render']
    sav = cfg['save']
    caps = cfg.get('caps') or {}
    num_frames = args.num_frames if args.num_frames is not None else caps.get('num_frames')

    # ar_kv only: the output layout drops the per-mode sub-dir, so running more
    # than one mode would overwrite. Enforce a single mode.
    modes = smp['modes']
    if len(modes) != 1:
        raise ValueError(
            f"This video-only script supports exactly one mode (ar_kv); "
            f"config has modes={modes}.")
    mode = modes[0]

    scenes = _resolve_scenes(args.input)

    torch.manual_seed(smp['seed'])
    np.random.seed(smp['seed'])
    os.makedirs(args.output_dir, exist_ok=True)

    bg_color: Tuple[float, float, float] = (
        (1.0, 1.0, 1.0) if rdr['bg_color'] == 'white' else (0.0, 0.0, 0.0)
    )

    print('=' * 60)
    print('Video-only SS+SLat inference')
    print(f'  config       : {args.config}')
    print(f'  input        : {args.input}  ({len(scenes)} scenes)')
    print(f'  output_dir   : {args.output_dir}')
    print(f'  mode         : {mode}')
    print(f'  image_size   : {args.image_size}')
    print(f'  steps        : ss={smp["steps_ss"]}, slat={smp["steps_slat"]}')
    print(f'  cfg          : ss={smp["cfg_ss"]}, slat={smp["cfg_slat"]}')
    print(f'  bg/res       : {rdr["bg_color"]} / {rdr["resolution"]}')
    print('  camera       : DEFAULT pose')
    print('=' * 60)

    pipeline = VideoTo4DPipeline.from_ckpts(
        ss_config=cks['ss_config'],
        ss_weights=cks['ss_weights'],
        slat_config=cks['slat_config'],
        slat_weights=cks['slat_weights'],
        rescale_t=smp['rescale_t'],
        cfg_interval=tuple(smp['cfg_interval']),
    )

    for scene in scenes:
        scene_id = scene['scene_id']
        cond_video = _load_scene_cond(scene, image_size=args.image_size)
        if num_frames is not None:
            cond_video = cond_video[:num_frames]
        T = cond_video.shape[0]
        print(f'\n=== {scene_id}  ({scene["kind"]}, {T} frames) ===')

        cond_per_frame = pipeline.encode_cond_video(cond_video)

        try:
            ss_samples = pipeline.sample_ss_sequence(
                cond_per_frame,
                steps=smp['steps_ss'], cfg_strength=smp['cfg_ss'],
            )
            ss_coords = [
                pipeline.ss_to_coords(s, threshold=smp['ss_threshold'])
                for s in ss_samples
            ]
            slat_samples = pipeline.sample_slat_sequence(
                ss_coords, cond_per_frame,
                steps=smp['steps_slat'], cfg_strength=smp['cfg_slat'],
            )
            # dataset_name="" and mode="" collapse decode_and_save's output to
            # {output_dir}/{appearance,geometry,slat}/{scene_id}/... (ar_kv only).
            pipeline.decode_and_save(
                slat_samples, scene_id, '', args.output_dir,
                dataset_name='',
                save_mesh=sav['save_mesh'],
                save_slat=sav['save_slat'],
                textured=sav['textured'],
                mesh_simplify=sav['mesh_simplify'],
                mesh_texture_size=sav['mesh_texture_size'],
                render_resolution=rdr['resolution'],
                bg_color=bg_color,
                # Remap rendered azimuth under the dir labels: 180->0, 270->90,
                # 0->180, 90->270 (label 0 now shows the previous yaw=180 view).
                appearance_yaw_offset_deg=180.0,
            )
            # Concatenate the 4 azimuth views into a 2x2 grid video per scene.
            _save_appearance_grid_video(
                os.path.join(args.output_dir, 'appearance', scene_id),
                fps=args.fps,
            )
        except NotImplementedError as e:
            print(f'  [skip] {scene_id} mode={mode}: {e}')

        torch.cuda.empty_cache()

    print(f'\nDone. Outputs in {args.output_dir}/')


if __name__ == '__main__':
    main()
