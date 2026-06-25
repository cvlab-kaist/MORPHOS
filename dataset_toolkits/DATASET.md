# T-SLat training data preparation

This page is for **reproducing the T-SLat-10K dataset or building a new one from your own animated meshes.**
This pipeline converts a set of animated `.glb` assets into the T-SLat dataset for MORPHOS SS and SLat training.

> ### ⚠ Read first:
>
> **CPU bottleneck** — Blender CYCLES rendering on the
> CPU dominates the wall-clock. For the scalable pipeline, we rendered **60
> multi-view images per frame**, half of the 120 used by the
> original TRELLIS. You can change `NUM_VIEWS` to control the number of multi-view images.
>
> **SLat normalization mismatch.** — The normalization stat (`mean` and `std`) in
> `config/temporal_slat_flow_dit_w3_diffforcing.json` were computed on an
> **earlier internal training subset**. The
> public release (`wlsgur1018/T-SLat-10K`) was prepared later from a
> larger object pool, so its empirical stats will differ from what MORPHOS trained on. Only recompute if you are retraining the model
> from scratch.



## Pipeline Overview

| Pass | Runtime | Purpose | Writes |
|---|---|---|---|
| **render** | Blender 3.0.1 | render multi-view images, condition videos, and normalized meshs | `renders_cond/<obj_id>/view_<VV>/frame_<TT>.png`, `meshes/<obj_id>/frame_<TT>.ply`, `render_done/<obj_id>.done` |
| **encode** | `morphos` conda env | voxelize each mesh to encode temporal SS and T-SLat latent | `ss_latents/<obj_id>/frame_<TT>.npz`, `slat_latents/<obj_id>/frame_<TT>.npz`, `encode_done/<obj_id>.done` |

Pretrained encoders are automatically downloaded once from
[`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large).



### Output layout

After a successful run the dataset root holds:

```
<DATA_ROOT>/
├── ss_latents/   <obj_id>/frame_<NNNN>.npz          # dense temporal SS latent (8, 16, 16, 16)
├── slat_latents/ <obj_id>/frame_<NNNN>.npz          # sparse T-SLat latent: coords [N,3] + feats [N,8]
├── renders_cond/ <obj_id>/view_<NN>/frame_<NNNN>.png # input video
├── encode_done/  <obj_id>.done                    # done marker
├── render_done/  <obj_id>.done                    # done marker
├── skipped_render.txt                             # GLBs rejected at render pass
├── skipped_encode.txt                             # objects rejected at encode pass
└── logs/                                          # per-rank logs
```

In input video, `view_00` is always canonical front view (yaw `3π/2`, pitch 10°).

- No `val/` subtree is produced. The pipeline only writes the training
tree above. You can manually split the validation set.

- Normalized meshes and multi-view images (`meshes/<obj_id>/frame_<TT>.ply` and
`renders/<obj_id>/frame_<TT>/<VV>.png`) are written during Pass 1 and deleted on Pass 2 to save storage.


## Installation and Setup

1. Prepare `morphos` conda env following `README.md`
2. Install Blender 3.0.1

    ```bash
    sudo apt-get update
    sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6
    wget 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz' -P <TARGET_DIR>
    tar -xvf <TARGET_DIR>/blender-3.0.1-linux-x64.tar.xz -C <TARGET_DIR>
    ```

3. Prepare your own dataset at `GLB_DIR`

    A flat directory of animated `.glb` files, one per object:
    ```
    GLB_DIR/
    ├── 0a1b2c3d.glb
    ├── 4e5f6a7b.glb
    └── ...
    ```

    For a quick test, download [Motion3-to-4 dataset](https://huggingface.co/datasets/River-Chen/Motion324):

    ```bash
    python dataset_toolkits/download_motion324_dataset.py \
      --parts 1 --local-dir <DATA_ROOT>
    # then set GLB_DIR=<DATA_ROOT>/train
    ```

## Configs

Set these by editing the top of `run_process_tslat_dataset.sh`, or pass them
inline as `VAR=value ./run_process_tslat_dataset.sh …`.

**Required paths**

| Var | Default | Set to |
|---|---|---|
| `OUTPUT_DIR` | `<PATH_TO_DATASET_OUTPUT_DIRECTORY>` | Dataset root to write (`<DATA_ROOT>`). |
| `GLB_DIR` | `<PATH_TO_ANIMATED_GLB_DIRECTORY>` | Flat directory of animated `.glb` files. |
| `CONDA_ENV` | `morphos` | Conda env name. |
| `BLENDER_PATH` | `<PATH_TO_BLENDER_3.0.1_BINARY>` | Absolute path to the Blender 3.0.1 binary. |

**Render configs**

| Var | Default | Meaning |
|---|---|---|
| `NUM_FRAMES` | 12 | Keyframes sampled per object. |
| `NUM_VIEWS` | 60 | Multi-view images per frame. |
| `NUM_COND_VIEWS` | 6 | Input video views per frame (first is always front). |
| `RESOLUTION` | 512 | Render side length in pixels (square). |

**Runtime configs**

| Var | Default | Meaning |
|---|---|---|
| `GPU_IDS` | `0,1` | CUDA device ids. |
| `RANKS_PER_GPU` | 1 | Workers per GPU. Render: ~2 GB each. Encode: keep at 1 unless GPU ≥ 24 GB. |
| `SEED` | 42 | Frame-sampling RNG seed. |
| `ADD` | 0 | `1` for an additional data for same obj with different keyframes. |
| `KEEP_INTERMEDIATES` | 0 | `1` keeps Pass-1 intermediates (`renders/`, `meshes/`) after Pass 2. |

**Quality filters**

| Var | Default | Meaning |
|---|---|---|
| `MAX_EXTENT_RATIO` | 15 | Reject GLBs whose pre-normalization bbox is dominated by one submesh. |
| `MIN_VOXELS` | 500 | Reject objects whose post-normalization voxelization is too sparse. |


## Run the Pipeline

```bash
bash dataset_toolkits/run_process_tslat_dataset.sh <pass> [N_objects]
```

| `<pass>` | Action |
|---|---|
| `render` | Pass 1 only. |
| `encode` | Pass 2 only (needs `render_done/<obj_id>.done` from a prior render). |
| `all`    | `render` → `encode`. |
| `status` | Print `render_done/` / `encode_done/` counts and per-rank status. |
| `kill`   | `SIGTERM` (then `SIGKILL`) every worker in `<DATA_ROOT>/logs/pids.txt`. |

`N_objects` (optional) caps how many GLBs each worker processes — `-1`
(default) processes every file in `GLB_DIR`.

**GPU sharding** — `WORLD_SIZE = NUM_GPUS × RANKS_PER_GPU` workers are spawned and pinned via `CUDA_VISIBLE_DEVICES`.

```bash
# 16-way render across 8 GPUs
RANKS_PER_GPU=2 GPU_IDS="0,1,2,3,4,5,6,7" \
  bash dataset_toolkits/run_process_tslat_dataset.sh render

# Encode: one rank per GPU
RANKS_PER_GPU=1 GPU_IDS="0,1,2,3" \
  bash dataset_toolkits/run_process_tslat_dataset.sh encode
```

**Extra frame sampling** — add a second sampling without overwriting:

```bash
SEED=123 ADD=1 bash dataset_toolkits/run_process_tslat_dataset.sh render
```

**Monitor**

```bash
bash dataset_toolkits/run_process_tslat_dataset.sh status
bash dataset_toolkits/run_process_tslat_dataset.sh kill # kill running processes
```


## Post-Process Filtering

`detect_failure.py` inspects video quality in `renders_cond/` for two render failures:

- **cropped** — silhouette touches the image border
- **blank** — near-zero alpha coverage across every view/frame.

```bash
python dataset_toolkits/detect_failure.py --data_dir <DATA_ROOT>
```

It reads `<DATA_ROOT>/renders_cond/` and writes both lists back under `<DATA_ROOT>/`:

| Output | Contents |
|---|---|
| `blank_ids.txt` | object ids with near-zero alpha coverage everywhere |
| `cropped_ids.txt` | object ids whose worst-frame `border_ratio` exceeds the candidate threshold (`0.02`–`0.50`) that flags the most ids |

Use these sample ids to curate which objects to drop.

## Split Validation Set

Split validation set from processed dataset:

```bash
N_VAL=20 DATA_DIR=<DATA_ROOT> bash dataset_toolkits/split_val.sh
```

| Env | Default | Meaning |
|---|---|---|
| `DATA_DIR` | — (required) | Dataset root to split. |
| `N_VAL` | 20 | Number of objects to move into `val/`. |
| `SEED` | 42 | RNG seed for the random pick (reproducible). |
| `DRY_RUN` | 0 | `1` lists the chosen ids without moving anything. |


## Compute Normalization Stats

Compute normalization stat on dataset:

```bash
python dataset_toolkits/compute_norm_stats.py --data_dir <DATA_ROOT>
```

It writes the result to `<DATA_ROOT>/slat_norm_stats.json`.

Place the stat in the `dataset.args.normalization` block of
`config/temporal_slat_flow_dit_w3_diffforcing.json`:

```json
"normalization": {
    "mean": [ ... ],
    "std":  [ ... ]
}
```
