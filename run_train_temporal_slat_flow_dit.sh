#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.4

# Wandb. Set WANDB_API_KEY in your shell (`export WANDB_API_KEY=...`) before
# launching, or `wandb login` once. Unset → wandb runs in offline/silent mode.
export WANDB_PROJECT="${WANDB_PROJECT:-morphos-slat-temporal}"

RUN_NAME=slat_diffforcing_w3_uniform_$(date +%Y%m%d_%H%M%S)
export WANDB_RUN_NAME="${RUN_NAME}"

CONFIG=./config/temporal_slat_flow_dit_w3_diffforcing.json
# SLat dataset root: same on-disk tree the SS trainer reads. Generate it with
# `dataset_toolkits/run_process_tslat_dataset.sh all` (encode pass writes
# `latents/<obj_id>/frame_XXXX.npz`).
DATA_DIR="${DATA_DIR:-<PATH_TO_TRAINING_DATA_ROOT>}"
OUTPUT_DIR="./outputs/slat_diffforcing/${RUN_NAME}"

# Resume: CKPT=<step> [LOAD_DIR=./outputs/slat_diffforcing/<run>] ./run_train_temporal_slat_flow_dit.sh
CKPT=${CKPT:-}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONWARNINGS=ignore
export ATTN_BACKEND=flash_attn

# triton / inductor caches. Defaults to ~/.cache; override via env.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

echo "============================================================"
echo "Config:     $CONFIG"
echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Run name:   $RUN_NAME"
echo "GPUs:       $CUDA_VISIBLE_DEVICES (num_gpus=$NUM_GPUS)"
echo "Resume from: ${LOAD_DIR:-"<from scratch>"} ${CKPT:+(step $CKPT)}"
echo "============================================================"

export SPCONV_ALGO=native
CMD="python3 train.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --num_gpus $NUM_GPUS \
    --master_port ${MASTER_PORT:-29500}"
if [ -n "${LOAD_DIR:-}" ]; then CMD="$CMD --load_dir $LOAD_DIR"; fi
if [ -n "$CKPT" ];           then CMD="$CMD --ckpt $CKPT"; fi

echo "Running: $CMD"
eval $CMD
