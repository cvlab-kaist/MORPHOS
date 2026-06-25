#!/bin/bash
# Stage-1 SS diffusion-forcing training (block-causal temporal flow matching).
# Snapshot/validation uses the dual cond/uncond KV-cache sampler
# (FlowEulerKVCacheGuidanceIntervalSampler) — no history guidance, no TF.
export CUDA_HOME=/usr/local/cuda-12.4

# Attention backend: flash_attn end-to-end. The SS diff-forcing self-attn calls
# flash_attn.flash_attn_func directly (per-frame slicing, no mask); cross-attn
# goes through the wrapper which honors this env var. REQUIRED.
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export PYTHONWARNINGS=ignore

# Wandb. Set WANDB_API_KEY in your shell (`export WANDB_API_KEY=...`) before
# launching, or `wandb login` once. Unset → wandb runs in offline/silent mode.
export WANDB_PROJECT="${WANDB_PROJECT:-morphos-ss-temporal}"

RUN_NAME=ss_diffforcing_w3_uniform_$(date +%Y%m%d_%H%M%S)
export WANDB_RUN_NAME="${RUN_NAME}"

CONFIG=./config/temporal_ss_flow_dit_w3_diffforcing.json
# SS dataset root: needs <root>/{ss_latents, renders_cond, encode_done} for
# train and <root>/val/{ss_latents, renders_cond} for val. Generate this tree
# with `dataset_toolkits/run_process_tslat_dataset.sh all`.
DATA_DIR="${DATA_DIR:-<PATH_TO_TRAINING_DATA_ROOT>}"
OUTPUT_DIR="./outputs/ss_diffforcing/${RUN_NAME}"

# triton / inductor caches. Defaults to ~/.cache; override via env.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

# Resume: CKPT=<step> LOAD_DIR=./outputs/ss_diffforcing/<run> ./run_train_temporal_ss_flow_dit.sh
CKPT=${CKPT:-}
LOAD_DIR=${LOAD_DIR:-}

echo "============================================================"
echo "Config:     $CONFIG"
echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Run name:   $RUN_NAME"
echo "GPUs:       $CUDA_VISIBLE_DEVICES (num_gpus=$NUM_GPUS)"
echo "ATTN_BACKEND: $ATTN_BACKEND"
echo "Resume from: ${LOAD_DIR:-"<from scratch>"} ${CKPT:+(step $CKPT)}"
echo "============================================================"

CMD="python3 train.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --num_gpus $NUM_GPUS \
    --master_port ${MASTER_PORT:-29500}"
if [ -n "$LOAD_DIR" ]; then CMD="$CMD --load_dir $LOAD_DIR"; fi
if [ -n "$CKPT" ]; then CMD="$CMD --ckpt $CKPT"; fi

echo "Running: $CMD"
eval $CMD
