#!/bin/bash
# Standalone video-only SS+SLat Diffusion-Forcing inference launcher.
#
# Runs the pipeline on plain videos / frame folders. Rendering always uses the
# pipeline's default camera pose.
#
# Ckpt paths + sampling/render/save knobs live in the JSON config
# `config/inference_video_to_4d.json` (override with CONFIG=...). This script
# only wires up the input + output dir.
#
# Usage:
#   ./run_inference_video_to_4d.sh <input> [output_dir]
#
# Positional:
#   $1   input: a *.mp4 file, OR a directory of sub-dirs / *.mp4 / *.png
#          - sub-dirs        -> one scene per sub-dir (its *.png frames)
#          - loose *.mp4     -> one scene per video
#          - loose *.png     -> a single scene (the directory)
#   $2   output dir (optional). Default: ${OUTPUT_DIR} env or ./results/video_only.
#
# Optional env:
#   CONFIG                 default ./config/inference_video_to_4d.json
#   IMAGE_SIZE             conditioning resize (default: 518)
#   NUM_FRAMES             cap frames per scene (default: all)
#   FPS                    grid_video.mp4 frame rate (default: 10)
#   CUDA_VISIBLE_DEVICES   default 7
#
# Output layout (ar_kv mode only):
#   {OUTPUT_DIR}/appearance/{scene}/{0|90|180|270}/frame_NNNN.png
#   {OUTPUT_DIR}/appearance/{scene}/grid_video.mp4   # 2x2 azimuth grid
#   {OUTPUT_DIR}/geometry/{scene}/frame_NNNN.glb
#   {OUTPUT_DIR}/slat/{scene}/frame_NNNN.npz

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export SPCONV_ALGO=native
export ATTN_BACKEND=flash_attn

INPUT="${1:?Usage: $0 <input: a *.mp4 file or a dir of sub-dirs/*.mp4/*.png> [output_dir]}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-./results/video_only}}"

CONFIG="${CONFIG:-./config/inference_video_to_4d.json}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Video-only SS+SLat inference"
echo "  config        : $CONFIG"
echo "  input         : $INPUT"
echo "  output_dir    : $OUTPUT_DIR"
echo "  image_size    : $IMAGE_SIZE"
echo "  cuda          : $CUDA_VISIBLE_DEVICES"
echo "============================================================"

EXTRA_ARGS=()
if [ -n "${NUM_FRAMES:-}" ] ; then
    EXTRA_ARGS+=(--num_frames "$NUM_FRAMES")
fi
if [ -n "${FPS:-}" ] ; then
    EXTRA_ARGS+=(--fps "$FPS")
fi

python3 inference_video_to_4d.py \
    --input "$INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --image_size "$IMAGE_SIZE" \
    "${EXTRA_ARGS[@]}"
