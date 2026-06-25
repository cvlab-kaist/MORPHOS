#!/bin/bash
# Prepare dynamic mesh data for TRELLIS training.
#
# Usage:
#   bash dataset_toolkits/run_process_tslat_dataset.sh render [N]
#   bash dataset_toolkits/run_process_tslat_dataset.sh encode [N]
#   bash dataset_toolkits/run_process_tslat_dataset.sh all [N]
#   bash dataset_toolkits/run_process_tslat_dataset.sh status
#   bash dataset_toolkits/run_process_tslat_dataset.sh kill   # kill all workers spawned by this script
#
# GPU config: GPU_IDS="0,1,4,6" bash dataset_toolkits/run_process_tslat_dataset.sh render
#
# Multi-rank per GPU (oversubscribe):
#   Blender render uses ~2GB VRAM per process, so multiple ranks can share one GPU.
#   RANKS_PER_GPU=2 GPU_IDS="0,1,2,3,4,5,6,7" \
#     bash dataset_toolkits/run_process_tslat_dataset.sh render
#     # 16 render workers across 8 GPUs (rank 0,8 -> GPU 0; rank 1,9 -> GPU 1; ...)
#
#   RANKS_PER_GPU=4 GPU_IDS="0,1,2,3" \
#     bash dataset_toolkits/run_process_tslat_dataset.sh render
#     # 16 render workers across 4 GPUs
#
#   Encode is VRAM-heavy (~8-10GB per rank). Use RANKS_PER_GPU=1 unless GPU has >=24GB.
#     RANKS_PER_GPU=1 bash dataset_toolkits/run_process_tslat_dataset.sh encode
#
#   Watch CPU/RAM too — each Blender worker needs 1+ core & ~500MB RAM.
#
# Seed/add: SEED=123 ADD=1 bash dataset_toolkits/run_process_tslat_dataset.sh render
#           Re-renders same objects with different frame sampling.
#           Outputs go to {obj_id}__run_123/ subdirs (don't overwrite originals).

set -e
export WARP_CACHE_PATH=./tmp/.cache/warp_user
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# === REQUIRED user-fillable paths ===
# Where the prepared dataset tree will live (encode_done/, ss_latents/,
# latents/, renders_cond/, val/, etc.).
OUTPUT_DIR="${OUTPUT_DIR:-<PATH_TO_DATASET_OUTPUT_DIRECTORY>}"
# Directory of animated .glb files to process.
GLB_DIR="${GLB_DIR:-<PATH_TO_ANIMATED_GLB_DIRECTORY>}"
# Conda env with torch/spconv/open3d (created by repo-root setup.sh / setup_b200.sh).
CONDA_ENV="${CONDA_ENV:-morphos}"
# Blender 3.0.1 executable.
BLENDER_PATH="${BLENDER_PATH:-<PATH_TO_BLENDER_3.0.1_BINARY>}"
# Render-shape knobs (env-overridable). Changing NUM_VIEWS or NUM_FRAMES
# changes the SLat-feature distribution → normalization stats must be
# recomputed if you retrain; see dataset_toolkits/DATASET.md §7.
NUM_FRAMES="${NUM_FRAMES:-12}"
NUM_VIEWS="${NUM_VIEWS:-60}"
NUM_COND_VIEWS="${NUM_COND_VIEWS:-6}"
RESOLUTION="${RESOLUTION:-512}"
# Keep the per-frame supervision renders + meshes after encoding. Default
# = drop them to save disk (~600 MB / scene at default settings).
KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-0}"
GPU_IDS="${GPU_IDS:-0,1}"
SEED="${SEED:-42}"
ADD="${ADD:-0}"
MAX_EXTENT_RATIO="${MAX_EXTENT_RATIO:-15}"
MIN_VOXELS="${MIN_VOXELS:-500}"

if [ ! -f "$BLENDER_PATH" ]; then
    echo "Blender not found at $BLENDER_PATH"; exit 1
fi

# Build --add / --keep_intermediates flag strings
ADD_FLAG=""
[ "$ADD" = "1" ] && ADD_FLAG="--add"
KEEP_FLAG=""
[ "$KEEP_INTERMEDIATES" = "1" ] && KEEP_FLAG="--keep_intermediates"

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}
RANKS_PER_GPU="${RANKS_PER_GPU:-1}"
WORLD_SIZE=$((NUM_GPUS * RANKS_PER_GPU))
PASS=${1:-all}
MAX_OBJECTS=${2:--1}

echo "Config: GPUs=${GPU_IDS} (${NUM_GPUS}) x ranks/gpu=${RANKS_PER_GPU} -> world_size=${WORLD_SIZE}"
echo "        output=${OUTPUT_DIR}, seed=${SEED}, add=${ADD}"

PID_FILE="$OUTPUT_DIR/logs/pids.txt"

run_render() {
    echo "=== Render: ${WORLD_SIZE} workers on ${NUM_GPUS} GPUs, ${NUM_FRAMES}f x ${NUM_VIEWS}v ==="
    mkdir -p "$OUTPUT_DIR/logs"
    : > "$PID_FILE"
    for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
        GPU_ID=${GPU_ARRAY[$((RANK % NUM_GPUS))]}
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        "$BLENDER_PATH" --background --python "$SCRIPT_DIR/process_tlsat_dataset.py" -- \
            --pass render --glb_dir "$GLB_DIR" --output_dir "$OUTPUT_DIR" \
            --num_frames $NUM_FRAMES --num_views $NUM_VIEWS \
            --num_cond_views $NUM_COND_VIEWS --resolution $RESOLUTION \
            --rank $RANK --world_size $WORLD_SIZE -n $MAX_OBJECTS \
            --seed $SEED --max_extent_ratio $MAX_EXTENT_RATIO $ADD_FLAG \
            $KEEP_FLAG \
            > "$OUTPUT_DIR/logs/render_rank${RANK}.log" 2>&1 &
        echo "$!" >> "$PID_FILE"
        echo "  Rank $RANK on GPU $GPU_ID (PID $!)"
    done
    echo "Monitor: tail -f $OUTPUT_DIR/logs/render_rank*.log"
    echo "Kill:    bash $0 kill"
    wait; echo "Render done."
}

run_encode() {
    echo "=== Encode: ${WORLD_SIZE} workers on ${NUM_GPUS} GPUs ==="
    mkdir -p "$OUTPUT_DIR/logs"
    : > "$PID_FILE"
    for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
        GPU_ID=${GPU_ARRAY[$((RANK % NUM_GPUS))]}
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        ${CONDA_ENV}/bin/python \
        "$SCRIPT_DIR/process_tlsat_dataset.py" \
            --pass encode --glb_dir "$GLB_DIR" --output_dir "$OUTPUT_DIR" \
            --rank $RANK --world_size $WORLD_SIZE -n $MAX_OBJECTS \
            --min_voxels $MIN_VOXELS $KEEP_FLAG \
            > "$OUTPUT_DIR/logs/encode_rank${RANK}.log" 2>&1 &
        echo "$!" >> "$PID_FILE"
        echo "  Rank $RANK on GPU $GPU_ID (PID $!)"
    done
    echo "Monitor: tail -f $OUTPUT_DIR/logs/encode_rank*.log"
    echo "Kill:    bash $0 kill"
    wait; echo "Encode done."
}

run_status() {
    echo "=== Status ==="
    for f in "$OUTPUT_DIR"/logs/status_*.txt; do [ -f "$f" ] && cat "$f"; done
    echo "Rendered: $(ls "$OUTPUT_DIR/render_done/" 2>/dev/null | wc -l)"
    echo "Encoded:  $(ls "$OUTPUT_DIR/encode_done/" 2>/dev/null | wc -l)"
}

run_kill() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file at $PID_FILE — nothing spawned by this script."
        return
    fi
    local ALIVE=()
    while read -r PID; do
        [ -z "$PID" ] && continue
        if kill -0 "$PID" 2>/dev/null; then
            ALIVE+=("$PID")
        fi
    done < "$PID_FILE"
    if [ ${#ALIVE[@]} -eq 0 ]; then
        echo "No live workers tracked in $PID_FILE."
        : > "$PID_FILE"
        return
    fi
    echo "Sending SIGTERM to ${#ALIVE[@]} workers: ${ALIVE[*]}"
    kill -TERM "${ALIVE[@]}" 2>/dev/null
    sleep 3
    local STILL=()
    for PID in "${ALIVE[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then STILL+=("$PID"); fi
    done
    if [ ${#STILL[@]} -gt 0 ]; then
        echo "SIGKILL to stragglers: ${STILL[*]}"
        kill -KILL "${STILL[@]}" 2>/dev/null
    fi
    : > "$PID_FILE"
    echo "Killed."
}

case "$PASS" in
    render) run_render ;;
    encode) run_encode ;;
    all) run_render; echo "---"; run_encode ;;
    status) run_status ;;
    kill) run_kill ;;
    *) echo "Usage: $0 {render|encode|all|status|kill} [N_objects]"; exit 1 ;;
esac
