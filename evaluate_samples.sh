#!/bin/bash
# Combined evaluation launcher for the per-frame inference output tree.
#
# Runs the geometry and appearance metric suites against a predicted output
# tree (the layout produced by run_inference_video_to_4d.sh):
#
#   {pred_root}/appearance/{scene_id}/{0|90|180|270}/frame_NNNN.png
#   {pred_root}/geometry/{scene_id}/frame_NNNN.glb
#
# The GT side is described entirely by a generic, path-templated JSON config —
# no dataset layout is hard-coded. See config/eval_gt_dataset.example.json.
# Works for ActionBench / Motion80 / Consist4D or any custom data: just point
# the templates at your GT.
#
# Which metrics run depends on what the gt_config provides:
#   gt_4view     -> appearance (LPIPS / CLIP / DreamSim / FVD)
#   gt_pcd       -> geometry CD / F-score
#   gt_face_glb  -> P2S (omit to skip P2S; CD / F-score still run)
#
# Usage:
#   ./evaluate_samples.sh <gt_config.json>
#
# Positional:
#   $1   path to the GT dataset JSON config
#
# Required env:
#   PRED_PATH      root of one inference run — must contain
#                  appearance/<scene>/...  and/or  geometry/<scene>/...
#
# Optional env overrides:
#   CUDA_VISIBLE_DEVICES   default 0
#   METRICS                space-separated subset of: lpips clip dreamsim fvd. Default: all.
#   OUTPUT_CSV_APPEARANCE  default <PRED_PATH>/metrics_appearance.csv
#   OUTPUT_CSV_GEOMETRY    default <PRED_PATH>/metrics_geometry.csv
#   GRID_VIDEO_FPS         default 10  (per-scene comparison.mp4 fps)
#   NO_GRID_VIDEO=1        skip the per-scene comparison.mp4
#   RECOMPUTE=1            force recompute even if rows already in output CSV
#   GEOMETRY_ONLY=1        only run metric_geometry
#   APPEARANCE_ONLY=1      only run metric_appearance
#
# Outputs:
#   <PRED_PATH>/metrics_appearance.csv     per-scene LPIPS/CLIP/DreamSim/FVD
#   <PRED_PATH>/metrics_geometry.csv       per-scene Chamfer/F-score/P2S
#   <PRED_PATH>/<scene>/comparison.mp4     (unless NO_GRID_VIDEO=1)
#
# Required vendored asset:
#   trellis/evaluation/fvd/styleganv/i3d_torchscript.pt  (49 MB)
#   FVD will silently fall back if missing — copy it in to enable.

set -e

GT_CONFIG="${1:?Usage: $0 <gt_config.json>   (env: PRED_PATH required)}"

: "${PRED_PATH:?PRED_PATH must be set (root containing appearance/ and/or geometry/)}"
[[ -f "$GT_CONFIG" ]] || { echo "gt_config not found: $GT_CONFIG" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

METRICS="${METRICS:-lpips clip dreamsim fvd}"
OUTPUT_CSV_APPEARANCE="${OUTPUT_CSV_APPEARANCE:-${PRED_PATH}/metrics_appearance.csv}"
OUTPUT_CSV_GEOMETRY="${OUTPUT_CSV_GEOMETRY:-${PRED_PATH}/metrics_geometry.csv}"
GRID_VIDEO_FPS="${GRID_VIDEO_FPS:-10}"

EXTRA_APP=""
[[ -n "${NO_GRID_VIDEO:-}" ]] && EXTRA_APP+=" --no_grid_video"
[[ -n "${RECOMPUTE:-}"     ]] && EXTRA_APP+=" --recompute"
EXTRA_APP+=" --grid_video_fps ${GRID_VIDEO_FPS}"

EXTRA_GEO=""
[[ -n "${RECOMPUTE:-}" ]] && EXTRA_GEO+=" --recompute"

if [[ ! -d "${PRED_PATH}/appearance" ]] && [[ -z "${GEOMETRY_ONLY:-}" ]]; then
    echo "WARNING: ${PRED_PATH}/appearance not found — skipping appearance metrics" >&2
    APPEARANCE_ONLY=""
    GEOMETRY_ONLY=1
fi
if [[ ! -d "${PRED_PATH}/geometry" ]] && [[ -z "${APPEARANCE_ONLY:-}" ]]; then
    echo "WARNING: ${PRED_PATH}/geometry not found — skipping geometry metrics" >&2
    GEOMETRY_ONLY=""
    APPEARANCE_ONLY=1
fi

echo "============================================================"
echo "Eval"
echo "  gt_config           : $GT_CONFIG"
echo "  pred_path           : $PRED_PATH"
echo "  appearance metrics  : $METRICS"
echo "  appearance csv      : $OUTPUT_CSV_APPEARANCE"
echo "  geometry  csv       : $OUTPUT_CSV_GEOMETRY"
echo "  cuda                : $CUDA_VISIBLE_DEVICES"
echo "============================================================"

if [[ -z "${APPEARANCE_ONLY:-}" ]]; then
    echo
    echo "--- metric_geometry ---"
    python3 -m trellis.evaluation.metric_geometry \
        --gt_config "$GT_CONFIG" \
        --pred_path "$PRED_PATH/geometry" \
        --output_csv "$OUTPUT_CSV_GEOMETRY" \
        $EXTRA_GEO
fi

if [[ -z "${GEOMETRY_ONLY:-}" ]]; then
    echo
    echo "--- metric_appearance ---"
    python3 -m trellis.evaluation.metric_appearance \
        --gt_config "$GT_CONFIG" \
        --pred_path "$PRED_PATH/appearance" \
        --metrics $METRICS \
        --output_csv "$OUTPUT_CSV_APPEARANCE" \
        $EXTRA_APP
fi

echo
echo "Done. CSVs written under: $PRED_PATH/"
