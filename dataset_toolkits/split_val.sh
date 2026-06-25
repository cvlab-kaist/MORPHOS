#!/bin/bash
# Carve a validation split out of a processed T-SLat dataset tree.
#
# Trainers read validation from
#   <DATA_DIR>/val/{ss_latents,slat_latents,renders_cond}/
# and treat everything still under <DATA_DIR>/{ss_latents,slat_latents,
# renders_cond}/ as training. This script moves N randomly-chosen object ids
# into val/ so the trainers pick them up. Object ids are taken from ss_latents/.
#
# Usage:
#   N_VAL=20 DATA_DIR=/path/to/dataset bash dataset_toolkits/split_val.sh
#
# Env:
#   DATA_DIR   (required) dataset root produced by run_process_tslat_dataset.sh
#   N_VAL      number of validation objects to move        (default: 20)
#   SEED       RNG seed for the random pick (reproducible)  (default: 42)
#   DRY_RUN    set to 1 to list the chosen ids without moving anything
set -euo pipefail

DATA_DIR="${DATA_DIR:?set DATA_DIR to the dataset root}"
N_VAL="${N_VAL:-20}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

SUBSETS=(ss_latents slat_latents renders_cond)

src="$DATA_DIR/ss_latents"
[[ -d "$src" ]] || { echo "no ss_latents/ under $DATA_DIR" >&2; exit 1; }

# All training object ids = top-level dirs under ss_latents/.
mapfile -t ALL < <(find "$src" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
total="${#ALL[@]}"
(( total > 0 ))       || { echo "no object dirs under $src" >&2; exit 1; }
(( N_VAL <= total ))  || { echo "N_VAL=$N_VAL > available objects ($total)" >&2; exit 1; }

# Deterministic shuffle keyed by SEED, take the first N_VAL ids.
mapfile -t PICK < <(printf '%s\n' "${ALL[@]}" | shuf -n "$N_VAL" --random-source=<(yes "$SEED"))

echo "Splitting $N_VAL / $total objects into $DATA_DIR/val/  (seed=$SEED)"
for sid in "${PICK[@]}"; do
    echo "  $sid"
    [[ "$DRY_RUN" == 1 ]] && continue
    for sub in "${SUBSETS[@]}"; do
        s="$DATA_DIR/$sub/$sid"
        if [[ -d "$s" ]]; then
            mkdir -p "$DATA_DIR/val/$sub"
            mv "$s" "$DATA_DIR/val/$sub/$sid"
        fi
    done
done

if [[ "$DRY_RUN" == 1 ]]; then
    echo "(dry run — nothing moved)"
else
    echo "Done. Moved ${#PICK[@]} objects to $DATA_DIR/val/"
fi
