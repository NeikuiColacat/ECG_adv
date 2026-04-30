#!/bin/bash
# Train CT v2 on 3 centers sequentially (per-center INDEPENDENT, no joint training).
# Each center is independent — new centers can be enrolled later by running this
# script (or train_center_token_v2.py directly) on their data.
#
# Plan Rev 13. ~5 min per center on RTX 4090, ~15 min total.
#
# Usage:
#   bash scripts/ecgtwin_gen/run_train_ct_v2_3centers.sh

set -e

PYTHON=/root/miniforge3/envs/ECGTwin/bin/python
REPO=/root/ECG_adv_Gen
DATA_ROOT=/root/autodl-tmp/center_token_super5
OUT_ROOT=${OUT_ROOT:-/root/autodl-tmp/center_token_super5_v2}

TOTAL_STEPS=${TOTAL_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-8}
SPHERE_NORM=${SPHERE_NORM:-1.0}

mkdir -p "$OUT_ROOT"

for center in extra nin geo; do
    DATASET="$DATA_ROOT/${center}_k200.pt"
    SAVE_DIR="$OUT_ROOT/${center}_k200"

    if [ ! -f "$DATASET" ]; then
        echo "[skip] $DATASET not found"
        continue
    fi

    echo
    echo "============================================================"
    echo "[train] center=$center steps=$TOTAL_STEPS batch=$BATCH_SIZE sphere=$SPHERE_NORM"
    echo "[train] dataset=$DATASET"
    echo "[train] save_dir=$SAVE_DIR"
    echo "============================================================"

    "$PYTHON" "$REPO/scripts/ecgtwin_gen/train_center_token_v2.py" \
        --dataset "$DATASET" \
        --save_dir "$SAVE_DIR" \
        --total_steps "$TOTAL_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --sphere_target_norm "$SPHERE_NORM" \
        --log_every 100 \
        2>&1 | tee "$SAVE_DIR.log"
done

echo
echo "[all done] CT v2 trained for 3 centers; smoothed snapshots at:"
ls -la "$OUT_ROOT"/*/center_token_smoothed.pth
