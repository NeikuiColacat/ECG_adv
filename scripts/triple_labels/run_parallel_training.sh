#!/bin/bash
# Triple-label training orchestrator — runs 3 schemes in parallel or sequentially.
#
# Usage:
#   bash scripts/triple_labels/run_parallel_training.sh [parallel|sequential]
#
# - parallel: 3 trainers concurrently on the same GPU, batch=96, no compile.
#             Memory budget ~7-8GB; safe on RTX 4090 24GB.
# - sequential: 1 trainer at a time, batch=192 + torch.compile=true (faster
#               per-model but takes 3x wall time).
#
# After training, runs 3 evaluators in parallel and aggregates results.

set -euo pipefail

ROOT=/root/ECG_adv_Gen
OUT=/root/autodl-tmp/triple_labels
PY=/root/miniforge3/envs/ECGTwin/bin/python
MODE=${1:-parallel}

cd "$ROOT"
mkdir -p "$OUT"/{super5,sub23,pn26}

launch_train() {
    local scheme=$1 batch=$2 compile=$3 epochs=${4:-50}
    "$PY" scripts/triple_labels/train_ptbxl.py \
        --scheme "$scheme" --batch_size "$batch" --compile "$compile" \
        --epochs "$epochs" \
        --output_dir "$OUT/$scheme" \
        > "$OUT/$scheme/train.log" 2>&1
}

launch_eval() {
    local scheme=$1
    "$PY" scripts/triple_labels/eval_crosscenter.py \
        --scheme "$scheme" --model_dir "$OUT/$scheme" \
        --batch_size 256 --num_workers 4 \
        > "$OUT/$scheme/eval.log" 2>&1
}

echo "[runner] mode=$MODE  output=$OUT"
echo "[runner] starting at $(date)"
T_START=$SECONDS

if [ "$MODE" = "parallel" ]; then
    echo "[runner] launching 3 parallel trainers (batch 96, no compile)"
    launch_train super5 96 false &
    P_S5=$!
    launch_train sub23 96 false &
    P_S23=$!
    launch_train pn26 96 false &
    P_P26=$!
    trap "kill $P_S5 $P_S23 $P_P26 2>/dev/null || true" EXIT
    echo "[runner] PIDs: super5=$P_S5  sub23=$P_S23  pn26=$P_P26"
    rc_s5=0; rc_s23=0; rc_p26=0
    wait $P_S5  || rc_s5=$?
    wait $P_S23 || rc_s23=$?
    wait $P_P26 || rc_p26=$?
    trap - EXIT
    if [ $rc_s5 -ne 0 ] || [ $rc_s23 -ne 0 ] || [ $rc_p26 -ne 0 ]; then
        echo "[runner] trainer failures: super5=$rc_s5 sub23=$rc_s23 pn26=$rc_p26" >&2
        exit 1
    fi
elif [ "$MODE" = "sequential" ]; then
    echo "[runner] sequential trainers (batch 192, compile true)"
    for s in super5 sub23 pn26; do
        echo "[runner] training $s..."
        launch_train "$s" 192 true
    done
else
    echo "[runner] unknown mode: $MODE (use parallel|sequential)" >&2
    exit 2
fi

T_TRAIN=$((SECONDS - T_START))
echo "[runner] training done in ${T_TRAIN}s ($((T_TRAIN/60))m)"

echo "[runner] launching 3 evaluators in parallel"
T_EVAL_START=$SECONDS
launch_eval super5 &
E_S5=$!
launch_eval sub23 &
E_S23=$!
launch_eval pn26 &
E_P26=$!
rc_e_s5=0; rc_e_s23=0; rc_e_p26=0
wait $E_S5  || rc_e_s5=$?
wait $E_S23 || rc_e_s23=$?
wait $E_P26 || rc_e_p26=$?
if [ $rc_e_s5 -ne 0 ] || [ $rc_e_s23 -ne 0 ] || [ $rc_e_p26 -ne 0 ]; then
    echo "[runner] eval failures: super5=$rc_e_s5 sub23=$rc_e_s23 pn26=$rc_e_p26" >&2
    exit 1
fi
T_EVAL=$((SECONDS - T_EVAL_START))
echo "[runner] eval done in ${T_EVAL}s"

echo "[runner] aggregating results"
"$PY" scripts/triple_labels/aggregate_results.py --out_dir "$OUT" \
    > "$OUT/aggregate.log" 2>&1 || echo "[runner] aggregate FAILED, check log"

T_TOTAL=$((SECONDS - T_START))
echo "[runner] all done in ${T_TOTAL}s ($((T_TOTAL/60))m) at $(date)"
echo "[runner] results: $OUT"
echo "[runner] report:  $ROOT/docs/triple_labels_experiment.md"
