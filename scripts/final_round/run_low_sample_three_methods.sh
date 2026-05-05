#!/usr/bin/env bash
set -euo pipefail

# Reproduce the low-sample thesis comparison:
#   1) real2000 random-init baseline
#   2) no-token hard-label pretrain -> real2000 fine-tune
#   3) center-token hard-label pretrain -> real2000 fine-tune

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/miniforge3/envs/ECGTwin/bin/python}"

export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/graduate_project}"

DATA_PATH="${DATA_PATH:-/root/autodl-tmp/ptbxl/raw100.npy}"
CSV_PATH="${CSV_PATH:-/root/autodl-tmp/ptbxl/ptbxl_database.csv}"
CACHE_PATH="${CACHE_PATH:-/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy}"
SPLIT_JSON="${SPLIT_JSON:-/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json}"

NO_TOKEN_INIT="${NO_TOKEN_INIT:-/root/autodl-tmp/graduate_project/self_distill_v2_e21_v46_no_token_hardlabel_r10_seed8042_auroc/best_model.pt}"
CENTER_TOKEN_INIT="${CENTER_TOKEN_INIT:-/root/autodl-tmp/graduate_project/self_distill_v2_e18_v46_class_oracle_hardlabel_r10_seed42_auroc/best_model.pt}"

NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_ROOT"

common_train_args=(
  --scheme super5
  --data_path "$DATA_PATH"
  --csv_path "$CSV_PATH"
  --cache_path "$CACHE_PATH"
  --split_json "$SPLIT_JSON"
  --preprocess_mode minimal_resample
  --norm_mode per_sample_global
  --device "$DEVICE"
  --crop_len 1000
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --checkpoint_metric auroc
)

common_eval_args=(
  --scheme super5
  --device "$DEVICE"
  --crop_len 1000
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --ptbxl_csv "$CSV_PATH"
  --ptbxl_cache "$CACHE_PATH"
  --preprocess_mode minimal_resample
  --norm_mode per_sample_global
  --skip_mimic
)

eval_model() {
  local model_dir="$1"
  "$PYTHON" "$ROOT/scripts/triple_labels/eval_crosscenter.py" \
    "${common_eval_args[@]}" \
    --model_dir "$model_dir" \
    --output_path "$model_dir/eval_result_current_code.json"
}

REAL_DIR="$OUT_ROOT/method_a_real2000_seed42_rerun_${RUN_TAG}"
NO_TOKEN_DIR="$OUT_ROOT/self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc_rerun_${RUN_TAG}"
CENTER_TOKEN_DIR="$OUT_ROOT/self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc_rerun_${RUN_TAG}"

echo "[run] tag=$RUN_TAG"
echo "[run] real2000 -> $REAL_DIR"
"$PYTHON" "$ROOT/scripts/triple_labels/train_ptbxl.py" \
  "${common_train_args[@]}" \
  --output_dir "$REAL_DIR" \
  --epochs 50 \
  --lr 0.01 \
  --weight_decay 0.01 \
  --cosine_tmax 15 \
  --patience 10 \
  --seed 42
eval_model "$REAL_DIR"

echo "[run] no-token hard-label pretrain -> real2000 fine-tune -> $NO_TOKEN_DIR"
test -f "$NO_TOKEN_INIT"
"$PYTHON" "$ROOT/scripts/triple_labels/train_ptbxl.py" \
  "${common_train_args[@]}" \
  --output_dir "$NO_TOKEN_DIR" \
  --epochs 25 \
  --lr 0.0001 \
  --weight_decay 0.01 \
  --cosine_tmax 10 \
  --patience 8 \
  --seed 9042 \
  --init_ckpt "$NO_TOKEN_INIT"
eval_model "$NO_TOKEN_DIR"

echo "[run] center-token hard-label pretrain -> real2000 fine-tune -> $CENTER_TOKEN_DIR"
test -f "$CENTER_TOKEN_INIT"
"$PYTHON" "$ROOT/scripts/triple_labels/train_ptbxl.py" \
  "${common_train_args[@]}" \
  --output_dir "$CENTER_TOKEN_DIR" \
  --epochs 25 \
  --lr 0.0001 \
  --weight_decay 0.01 \
  --cosine_tmax 10 \
  --patience 8 \
  --seed 9042 \
  --init_ckpt "$CENTER_TOKEN_INIT"
eval_model "$CENTER_TOKEN_DIR"

echo "[done] low-sample three-method reproduction complete"
echo "  real2000:     $REAL_DIR"
echo "  no-token:     $NO_TOKEN_DIR"
echo "  center-token: $CENTER_TOKEN_DIR"
