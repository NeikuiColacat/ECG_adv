#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("$PYTHON")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python)
fi

DATA_ROOT="${ECG_ADV_DATA_ROOT:-${HOME}/autodl-tmp}"
PTBXL_ROOT="${ECG_ADV_PTBXL_ROOT:-${DATA_ROOT}/ptbxl}"
GRAD_ROOT="${ECG_ADV_GRAD_ROOT:-${DATA_ROOT}/graduate_project}"
TRIPLE_ROOT="${ECG_ADV_TRIPLE_ROOT:-${DATA_ROOT}/triple_labels}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${GRAD_ROOT}/table_6_8_ablation_${RUN_TAG}}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-50}"

DATA_PATH="${DATA_PATH:-${PTBXL_ROOT}/raw100.npy}"
CSV_PATH="${CSV_PATH:-${PTBXL_ROOT}/ptbxl_database.csv}"
CACHE_PATH="${CACHE_PATH:-${TRIPLE_ROOT}/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy}"
SPLIT_JSON="${SPLIT_JSON:-${GRAD_ROOT}/splits/ptbxl_super5_seed42_train2000_val2000.json}"

CENTER_TOKEN_SYNTH_NPZ="${CENTER_TOKEN_SYNTH_NPZ:-${DATA_ROOT}/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_token_scale_large_20260504/target_token_s05/ningbo/gated/gated_samples.npz}"
NO_TOKEN_SYNTH_NPZ="${NO_TOKEN_SYNTH_NPZ:-${DATA_ROOT}/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_no_token_large_20260504/no_token/ningbo/gated/gated_samples.npz}"
REPORT_TEXT_SYNTH_NPZ="${REPORT_TEXT_SYNTH_NPZ:-${GRAD_ROOT}/method_b_synth_candidates_actual_report_mv4_step2000_seed42/ptbxl/gated_max1000/gated_samples.npz}"
DEFAULT_TEXT_SYNTH_NPZ="${DEFAULT_TEXT_SYNTH_NPZ:-${DATA_ROOT}/ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced/ningbo/gated/gated_samples.npz}"
CENTER_TOKEN_INIT="${CENTER_TOKEN_INIT:-${GRAD_ROOT}/self_distill_v2_e18_v46_class_oracle_hardlabel_r10_seed42_auroc/best_model.pt}"

export TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DATA_ROOT}/cache}"

mkdir -p "$OUT_ROOT"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[missing] $path" >&2
    exit 1
  fi
}

run_train() {
  local output_dir="$1"
  shift
  "${PYTHON_CMD[@]}" "$ROOT/scripts/triple_labels/train_ptbxl.py" \
    --scheme super5 \
    --data_path "$DATA_PATH" \
    --csv_path "$CSV_PATH" \
    --cache_path "$CACHE_PATH" \
    --split_json "$SPLIT_JSON" \
    --preprocess_mode minimal_resample \
    --norm_mode per_sample_global \
    --device "$DEVICE" \
    --crop_len 1000 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_metric auroc \
    --epochs "$EPOCHS" \
    --lr 0.01 \
    --weight_decay 0.01 \
    --cosine_tmax 15 \
    --patience 10 \
    --output_dir "$output_dir" \
    "$@"
}

require_file "$DATA_PATH"
require_file "$CSV_PATH"
require_file "$CACHE_PATH"
require_file "$SPLIT_JSON"
require_file "$CENTER_TOKEN_SYNTH_NPZ"
require_file "$NO_TOKEN_SYNTH_NPZ"
require_file "$REPORT_TEXT_SYNTH_NPZ"
require_file "$DEFAULT_TEXT_SYNTH_NPZ"

echo "[table6.8] output root: $OUT_ROOT"

run_train "$OUT_ROOT/center_token_joint_seed42" \
  --synth_npz "$CENTER_TOKEN_SYNTH_NPZ" \
  --synth_ratio 0.25 \
  --seed 42

run_train "$OUT_ROOT/report_text_joint_seed42" \
  --synth_npz "$REPORT_TEXT_SYNTH_NPZ" \
  --synth_ratio 0.25 \
  --seed 42

run_train "$OUT_ROOT/default_text_joint_seed42" \
  --synth_npz "$DEFAULT_TEXT_SYNTH_NPZ" \
  --synth_ratio 0.25 \
  --seed 42

run_train "$OUT_ROOT/no_token_synthetic_only_seed42" \
  --synth_npz "$NO_TOKEN_SYNTH_NPZ" \
  --synthetic_only \
  --seed 42

require_file "$CENTER_TOKEN_INIT"
run_train "$OUT_ROOT/center_token_pretrain_realfine_seed42" \
  --init_ckpt "$CENTER_TOKEN_INIT" \
  --lr 0.0001 \
  --cosine_tmax 10 \
  --patience 8 \
  --epochs 25 \
  --seed 9042

echo "[done] Table 6.8 ablation commands complete: $OUT_ROOT"
