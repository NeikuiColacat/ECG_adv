#!/usr/bin/env bash
set -euo pipefail

# Rebuild the synthetic-pretrain initialization checkpoints used by the
# Table 6.6/6.7 real2000 fine-tuning runs. The historical artifact names are
# preserved because downstream scripts refer to these directories.

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

DATA_PATH="${DATA_PATH:-${PTBXL_ROOT}/raw100.npy}"
CSV_PATH="${CSV_PATH:-${PTBXL_ROOT}/ptbxl_database.csv}"
CACHE_PATH="${CACHE_PATH:-${TRIPLE_ROOT}/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy}"
SPLIT_JSON="${SPLIT_JSON:-${GRAD_ROOT}/splits/ptbxl_super5_seed42_train2000_val2000.json}"

NO_TOKEN_PRETRAIN_SYNTH_NPZ="${NO_TOKEN_PRETRAIN_SYNTH_NPZ:-${DATA_ROOT}/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_no_token_large_20260504/no_token/ningbo/gated/gated_samples.npz}"
CENTER_TOKEN_PRETRAIN_SYNTH_NPZ="${CENTER_TOKEN_PRETRAIN_SYNTH_NPZ:-${DATA_ROOT}/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_token_scale_large_20260504/target_token_s05/ningbo/gated/gated_samples.npz}"

NO_TOKEN_OUT="${NO_TOKEN_OUT:-${GRAD_ROOT}/self_distill_v2_e21_v46_no_token_hardlabel_r10_seed8042_auroc}"
CENTER_TOKEN_OUT="${CENTER_TOKEN_OUT:-${GRAD_ROOT}/self_distill_v2_e18_v46_class_oracle_hardlabel_r10_seed42_auroc}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-10}"

export TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DATA_ROOT}/cache}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[missing] $path" >&2
    exit 1
  fi
}

run_synth_pretrain() {
  local output_dir="$1"
  local synth_npz="$2"
  local seed="$3"
  echo "[run] synthetic pretrain -> $output_dir"
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
    --epochs "$PRETRAIN_EPOCHS" \
    --lr 0.01 \
    --weight_decay 0.01 \
    --cosine_tmax 15 \
    --patience "$PRETRAIN_PATIENCE" \
    --output_dir "$output_dir" \
    --synth_npz "$synth_npz" \
    --synthetic_only \
    --seed "$seed"
}

require_file "$DATA_PATH"
require_file "$CSV_PATH"
require_file "$CACHE_PATH"
require_file "$SPLIT_JSON"
require_file "$NO_TOKEN_PRETRAIN_SYNTH_NPZ"
require_file "$CENTER_TOKEN_PRETRAIN_SYNTH_NPZ"

run_synth_pretrain "$NO_TOKEN_OUT" "$NO_TOKEN_PRETRAIN_SYNTH_NPZ" 8042
run_synth_pretrain "$CENTER_TOKEN_OUT" "$CENTER_TOKEN_PRETRAIN_SYNTH_NPZ" 42

echo "[done] synthetic-pretrain init checkpoints:"
echo "  no-token:     ${NO_TOKEN_OUT}/best_model.pt"
echo "  center-token: ${CENTER_TOKEN_OUT}/best_model.pt"
