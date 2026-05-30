#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${ECG_ADV_DATA_ROOT:-${HOME}/autodl-tmp}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("${PYTHON}")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python)
fi
ROOT="${ROOT:-${DATA_ROOT}/ecgtwin_author_repro/full_dit_main}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AMP_ARGS="${AMP_ARGS:---amp --amp_dtype bf16 --matmul_precision high}"
DATALOADER_ARGS="${DATALOADER_ARGS:---pin_memory --persistent_workers --prefetch_factor 2}"

mkdir -p "${ROOT}"

echo "$(date -Is) [pipeline] starting IBE stage" | tee -a "${ROOT}/pipeline.log"
"${PYTHON_CMD[@]}" -u scripts/ecgtwin_author_repro/train_ibe_repro.py \
  --output_dir "${ROOT}/ibe_stage1" \
  --epochs 40 \
  --batch_size 65536 \
  --mini_batch_size 512 \
  --val_batch_size 256 \
  --num_workers "${NUM_WORKERS}" \
  --max_val_batches 64 \
  --log_every 1 \
  ${DATALOADER_ARGS} \
  ${AMP_ARGS} \
  --device cuda 2>&1 | tee -a "${ROOT}/pipeline.log"

echo "$(date -Is) [pipeline] starting DiT stage" | tee -a "${ROOT}/pipeline.log"
"${PYTHON_CMD[@]}" -u scripts/ecgtwin_author_repro/train_dit_repro.py \
  --output_dir "${ROOT}/dit_stage2" \
  --ibe_path "${ROOT}/ibe_stage1/checkpoints/IBE_best.pth" \
  --epochs 30 \
  --batch_size 512 \
  --val_batch_size 512 \
  --num_workers "${NUM_WORKERS}" \
  --max_val_batches 64 \
  --log_every 20 \
  ${DATALOADER_ARGS} \
  ${AMP_ARGS} \
  --device cuda 2>&1 | tee -a "${ROOT}/pipeline.log"

echo "$(date -Is) [pipeline] done" | tee -a "${ROOT}/pipeline.log"
