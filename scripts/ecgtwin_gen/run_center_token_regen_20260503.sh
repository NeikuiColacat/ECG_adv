#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/ECGTwin/bin/python}"
ROOT="${ROOT:-/root/ECG_adv_Gen}"

export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

cd "${ROOT}"

PROMPT_BANK="${PROMPT_BANK:-/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt}"

PN_CACHE="${PN_CACHE:-/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1}"
PN_RUN="${PN_RUN:-/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v40_pn2021_big4_direct_mv4_actual_steps2500}"
PN_LOG_DIR="${PN_LOG_DIR:-/root/autodl-tmp/ecgtwin_prompt_token_super5/logs}"

PTB_CACHE="${PTB_CACHE:-/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/cache_v1}"
PTB_RUN="${PTB_RUN:-/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_k500_direct_mv4_actual_steps2500_20260503}"
PTB_LOG_DIR="${PTB_LOG_DIR:-/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/logs}"

DIAG_DIR="${DIAG_DIR:-/root/autodl-tmp/ecgtwin_prompt_token_super5/diagnostics/regen_20260503}"

mkdir -p "${PN_LOG_DIR}" "${PTB_LOG_DIR}" "${DIAG_DIR}"

echo "$(date -Is) [stage 1] train PN2021 big4 direct MV4 prompt tokens"
"${PYTHON_BIN}" -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --cache_root "${PN_CACHE}" \
  --prompt_bank "${PROMPT_BANK}" \
  --save_dir "${PN_RUN}" \
  --K 500 \
  --seed 42 \
  --total_steps 2500 \
  --batch_size 16 \
  --num_workers 4 \
  --sample_strategy center_class_balanced \
  --amp_dtype bf16 \
  --token_mode direct \
  --n_token_vectors 4 \
  --token_orth_weight 1e-4 \
  --ref_text_mode actual_report \
  --log_every 50 \
  --save_every 500 \
  2>&1 | tee "${PN_LOG_DIR}/v40_pn2021_big4_direct_mv4_actual_steps2500.log"

echo "$(date -Is) [stage 2] prepare PTB-XL source K500 balanced selection"
"${PYTHON_BIN}" -u scripts/ecgtwin_gen/build_ptbxl_prompt_token_cache.py \
  --out_root "${PTB_CACHE}" \
  --center ptbxl_source \
  --folds 1 2 3 4 5 6 7 8 \
  --per_class 100 \
  --seed 42 \
  2>&1 | tee "${PTB_LOG_DIR}/07_build_ptbxl_source_k500_selection.log"

echo "$(date -Is) [stage 3] train PTB-XL source direct MV4 prompt tokens"
"${PYTHON_BIN}" -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ptbxl_source \
  --cache_root "${PTB_CACHE}" \
  --prompt_bank "${PROMPT_BANK}" \
  --save_dir "${PTB_RUN}" \
  --K 500 \
  --seed 42 \
  --total_steps 2500 \
  --batch_size 16 \
  --num_workers 4 \
  --sample_strategy class_balanced \
  --amp_dtype bf16 \
  --token_mode direct \
  --n_token_vectors 4 \
  --token_orth_weight 1e-4 \
  --ref_text_mode actual_report \
  --log_every 50 \
  --save_every 500 \
  2>&1 | tee "${PTB_LOG_DIR}/08_train_ptbxl_source_k500_direct_mv4_actual_steps2500.log"

echo "$(date -Is) [stage 4] export lightweight token diagnostics"
"${PYTHON_BIN}" -u scripts/ecgtwin_gen/export_prompt_token_diagnostics.py \
  --patterns \
    "${PN_RUN}/prompt_token_bank*.pt" \
    "${PTB_RUN}/prompt_token_bank*.pt" \
  --out_dir "${DIAG_DIR}" \
  --root /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs \
  2>&1 | tee "${DIAG_DIR}/export.log"

echo "$(date -Is) [done] regenerated prompt-token banks"
echo "PN2021: ${PN_RUN}/prompt_token_bank.pt"
echo "PTBXL:  ${PTB_RUN}/prompt_token_bank.pt"
echo "DIAG:   ${DIAG_DIR}"
