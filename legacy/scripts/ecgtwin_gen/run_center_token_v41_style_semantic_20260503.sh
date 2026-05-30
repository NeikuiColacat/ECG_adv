#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/ECGTwin/bin/python}"
ROOT="${ROOT:-/root/ECG_adv_Gen}"
PN_CACHE="${PN_CACHE:-/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1}"
PROMPT_BANK="${PROMPT_BANK:-${PN_CACHE}/text_prompt_bank.pt}"
STYLE_CKPT="${STYLE_CKPT:-/root/autodl-tmp/per_center_style_classifier_full10s_max3000/best_model.pt}"
SEMANTIC_CKPT="${SEMANTIC_CKPT:-/root/autodl-tmp/triple_labels/super5/best_model.pt}"

TOKEN_RUN="${TOKEN_RUN:-/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v41_style_semantic_direct_mv4_actual_steps1500_20260503}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v41_style_semantic_20260503}"
DIAG_DIR="${DIAG_DIR:-/root/autodl-tmp/ecgtwin_prompt_token_super5/diagnostics/v41_style_semantic_20260503}"

TOTAL_STEPS="${TOTAL_STEPS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STYLE_LOSS_WEIGHT="${STYLE_LOSS_WEIGHT:-0.01}"
SEMANTIC_LOSS_WEIGHT="${SEMANTIC_LOSS_WEIGHT:-0.05}"
AUX_TIMESTEP_MAX="${AUX_TIMESTEP_MAX:-250}"
N_PER_CLASS="${N_PER_CLASS:-8}"
DDPM_STEPS="${DDPM_STEPS:-25}"

export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"
mkdir -p "${TOKEN_RUN}" "${OUT_ROOT}" "${DIAG_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}"

cd "${ROOT}"

if [[ ! -s "${TOKEN_RUN}/prompt_token_bank.pt" ]]; then
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
    --centers ningbo chapman_shaoxing cpsc_2018 georgia \
    --cache_root "${PN_CACHE}" \
    --prompt_bank "${PROMPT_BANK}" \
    --save_dir "${TOKEN_RUN}" \
    --K 500 \
    --seed 42 \
    --total_steps "${TOTAL_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --reg_init_weight 1e-3 \
    --sample_strategy center_class_balanced \
    --amp_dtype bf16 \
    --token_mode direct \
    --n_token_vectors 4 \
    --token_orth_weight 1e-4 \
    --ref_text_mode actual_report \
    --style_ckpt "${STYLE_CKPT}" \
    --style_loss_weight "${STYLE_LOSS_WEIGHT}" \
    --style_crop_len 1000 \
    --semantic_ckpt "${SEMANTIC_CKPT}" \
    --semantic_loss_weight "${SEMANTIC_LOSS_WEIGHT}" \
    --semantic_crop_len 250 \
    --aux_timestep_max "${AUX_TIMESTEP_MAX}" \
    --aux_amp_clamp 6.0 \
    --log_every 50 \
    --save_every 500 \
    --device cuda:0
else
  echo "$(date -Is) [skip-train] existing ${TOKEN_RUN}/prompt_token_bank.pt"
fi

"${PYTHON_BIN}" -u scripts/ecgtwin_gen/export_prompt_token_diagnostics.py \
  --token_banks "${TOKEN_RUN}/prompt_token_bank.pt" \
  --output_dir "${DIAG_DIR}"

PN_TOKEN="${TOKEN_RUN}/prompt_token_bank.pt" \
OUT_ROOT="${OUT_ROOT}" \
N_PER_CLASS="${N_PER_CLASS}" \
DDPM_STEPS="${DDPM_STEPS}" \
bash scripts/ecgtwin_gen/run_center_token_effectiveness_pilot_20260503.sh

echo "$(date -Is) [done] v41 style-semantic center-token pilot: ${OUT_ROOT}"
