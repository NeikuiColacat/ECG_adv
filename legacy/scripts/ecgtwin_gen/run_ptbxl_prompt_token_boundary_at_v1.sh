#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/ECGTwin/bin/python}"
ROOT="${ROOT:-/root/ECG_adv_Gen}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1}"
CACHE_ROOT="${CACHE_ROOT:-${OUT_ROOT}/cache_v1}"
PROMPT_BANK="${PROMPT_BANK:-/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt}"
TOKEN_RUN="${TOKEN_RUN:-${OUT_ROOT}/prompt_token_runs/ptbxl_source_5class_steps2000}"
GEN_ROOT="${GEN_ROOT:-${OUT_ROOT}/generated_pool_v1}"
GATED_DIR="${GATED_DIR:-${GEN_ROOT}/ptbxl_source/gated_all5}"
AT_RUN="${AT_RUN:-${OUT_ROOT}/online_at/ptbxl_source_all5_boundary_p050_060_M10_ep20}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"

mkdir -p "${OUT_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}" /root/autodl-tmp/tmp /root/autodl-tmp/cache
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"

cd "${ROOT}"

echo "[stage 1] Build PTB-XL source prompt-token cache"
"${PYTHON_BIN}" -u scripts/ecgtwin_gen/build_ptbxl_prompt_token_cache.py \
  --out_root "${CACHE_ROOT}" \
  --center ptbxl_source \
  --folds 1 2 3 4 5 6 7 8 \
  --per_class 200 \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/01_build_ptbxl_cache.log"

echo "[stage 2] Train five PTB-XL source-style prompt tokens"
"${PYTHON_BIN}" -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ptbxl_source \
  --cache_root "${CACHE_ROOT}" \
  --prompt_bank "${PROMPT_BANK}" \
  --save_dir "${TOKEN_RUN}" \
  --K 1000 \
  --seed 42 \
  --total_steps 2000 \
  --batch_size 16 \
  --num_workers 4 \
  --sample_strategy class_balanced \
  --amp_dtype bf16 \
  --token_mode direct \
  --n_token_vectors 1 \
  --token_repeat 1 \
  --log_every 50 \
  --save_every 500 \
  2>&1 | tee "${LOG_DIR}/02_train_ptbxl_prompt_tokens.log"

echo "[stage 3] Generate five-class PTB-XL-style prompt-token ECGTwin pool"
"${PYTHON_BIN}" -u scripts/ecgtwin_gen/generate_center_prompt_token_synth.py \
  --center ptbxl_source \
  --classes CD HYP MI NORM STTC \
  --cache_root "${CACHE_ROOT}" \
  --token_bank "${TOKEN_RUN}/prompt_token_bank.pt" \
  --prompt_bank "${PROMPT_BANK}" \
  --out_dir "${GEN_ROOT}" \
  --K 1000 \
  --selection_seed 42 \
  --n_per_class "${N_PER_CLASS:-80}" \
  --steps "${DDPM_STEPS:-25}" \
  --seed 20260502 \
  2>&1 | tee "${LOG_DIR}/03_generate_ptbxl_prompt_token_pool.log"

echo "[stage 4] Gate generated pool"
"${PYTHON_BIN}" -u scripts/ecgtwin_gen/gate_prompt_token_synth.py \
  --input_dir "${GEN_ROOT}/ptbxl_source" \
  --out_dir "${GATED_DIR}" \
  --classes CD HYP MI NORM STTC \
  --min_target_prob 0.30 \
  --cache_root "${CACHE_ROOT}" \
  --K 1000 \
  --selection_seed 42 \
  --allow_hyp_cd_trust \
  2>&1 | tee "${LOG_DIR}/04_gate_ptbxl_prompt_token_pool.log"

echo "[stage 5] Boundary-confidence Latent-Hull online AT"
"${PYTHON_BIN}" -u scripts/pgd_cross_center/synth_online_at_super5.py \
  --center_name ptbxl_source \
  --synth_npz "${GATED_DIR}/gated_samples.latent.npz" \
  --class_trust "${GATED_DIR}/gated_samples.class_trust.json" \
  --ref_meta_json "${GATED_DIR}/gated_samples.ref_meta.json" \
  --output_dir "${AT_RUN}" \
  --attack_mode latent_hull \
  --classes_in_scope CD HYP MI NORM STTC \
  --allow_hyp_cd_trust \
  --boundary_prob_min 0.50 \
  --boundary_prob_max 0.60 \
  --hull_M 10 \
  --hull_lambda 0.25 \
  --hull_steps 5 \
  --hull_lr 0.3 \
  --hull_label_mode primary \
  --K_anchor 150 \
  --pgd_batch 32 \
  --n_epochs 20 \
  --eval_every 3 \
  --patience 9 \
  --lr 2e-5 \
  --weight_decay 1e-4 \
  --adv_weight 0.10 \
  --roundtrip_weight 0.5 \
  --roundtrip_anchor_n 1500 \
  --anchor_lambda 0.10 \
  --ewa_decay 0.999 \
  --batch_size 128 \
  --num_workers 6 \
  --quick_eval_n_per_center 1000 \
  --asr_low_threshold 0.0 \
  --device cuda:0 \
  --seed 20260502 \
  2>&1 | tee "${LOG_DIR}/05_online_at_boundary.log"

echo "[stage 6] Full PTB-XL fold10 + PN2021 v3 evaluation"
"${PYTHON_BIN}" -u scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir "${AT_RUN}" \
  --skip_mimic \
  --num_workers 4 \
  --output_path "${AT_RUN}/eval_result_v3_super5_normsuppress.json" \
  2>&1 | tee "${LOG_DIR}/06_eval_crosscenter.log"

echo "[done] ${OUT_ROOT}"
