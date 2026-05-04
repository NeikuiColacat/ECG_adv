#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/ECGTwin/bin/python}"
ROOT="${ROOT:-/root/ECG_adv_Gen}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503}"
PN_CACHE="${PN_CACHE:-/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1}"
PN_TOKEN="${PN_TOKEN:-/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v40_pn2021_big4_direct_mv4_actual_steps2500/prompt_token_bank.pt}"
PTB_CACHE="${PTB_CACHE:-/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/cache_v1}"
PTB_TOKEN="${PTB_TOKEN:-/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_k500_direct_mv4_actual_steps2500_20260503/prompt_token_bank.pt}"
PROMPT_BANK="${PROMPT_BANK:-/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt}"
STYLE_CLASSIFIER_DIR="${STYLE_CLASSIFIER_DIR:-/root/autodl-tmp/per_center_style_classifier_full10s_max3000}"
NOLEAK_MMAP_ROOT="${NOLEAK_MMAP_ROOT:-/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap}"
NOLEAK_FEATURE_CKPT="${NOLEAK_FEATURE_CKPT:-/root/autodl-tmp/triple_labels/super5/best_model.pt}"
NOLEAK_FEATURE_CROP_LEN="${NOLEAK_FEATURE_CROP_LEN:-250}"
NOLEAK_FEATURE_BATCH_SIZE="${NOLEAK_FEATURE_BATCH_SIZE:-256}"
GEN_VICTIM_CKPT="${GEN_VICTIM_CKPT:-/root/autodl-tmp/triple_labels/super5/best_model.pt}"
GEN_VICTIM_CROP_LEN="${GEN_VICTIM_CROP_LEN:-250}"
GEN_TOKEN_SCALE="${GEN_TOKEN_SCALE:-1.0}"
N_PER_CLASS="${N_PER_CLASS:-10}"
DDPM_STEPS="${DDPM_STEPS:-25}"

export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"
mkdir -p "${OUT_ROOT}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${OUT_ROOT}/logs"

cd "${ROOT}"

wrong_center_for() {
  case "$1" in
    ningbo) echo "georgia" ;;
    georgia) echo "ningbo" ;;
    chapman_shaoxing) echo "ningbo" ;;
    cpsc_2018) echo "georgia" ;;
    *) echo "ningbo" ;;
  esac
}

run_gen() {
  local center="$1"
  local arm="$2"
  shift 2
  local out="${OUT_ROOT}/${arm}"
  local log="${OUT_ROOT}/logs/${center}_${arm}.log"
  if [[ -s "${out}/${center}/samples.npz" ]]; then
    echo "$(date -Is) [skip-generate] ${center} ${arm}"
    return
  fi
  echo "$(date -Is) [generate] ${center} ${arm}"
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/generate_center_prompt_token_synth.py \
    --center "${center}" \
    --classes NORM MI STTC \
    --cache_root "${PN_CACHE}" \
    --prompt_bank "${PROMPT_BANK}" \
    --out_dir "${out}" \
    --K 500 \
    --selection_seed 42 \
    --n_per_class "${N_PER_CLASS}" \
    --steps "${DDPM_STEPS}" \
    --seed 20260503 \
    --device cuda:0 \
    --victim_ckpt "${GEN_VICTIM_CKPT}" \
    --victim_crop_len "${GEN_VICTIM_CROP_LEN}" \
    --token_scale "${GEN_TOKEN_SCALE}" \
    --arm "${arm}" \
    "$@" 2>&1 | tee "${log}"
}

run_gate() {
  local center="$1"
  local arm="$2"
  local in_dir="${OUT_ROOT}/${arm}/${center}"
  local gate_dir="${in_dir}/gated"
  local log="${OUT_ROOT}/logs/${center}_${arm}_gate.log"
  if [[ -s "${gate_dir}/gated_samples.npz" ]]; then
    echo "$(date -Is) [skip-gate] ${center} ${arm}"
    return
  fi
  if [[ ! -s "${in_dir}/samples.npz" ]]; then
    echo "$(date -Is) [skip-gate-missing] ${in_dir}/samples.npz"
    return
  fi
  echo "$(date -Is) [gate] ${center} ${arm}"
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/gate_prompt_token_synth.py \
    --input_dir "${in_dir}" \
    --out_dir "${gate_dir}" \
    --classes NORM MI STTC \
    --min_target_prob 0.30 \
    --cache_root "${PN_CACHE}" \
    --K 500 \
    --selection_seed 42 2>&1 | tee "${log}" || true
}

if [[ -n "${CENTERS:-}" ]]; then
  # shellcheck disable=SC2206
  CENTERS_ARR=(${CENTERS})
else
  CENTERS_ARR=(ningbo chapman_shaoxing cpsc_2018 georgia)
fi

for center in "${CENTERS_ARR[@]}"; do
  wrong="$(wrong_center_for "${center}")"
  run_gen "${center}" "vanilla" --token_bank "${PN_TOKEN}" --no_token
  run_gen "${center}" "target_token" --token_bank "${PN_TOKEN}" --token_center "${center}"
  run_gen "${center}" "wrong_center_token" --token_bank "${PN_TOKEN}" --token_center "${wrong}"
  run_gen "${center}" "ptbxl_source_token" --token_bank "${PTB_TOKEN}" --token_center ptbxl_source
  for arm in vanilla target_token wrong_center_token ptbxl_source_token; do
    run_gate "${center}" "${arm}"
  done
done

mapfile -t raw_dirs < <(find "${OUT_ROOT}" -mindepth 2 -maxdepth 2 -type d | sort)
"${PYTHON_BIN}" -u scripts/ecgtwin_gen/score_prompt_token_style_probe.py \
  --input_dirs "${raw_dirs[@]}" \
  --npz_name samples.npz \
  --output_dir "${OUT_ROOT}/style_probe" \
  --device cuda:0 \
  2>&1 | tee "${OUT_ROOT}/logs/style_probe_raw.log"

mapfile -t gated_dirs < <(find "${OUT_ROOT}" -mindepth 3 -maxdepth 3 -type d -name gated | sort)
if [[ "${#gated_dirs[@]}" -gt 0 ]]; then
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/score_prompt_token_style_probe.py \
    --input_dirs "${gated_dirs[@]}" \
    --npz_name gated_samples.npz \
    --output_dir "${OUT_ROOT}/style_probe" \
    --device cuda:0 \
    2>&1 | tee "${OUT_ROOT}/logs/style_probe_gated.log" || true
fi

if [[ -s "${STYLE_CLASSIFIER_DIR}/best_model.pt" ]]; then
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/score_prompt_token_style_probe.py \
    --classifier_dir "${STYLE_CLASSIFIER_DIR}" \
    --input_dirs "${raw_dirs[@]}" \
    --npz_name samples.npz \
    --output_dir "${OUT_ROOT}/style_probe_full10s" \
    --crop_len 1000 \
    --device cuda:0 \
    2>&1 | tee "${OUT_ROOT}/logs/style_probe_full10s_raw.log"

  if [[ "${#gated_dirs[@]}" -gt 0 ]]; then
    "${PYTHON_BIN}" -u scripts/ecgtwin_gen/score_prompt_token_style_probe.py \
      --classifier_dir "${STYLE_CLASSIFIER_DIR}" \
      --input_dirs "${gated_dirs[@]}" \
      --npz_name gated_samples.npz \
      --output_dir "${OUT_ROOT}/style_probe_full10s" \
      --crop_len 1000 \
      --device cuda:0 \
      2>&1 | tee "${OUT_ROOT}/logs/style_probe_full10s_gated.log" || true
  fi
fi

"${PYTHON_BIN}" -u scripts/ecgtwin_gen/run_prompt_token_c2st_proxy.py \
  --input_dirs "${raw_dirs[@]}" \
  --npz_name samples.npz \
  --output_csv "${OUT_ROOT}/c2st_proxy/c2st_raw.csv" \
  --repeats 5 \
  --stride 10 \
  2>&1 | tee "${OUT_ROOT}/logs/c2st_raw.log"

if [[ "${#gated_dirs[@]}" -gt 0 ]]; then
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/run_prompt_token_c2st_proxy.py \
    --input_dirs "${gated_dirs[@]}" \
    --npz_name gated_samples.npz \
    --output_csv "${OUT_ROOT}/c2st_proxy/c2st_gated.csv" \
    --repeats 5 \
    --stride 10 \
    2>&1 | tee "${OUT_ROOT}/logs/c2st_gated.log" || true
fi

"${PYTHON_BIN}" -u scripts/ecgtwin_gen/run_prompt_token_noleak_style_validation.py \
  --mmap_root "${NOLEAK_MMAP_ROOT}" \
  --input_dirs "${raw_dirs[@]}" \
  --npz_name samples.npz \
  --output_dir "${OUT_ROOT}/noleak_validation_raw" \
  --classes NORM MI STTC \
  --feature_stride 10 \
  --max_train_per_class 600 \
  --max_eval_per_class 300 \
  --max_c2st_real_per_class 80 \
  --c2st_repeats 5 \
  2>&1 | tee "${OUT_ROOT}/logs/noleak_validation_raw.log"

if [[ "${#gated_dirs[@]}" -gt 0 ]]; then
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/run_prompt_token_noleak_style_validation.py \
    --mmap_root "${NOLEAK_MMAP_ROOT}" \
    --input_dirs "${gated_dirs[@]}" \
    --npz_name gated_samples.npz \
    --output_dir "${OUT_ROOT}/noleak_validation_gated" \
    --classes NORM MI STTC \
    --feature_stride 10 \
    --max_train_per_class 600 \
    --max_eval_per_class 300 \
    --max_c2st_real_per_class 80 \
    --c2st_repeats 5 \
    2>&1 | tee "${OUT_ROOT}/logs/noleak_validation_gated.log" || true
fi

"${PYTHON_BIN}" -u scripts/ecgtwin_gen/run_prompt_token_noleak_style_validation.py \
  --mmap_root "${NOLEAK_MMAP_ROOT}" \
  --input_dirs "${raw_dirs[@]}" \
  --npz_name samples.npz \
  --output_dir "${OUT_ROOT}/noleak_validation_raw_effnet" \
  --classes NORM MI STTC \
  --feature_backend efficientnet \
  --feature_ckpt "${NOLEAK_FEATURE_CKPT}" \
  --feature_device cuda:0 \
  --feature_batch_size "${NOLEAK_FEATURE_BATCH_SIZE}" \
  --feature_crop_len "${NOLEAK_FEATURE_CROP_LEN}" \
  --max_train_per_class 600 \
  --max_eval_per_class 300 \
  --max_c2st_real_per_class 80 \
  --c2st_repeats 5 \
  2>&1 | tee "${OUT_ROOT}/logs/noleak_validation_raw_effnet.log"

if [[ "${#gated_dirs[@]}" -gt 0 ]]; then
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/run_prompt_token_noleak_style_validation.py \
    --mmap_root "${NOLEAK_MMAP_ROOT}" \
    --input_dirs "${gated_dirs[@]}" \
    --npz_name gated_samples.npz \
    --output_dir "${OUT_ROOT}/noleak_validation_gated_effnet" \
    --classes NORM MI STTC \
    --feature_backend efficientnet \
    --feature_ckpt "${NOLEAK_FEATURE_CKPT}" \
    --feature_device cuda:0 \
    --feature_batch_size "${NOLEAK_FEATURE_BATCH_SIZE}" \
    --feature_crop_len "${NOLEAK_FEATURE_CROP_LEN}" \
    --max_train_per_class 600 \
    --max_eval_per_class 300 \
    --max_c2st_real_per_class 80 \
    --c2st_repeats 5 \
    2>&1 | tee "${OUT_ROOT}/logs/noleak_validation_gated_effnet.log" || true
fi

echo "$(date -Is) [done] center-token effectiveness pilot: ${OUT_ROOT}"
