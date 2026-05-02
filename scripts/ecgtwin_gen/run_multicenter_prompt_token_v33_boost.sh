#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/ECGTwin/bin/python}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"

TOKEN_BANK="${TOKEN_BANK:-/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v8_seed42_direct_mv4_steps2500/prompt_token_bank.pt}"
BASE_ROOT="${BASE_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v32_seed42_direct_mv4_multicenter_q50}"
BOOST_ROOT="${BOOST_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v33_seed42_direct_mv4_multicenter_boost}"
MERGED_ROOT="${MERGED_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v34_seed42_direct_mv4_multicenter_merged}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/ecgtwin_prompt_token_super5/logs}"
mkdir -p "${LOG_DIR}"

run_boost() {
  local center="$1"
  shift
  local classes=("$@")
  local center_dir="${BOOST_ROOT}/${center}"
  local gate_dir="${center_dir}/gated"
  local gen_log="${LOG_DIR}/v33_${center}_generate.log"
  local gate_log="${LOG_DIR}/v33_${center}_gate.log"

  if [[ ! -s "${center_dir}/samples.npz" ]]; then
    echo "$(date -Is) [boost-generate] ${center}: ${classes[*]}"
    "${PYTHON_BIN}" -u scripts/ecgtwin_gen/generate_center_prompt_token_synth.py \
      --center "${center}" \
      --classes "${classes[@]}" \
      --token_bank "${TOKEN_BANK}" \
      --out_dir "${BOOST_ROOT}" \
      --n_per_class 80 \
      --steps 25 \
      --selection_seed 42 \
      --seed 1042 \
      --device cuda:0 2>&1 | tee "${gen_log}"
  else
    echo "$(date -Is) [skip-boost-generate] existing ${center_dir}/samples.npz"
  fi

  if [[ ! -s "${gate_dir}/gated_samples.latent.npz" ]]; then
    echo "$(date -Is) [boost-gate] ${center}"
    "${PYTHON_BIN}" -u scripts/ecgtwin_gen/gate_prompt_token_synth.py \
      --input_dir "${center_dir}" \
      --out_dir "${gate_dir}" \
      --classes "${classes[@]}" \
      --min_target_prob 0.30 \
      --K 500 \
      --selection_seed 42 2>&1 | tee "${gate_log}"
  else
    echo "$(date -Is) [skip-boost-gate] existing ${gate_dir}/gated_samples.latent.npz"
  fi

  local merged_dir="${MERGED_ROOT}/${center}/gated"
  if [[ ! -s "${merged_dir}/gated_samples.latent.npz" ]]; then
    echo "$(date -Is) [merge] ${center}"
    "${PYTHON_BIN}" scripts/ecgtwin_gen/merge_gated_prompt_token_pools.py \
      --input_dirs "${BASE_ROOT}/${center}/gated" "${gate_dir}" \
      --out_dir "${merged_dir}" \
      --tag "v34_${center}_v32_plus_v33"
  else
    echo "$(date -Is) [skip-merge] existing ${merged_dir}/gated_samples.latent.npz"
  fi
}

run_boost chapman_shaoxing STTC
run_boost cpsc_2018 STTC
run_boost georgia MI STTC

echo "$(date -Is) [done] multicenter prompt-token v33 boost + v34 merge complete"
