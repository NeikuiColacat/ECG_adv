#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/ECGTwin/bin/python}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"

TOKEN_BANK="${TOKEN_BANK:-/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v8_seed42_direct_mv4_steps2500/prompt_token_bank.pt}"
BASE_ROOT="${BASE_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v34_seed42_direct_mv4_multicenter_merged}"
BOOST_ROOT="${BOOST_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v35_seed42_direct_mv4_georgia_sttc_boost2}"
MERGED_ROOT="${MERGED_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v36_seed42_direct_mv4_multicenter_merged2}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/ecgtwin_prompt_token_super5/logs}"
mkdir -p "${LOG_DIR}"

center="georgia"
center_dir="${BOOST_ROOT}/${center}"
gate_dir="${center_dir}/gated"

if [[ ! -s "${center_dir}/samples.npz" ]]; then
  echo "$(date -Is) [generate] ${center} STTC boost2"
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/generate_center_prompt_token_synth.py \
    --center "${center}" \
    --classes STTC \
    --token_bank "${TOKEN_BANK}" \
    --out_dir "${BOOST_ROOT}" \
    --n_per_class 80 \
    --steps 25 \
    --selection_seed 42 \
    --seed 2042 \
    --device cuda:0 2>&1 | tee "${LOG_DIR}/v35_${center}_sttc_generate.log"
else
  echo "$(date -Is) [skip-generate] existing ${center_dir}/samples.npz"
fi

if [[ ! -s "${gate_dir}/gated_samples.latent.npz" ]]; then
  echo "$(date -Is) [gate] ${center} STTC boost2"
  "${PYTHON_BIN}" -u scripts/ecgtwin_gen/gate_prompt_token_synth.py \
    --input_dir "${center_dir}" \
    --out_dir "${gate_dir}" \
    --classes STTC \
    --min_target_prob 0.30 \
    --K 500 \
    --selection_seed 42 2>&1 | tee "${LOG_DIR}/v35_${center}_sttc_gate.log"
else
  echo "$(date -Is) [skip-gate] existing ${gate_dir}/gated_samples.latent.npz"
fi

for c in chapman_shaoxing cpsc_2018; do
  mkdir -p "${MERGED_ROOT}/${c}"
  if [[ ! -e "${MERGED_ROOT}/${c}/gated" ]]; then
    ln -s "${BASE_ROOT}/${c}/gated" "${MERGED_ROOT}/${c}/gated"
  fi
done

merged_dir="${MERGED_ROOT}/${center}/gated"
if [[ ! -s "${merged_dir}/gated_samples.latent.npz" ]]; then
  echo "$(date -Is) [merge] ${center}"
  "${PYTHON_BIN}" scripts/ecgtwin_gen/merge_gated_prompt_token_pools.py \
    --input_dirs "${BASE_ROOT}/${center}/gated" "${gate_dir}" \
    --out_dir "${merged_dir}" \
    --tag "v36_${center}_v34_plus_v35"
else
  echo "$(date -Is) [skip-merge] existing ${merged_dir}/gated_samples.latent.npz"
fi

echo "$(date -Is) [done] georgia STTC boost2 complete"
