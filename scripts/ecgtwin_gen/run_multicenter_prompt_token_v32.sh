#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/ECGTwin/bin/python}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"

TOKEN_BANK="${TOKEN_BANK:-/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v8_seed42_direct_mv4_steps2500/prompt_token_bank.pt}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v32_seed42_direct_mv4_multicenter_q50}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/ecgtwin_prompt_token_super5/logs}"
mkdir -p "${LOG_DIR}"

CENTERS=(chapman_shaoxing cpsc_2018 georgia)
CLASSES=(NORM MI STTC)

for center in "${CENTERS[@]}"; do
  center_dir="${OUT_ROOT}/${center}"
  gate_dir="${center_dir}/gated"
  gen_log="${LOG_DIR}/v32_${center}_generate.log"
  gate_log="${LOG_DIR}/v32_${center}_gate.log"

  if [[ ! -s "${center_dir}/samples.npz" ]]; then
    echo "$(date -Is) [generate] ${center}"
    "${PYTHON_BIN}" -u scripts/ecgtwin_gen/generate_center_prompt_token_synth.py \
      --center "${center}" \
      --classes "${CLASSES[@]}" \
      --token_bank "${TOKEN_BANK}" \
      --out_dir "${OUT_ROOT}" \
      --n_per_class 50 \
      --steps 25 \
      --selection_seed 42 \
      --seed 42 \
      --device cuda:0 2>&1 | tee "${gen_log}"
  else
    echo "$(date -Is) [skip-generate] existing ${center_dir}/samples.npz"
  fi

  if [[ ! -s "${gate_dir}/gated_samples.latent.npz" ]]; then
    echo "$(date -Is) [gate] ${center}"
    "${PYTHON_BIN}" -u scripts/ecgtwin_gen/gate_prompt_token_synth.py \
      --input_dir "${center_dir}" \
      --out_dir "${gate_dir}" \
      --classes "${CLASSES[@]}" \
      --min_target_prob 0.30 \
      --K 500 \
      --selection_seed 42 2>&1 | tee "${gate_log}"
  else
    echo "$(date -Is) [skip-gate] existing ${gate_dir}/gated_samples.latent.npz"
  fi
done

echo "$(date -Is) [done] multicenter prompt-token v32 generation + gating complete"
