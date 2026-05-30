#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/ECGTwin/bin/python}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"

CENTERS=(ningbo chapman_shaoxing cpsc_2018 georgia)
CORRUPTIONS=(powerline_noise emg_noise baseline_wander baseline_shift random_leads_masking)
SEVERITIES=(1 2 3 4 5)

MODEL_DIRS=(
  "/root/autodl-tmp/latent_hull_super5_pilot/nin_M5_lambda025_ep20"
  "/root/autodl-tmp/latent_hull_super5_pilot/extra_M5_lambda025_ep20"
  "/root/autodl-tmp/latent_hull_super5_pilot/geo_M20_lambda025_ep20"
)

for model_dir in "${MODEL_DIRS[@]}"; do
  clean_json="${model_dir}/eval_result_v3_super5_normsuppress.json"
  out_json="${model_dir}/eval_pn2021_c_stream_v1_selfclean_all4.json"
  log_path="${model_dir}/eval_pn2021_c_stream_v1_selfclean_all4.log"

  if [[ -s "${out_json}" ]]; then
    echo "$(date -Is) [skip] existing ${out_json}"
    continue
  fi
  if [[ ! -s "${model_dir}/best_model.pt" ]]; then
    echo "$(date -Is) [error] missing best_model.pt under ${model_dir}" >&2
    exit 1
  fi
  if [[ ! -s "${clean_json}" ]]; then
    echo "$(date -Is) [error] missing clean eval ${clean_json}" >&2
    exit 1
  fi

  echo "$(date -Is) [run] PN2021-C ${model_dir}"
  "${PYTHON_BIN}" -u scripts/triple_labels/eval_pn2021_corruptions.py \
    --scheme super5 \
    --mode stream \
    --model_dir "${model_dir}" \
    --clean_eval_json "${clean_json}" \
    --centers "${CENTERS[@]}" \
    --corruptions "${CORRUPTIONS[@]}" \
    --severities "${SEVERITIES[@]}" \
    --batch_size 512 \
    --num_workers 8 \
    --device cuda \
    --output_path "${out_json}" 2>&1 | tee "${log_path}"
done

"${PYTHON_BIN}" scripts/triple_labels/export_pn2021_c_summaries.py \
  --out_dir /root/autodl-tmp/triple_labels
"${PYTHON_BIN}" scripts/triple_labels/export_pn2021_c_candidate_coverage.py \
  --out_path /root/autodl-tmp/triple_labels/pn2021_c_candidate_coverage.csv

echo "$(date -Is) [done] selected PN2021-C queue complete"
