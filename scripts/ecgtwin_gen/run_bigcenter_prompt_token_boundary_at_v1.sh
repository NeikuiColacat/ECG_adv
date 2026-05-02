#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/ECGTwin/bin/python}"
ROOT="${ROOT:-/root/ECG_adv_Gen}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/ecgtwin_prompt_token_super5/boundary_at_bigcenters_v1}"
INIT_CKPT="${INIT_CKPT:-/root/autodl-tmp/triple_labels/super5/best_model.pt}"

export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"
mkdir -p "${OUT_ROOT}" "${TMPDIR}" "${XDG_CACHE_HOME}"

cd "${ROOT}"

run_one() {
  local center="$1"
  local tag="$2"
  local pool_dir="$3"
  shift 3
  local classes=("$@")
  local out_dir="${OUT_ROOT}/${tag}_boundary_p050_060_M10_ep20_seed03"
  local classes_arg="${classes[*]}"

  if [[ ! -s "${pool_dir}/gated_samples.latent.npz" ]]; then
    echo "$(date -Is) [error] missing latent pool ${pool_dir}/gated_samples.latent.npz" >&2
    exit 1
  fi

  mkdir -p "${out_dir}"
  if [[ ! -s "${out_dir}/best_model.pt" ]]; then
    echo "$(date -Is) [train] ${center} classes=${classes_arg} -> ${out_dir}"
    "${PYTHON_BIN}" -u scripts/pgd_cross_center/synth_online_at_super5.py \
      --center_name "${center}" \
      --ref_meta_json "${pool_dir}/gated_samples.ref_meta.json" \
      --synth_npz "${pool_dir}/gated_samples.latent.npz" \
      --class_trust "${pool_dir}/gated_samples.class_trust.json" \
      --init_ckpt "${INIT_CKPT}" \
      --output_dir "${out_dir}" \
      --quick_eval_centers chapman_shaoxing cpsc_2018 georgia ningbo \
      --quick_eval_n_per_center 300 \
      --attack_mode latent_hull \
      --classes_in_scope "${classes[@]}" \
      --boundary_prob_min 0.50 \
      --boundary_prob_max 0.60 \
      --hull_M 10 \
      --hull_lambda 0.25 \
      --hull_steps 5 \
      --hull_lr 0.3 \
      --K_anchor 90 \
      --pgd_batch 16 \
      --n_epochs 20 \
      --batch_size 128 \
      --num_workers 8 \
      --eval_every 2 \
      --patience 10 \
      --adv_weight 0.10 \
      --anchor_lambda 0.10 \
      --lr 2e-5 \
      --weight_decay 1e-4 \
      --asr_low_threshold 0.0 \
      --seed 20260503 \
      --device cuda:0 \
      2>&1 | tee "${out_dir}/train.log"
  else
    echo "$(date -Is) [skip-train] existing ${out_dir}/best_model.pt"
  fi

  if [[ ! -s "${out_dir}/eval_result_v3_super5_normsuppress.json" ]]; then
    echo "$(date -Is) [eval] ${center}"
    "${PYTHON_BIN}" -u scripts/triple_labels/eval_crosscenter.py \
      --scheme super5 \
      --model_dir "${out_dir}" \
      --batch_size 256 \
      --num_workers 8 \
      --skip_mimic \
      --exclude_ref_ids "${pool_dir}/gated_samples.ref_meta.json" \
      --output_path "${out_dir}/eval_result_v3_super5_normsuppress.json" \
      2>&1 | tee "${out_dir}/eval_crosscenter_v3.log"
  else
    echo "$(date -Is) [skip-eval] existing ${out_dir}/eval_result_v3_super5_normsuppress.json"
  fi
}

run_one \
  ningbo \
  ningbo_v4 \
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced/ningbo/gated \
  NORM MI STTC

run_one \
  chapman_shaoxing \
  chapman_v34 \
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v34_seed42_direct_mv4_multicenter_merged/chapman_shaoxing/gated \
  NORM MI STTC

run_one \
  cpsc_2018 \
  cpsc_v34 \
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v34_seed42_direct_mv4_multicenter_merged/cpsc_2018/gated \
  NORM STTC

run_one \
  georgia \
  georgia_v36 \
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v36_seed42_direct_mv4_multicenter_merged2/georgia/gated \
  NORM MI STTC

echo "$(date -Is) [done] big-center prompt-token boundary AT v1 complete: ${OUT_ROOT}"
