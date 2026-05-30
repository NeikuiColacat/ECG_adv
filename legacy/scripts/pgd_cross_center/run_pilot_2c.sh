#!/bin/bash
# Synth-anchored online AT pilot — 2 cells (cpsc_2018_extra + ningbo) × K=200.
# Plan Rev 8 — sequential GPU runs.
set -euo pipefail

PYBIN=/root/miniforge3/envs/ECGTwin/bin/python
REPO=/root/ECG_adv_Gen
ROOT=/root/autodl-tmp
SUPER5_VICTIM=${ROOT}/triple_labels/super5/best_model.pt

OUT_BASE=${ROOT}/synth_anchored_super5/runs
mkdir -p "${OUT_BASE}"

run_cell() {
    local TAG=$1
    local CENTER=$2
    local OUT_DIR="${OUT_BASE}/${TAG}_k200"
    mkdir -p "${OUT_DIR}"
    echo
    echo "============================================================"
    echo "[pilot] cell: ${CENTER} (tag=${TAG}_k200)"
    echo "============================================================"
    cd "${REPO}"
    "${PYBIN}" scripts/pgd_cross_center/synth_online_at_super5.py \
        --center_name "${CENTER}" \
        --ref_meta_json "${ROOT}/center_token_super5/${TAG}_k200.meta.json" \
        --synth_npz "${ROOT}/synth_anchored_super5/synth_latents/${TAG}_k200.latent.npz" \
        --init_ckpt "${SUPER5_VICTIM}" \
        --output_dir "${OUT_DIR}" \
        --n_epochs 50 \
        --K_anchor 100 \
        --pgd_K 5 \
        --pgd_eps 1.0 \
        --pgd_batch 32 \
        --batch_size 128 \
        --num_workers 12 \
        --roundtrip_anchor_n 1500 \
        --roundtrip_weight 0.5 \
        --adv_weight 2.0 \
        --eval_every 3 \
        --quick_eval_n_per_center 1000 \
        --anchor_lambda 0.05 \
        --ewa_decay 0.999 \
        --lr 5e-5 \
        --weight_decay 1e-4 \
        --seed 42 \
        --device cuda:0 \
        2>&1 | tee "${OUT_DIR}/run.log"
}

run_cell extra cpsc_2018_extra
run_cell nin   ningbo

echo
echo "============================================================"
echo "[pilot] all cells complete — running aggregator"
echo "============================================================"
"${PYBIN}" "${REPO}/scripts/pgd_cross_center/aggregate_pilot_results.py" \
    --runs_dir "${OUT_BASE}" \
    --tags extra_k200 nin_k200 \
    --out "${REPO}/docs/synth_anchored_super5_pilot.md"
echo "[pilot] done."
