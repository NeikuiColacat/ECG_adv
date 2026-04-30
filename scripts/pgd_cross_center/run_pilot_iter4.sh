#!/usr/bin/env bash
# Synth-anchored online AT pilot iter 4 (Plan Rev 13.2).
#
# 3 cells (extra, nin, geo) × ≤100 epoch (early-stop patience=20 on val_macro_auroc)
# Stage 0.4 sanity → class_trust.json → Stage 2 training.
#
# Usage:
#   bash scripts/pgd_cross_center/run_pilot_iter4.sh

set -euo pipefail

PYTHON=/root/miniforge3/envs/ECGTwin/bin/python
ROOT=/root/ECG_adv_Gen
OUT_ROOT=/root/autodl-tmp/synth_anchored_super5_v13
SCRIPT="$ROOT/scripts/pgd_cross_center/synth_online_at_super5.py"

declare -A CENTER_BY_TAG=(
  [extra]=cpsc_2018_extra
  [nin]=ningbo
  [geo]=georgia
)

echo "============================================================"
echo "Synth-anchored online AT pilot iter 4  (Plan Rev 13.2)"
echo "  3 cells × ≤100 epoch  + early-stop patience=20"
echo "  K_pgd=10  K_anchor=300  ε=2.0  adv_w=0.5"
echo "  Out:    $OUT_ROOT/{tag}_k200/"
echo "============================================================"

# ── Phase 3: class_trust.json sanity for each cell ─────────────────
for tag in extra nin geo; do
    cn=${CENTER_BY_TAG[$tag]}
    syn="$OUT_ROOT/synth_latents/${tag}_k200_pool300.npz"
    ct="$OUT_ROOT/synth_latents/${tag}_k200_pool300.class_trust.json"
    if [[ -f "$ct" ]]; then
        echo "[ct] $tag: cache hit ($ct)"
        continue
    fi
    echo "[ct] $tag → building class_trust.json"
    "$PYTHON" "$SCRIPT" \
        --center_name "$cn" \
        --synth_npz "$syn" \
        --output_dir "$OUT_ROOT/${tag}_k200" \
        --class_trust "$ct" \
        --build_class_trust
done

# ── Phase 6: Train each cell with early-stop ─────────────────────
for tag in extra nin geo; do
    cn=${CENTER_BY_TAG[$tag]}
    syn_lat="$OUT_ROOT/synth_latents/${tag}_k200_pool300.latent.npz"
    ct="$OUT_ROOT/synth_latents/${tag}_k200_pool300.class_trust.json"
    out="$OUT_ROOT/${tag}_k200"
    meta="/root/autodl-tmp/center_token_super5/${tag}_k200.meta.json"
    echo
    echo "============================================================"
    echo "[train] cell=$tag center=$cn out=$out"
    echo "============================================================"
    "$PYTHON" "$SCRIPT" \
        --center_name "$cn" \
        --ref_meta_json "$meta" \
        --synth_npz "$syn_lat" \
        --class_trust "$ct" \
        --output_dir "$out" \
        --K_anchor 300 --pgd_K 10 --pgd_eps 2.0 \
        --adv_weight 0.5 --roundtrip_weight 0.5 \
        --n_epochs 100 --patience 20 \
        --eval_every 3 --rescore_interval 3 \
        --seed 42
done

echo
echo "============================================================"
echo "All 3 cells done. Outputs:"
echo "  $OUT_ROOT/extra_k200/best_model.pt"
echo "  $OUT_ROOT/nin_k200/best_model.pt"
echo "  $OUT_ROOT/geo_k200/best_model.pt"
echo
echo "Run aggregate + Stage 3 eval next:"
echo "  $PYTHON $ROOT/scripts/pgd_cross_center/aggregate_pilot_results.py \\"
echo "      --out_root $OUT_ROOT"
echo "============================================================"
