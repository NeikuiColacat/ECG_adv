#!/usr/bin/env bash
# Module Ablation Study #1 (2026-04-27): Real-anchored target-center PGD.
#
# Strips ECGTwin diffusion module entirely. Uses K=200 real target-center
# records (already VAE-encoded in existing .pt ref pools) as PGD anchors.
# Same PGD config as Plan Rev 13.2 (K_pgd=10, ε=2.0, K_anchor=300, adv_w=0.5)
# for direct head-to-head comparison.
#
# Decision target: Does real-anchored beat synth-anchored on the strict
# avg gate (>+0.30pp AUROC AND >+0.50pp AUPRC)?
#
# Usage:
#   bash scripts/pgd_cross_center/run_module_ablation_1_real_anchored.sh

set -euo pipefail

PYTHON=/root/miniforge3/envs/ECGTwin/bin/python
ROOT=/root/ECG_adv_Gen
PREP=$ROOT/scripts/pgd_cross_center/prep_real_anchor_npz.py
SCRIPT=$ROOT/scripts/pgd_cross_center/synth_online_at_super5.py

REF_DIR=/root/autodl-tmp/center_token_super5
OUT_ROOT=/root/autodl-tmp/real_anchored_super5

declare -A CENTER_BY_TAG=(
  [extra]=cpsc_2018_extra
  [nin]=ningbo
  [geo]=georgia
)

mkdir -p "$OUT_ROOT/synth_latents"

echo "============================================================"
echo "Module Ablation #1 — real-anchored target-center PGD (no diffusion)"
echo "============================================================"

# ── Phase 1: Convert existing .pt → .latent.npz with one-hot primary labels ─
for tag in extra nin geo; do
  cn=${CENTER_BY_TAG[$tag]}
  ref_pt="$REF_DIR/${tag}_k200.pt"
  out_npz="$OUT_ROOT/synth_latents/${tag}_real_k200.latent.npz"
  out_ct="$OUT_ROOT/synth_latents/${tag}_real_k200.class_trust.json"
  if [[ -f "$out_npz" && -f "$out_ct" ]]; then
    echo "[prep] $tag: cache hit ($out_npz)"
    continue
  fi
  if [[ ! -f "$ref_pt" ]]; then
    echo "[prep] $tag: SKIP (ref_pt not found: $ref_pt)" >&2
    continue
  fi
  echo "[prep] $tag → real-anchor latent npz"
  "$PYTHON" "$PREP" \
    --ref_pt "$ref_pt" \
    --center_name "$cn" \
    --out_npz "$out_npz" \
    --out_class_trust "$out_ct"
done

# ── Phase 2: Online AT (reuse synth_online_at_super5.py with real latents) ─
for tag in extra nin geo; do
  cn=${CENTER_BY_TAG[$tag]}
  syn="$OUT_ROOT/synth_latents/${tag}_real_k200.latent.npz"
  ct="$OUT_ROOT/synth_latents/${tag}_real_k200.class_trust.json"
  meta="$REF_DIR/${tag}_k200.meta.json"
  out="$OUT_ROOT/${tag}_real_k200"
  if [[ -f "$out/best_model.pt" ]]; then
    echo "[train] $tag: cache hit ($out/best_model.pt)"
    continue
  fi
  if [[ ! -f "$syn" || ! -f "$ct" || ! -f "$meta" ]]; then
    echo "[train] $tag: SKIP (missing prereqs)" >&2
    continue
  fi
  echo
  echo "============================================================"
  echo "[train] cell=${tag}_real_k200 center=$cn out=$out"
  echo "============================================================"
  "$PYTHON" "$SCRIPT" \
    --center_name "$cn" \
    --ref_meta_json "$meta" \
    --synth_npz "$syn" --class_trust "$ct" \
    --output_dir "$out" \
    --K_anchor 300 --pgd_K 10 --pgd_eps 2.0 \
    --adv_weight 0.5 --roundtrip_weight 0.5 \
    --n_epochs 100 --patience 20 \
    --eval_every 3 --rescore_interval 3 \
    --seed 42
done

echo "============================================================"
echo "Module Ablation #1 done. Outputs at $OUT_ROOT/{tag}_real_k200/"
echo "Aggregate next:"
echo "  $PYTHON $ROOT/scripts/pgd_cross_center/aggregate_pilot_results.py \\"
echo "      --out_root $OUT_ROOT --meta_dir $REF_DIR \\"
echo "      --cell_suffix real_k200 \\"
echo "      --report_md $OUT_ROOT/module_ablation_1_report.md"
echo "============================================================"
