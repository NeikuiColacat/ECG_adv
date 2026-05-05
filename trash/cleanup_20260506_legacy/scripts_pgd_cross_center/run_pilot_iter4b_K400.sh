#!/usr/bin/env bash
# Backup pilot iter 4b (Plan Rev 13.2 escalation):
#   - K=400 ref pool (vs iter 4 K=200) for richer base_vector diversity
#   - synth pool 600/cell (vs iter 4 300; 2x scale, still real-dominated mix)
#   - Same K_pgd=10, ε=2.0, adv_w=0.5
#
# Trigger: only run if iter 4 decision gate fails (0/3 or 1/3 cells pass).
#
# Usage:
#   bash scripts/pgd_cross_center/run_pilot_iter4b_K400.sh

set -euo pipefail

PYTHON=/root/miniforge3/envs/ECGTwin/bin/python
ROOT=/root/ECG_adv_Gen
PREP=$ROOT/scripts/ecgtwin_gen/prep_center_dataset_super5.py
GEN=$ROOT/scripts/ecgtwin_gen/generate_center_synth.py
SCRIPT=$ROOT/scripts/pgd_cross_center/synth_online_at_super5.py
TXT_EMB=/root/autodl-tmp/center_token_super5/super5_text_embeds.pt

REF_DIR=/root/autodl-tmp/center_token_super5
OUT_ROOT=/root/autodl-tmp/synth_anchored_super5_v13b_K400

declare -A CENTER_BY_TAG=(
  [extra]=cpsc_2018_extra
  [nin]=ningbo
  [geo]=georgia
)

mkdir -p "$OUT_ROOT/synth_latents"

echo "============================================================"
echo "Pilot iter 4b backup — K=400 ref pool + 600/cell synth pool"
echo "============================================================"

# ── Phase 1: K=400 ref pool prep ──────────────────────────────────
for tag in extra nin geo; do
  cn=${CENTER_BY_TAG[$tag]}
  ref="$REF_DIR/${tag}_k400.pt"
  if [[ -f "$ref" ]]; then
    echo "[prep] $tag: cache hit ($ref)"
    continue
  fi
  echo "[prep] $tag → K=400 ref pool"
  "$PYTHON" "$PREP" \
    --center "$cn" --K 400 --seed 42 \
    --text_embeds "$TXT_EMB" --out "$ref"
done

# ── Phase 2: 600/cell synth pool ──────────────────────────────────
for tag in extra nin geo; do
  cn=${CENTER_BY_TAG[$tag]}
  ref="$REF_DIR/${tag}_k400.pt"
  syn="$OUT_ROOT/synth_latents/${tag}_k400_pool600.npz"
  if [[ -f "$syn" ]]; then
    echo "[gen] $tag: cache hit ($syn)"
    continue
  fi
  echo "[gen] $tag → 600 synth"
  "$PYTHON" "$GEN" \
    --scheme super5 --ref_pt "$ref" --center_name "$cn" \
    --out "$syn" \
    --n_per_class 200 --batch_size 100 --num_steps 50 \
    --save_latent --seed 42
done

# ── Phase 3: class_trust ──────────────────────────────────────────
for tag in extra nin geo; do
  cn=${CENTER_BY_TAG[$tag]}
  syn="$OUT_ROOT/synth_latents/${tag}_k400_pool600.npz"
  ct="$OUT_ROOT/synth_latents/${tag}_k400_pool600.class_trust.json"
  if [[ -f "$ct" ]]; then
    echo "[ct] $tag: cache hit"
    continue
  fi
  "$PYTHON" "$SCRIPT" \
    --center_name "$cn" --synth_npz "$syn" \
    --output_dir "$OUT_ROOT/${tag}_k400" \
    --class_trust "$ct" --build_class_trust
done

# ── Phase 4: train ─────────────────────────────────────────────────
for tag in extra nin geo; do
  cn=${CENTER_BY_TAG[$tag]}
  syn="$OUT_ROOT/synth_latents/${tag}_k400_pool600.latent.npz"
  ct="$OUT_ROOT/synth_latents/${tag}_k400_pool600.class_trust.json"
  meta="$REF_DIR/${tag}_k400.meta.json"
  out="$OUT_ROOT/${tag}_k400"
  echo
  echo "============================================================"
  echo "[train] cell=$tag center=$cn out=$out"
  echo "============================================================"
  "$PYTHON" "$SCRIPT" \
    --center_name "$cn" \
    --ref_meta_json "$meta" \
    --synth_npz "$syn" --class_trust "$ct" \
    --output_dir "$out" \
    --K_anchor 600 --pgd_K 10 --pgd_eps 2.0 \
    --adv_weight 0.5 --roundtrip_weight 0.5 \
    --n_epochs 100 --patience 20 \
    --eval_every 3 --rescore_interval 3 \
    --seed 42
done

echo "============================================================"
echo "Iter 4b done. Outputs at $OUT_ROOT/{tag}_k400/"
echo "Aggregate next:"
echo "  $PYTHON $ROOT/scripts/pgd_cross_center/aggregate_pilot_results.py \\"
echo "      --out_root $OUT_ROOT --report_md $OUT_ROOT/pilot_iter4b_report.md"
echo "============================================================"
