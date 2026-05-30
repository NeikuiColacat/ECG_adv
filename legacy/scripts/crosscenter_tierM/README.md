# Cross-Center Tier-M 6-Class Training

ECGTwin-friendly 6-class victim baseline for the AdvDiff + AugMix generalization pipeline.
Forked from `scripts/crosscenter_v2/` with the classification head reduced from 26 →
**6 classes**: `NSR / STach / AF / IAVB / LBBB / RBBB`.

## Why 6 classes (Tier-M)

See `docs/label_selection_research.md` §7.6 for the full rationale. Short version:

1. **ECGTwin-friendly** — all 6 have validated demo disease directories in
   `model/ECGTwin/generation_result_by_disease/`, so downstream AdvDiff can actually
   generate each class.
2. **4 ECG semantic categories** — sinus rhythm (NSR), tachyarrhythmias (STach/AF),
   three major conduction blocks (IAVB/LBBB/RBBB).
3. **PTBXL fully covers all 6** — no `-1` mask needed, so training uses plain
   `BCEWithLogitsLoss(pos_weight=...)` instead of `MaskedFocalLoss`.
4. **Cross-center evaluable after 1 patch** — see "ningbo AF↔AFL patch" below.

## What was patched upstream

Two in-place additions to `scripts/crosscenter_v2/label_alignment_v2.py`:

- **Tier-M constants**: `TIER_M`, `TIER_M_IDX`, `NUM_CLASSES_TIER_M = 6`.
- **Ningbo AF quirk fix**: PN2021 ningbo center uses SNOMED `164890007`
  (officially AFL) to label ~7,615 AF+AFL cases. Without a patch, ningbo's AF
  column is entirely zero in cross-center eval. The patch adds a dual mapping so
  `164890007` → {AFL idx, AF idx}. Pure AF (`164889003`) is unaffected.
  Rationale: `/root/.claude/projects/-root-ECG-adv-Gen/memory/pn2021_labeling_quirks.md`.

Verify both patches:
```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_v2/label_alignment_v2.py
```

## Files

```
scripts/crosscenter_tierM/
├── README.md                        # This file
├── train_ptbxl_tierM.py             # Training (PTBXL folds 1-8 / 9 / 10)
└── eval_crosscenter_tierM.py        # PN2021 5-main-center + 2-small-center eval
```

**Big outputs** live under `/root/autodl-tmp/crosscenter_tierM/`:

- `best_model.pt` — early-stopped on val 6-class macro AUROC
- `training_log.json` — per-epoch loss / AUROC / AUPRC
- `train_result.json` — final PTBXL test metrics + config snapshot + `pos_weight`
- `eval_crosscenter.json` — PN2021 per-class per-center + aggregate

PTBXL preprocessing cache is reused from
`/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy` (same
`unified_preprocess_to_1000` pipeline as v2). Override with `--cache_path`.

## Run

```bash
PY=/root/miniforge3/envs/ECGTwin/bin/python
cd /root/ECG_adv_Gen

# Train (~15-25 min on RTX 4090, early stopping)
$PY scripts/crosscenter_tierM/train_ptbxl_tierM.py \
    --output_dir /root/autodl-tmp/crosscenter_tierM

# Cross-center eval
$PY scripts/crosscenter_tierM/eval_crosscenter_tierM.py \
    --model_dir /root/autodl-tmp/crosscenter_tierM
```

## Actual results (seed=42, 50 epochs, best val ep 48)

### PTBXL test (fold 10)
- Test macro AUROC **0.9741**, AUPRC **0.8447**
- Per-class AUROC:

  | Class | AUROC | AUPRC | n_pos |
  |---|---:|---:|---:|
  | NSR   | 0.9037 | 0.9726 | 1822 |
  | STach | 0.9927 | 0.9027 | 82 |
  | AF    | 0.9764 | 0.8560 | 152 |
  | IAVB  | 0.9778 | 0.5993 | 79 |
  | LBBB  | 0.9971 | 0.9090 | 54 |
  | RBBB  | 0.9966 | 0.8283 | 54 |

### PN2021 cross-center (main 5 centers, AUROC|n_pos per cell)

| Center | N | Macro AUROC | Macro AUPRC | NSR | STach | AF | IAVB | LBBB | RBBB |
|---|---:|---:|---:|---|---|---|---|---|---|
| chapman_shaoxing | 9709 | 0.9575 | 0.7845 | 0.901\|1826 | 0.994\|1568 | 0.992\|2225 | 0.976\|247 | 0.916\|205 | 0.966\|454 |
| cpsc_2018 | 5279 | 0.9166 | 0.7849 | 0.878\|918 | **N/A\|0** | 0.981\|1221 | 0.945\|722 | 0.937\|236 | 0.842\|1857 |
| cpsc_2018_extra | 1296 | 0.8538 | 0.5638 | 0.742\|4 | 0.971\|303 | 0.963\|207 | 0.912\|106 | 0.566\|38 | 0.969\|114 |
| georgia | 9320 | 0.9482 | 0.7553 | 0.878\|1752 | 0.990\|1261 | 0.927\|741 | 0.966\|769 | 0.970\|231 | 0.959\|556 |
| ningbo | 34470 | 0.9695 | 0.8111 | 0.872\|6299 | 0.991\|5687 | **0.993\|7615** | 0.972\|893 | 0.995\|248 | 0.994\|1291 |
| **MAIN avg (5)** | | **0.9291** | **0.7399** | | | | | | |

**Key validations**:
- ✅ **`ningbo × AF` = 0.993 AUROC (n_pos=7,615)** — confirms the 164890007→AF dual
  mapping patch worked. Without it, this cell would be `N/A|0`.
- ✅ `cpsc_2018 × STach` = `N/A|0` — structural absence (CPSC 2018 original 9-class
  task didn't include STach). Expected and unavoidable.
- ⚠️ `cpsc_2018_extra × NSR` = 0.742 AUROC, **n_pos=4** — technically non-null but
  metric is unstable at this sample size; treat as "soft N/A" when reporting.
- 29/30 cells computed; the one unavoidable null is the structural CPSC 2018
  STach gap.

### Source→target gap

| Metric | PTBXL test | PN2021 main avg | Δ |
|---|---:|---:|---:|
| Macro AUROC | 0.9741 | 0.9291 | **−4.50 pp** |
| Macro AUPRC | 0.8447 | 0.7399 | **−10.48 pp** |

Compared to the 26-class v2 baseline (Tier-1 AUROC gap 4.02 pp / AUPRC gap 13.72 pp,
see `docs/training/gap_report.md`):
- AUROC gap is **comparable** (slightly wider, but across more semantic diversity
  — Tier-1 was only 5 "easy" classes).
- AUPRC gap is **3.24 pp tighter**, consistent with every Tier-M class having
  adequate PTBXL positive samples (no mask-induced noise).

### Small centers (reported separately, not averaged)

| Center | N | Macro AUROC | Macro AUPRC | notes |
|---|---:|---:|---:|---|
| ptb | 116 | 0.8543 | 0.6327 | only NSR/STach/AF observable; STach has n_pos=1 |
| st_petersburg_incart | 33 | 0.9218 | 0.7729 | only STach/AF/RBBB observable |

Both CIs are wide (±5-10pp) due to tiny sample sizes; treat as sanity only.

## Comparison to 26-class v2 baseline

v2 baseline checkpoint at `/root/autodl-tmp/crosscenter_v2/best_model.pt` is
unchanged. When comparing:

- For `IAVB / LBBB / RBBB` (shared between v2 Tier-1 and Tier-M), Tier-M should
  be within ±2pp AUROC of v2 (slight regression OK, training target is more
  focused).
- For `AF / NSR / STach`, Tier-M is expected to match or slightly beat v2 —
  especially on ningbo AF where v2 can't compute the metric at all.

## Not covered

- No AdvDiff / AugMix finetuning — that lives in a follow-up experiment.
- No multi-dataset training (PTBXL-only, PN2021/MIMIC eval-only).
- MIMIC zero-shot eval is deferred — add a `eval_mimic_zeroshot_tierM.py` fork
  if/when needed.
