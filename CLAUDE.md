# ECG Adversarial Generation Project

## Overview
Cross-center adversarial training pipeline for improving DeepECG EfficientNet
generalization. Three concurrent research tracks:
- **VAE-latent PGD** (Mode A) — adversarial buffer generation against a frozen
  Tier-M victim, validated by ASR + medical-semantic gates
- **ECGTwin synthesis** — Center-Token / Style-Translator personalize ECGTwin
  to per-center style; output samples augment training
- **AugMix** — time/latent-domain mixing for in-domain robustness

## Environment
- Python: `/root/miniforge3/envs/ECGTwin/bin/python`
- Conda env: `ECGTwin` (activation not needed, use full python path)
- GPU: RTX 4090 24 GB
- Root disk: ~4 GB free → store large files in `/root/autodl-tmp/` (~28 GB free)

## Key Paths
- PTB-XL data: `/root/ECG_adv_Gen/datasets/PTBXL/` (raw) + `/root/autodl-tmp/ptbxl/` (preprocessed cache)
- PTB-XL VAE-encoded: `/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt`
- PN2021: `/root/autodl-tmp/physionet2021/training/<center>/`
- MIMIC: `/root/autodl-tmp/MIMIC/` (cart reports + signal cache)
- EfficientNet 77-class JIT: `/root/ECG_adv_Gen/model/DeepECG/weights/efficientnetv2_77_classes/efficientnet_deepecg_unscaled.pt`
- ECGTwin: loaded via `util/ecgtwin_utils.py::ECGTwinWrapper`
- Trained checkpoints (large, on autodl-tmp):
  - v2 26-class baseline: `/root/autodl-tmp/crosscenter_v2/best_model.pt`
  - Tier-M 6-class victim: `/root/autodl-tmp/crosscenter_tierM/best_model.pt`
  - Triple-label heads: `/root/autodl-tmp/triple_labels/{super5,sub23,pn26}/best_model.pt`
- Outputs: `/root/ECG_adv_Gen/outputs/` (small artifacts) + `/root/autodl-tmp/` (large)
- Docs: `/root/ECG_adv_Gen/docs/`
- Legacy archive: `/root/ECG_adv_Gen/trash/`

## Running Scripts
Always use the full Python path:
```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/train_ptbxl.py --scheme super5
```

## Architecture

### Core classifier pipelines
- `scripts/crosscenter_v2/` — original 26-class PN26 baseline + canonical
  utilities (`label_alignment_v2.py`, `preprocess_utils.py::unified_preprocess_to_1000`,
  `eval_crosscenter_v2.py`). **Reuse these from new code, do not fork.**
- `scripts/crosscenter_tierM/` — Tier-M 6-class (NSR/STach/AF/IAVB/LBBB/RBBB)
  PTB-XL/MIMIC training + cross-center eval (the original AT lineage).
- `scripts/triple_labels/` — 3 EfficientNet1DV2 heads (super5/sub23/pn26)
  trained on PTB-XL with masked BCE; eval on PTB-XL fold10 + PN2021 7 centers
  + MIMIC test. Canonical entry for new label-scheme experiments.

### Adversarial / augmentation
- `adversarial/` — VAE-latent PGD (Mode A) attack pipeline
  - `pgd_advdiff.py` — PGD generator (replaces failed BoundaryAdvDiff)
  - `efficientnet_victim_tierM.py` — frozen Tier-M victim wrapper
  - `adv_validation.py` — ASR + medical-semantic gates
    (ASR overall ≥ 70%, per-class ≥ 30%; Einthoven residual p95 < 0.5)
  - `tierM_labels.py` — ECGTwin prompts/dirs; class list **re-exported**
    from `scripts/crosscenter_v2/label_alignment_v2.py::TIER_M`
    (single source of truth — never redefine the list here)
  - `efficientnet_victim.py` / `efficientnet_adapter.py` / `adv_generate.py`
    / `finetune.py` — older 77-class JIT-adapter path (kept; not on the
    Tier-M flow)
- `scripts/pgd_cross_center/` — PGD adv-buffer generator (driver for `adversarial/pgd_advdiff.py`)
- `scripts/{augmix_validation,augmix_adv_combo,online_vs_offline}/` — ablation runners

### Synthesis (ECGTwin Center-Token / Style-Translator)
- `methods/ecgtwin_gen/center_token/` — Textual-Inversion-style 256-d center
  token embedding (model + trainer); class-agnostic
- `methods/ecgtwin_gen/style_translator/` — Style-Translator
  (Stage 0 **validated to fail**, see `docs/ecgtwin_gen/style_translator_stage0_results.md`;
  kept for archival)
- `scripts/ecgtwin_gen/` — prep_center_dataset / train_center_token /
  generate_center_synth / enroll_new_center / compare_*
- `methods/augmix/latent_viz/` — AugMix latent visualization + run pipeline

### Other
- `sub_experiment/tsne_clustering_v1/` — t-SNE clustering ablation
- `util/` — shared utilities: `ecgtwin_utils.py::ECGTwinWrapper`,
  `ecg_viz.py` (12-lead plot, `engine={auto,ecgplot,matplotlib}`),
  `lead_mapping.py` / `lead_utils.py` (`ECGTWIN_TO_PTBXL_INDICES`)
- `model/ECGTwin/` — ECGTwin DiT + VAE (submodule, gitignored under `model/`)
- `model/DeepECG/` — DeepECG EfficientNet JIT (gitignored)
- `docs/` — technical docs + per-experiment reports
- `docs/papers/` — reference paper PDFs (gitignored as too large; only
  `INDEX.md` is tracked)
- `trash/` — archived legacy code (not on any active path)

## Important Notes

### Data integrity
- **PN2021 cross-center eval must hard-exclude the `ptb-xl` shard**
  (`/root/autodl-tmp/physionet2021/training/ptb-xl/` is exactly PTB-XL
  training data). Use `assert center.lower() not in {'ptb-xl', 'ptbxl'}`
  in any new eval loop. ningbo `ptb` is the older small dataset, allowed.
- **Single source of truth for SNOMED parsing**: `parse_header_snomed`
  lives in `scripts/triple_labels/eval_crosscenter.py:53`. The PN2021 .hea
  format uses `# Dx:` (with a space) — naive `startswith('#Dx:')` silently
  returns no codes → all-zero labels → all-NaN AUROC.
- **ningbo AF↔AFL quirk**: SNOMED 164890007 is officially AFL but is used
  for AF in ningbo. `_SNOMED_TO_IDX` patches it to map to both classes.
- **Super5 NORM exclusivity guard** (`label_schemes.py::snomed_list_to_super5`):
  PTB-XL trains NORM as "no abnormality" (4.65% co-occurrence with abnormal
  classes); PN2021 emits sinus rhythm (426783006) alongside pathology codes
  (98% co-occurrence on cpsc_2018_extra). Without the guard, NORM AUROC
  collapses on PN2021. Sub23 / PN26 schemes do not yet have an analogous
  guard — flag if NORM/NSR participates in macro.

### Project conventions
- **Tier-M class list canonical source**: `scripts/crosscenter_v2/label_alignment_v2.py::TIER_M`.
  `adversarial/tierM_labels.py` re-exports it. Don't redefine the 6-class
  list anywhere else.
- **Per-scheme PTBXL label cache** is keyed by class count
  (`*.C5.all.npy`, `*.C23.all.npy`, `*.C26.all.npy`) so different schemes
  can share an `--output_dir` without silently loading each other's array.
- **`util/test_wave.py`** is an autodl GPU-utilization keep-alive script
  (despite the "test_" prefix), already `.gitignore`d. Not a unittest.

### Numerical / framework gotchas
- EfficientNet 77-class is JIT-compiled (`torch.jit.load()`); supports autograd
  but weights are frozen. Input shape: `(B, 12, 2500)` @ 250 Hz, scaled by
  `MHI_FACTOR = 1 / 0.0048`.
- EfficientNet1DV2 (s_v2) used for Tier-M / Super5 / Sub23 / PN26 takes
  `(B, 12, 250)` @ 100 Hz (z-scored) directly; no MHI factor.
- VAE decoder has in-place `/= 0.18215` that breaks autograd → use the
  out-of-place division in `efficientnet_victim*.py::_decode_latent_differentiable`.
- ECGTwin lead order ≠ PTB-XL lead order. Reorder via
  `ECGTWIN_TO_PTBXL_INDICES` from `util/lead_utils.py` before feeding the
  victim or computing PTB-XL-shaped metrics.
- `unified_preprocess_to_1000(target_fs=100, target_len=1000, apply_filter=True, apply_zscore=True)`
  is the cross-dataset canonical preprocessing (per-record global z-score,
  not per-lead). Bypass with `apply_*=False` only for benchmark
  cross-comparison against external models that own their own scaler.
