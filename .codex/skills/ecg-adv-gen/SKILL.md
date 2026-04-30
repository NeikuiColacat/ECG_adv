---
name: ecg-adv-gen
description: Work on /root/ECG_adv_Gen ECG generation, ECGTwin author reproduction, ECGTwin VAE latent augmentation, PTB-XL super5, EfficientNet1DV2, PN2021 cross-center evaluation, TA-OMAT/synthetic-anchor ablations, and related legacy no-IBE experiments. Use this whenever the task mentions ECGTwin, PTB-XL, PhysioNet 2021, super5, EfficientNetV2, center token, nomic prompts, TA-OMAT, or this graduation-project pipeline.
---

# ECG_adv_Gen Project Skill

This repo-tracked copy mirrors the active local Codex skill:

```text
/root/.codex/skills/ecg-adv-gen/SKILL.md
```

When moving to another AutoDL host, copy or symlink this file into Codex's skill
directory:

```bash
mkdir -p /root/.codex/skills/ecg-adv-gen
cp /root/ECG_adv_Gen/.codex/skills/ecg-adv-gen/SKILL.md \
  /root/.codex/skills/ecg-adv-gen/SKILL.md
```

## Environment

- Repo: `/root/ECG_adv_Gen`
- Python: `/root/miniforge3/envs/ECGTwin/bin/python`
- GPU target: RTX 4090D, 24 GB VRAM; CPU: 15 cores; RAM: 80 GB
- Store large outputs/checkpoints in `/root/autodl-tmp/`, not the repo.
- Prefer `find`/`grep` if `rg` is unavailable.

## Active Graduation Project Story

Mainline method:

```text
PTB-XL super5 real ECG
-> EfficientNet1DV2 super5 baseline
-> ECGTwin author training reproduction (IBE + DiT) with thesis-grade logs
-> ECGTwin VAE latent manifold for target-center augmentation
-> TA-OMAT / synth-anchor ablations
-> PN2021 7-center AUROC/AUPRC evaluation
```

Current preferred thesis route:

```text
reproduce ECGTwin author pipeline
-> use ECGTwin VAE latent space for target-anchored on-manifold augmentation
-> keep direct ECGTwin/no-IBE synthetic augmentation as an ablation or historical reference
```

Do not make no-IBE ECGTwin self-training mandatory for the thesis mainline unless the user explicitly revives it. Do not make MIMIC mandatory; PTB-XL + PN2021 gives the clean closed loop.

## Key Paths

- Current summary: `docs/experiment_summary_for_advisor.md`
- Current ablation: `docs/module_ablation_1_real_vs_synth.md`
- Historical no-IBE plan: `trash/docs_cleanup_20260501/historical_no_ibe/ecgtwin_no_ibe_diffusion_augmenter_recommended_plan.md`
- PTB-XL raw/preprocessed: `/root/ECG_adv_Gen/datasets/PTBXL/`, `/root/autodl-tmp/ptbxl/`
- PTB-XL VAE/nomic cache: `/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt`
- PN2021: `/root/autodl-tmp/physionet2021/training/<center>/`
- MIMIC: `/root/autodl-tmp/MIMIC/`
- ECGTwin repo: `model/ECGTwin/`
- ECGTwin author reproduction scripts: `scripts/ecgtwin_author_repro/`
- Super5 classifier scripts: `scripts/triple_labels/`
- TA-OMAT / synth-anchor ablations: `scripts/pgd_cross_center/`

## ECGTwin Facts

- Original ECGTwin VAE input/output is `(B, 1024, 12)` channels-last, raw mV, MIMIC lead order.
- VAE latent is `(B, 4, 128)` and PTB-XL latent cache is already scaled by `0.18215`.
- ECGTwin MIMIC lead order swaps PTB-XL aVL/aVF:
  `ECGTWIN_TO_PTBXL_INDICES = [0,1,2,3,5,4,6,7,8,9,10,11]`.
- Before classifier training/eval: decode, reorder leads, resample 1024 to 1000 at 100 Hz, then use the existing classifier preprocessing.
- Original DiT text path uses `text_embed` `(B,L,768)`, `text_embed_mask` `(B,L)`, and patient info `(hr, age, sex)` through `text_projector + CrossAttention`.
- Original IBE path supplies `base_vector` for AdaLN modulation. Preserve this for ECGTwin author reproduction; no-IBE replacements are historical unless explicitly revived.
- Never set `text_embed_mask` all zeros; original cross-attention can softmax all `-inf` and produce NaN. Use a null text embedding with mask `1`.

## Super5 Labels

Class order is:

```text
CD, HYP, MI, NORM, STTC
```

Use `scripts/triple_labels/label_schemes.py` as the source of truth.

Hard rules:

- PTB-XL super5 uses `diagnostic_class` from `scp_statements.csv`, class order `CD,HYP,MI,NORM,STTC`.
- Current PTB-XL implementation includes diagnostic SCP keys with likelihood/confidence `0` because it checks `conf >= 0.0`; do not change this threshold without invalidating old label caches and rerunning label distribution sanity checks.
- If label mapping, confidence threshold, `scp_statements.csv`, or class order changes, delete/regenerate `ptbxl_labels.C5.all.npy` and record new per-class positives.
- PN2021 super5 uses project-defined v3 semantic projection, not an official PN2021->PTB-XL crosswalk. Current source of truth is `SUPER5_PN2021_MAPPING_VERSION = v3_super5_normsuppress_20260501` in `scripts/triple_labels/label_schemes.py`.
- PN2021 v3 uses `SNOMED_TO_SUPER5_POSITIVE` for direct CD/HYP/MI/STTC positives, `NORM_POSITIVE_SNOMEDS` for strict normal candidates, and `NORM_SUPPRESS_SNOMEDS` for rhythm/axis/ectopy/low-voltage/boundary codes that cancel NORM without becoming a super5 positive.
- In PN2021 v3, `Q wave abnormal` and `early repolarization` are suppress-only by default, not direct STTC positives. `sinus bradycardia`, `sinus tachycardia`, and `sinus arrhythmia` are not PTB-XL-normal equivalents by default.
- PN2021 eval cache version is `v3_super5_normsuppress`; cache metadata includes the super5 mapping version/hash and must be rebuilt if mapping, parser, class order, or preprocessing changes.
- MIMIC super5 is weak regex labeling. Treat it as noisy external reference or optional pretraining/ablation, not as main supervised thesis evidence.

Historical Scheme B prompt table:

| class | prompt |
|---|---|
| NORM | `normal ecg|sinus rhythm` |
| MI | `myocardial infarction|pathological q wave` |
| STTC | `st-t change|st segment abnormality|t wave abnormality` |
| CD | `conduction disturbance|bundle branch block` |
| HYP | `ventricular hypertrophy|left ventricular hypertrophy` |

For multi-label samples, concatenate prompt fragments with `|`.

## PN2021 Evaluation Rules

- Main metrics: macro AUROC and macro AUPRC.
- Use PN2021 7 centers only:
  `chapman_shaoxing`, `cpsc_2018`, `cpsc_2018_extra`, `georgia`, `ningbo`, `ptb`, `st_petersburg_incart`.
- Hard-exclude `ptb-xl` / `ptbxl` shard because it leaks PTB-XL training data.
- Existing entry:

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /path/to/model_dir \
  --skip_mimic
```

## Implementation Guidance

Keep original `model/ECGTwin/` intact. Current active new code should live in:

```text
scripts/ecgtwin_author_repro/      # ECGTwin author IBE + DiT reproduction
scripts/pgd_cross_center/          # TA-OMAT / synth-anchor ablations
scripts/triple_labels/             # EfficientNet1DV2 training/eval
util/ecg_digital_features.py       # digital ECG validation
```

Archived/historical no-IBE files may exist under `trash/` and should not be restored into the mainline without an explicit user request.

If no-IBE Scheme B is explicitly revived, the old expected layout was:

```text
model.py              # fork DiT_ECGTwin, remove IBE, keep text path
dataset.py            # single-sample PTB-XL latent dataset, not paired ECGTwin dataset
prompt_table.py       # super5 prompt table + nomic embedding cache
train.py              # diffusion training with AMP/EMA/logging
sample.py             # DDIM/CFG sampling and npz export
center_token.py       # optional per-block token after mainline works
train_center_token.py # optional target-center token training
```

Reuse from ECGTwin:

- `DiTBlock_ECGTwin`
- `CrossAttention`
- `TimestepEmbedder`
- `RoPEEmbedder`
- `PositionalEmbedder`
- `process_pat_info`
- `_pad_text_embed`
- VAE encoder/decoder checkpoint

For that historical no-IBE branch, delete or bypass:

- `IBExtractor`
- `ibe_path`
- paired ref/target training logic
- `base_vector` dependency

Preferred forward signature:

```python
def forward(
    self,
    x,
    t,
    y_super5,
    text_embed=None,
    text_embed_mask=None,
    pat_info=None,
    center_token=None,
):
    ...
```

## Logging Requirements

For every training/reproduction run, write artifacts useful for the thesis:

- `run_config.yaml`, `config.yaml`, or `run_config.json`
- `train.log`
- `metrics.jsonl` with step/epoch/loss/lr/grad_norm/val_loss if available
- `loss_curve.csv`
- `loss_curve.png`
- `checkpoints/best.pt` and `checkpoints/latest.pt`
- `samples/*.npz` for generated ECG
- `figures/*.png` for 12-lead examples
- downstream `eval_result.json` from PN2021 cross-center eval

Use stable output roots like:

```text
/root/autodl-tmp/ecgtwin_author_repro/<run_name>/
/root/autodl-tmp/triple_labels/<run_name>/
```

## 8-Hour Execution Policy

An 8-hour window is realistic only for a reduced proof-of-concept, not the full thesis package.

Prioritize:

1. keep ECGTwin author reproduction logs/checkpoints/curves complete;
2. keep EfficientNet1DV2 super5 baseline and PN2021 eval reproducible;
3. run target-anchor/TA-OMAT or synthetic-anchor ablations only with saved configs;
4. save logs, curves, metrics JSON/CSV, and checkpoint paths needed for the thesis.

Defer MIMIC pretraining, no-IBE revival, and center token unless the current ECGTwin reproduction + EfficientNet evaluation loop is stable.

## Classifier Augmentation Format

Simplest synthetic classifier input is `.npz`:

```text
signals: (N, 1000, 12) float32, PTB-XL lead order, 100 Hz, 10 s
labels:  (N, 5) float32, class order CD/HYP/MI/NORM/STTC
```

If an existing generator writes `(N, 12, 1000)`, transpose before feeding `train_ptbxl.py`.

`train_ptbxl.py --synth_npz` does not run filtering, z-score, or lead reorder on synthetic signals; it only normalizes shape and crops. Synthetic `.npz` must already be in PTB-XL lead order and on the classifier-preprocessed scale, or be explicitly documented as a raw-mV ablation.
