# ECG_adv_Gen Agent Notes

This file is durable project memory for coding agents working in the ECG_adv_Gen
graduate-project worktree.

Repo-tracked Codex skill copy:

```text
.codex/skills/ecg-adv-gen/SKILL.md
```

The active runtime skill lives outside git at
`/root/.codex/skills/ecg-adv-gen/SKILL.md`. When moving to another AutoDL host,
copy the repo-tracked skill into that runtime location.

## Environment

- Current graduate worktree: `/root/autodl-tmp/ECG_adv_Gen_graduate`
- Original main worktree: `/root/ECG_adv_Gen`
- Python env: `/root/miniforge3/envs/ECGTwin/bin/python`
- Hardware target: RTX 4090D 24GB VRAM, 15 CPU cores, 80GB RAM.
- Optimize future training/preprocessing for this hardware profile:
  - prefer AMP/bf16 where numerically safe on the 4090D;
  - keep large arrays/checkpoints/caches under `/root/autodl-tmp/`;
  - use mmap or streaming for PN2021/PN2021-C instead of repeatedly
    decompressing large `.npz` files;
  - set DataLoader `num_workers` from CLI and tune around 4-8 before using all
    15 CPU cores;
  - use `pin_memory`, `persistent_workers`, and `prefetch_factor` for long
    training jobs when `num_workers > 0`;
  - avoid materializing every corrupted PN2021-C copy unless disk has been
    expanded substantially.
- System disk is small. Put large outputs, checkpoints, samples, caches, and logs under `/root/autodl-tmp/`, not the repo.
- Disk hygiene as of 2026-05-03 after user expansion and PN2021-C materialization: `/` has about 9.6 GB free and `/root/autodl-tmp` is about 350 GB total with about 54 GB free. Keep long-run temp/cache paths on the data disk, for example `TMPDIR=/root/autodl-tmp/tmp` and `XDG_CACHE_HOME=/root/autodl-tmp/cache` when safe for a command. Do not materialize another full PN2021-C copy without checking free space first.
- Do not move/delete repo historical artifacts or Git objects just to free disk unless the user explicitly approves; prefer package caches, bytecode caches, and new experiment outputs under `/root/autodl-tmp`.
- `rg` may be unavailable in this environment. Use `find`, `grep`, `sed`, `nl`, and `wc` when needed.
- Use `apply_patch` for manual file edits. Do not overwrite unrelated dirty worktree changes.

## Current Graduation Project Story

Mainline method:

```text
PTB-XL super5 real ECG
-> EfficientNet1DV2 super5 baseline
-> ECGTwin author training reproduction (IBE + DiT) with thesis-grade logs
-> ECGTwin VAE latent manifold for target-center augmentation
-> TA-OMAT / synth-anchor ablations
-> PhysioNet/CinC 2021 7-center AUROC/AUPRC evaluation
```

Current preferred thesis route:

```text
reproduce ECGTwin author pipeline
-> use ECGTwin VAE latent space for target-anchored on-manifold augmentation
-> keep direct ECGTwin synthetic augmentation as an ablation, not the main claim
```

Do not make no-IBE ECGTwin self-training mandatory for the thesis mainline. The
previous Scheme B/no-IBE implementation is archived under
`trash/cleanup_20260430/deprecated_no_ibe/` and should only be used as historical
reference unless the user explicitly revives it.

Do not make MIMIC mandatory for the thesis mainline. PTB-XL + PN2021 gives the
clean closed loop; MIMIC is an optional noisy OOD evaluation/pretraining source.

Primary plan document:

```text
docs/pipelines/graduate_project_branch_mainline.md
docs/experiment_summary_for_advisor.md
docs/module_ablation_1_real_vs_synth.md
trash/docs_cleanup_20260501/historical_no_ibe/ecgtwin_no_ibe_diffusion_augmenter_recommended_plan.md  # historical Scheme B plan
```

## Key Data And Artifact Paths

- ECGTwin original repo: `model/ECGTwin/`
- ECGTwin author reproduction scripts: `scripts/ecgtwin_author_repro/`
- Archived no-IBE Scheme B implementation: `trash/cleanup_20260430/deprecated_no_ibe/`
- Archived non-mainline code from the graduate branch cleanup:
  `trash/cleanup_20260506_legacy/`
- PTB-XL raw/preprocessed: `datasets/PTBXL/`, `/root/autodl-tmp/ptbxl/`
- PTB-XL VAE/nomic cache: `datasets/PTBXL/PTBXL_vae_multi_nomic.pt`
- PN2021 centers: `/root/autodl-tmp/physionet2021/training/<center>/`
- MIMIC raw/cache root: `/root/autodl-tmp/MIMIC/`
- ECGTwin latent MIMIC cache: `/root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt`
- ECGTwin paired MIMIC cache: `/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt`
- ECGTwin author repro outputs: `/root/autodl-tmp/ecgtwin_author_repro/<run_name>/`
- EfficientNet super5 outputs: `/root/autodl-tmp/triple_labels/<run_name>/`

External model repos are expected to live on the data disk and be linked into
`model/`:

```text
model/DeepECG                 -> /root/autodl-tmp/models/DeepECG
model/ECGTwin                 -> /root/autodl-tmp/models/ECGTwin
model/advdiff                 -> /root/autodl-tmp/models/advdiff
model/ecg_ptbxl_benchmarking  -> /root/autodl-tmp/models/ecg_ptbxl_benchmarking
```

On a new AutoDL host, run:

```bash
bash scripts/bootstrap_model_repos.sh
```

`.gitmodules` is currently an external-model URL manifest, not active gitlink
submodules; `git submodule update --init` should not be relied on unless
`git ls-files --stage | grep '^160000'` shows real gitlinks.

Previously trained baseline:

```text
/root/autodl-tmp/triple_labels/super5/best_model.pt
/root/autodl-tmp/triple_labels/super5/training_log.json
/root/autodl-tmp/triple_labels/super5/train_result.json
/root/autodl-tmp/triple_labels/super5/eval_result_NORMguard.json
```

Baseline metrics remembered:

```text
PTB-XL fold10 AUROC/AUPRC ~= 0.9064 / 0.7754
PN2021 7-center NORMguard avg AUROC/AUPRC ~= 0.8390 / 0.5889
```

Archived first Scheme B PTB-XL diffusion run:

```text
/root/autodl-tmp/ecgtwin_class_super5/ptbxl_scheme_b_init/
best val loss ~= 0.03080
samples_n300/synth_waveforms.npz: 1500 ECGs, (N,1000,12), PTB-XL lead order
```

First real+synth classifier run:

```text
/root/autodl-tmp/triple_labels/super5_scheme_b_n300_r025/
PTB-XL fold10 AUROC/AUPRC ~= 0.9059 / 0.7671
PN2021 avg AUROC/AUPRC ~= 0.8337 / 0.5756
```

That first augmentation run did not beat the baseline overall. Treat it as
engineering closure and ablation evidence, not final evidence of benefit.

Current main experimental conclusion:

```text
TA-OMAT real-anchor latent PGD > direct synth-anchor diffusion for stability/speed.
ECGTwin diffusion is still useful as the reproduced generative framework and as
an ablation, but the final EfficientNetV2 enhancement claim should rely on the
ECGTwin VAE latent manifold plus target-center anchors.
```

## ECGTwin Facts To Preserve

- Original ECGTwin VAE input/output: `(B, 1024, 12)`, channels-last, raw mV, ECGTwin/MIMIC lead order.
- VAE latent: `(B, 4, 128)`.
- PTB-XL latent cache is already scaled by `0.18215`.
- ECGTwin/MIMIC lead order differs from PTB-XL by aVL/aVF swap.
- Convert decoded ECGTwin output to PTB-XL order before classifier training:

```python
ECGTWIN_TO_PTBXL_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
```

- Classifier input format should be `(N, 1000, 12)`, PTB-XL lead order, 100Hz, 10 seconds.
- Original ECGTwin DiT text path uses:

```text
text_embed:      (B, L, 768)
text_embed_mask: (B, L)
pat_info:        (hr, age, sex)
text_projector + CrossAttention
```

- Original IBE provides `base_vector` for AdaLN modulation. Preserve this fact
  for ECGTwin reproduction. Do not introduce a no-IBE replacement unless the
  user explicitly revives the archived Scheme B branch.
- Never make `text_embed_mask` all zeros. Use `null_text_embed + mask=1` for unconditional text dropout; all-zero masks can produce softmax NaN.

## Super5 Labels

Class order:

```text
CD, HYP, MI, NORM, STTC
```

Source of truth:

```text
scripts/triple_labels/label_schemes.py
```

PN2021 super5 mapping:

```text
SUPER5_PN2021_MAPPING_VERSION = v3_super5_normsuppress_20260501
PN2021_EVAL_CACHE_VERSION = v3_super5_normsuppress
```

- PN2021 has no official `SNOMED -> PTB-XL super5` crosswalk. The mapping is a
  project-defined semantic projection for external-center evaluation.
- Current PN2021 v3 splits direct positives and NORM suppression:
  `SNOMED_TO_SUPER5_POSITIVE`, `NORM_POSITIVE_SNOMEDS`,
  `NORM_SUPPRESS_SNOMEDS`.
- `Q wave abnormal` and `early repolarization` are suppress-only by default, not
  direct STTC positives.
- `sinus bradycardia`, `sinus tachycardia`, and `sinus arrhythmia` are not
  treated as PTB-XL-normal equivalents by default; they suppress NORM.
- If PN2021 mapping, parser, preprocessing, or class order changes, bump cache
  version and rebuild `/root/autodl-tmp/triple_labels/pn2021_eval_cache`.

Historical Scheme B prompt fragments:

| class | prompt |
|---|---|
| NORM | `normal ecg|sinus rhythm` |
| MI | `myocardial infarction|pathological q wave` |
| STTC | `st-t change|st segment abnormality|t wave abnormality` |
| CD | `conduction disturbance|bundle branch block` |
| HYP | `ventricular hypertrophy|left ventricular hypertrophy` |

For multi-label samples, concatenate fragments with `|`.

## Implementation Layout

Keep original `model/ECGTwin/` intact. Current active new code should live in:

```text
scripts/ecgtwin_author_repro/      # ECGTwin author IBE + DiT reproduction
scripts/pgd_cross_center/          # TA-OMAT / synth-anchor ablations
scripts/triple_labels/             # EfficientNet1DV2 training/eval
util/ecg_digital_features.py       # digital ECG validation
```

Archived no-IBE implementation:

```text
trash/cleanup_20260430/deprecated_no_ibe/methods/ecgtwin_class_super5/
trash/cleanup_20260430/deprecated_no_ibe/scripts/ecgtwin_class_super5/
```

## Current Training And Sampling Commands

ECGTwin author reproduction, stage 1 IBE:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_author_repro/train_ibe_repro.py \
  --output_dir /root/autodl-tmp/ecgtwin_author_repro/<run>/ibe_stage1 \
  --epochs 40 --batch_size 65536 --mini_batch_size 512 \
  --num_workers 4 --pin_memory --persistent_workers --prefetch_factor 2 \
  --drop_last --amp --amp_dtype bf16 --device cuda
```

ECGTwin author reproduction, stage 2 DiT:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_author_repro/train_dit_repro.py \
  --output_dir /root/autodl-tmp/ecgtwin_author_repro/<run>/dit_stage2 \
  --ibe_path /root/autodl-tmp/ecgtwin_author_repro/<run>/ibe_stage1/checkpoints/IBE_best.pth \
  --epochs 30 --batch_size 512 --num_workers 4 \
  --pin_memory --persistent_workers --prefetch_factor 2 \
  --amp --amp_dtype bf16 --device cuda
```

One-shot pipeline:

```bash
bash scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh
```

Synthetic classifier input:

```text
signals: (N, 1000, 12) float32, PTB-XL lead order, 100Hz
labels:  (N, 5) float32, class order CD/HYP/MI/NORM/STTC
```

Classifier augmentation is implemented in `scripts/triple_labels/train_ptbxl.py` with:

```text
--synth_npz
--synth_ratio
```

PN2021 eval:

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /path/to/model_dir \
  --skip_mimic
```

## PN2021 Evaluation Rules

Use these 7 centers:

```text
chapman_shaoxing
cpsc_2018
cpsc_2018_extra
georgia
ningbo
ptb
st_petersburg_incart
```

Hard-exclude `ptb-xl` / `ptbxl` shards because they leak PTB-XL training data.

Main metrics:

```text
macro AUROC
macro AUPRC
per-center delta
per-class delta
```

## MIMIC Strategy

MIMIC should be optional pretraining:

```text
MIMIC single-sample latent pretrain
-> PTB-XL fine-tune
-> sample only from PTB-XL fine-tuned checkpoint
-> EfficientNetV2 real+synth evaluation
```

Use `/root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt` first. It has 744,372 single ECG latents, each with label fields:

```text
subject_id, ecg_time, text, hr, age, sex
```

Do not use full paired MIMIC as a default single-sample/no-IBE route. The paired file has about 6.4M pairs and exists for original IBE reference-target training. After removing IBE, the reference ECG has no clean role and the pair expansion wastes compute.

MIMIC report labels:

```python
from scripts.triple_labels.label_schemes import mimic_report_to_super5
```

Apply PTB-XL NORM semantics:

```text
if any abnormal class CD/HYP/MI/STTC is positive, force NORM=0
```

MIMIC rough guarded distribution:

```text
N = 744,372
mapped nonzero ~= 737,554
CD   ~= 178,311
HYP  ~=  86,608
MI   ~= 167,472
NORM ~= 257,351
STTC ~= 298,721
```

Use patient-level split by `subject_id`, not random ECG split.

## Generation Quality Validation

ECGTwin author's validation stack:

- Signal level: FID, improved Precision, Recall, F1 in ECG feature space.
- Feature/physiology level: HR-MAE between generated ECG HR and target condition HR.
- Diagnostic/semantic level: ECG-text CLIP Score.
- Personal consistency: base-vector t-SNE, similarity score, silhouette coefficient.
- Downstream utility: auto-diagnosis improvement.
- Qualitative/case analysis: 12-lead figures, attention maps, prompt-to-prompt editing cases.

For current super5 downstream utility and any historical no-IBE/synthetic ablation, use a task-adapted stack:

1. Basic signal sanity using `util/ecg_viz.sanity_check`:
   - NaN/Inf
   - flatline/saturation
   - DC offset
   - amplitude p2p
   - Einthoven and aVR residuals
   - HR estimate and HR range
2. Digital ECG criteria using `util/ecg_digital_features.py` and `docs/ecg_digital_thresholds.md`:
   - HR, RR-CV, P wave, PR, QRS, ST/T, voltage criteria.
   - Existing prior digital validation suggests NORM/MI/STTC are stronger; HYP/CD may fail absolute voltage or detailed conduction criteria.
3. Class/prompt consistency:
   - Use the frozen EfficientNet1DV2 super5 baseline as a victim classifier.
   - Generated class should raise the intended class probability.
   - For NORM, abnormal probabilities should stay low.
   - Use this as a filter, not as the only proof.
4. Feature distribution:
   - Compute real-vs-synth feature FID/rFID and precision/recall/F1 using EfficientNet penultimate features or ECGTwin CLIP features.
   - Compare synthetic samples to PTB-XL fold 1-8 and fold 9.
5. Downstream utility:
   - PTB-XL fold10 macro AUROC/AUPRC.
   - PN2021 7-center macro AUROC/AUPRC.
   - per-class and per-center deltas.

Expected `eval_synth.py` artifacts:

```text
synth_quality_summary.json
per_sample_quality.csv
per_class_quality.csv
victim_scores.csv
feature_distribution.json
digital_criteria_report.md
figures/*.png
```

Do not claim generated ECGs are clinically valid solely because they improve a classifier. The thesis argument should combine visual quality, physiology sanity, prompt/class consistency, feature distribution, and downstream external evaluation.

## Visualization

Use `util/ecg_viz.py`.

Important functions:

```text
plot_ecg
plot_with_report
sanity_check
plot_ecg_ecgtwin_style
plot_ecg_ecgtwin_gallery
```

`plot_ecg_ecgtwin_gallery` was added to show all 12 leads in one large figure with enough vertical row spacing to avoid peaks overlapping adjacent leads.

Gallery-safe output previously generated under:

```text
/root/autodl-tmp/ecgtwin_class_super5/ptbxl_scheme_b_init/ecgtwin_gallery_safe/
```

## Logging Requirements

Every diffusion run should save:

```text
run_config.json or config.yaml
train.log
metrics.jsonl
loss_curve.csv
loss_curve.png
checkpoints/latest.pt
checkpoints/best.pt
samples/*.npz
figures/*.png
```

Every downstream classifier run should save:

```text
training_log.json
train_result.json
eval_result.json
baseline_vs_synth_metrics.csv
pn2021_per_center_delta.csv
```

## Fallbacks

- If `class_text` is unstable or hurts fold9, use `class_only` as the thesis main result and keep text prompt as compatibility/demo.
- If HYP/CD synthetic samples fail quality gates, only use trusted classes such as NORM/MI/STTC for augmentation and report HYP/CD as limitations.
- If MIMIC pretraining does not improve downstream metrics, write it as an ablation: large-scale MIMIC improves/changes latent modeling but noisy report labels and single-center domain shift limit augmentation benefits.
- If PN2021 average does not improve, analyze per-center/per-class deltas and make the claim narrower.
