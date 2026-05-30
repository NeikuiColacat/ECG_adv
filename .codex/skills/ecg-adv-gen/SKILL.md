---
name: ecg-adv-gen
description: Work on the cleaned ECG_adv graduation-thesis reproduction repo: PTB-XL super5, EfficientNetV2, ECGTwin author IBE+DiT reproduction, prompt-token ECG generation, low-sample synthetic pretraining, ONNX/TensorRT deployment, Streamlit demo, and thesis artifact archival.
---

# ECG Adv Gen

Use this skill for the `ECG_adv` graduation-thesis archive under:

```text
/home/neiku/graduate_project/ECG_adv
```

The thesis source is:

```text
/home/neiku/graduate_project/thesis.md
```

## Active Thesis Route

Current mainline:

```text
PTB-XL super5 fixed train2000/val2000/test17799 split
-> EfficientNetV2 real2000 baseline
-> ECGTwin author IBE + DiT reproduction
-> ECGTwin no-token synthetic pretrain + real2000 fine-tune
-> ECGTwin center-prompt-token synthetic pretrain + real2000 fine-tune
-> synthetic ECG quality proxy and visual examples
-> ONNX/TensorRT inference benchmark
-> Streamlit demonstration system
```

PN2021-C, TA-OMAT, Latent-Hull, AdvDiff, AugMix, Tier-M, old cross-center
evaluation, and old self-distillation lines are non-mainline history. They
belong under `legacy/` and must not be imported by thesis reproduction
entrypoints unless the user explicitly revives them.

## Source Documents

Check these first:

```text
README.md
docs/thesis_archive_cleanup_standard.md
docs/thesis_reproduction.md
docs/thesis_repro_manifest.json
docs/artifact_manifest.json
```

## Code Map

- `apps/streamlit_ecg_demo/`: answer-defense demo UI.
- `scripts/final_round/`: thesis reproduction, preflight, package, evidence.
- `scripts/triple_labels/`: PTB-XL super5 split and EfficientNetV2 training.
- `scripts/ecgtwin_author_repro/`: ECGTwin IBE + DiT author-style reproduction.
- `scripts/ecgtwin_gen/`: prompt-token generation, gating, quality reports.
- `scripts/deploy/`: ONNX/TensorRT export and benchmark.
- `scripts/streamlit_demo/`: background commands used by the demo UI.
- `methods/ecgtwin_gen/prompt_token/`: prompt-token model and trainer.
- `util/`: shared preprocessing, lead mapping, plotting, Super5 victim, ECG
  signal-quality helpers.
- `legacy/`: non-mainline history only.

## Facts To Preserve

- Super5 class order: `CD, HYP, MI, NORM, STTC`.
- Label source of truth: `scripts/triple_labels/label_schemes.py`.
- ECGTwin VAE latent: `(B, 4, 128)`.
- ECGTwin decoded ECG: `(B, 1024, 12)`, raw mV, ECGTwin/MIMIC lead order.
- ECGTwin/MIMIC lead order and PTB-XL differ by aVL/aVF:

```python
ECGTWIN_TO_PTBXL_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
```

- Classifier-ready data uses PTB-XL lead order, 100 Hz, 1000 samples, and
  per-sample global normalization before the EfficientNetV2 crop.
- Do not overclaim center-token causality. The safe thesis claim is that
  ECGTwin synthetic pretraining plus real2000 fine-tuning improves the fixed
  PTB-XL low-resource protocol; center-prompt-token is the final method, with
  no-token and joint-training rows as controls.
- Full ECGTwin author reproduction may require MIMIC-derived ECGTwin caches as
  external `package: false` dependencies; MIMIC is not part of the Table
  6.5-6.8 downstream supervised training data.

## Environment

- Use `uv` from the repo root.
- Default artifact/data root: `${HOME}/autodl-tmp`, configurable with
  `ECG_ADV_DATA_ROOT`.
- Keep large datasets, checkpoints, generated pools, caches, ONNX exports,
  TensorRT engines, and logs under `ECG_ADV_DATA_ROOT`, not in git.
- TensorRT is an optional deploy extra tied to the target CUDA/TensorRT runtime.
- Recreate external model links with:

```bash
bash scripts/bootstrap_model_repos.sh
```

## Reproduction Entrypoint

Use:

```bash
bash scripts/final_round/run_thesis_reproduction.sh <stage>
```

Important stages include:

```text
env
restore_artifacts
preflight
preflight_full
thesis_assets
split
author_repro
low_sample
synthetic_pretrain_init
low_sample_rerun
ablation_6_8
medical_validity
figures
feature_dist
export_onnx
build_trt
benchmark
evidence
package_artifacts
streamlit
```

`preflight` checks the default archive/demo evidence scope.
`preflight_full` checks complete fresh-rerun dependencies and may report
authorized data or exact historical input pools that are intentionally not in
the default disc artifact package.

## Verification

Before claiming the archive is ready, run the relevant commands from
`docs/thesis_archive_cleanup_standard.md`, especially:

```bash
git status --short
uv lock --check
bash -n scripts/final_round/*.sh
uv run python -m compileall -q \
  apps scripts/final_round scripts/triple_labels scripts/ecgtwin_author_repro \
  scripts/ecgtwin_gen scripts/deploy scripts/streamlit_demo methods/ecgtwin_gen util
uv run pytest apps/streamlit_ecg_demo/tests scripts/final_round/tests util/tests -q
uv run ruff check --select F apps scripts methods util
uv run python -m json.tool docs/thesis_repro_manifest.json >/dev/null
uv run python -m json.tool docs/artifact_manifest.json >/dev/null
bash scripts/final_round/run_thesis_reproduction.sh preflight
```
