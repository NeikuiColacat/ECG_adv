# ECG_adv Thesis Archive Agent Notes

This file is durable project memory for coding agents working in the cleaned
`ECG_adv` graduation-thesis archive branch.

## Current Branch Purpose

The goal of this branch is a clean, reproducible repository for `thesis.md` and
disc archival. It should be understandable from `README.md` and
`docs/thesis_reproduction.md` without reading historical chat logs.

Current worktree in this environment:

```text
/home/neiku/graduate_project/ECG_adv
```

Main thesis source:

```text
/home/neiku/graduate_project/thesis.md
```

## Active Thesis Route

The active reproduction story is:

```text
PTB-XL super5 fixed low-sample split
-> EfficientNetV2 real2000 baseline
-> ECGTwin author IBE + DiT reproduction
-> ECGTwin no-token synthetic pretrain + real2000 fine-tune
-> ECGTwin center-prompt-token synthetic pretrain + real2000 fine-tune
-> synthetic ECG quality proxy and visual examples
-> ONNX/TensorRT inference benchmark
-> Streamlit demonstration system
```

Super5 class order is fixed as:

```text
CD, HYP, MI, NORM, STTC
```

Source of truth:

```text
scripts/triple_labels/label_schemes.py
```

## Non-Mainline Boundary

These are historical or optional research lines, not thesis-mainline code:

- PN2021-C robustness and PN2021 cross-center result chasing.
- TA-OMAT, Latent-Hull, real-anchor/synth-anchor adversarial training.
- AdvDiff, PGD, online/offline adversarial training.
- AugMix.
- Tier-M and old 26-class/cross-center pipelines.
- Old self-distillation experiment lines.

They are kept under `legacy/` only for traceability. Mainline scripts and apps
must not import from `legacy/`.

## Key Entrypoints

Run everything through:

```bash
bash scripts/final_round/run_thesis_reproduction.sh <stage>
```

Important stages:

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

Human-readable mapping:

```text
docs/thesis_reproduction.md
```

Machine-readable mapping:

```text
docs/thesis_repro_manifest.json
docs/artifact_manifest.json
```

Cleanup standard:

```text
docs/thesis_archive_cleanup_standard.md
```

## Environment And Data

- Use `uv`; do not rely on a hard-coded conda environment.
- Default artifact/data root is `${HOME}/autodl-tmp`, configurable through
  `ECG_ADV_DATA_ROOT`.
- Keep large arrays, checkpoints, generated pools, caches, ONNX exports,
  TensorRT engines, and logs under `ECG_ADV_DATA_ROOT`, not in git.
- `migrate_files/` and the parent-workspace
  `ecg_grad_repro_no_pn2021_*.tar.gz` bundle are external artifact sources.
  Restore them with:

```bash
bash scripts/final_round/run_thesis_reproduction.sh restore_artifacts
```

External model repos are expected under `model/` as local links or directories.
Recreate them with:

```bash
bash scripts/bootstrap_model_repos.sh
```

Do not commit host-specific symlink target changes under `model/`.

## ECGTwin Facts To Preserve

- ECGTwin VAE input/output: `(B, 1024, 12)`, channels-last, raw mV.
- ECGTwin latent: `(B, 4, 128)`.
- ECGTwin/MIMIC lead order differs from PTB-XL by aVL/aVF order.
- Convert decoded ECGTwin output to PTB-XL order before classifier training:

```python
ECGTWIN_TO_PTBXL_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
```

- Classifier-ready synthetic input is `(N, 12, 1000)` or `(N, 1000, 12)`
  depending on script boundary; verify the local docstring before changing a
  callsite.
- EfficientNetV2 training uses PTB-XL super5 labels, 100 Hz, 1000 points, and
  per-sample global normalization before the 250-sample model crop.
- Never make an all-zero `text_embed_mask`; use a real null text embedding with
  mask set when unconditional text dropout is needed.

## Mainline Code Map

- `apps/streamlit_ecg_demo/`: answer-defense demo UI.
- `scripts/final_round/`: thesis reproduction, preflight, packaging, evidence.
- `scripts/triple_labels/`: PTB-XL super5 split and EfficientNetV2 training.
- `scripts/ecgtwin_author_repro/`: ECGTwin IBE + DiT author-style reproduction.
- `scripts/ecgtwin_gen/`: prompt-token generation, gating, quality reports.
- `scripts/deploy/`: ONNX/TensorRT export and benchmark.
- `scripts/streamlit_demo/`: background commands used by the demo UI.
- `methods/ecgtwin_gen/prompt_token/`: prompt-token model/trainer.
- `util/`: shared preprocessing, lead mapping, plotting, signal quality, and
  Super5 victim helpers.
- `legacy/`: non-mainline history only.

## Verification Before Completion

Use the standard in `docs/thesis_archive_cleanup_standard.md`. At minimum:

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

`preflight` is the default archive/demo evidence gate. `preflight_full` checks
full fresh-rerun dependencies such as authorized PTB-XL/MIMIC-derived data and
exact historical input pools; missing full-rerun files must remain visible in
the generated missing-artifact reports.

## Editing Rules

- Use `apply_patch` for manual edits.
- Do not revert user changes unless explicitly asked.
- Do not commit large weights, datasets, ONNX files, TensorRT engines, local
  caches, or host-specific symlinks.
- Prefer `rg` for search when available.
