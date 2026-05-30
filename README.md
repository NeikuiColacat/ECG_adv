# ECG_adv Graduate Thesis Archive

This branch is the cleaned reproduction branch for the graduation thesis
`thesis.md`.

The thesis route reproduced by this repository is:

```text
PTB-XL super5 fixed low-sample split
-> EfficientNetV2 real2000 baseline
-> ECGTwin no-token synthetic pretrain + real2000 fine-tune
-> ECGTwin center-prompt-token synthetic pretrain + real2000 fine-tune
-> synthetic ECG quality proxy analysis
-> ONNX/TensorRT inference benchmark
-> Streamlit demonstration system
```

## Quick Start

Install the Python environment with uv:

```bash
uv sync --all-groups
```

Set the data/artifact root. Large datasets, checkpoints, generated ECG pools,
ONNX files, and TensorRT engines live here instead of in git:

```bash
export ECG_ADV_DATA_ROOT="${HOME}/autodl-tmp"
```

If the `migrate_files/` archives are present on a fresh machine, restore the
packaged runtime/reproduction artifacts before running preflight. If the parent
workspace also contains `ecg_grad_repro_no_pn2021_*.tar.gz`, the same command
restores that external no-PN2021 reproducibility bundle as well:

```bash
bash scripts/final_round/run_thesis_reproduction.sh restore_artifacts
```

Check the resolved paths and required artifacts:

```bash
bash scripts/final_round/run_thesis_reproduction.sh env
bash scripts/final_round/run_thesis_reproduction.sh preflight
```

Run the Streamlit demo:

```bash
bash scripts/final_round/run_thesis_reproduction.sh streamlit
```

## Reproduction Entrypoints

| purpose | command |
|---|---|
| restore `migrate_files` tar archives | `bash scripts/final_round/run_thesis_reproduction.sh restore_artifacts` |
| thesis image/link check | `bash scripts/final_round/run_thesis_reproduction.sh thesis_assets` |
| archive artifact preflight | `bash scripts/final_round/run_thesis_reproduction.sh preflight` |
| full rerun data/checkpoint preflight | `bash scripts/final_round/run_thesis_reproduction.sh preflight_full` |
| PTB-XL train2000/val2000/test17799 split | `bash scripts/final_round/run_thesis_reproduction.sh split` |
| ECGTwin IBE + DiT author reproduction | `bash scripts/final_round/run_thesis_reproduction.sh author_repro` |
| Table 6.5-6.7 archived low-sample summary | `bash scripts/final_round/run_thesis_reproduction.sh low_sample` |
| synthetic-pretrain init checkpoints | `bash scripts/final_round/run_thesis_reproduction.sh synthetic_pretrain_init` |
| Table 6.5-6.7 full training rerun | `bash scripts/final_round/run_thesis_reproduction.sh low_sample_rerun` |
| Table 6.8 ablations | `bash scripts/final_round/run_thesis_reproduction.sh ablation_6_8` |
| Table 6.4 synthetic ECG quality proxy | `bash scripts/final_round/run_thesis_reproduction.sh medical_validity` |
| Five-class ECG visualization examples | `bash scripts/final_round/run_thesis_reproduction.sh figures` |
| feature distribution analysis | `bash scripts/final_round/run_thesis_reproduction.sh feature_dist` |
| ONNX export | `bash scripts/final_round/run_thesis_reproduction.sh export_onnx` |
| TensorRT FP16 engine build | `bash scripts/final_round/run_thesis_reproduction.sh build_trt` |
| PyTorch/ONNX/TensorRT benchmark | `bash scripts/final_round/run_thesis_reproduction.sh benchmark` |
| Thesis evidence summary | `bash scripts/final_round/run_thesis_reproduction.sh evidence` |
| package weights/demo artifacts | `bash scripts/final_round/run_thesis_reproduction.sh package_artifacts` |

Detailed mapping from paper tables/figures to code and artifacts is in
`docs/thesis_reproduction.md`, `docs/thesis_repro_manifest.json`, and
`docs/artifact_manifest.json`.

`preflight` checks the default archive scope: files that are packageable and
required for the delivered demo, evidence tables, figures, ONNX/TensorRT demo
path, and shipped ECGTwin weights. `preflight_full` checks the full fresh-rerun
scope, including authorized PTB-XL/MIMIC-derived data and exact Table 6.8 input
pools or synthetic-pretrain initialization checkpoints that may be too large or
license-sensitive for the default archive bundle. `package_artifacts` copies the
archive-required packageable entries from `docs/artifact_manifest.json`,
including the required model weights, ECGTwin checkpoints, demo samples,
generated pools, ONNX export, optional TensorRT engine, and thesis evidence
files. It writes
`artifact_manifest.resolved.json`, `checksums.sha256`,
`missing_artifacts.json`, and `missing_artifacts.md` under
`${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/` so missing required files
remain visible in both machine-readable and human-readable form. Manifest
entries marked `package: false` are external data dependencies rather than
default disc artifacts; entries marked `archive_required: false` remain required
for a full rerun but are skipped by the default archive package unless
`package_thesis_artifacts.py --include-rerun` is used.
When full-scope dependencies are missing, `preflight_full` also writes
`${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/full_preflight_missing_artifacts.json`
and `.md` for machine and human review.

`low_sample` is the light archive verification path: it summarizes the shipped
`train_result.json` files from the fixed PTB-XL custom split, falling back to
the committed `artifacts/evidence_pack/` result JSONs when the original data-disk
paths are absent. Use `low_sample_rerun` only when the full synthetic-pretrain
initialization checkpoints are present.

For a quick code and UI smoke check:

```bash
uv run pytest apps/streamlit_ecg_demo/tests scripts/final_round/tests util/tests -q
```

## External Model Repos

External repositories are linked under `model/` and stored on the data disk.
They are not tracked by git. Recreate links on a new host with:

```bash
bash scripts/bootstrap_model_repos.sh
```

Do not commit host-specific symlink target changes such as
`model/ECGTwin -> /home/<user>/autodl-tmp/...`.

## Archive Boundary

Tracked source:

- `apps/streamlit_ecg_demo/`
- `scripts/final_round/`
- `scripts/triple_labels/`
- `scripts/ecgtwin_author_repro/`
- `scripts/ecgtwin_gen/`
- `scripts/deploy/`
- `methods/ecgtwin_gen/prompt_token/`
- `util/`
- `artifacts/figures/` for thesis-ready screenshots and ECG example figures
- `artifacts/evidence_pack/` for thesis tables, small result summaries, and figures referenced by `thesis.md`
- `legacy/` for non-mainline historical experiments
- `pyproject.toml` and `uv.lock`

Ignored or external:

- `.venv/`
- `migrate_files/`
- `__pycache__/`
- large datasets and checkpoints
- generated ECG pools
- ONNX and TensorRT engine binaries

The strict thesis route must not import from `legacy/`. Legacy code is kept only
for traceability of exploratory experiments that are not part of `thesis.md`.

For durable project memory and ECG-specific caveats, see `AGENTS.md`.
