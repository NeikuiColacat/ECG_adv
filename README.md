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
| artifact/data preflight | `bash scripts/final_round/run_thesis_reproduction.sh preflight` |
| PTB-XL train2000/val2000/test17799 split | `bash scripts/final_round/run_thesis_reproduction.sh split` |
| ECGTwin IBE + DiT author reproduction | `bash scripts/final_round/run_thesis_reproduction.sh author_repro` |
| Table 6.5-6.7 archived low-sample summary | `bash scripts/final_round/run_thesis_reproduction.sh low_sample` |
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

`preflight` checks the full reproduction environment, including authorized
PTB-XL/MIMIC-derived data that is not committed to git. `package_artifacts`
copies the packageable entries from `docs/artifact_manifest.json`, including
the required model weights, ECGTwin checkpoints, demo samples, generated pools,
ONNX export, optional TensorRT engine, and thesis evidence files. Manifest
entries marked `package: false` are external data dependencies rather than
default disc artifacts.

`low_sample` is the light archive verification path: it summarizes the shipped
`train_result.json` files from the fixed PTB-XL custom split. Use
`low_sample_rerun` only when the full synthetic-pretrain initialization
checkpoints are present.

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
