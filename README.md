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

Check the resolved paths:

```bash
bash scripts/final_round/run_thesis_reproduction.sh env
```

Run the Streamlit demo:

```bash
bash scripts/final_round/run_thesis_reproduction.sh streamlit
```

## Reproduction Entrypoints

| purpose | command |
|---|---|
| PTB-XL train2000/val2000/test17799 split | `bash scripts/final_round/run_thesis_reproduction.sh split` |
| Table 6.5-6.7 low-sample comparison | `bash scripts/final_round/run_thesis_reproduction.sh low_sample` |
| Table 6.4 synthetic ECG quality proxy | `bash scripts/final_round/run_thesis_reproduction.sh medical_validity` |
| Five-class ECG visualization examples | `bash scripts/final_round/run_thesis_reproduction.sh figures` |
| ONNX export | `bash scripts/final_round/run_thesis_reproduction.sh export_onnx` |
| TensorRT FP16 engine build | `bash scripts/final_round/run_thesis_reproduction.sh build_trt` |
| PyTorch/ONNX/TensorRT benchmark | `bash scripts/final_round/run_thesis_reproduction.sh benchmark` |
| Thesis evidence summary | `bash scripts/final_round/run_thesis_reproduction.sh evidence` |

Detailed mapping from paper tables/figures to code and artifacts is in
`docs/thesis_reproduction.md` and `docs/thesis_repro_manifest.json`.

## External Model Repos

External repositories are linked under `model/` and stored on the data disk.
Recreate links on a new host with:

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
- `pyproject.toml` and `uv.lock`

Ignored or external:

- `.venv/`
- `migrate_files/`
- `__pycache__/`
- large datasets and checkpoints
- generated ECG pools
- ONNX and TensorRT engine binaries

For durable project memory and ECG-specific caveats, see `AGENTS.md`.
