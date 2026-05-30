# ECG_adv Thesis Archive Notes

This worktree is the cleaned thesis-reproduction branch for the graduation
project. The durable agent instructions are in `AGENTS.md`; this file is a short
compatibility note for Claude-style coding agents.

## Active Route

The `thesis.md` reproduction route is:

```text
PTB-XL super5 fixed train2000/val2000/test17799 split
-> EfficientNetV2 real2000 baseline
-> ECGTwin no-token synthetic pretrain + real2000 fine-tune
-> ECGTwin center-prompt-token synthetic pretrain + real2000 fine-tune
-> synthetic ECG quality proxy and visual examples
-> ONNX/TensorRT inference benchmark
-> Streamlit demonstration system
```

PN2021-C, TA-OMAT, Latent-Hull, AdvDiff, AugMix, Tier-M, old cross-center
evaluation, and old self-distillation notes are not thesis-mainline work. They
live under `legacy/` for traceability and must not be imported by mainline
entrypoints.

## Environment

- Use `uv` from the repository root: `uv sync --all-groups`.
- Put large datasets, checkpoints, generated pools, caches, and logs under
  `ECG_ADV_DATA_ROOT`, defaulting to `${HOME}/autodl-tmp`.
- Do not hard-code `/root/...` or `/home/<user>/...` paths in source files.
- External model repos are linked under `model/`; recreate links with
  `bash scripts/bootstrap_model_repos.sh`.

## Reproduction Entrypoint

Use:

```bash
bash scripts/final_round/run_thesis_reproduction.sh env
bash scripts/final_round/run_thesis_reproduction.sh preflight
```

The detailed table/figure mapping is in `docs/thesis_reproduction.md`; the
machine-readable reproduction index is `docs/thesis_repro_manifest.json`; large
file requirements and package policy are in `docs/artifact_manifest.json`.

## Mainline Code Map

- `apps/streamlit_ecg_demo/`: demo UI and default inference/training workflow.
- `scripts/final_round/`: thesis reproduction, preflight, packaging, evidence.
- `scripts/triple_labels/`: PTB-XL super5 labels, split, EfficientNetV2 training.
- `scripts/ecgtwin_author_repro/`: ECGTwin IBE + DiT author-style reproduction.
- `scripts/ecgtwin_gen/`: prompt-token training, generation, gating, quality.
- `scripts/deploy/`: ONNX/TensorRT export and inference benchmark.
- `scripts/streamlit_demo/`: background commands used by the demo UI.
- `methods/ecgtwin_gen/prompt_token/`: prompt-token model/trainer.
- `util/`: shared preprocessing, lead mapping, ECG plotting, Super5 victim and
  signal-quality helpers.
- `legacy/`: non-mainline historical experiments only.

## Verification

Before claiming the archive is ready, run the relevant checks from
`docs/thesis_archive_cleanup_standard.md`, especially:

```bash
uv lock --check
bash -n scripts/final_round/*.sh
uv run pytest apps/streamlit_ecg_demo/tests scripts/final_round/tests util/tests -q
uv run ruff check --select F apps scripts methods util
uv run python -m json.tool docs/thesis_repro_manifest.json >/dev/null
uv run python -m json.tool docs/artifact_manifest.json >/dev/null
bash scripts/final_round/run_thesis_reproduction.sh preflight
```
