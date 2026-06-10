# Thesis Archive Completion Audit

This document audits the cleaned archive branch against
`docs/thesis_archive_cleanup_standard.md`. It is intentionally evidence-based:
each item points to a current file, command, or artifact that proves the status.

## Scope

Branch:

```text
archive/thesis-repro-cleanup
```

The archive has two scopes:

- **Archive/demo scope**: source code, thesis evidence files, default Streamlit
  demo data, default model weights, ECGTwin runtime weights, ONNX/TensorRT demo
  artifacts, generated pools, screenshots, checksums, and missing-artifact
  reports that can be shipped with the graduation disc.
- **Full rerun scope**: all authorized raw/PTB-XL/MIMIC-derived data and exact
  historical intermediate pools/checkpoints needed to rerun every long training
  job from scratch.

The default graduation archive is judged by archive/demo scope. Full rerun
missing files are deliberately reported by `preflight_full` instead of being
silently ignored.

## Requirement Audit

| Standard item | Status | Evidence |
|---|---|---|
| README explains install, artifact placement, demo startup, and reproduction commands | Pass | `README.md`; `uv sync --all-groups`; `run_thesis_reproduction.sh` table |
| `docs/thesis_reproduction.md` maps thesis tables/figures to code and artifacts | Pass | `docs/thesis_reproduction.md`, section 3; Table 6.8 has per-row inputs/outputs/args |
| Machine-readable reproduction index exists | Pass | `docs/thesis_repro_manifest.json` parses with `python -m json.tool` |
| Artifact manifest covers required datasets, weights, pools, ONNX/TensorRT, results, figures | Pass | `docs/artifact_manifest.json`; archive preflight reports `missing_required=0` |
| Table 6.3 IBE/DiT metrics are traceable | Pass | `artifacts/evidence_pack/raw/ecgtwin_author_repro_summary.json`, loss-curve PNGs |
| Table 6.5-6.7 low-sample results are traceable | Pass | `artifacts/evidence_pack/raw/train_results/*.json`, `tables/per_class_main_results.csv`, `low_sample` stage |
| Table 6.8 ablation rows are explicit | Pass | `docs/thesis_reproduction.md` Table 6.8 section and `run_table_6_8_ablations.sh` |
| Streamlit demo entrypoint is available | Pass | `bash scripts/final_round/run_thesis_reproduction.sh streamlit` |
| Streamlit app smoke test covers startup/tabs/backend options | Pass | `apps/streamlit_ecg_demo/tests/test_app_smoke.py`; pytest passes |
| TensorRT is optional and can fall back when unavailable | Pass | `pyproject.toml` deploy extra; `apps/streamlit_ecg_demo/app.py` fallback to torch |
| Mainline code does not import `legacy/` or top-level `adversarial/` | Pass | `test_mainline_sources_do_not_import_top_level_adversarial`; grep audit clean |
| Non-mainline historical material is separated or labeled | Pass | `legacy/README.md`, `docs/final_round/README.md`, cleaned `AGENTS.md`/`CLAUDE.md` |
| Large model/data artifacts are outside git | Pass | `.gitignore`; `git ls-files` only contains one tiny curated Figure 3.3 `.npz` sample |
| Package command emits checksums and missing reports | Pass | `package_artifacts` writes `artifact_manifest.resolved.json`, `checksums.sha256`, `missing_artifacts.json`, `missing_artifacts.md` |
| Clean source package can be produced for the graduation disc | Pass | `package_source` writes `ECG_adv_source_*.tar.gz` and `.sha256` under `${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/source_code/` |
| Archive preflight can tell what is missing | Pass | `preflight` and `preflight_full` stages; full report paths under `${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/` |

## Verification Snapshot

The following commands were run on this branch after the final archive-boundary
cleanup:

```bash
uv lock --check
bash -n scripts/final_round/*.sh
uv run python -m compileall -q \
  apps scripts/final_round scripts/triple_labels scripts/deploy scripts/streamlit_demo \
  scripts/ecgtwin_author_repro scripts/ecgtwin_gen methods/ecgtwin_gen util
uv run ruff check --select F apps scripts methods util
uv run pytest apps/streamlit_ecg_demo/tests scripts/final_round/tests util/tests -q
uv run python -m json.tool docs/thesis_repro_manifest.json >/dev/null
uv run python -m json.tool docs/artifact_manifest.json >/dev/null
bash scripts/final_round/run_thesis_reproduction.sh thesis_assets
bash scripts/final_round/run_thesis_reproduction.sh preflight
bash scripts/final_round/run_thesis_reproduction.sh package_source
bash scripts/final_round/run_thesis_reproduction.sh package_artifacts
bash scripts/final_round/run_thesis_reproduction.sh env
bash scripts/final_round/run_thesis_reproduction.sh low_sample
bash scripts/final_round/run_thesis_reproduction.sh evidence
```

Observed results:

```text
pytest: 49 passed
ruff: All checks passed
thesis_assets: checked=10 missing=0
preflight: ok=50 missing_required=0 missing_optional=0 bad=0 skipped=21
package_source: wrote ECG_adv_source_20260609T1915.tar.gz, sha256 OK
package_artifacts: packaged 59 artifacts, missing=0
low_sample: wrote low_sample_summary.json and low_sample_summary.md
evidence: wrote final_thesis_evidence_summary.json and .md
```

## Full Rerun Gaps

`preflight_full` is expected to fail on a clean archive machine unless the
authorized raw datasets and exact historical intermediate pools are also
present. The currently reported full-scope missing set is:

- PTB-XL raw/cache metadata: `raw100.npy`, `ptbxl_database.csv`,
  `scp_statements.csv`, and the minimal preprocessing cache.
- Exact Table 6.8 historical generated pools for some rerun-only rows.
- ECGTwin author reproduction train/validation caches derived from MIMIC.
- Full-rerun synthetic-pretrain init checkpoints e18/e21 if not rebuilt.

These are not hidden failures: `preflight_full` writes machine-readable and
human-readable missing reports under:

```text
${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/full_preflight_missing_artifacts.json
${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/full_preflight_missing_artifacts.md
```

For the e18/e21 initialization checkpoints, the archive provides a rebuild
stage:

```bash
bash scripts/final_round/run_thesis_reproduction.sh synthetic_pretrain_init
```

The remaining raw/cache files require authorized dataset sources and are kept
out of the default disc artifact package by design.

## Current Completion Judgment

Archive/demo scope is complete: the repo contains the cleaned source tree,
paper-to-code mapping, machine-readable manifests, evidence pack, artifact
restore/package/preflight tooling, Streamlit demo entrypoint, and checksums.

Full rerun scope is transparent but not self-contained without external
authorized datasets and exact historical intermediate pools. This matches the
archive standard: those files are listed in `docs/artifact_manifest.json` and
reported by `preflight_full`, but are not committed to git.
