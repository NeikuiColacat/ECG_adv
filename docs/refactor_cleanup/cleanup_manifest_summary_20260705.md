# Mainline Cleanup Manifest Summary

Generated: 2026-07-05

This is phase 1 only: no source files were deleted or moved. The keep list is conservative and is meant to be reviewed before any archive/delete pass.

## Inputs

- Source of truth: `configs/active_scripts.yaml:latest_mainline`
- Launcher: `scripts/run_experiment.py`
- Method: PN2021/PN2021-C VAE-LHAT + three-chain AugMix
- Mapping: `v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`
- Class order: `CD,HYP,MI,NORM,STTC`

## Outputs

- Keep list: `docs/refactor_cleanup/mainline_keep_list_20260705.txt`
- Archive candidates: `docs/refactor_cleanup/archive_candidates_20260705.csv`

## Counts

- Tracked text/config/code lines scanned: 216681
- Conservative keep-list files: 150
- Conservative keep-list text lines: 44363
- Archive/review candidate files: 1154
- Archive/review candidate text lines: 172318

## Keep Reasons

- `python_import_closure`: 90 files
- `latest_mainline_yaml_or_extend`: 18 files
- `static_public_surface`: 11 files
- `core_regression_test`: 11 files
- `runner_adapter`: 6 files
- `runner_entrypoint`: 6 files
- `data_or_label_protocol`: 3 files
- `package_init`: 3 files
- `postprocess_entrypoint`: 2 files

## Archive Candidate Categories

- `archive_historical_config`: 491 files, 15942 text lines
- `archive_remove_from_public`: 463 files, 86314 text lines
- `review`: 70 files, 13442 text lines
- `archive_noncore_test`: 63 files, 13663 text lines
- `archive_doc_or_report`: 53 files, 33489 text lines
- `review_runner_not_in_closure`: 14 files, 9468 text lines

## First Safe Archive Pass

1. Remove `trash/` from the public release tree after preserving this CSV and the existing `trash/legacy_scripts_20260701/README.md` provenance note.
2. Move `docs/reports/archive/`, `docs/tmp_html/`, and `docs/papers/` out of the public release tree or replace them with a compact bibliography/provenance manifest.
3. Move `configs/experiments/*` not listed in the keep list into a generated config archive index instead of keeping hundreds of historical candidate YAMLs in the active config folder.
4. Review `ecg_adv_gen/runner/*` and `ecg_adv_gen/config/adapters/*` candidates that are not in the import closure before deleting; these are code, not pure archive.

## Required Verification Before Deleting

```bash
git diff --check
pytest -q util/tests/test_config_loader.py util/tests/test_active_script_index.py util/tests/test_pn2021c_eval_adapter.py util/tests/test_ecgfounder_pn2021c_evaluator.py util/tests/test_data_contracts.py util/tests/test_labels_super5.py util/tests/test_pn2021_eval_cache.py util/tests/test_ref_exclusion.py util/tests/test_pn2021c_protocol.py util/tests/test_pn2021_corruptions.py util/tests/test_pn2021_metric_views.py
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
```
