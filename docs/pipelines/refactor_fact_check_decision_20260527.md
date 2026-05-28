# Refactor Fact Check Decision 2026-05-27

This note records the multi-agent fact check before continuing the
configuration refactor. It is a planning checkpoint, not an experiment result.

## Checked Areas

- YAML config loader, path resolver, schemas, and launcher.
- Managed mainlines:
  - `effnet_direct_k500_v6`
  - `effnet_vae_lhat_k500_v6`
  - `ecgfounder_direct_k500_v6`
  - `ecgfounder_inithead_fullft_k500_v6`
  - `ecgfounder_vae_lhat_k500_v6`
  - `pn2021_eval_v6_refexcluded`
- Metrics exporter and paper table exporter.
- Shared-server safety gates and paper-protocol leakage risks.

## Facts Found

Phase 0 and Phase 1 are sound enough to keep:

- Tracked YAML and local YAML are separated.
- Local paths are checked against `/home/linbinhao`.
- Mapping version/hash/class order are checked against code.
- `--execute` requires explicit `CUDA_VISIBLE_DEVICES` and `nvidia-smi`.

Phase 2 was initially only a wrapper skeleton. The fact check found concrete
protocol gaps:

- EfficientNet VAE-LHAT used invalid legacy argparse choices and would have
fallen back to old seed42 anchors unless `--anchor_base` was explicit.
- ECGFounder init-head full fine-tune missed required `--ref_meta_json`, missed
  `--k 500`, and did not pass `--init_head_path`.
- `command.sh` did not include environment prefixes, so it was not directly
  reproducible.
- Direct EfficientNet evaluation declared drop-all-zero reporting but did not
  request that PN2021 view from `eval_crosscenter.py`.
- Metrics export could attach the all-zero-kept sample count to drop-all-zero
  rows unless drop-specific stats were preferred.
- EfficientNet VAE-LHAT initially launched from the PTB-XL source checkpoint
  instead of the direct-K500 checkpoint and did not encode the current
  compatsoft direct-init parameter set.
- The first manifest version recorded raw commands but did not structure K500
  ref-meta files, init checkpoints/heads, child output roots, or expected eval
  artifacts.

## Decisions

Continue the refactor as a conservative wrapper-first migration.

Do not move large legacy scripts yet. The next useful milestone is not package
extraction; it is making every managed run protocol-equivalent, manifest-backed,
and table-exportable.

Concrete decisions after the fact check:

- Keep `scripts/paper/` as execution source for now.
- Treat YAML as the paper protocol source.
- Use explicit K500 seed20260531 anchors/ref-meta paths in wrappers.
- Use K500 internal validation plus source floor for selection; do not use
  held-out target labels for model selection.
- Keep all-zero-kept and drop-all-zero as separate `view` values.
- Require complete mapping metadata for metric artifacts.

## Current Fixed State

The wrapper layer now has CPU-only checks for:

- direct command uses K500 seed20260531 and managed `--out_root`;
- VAE-LHAT command uses seed20260531 `--anchor_base`, direct-K500
  `--init_ckpt`, valid `standardized/local_random` choices, compatsoft
  direct-init parameters, and `--quick_eval_source target_real_val`;
- ECGFounder frozen-feature direct K500 command uses the v6 linear-probe
  head/features, `--k 500`, `--source_k 500`, `--subset_seed 20260531`,
  `--val_fraction 0.2`, and no `--reset_head`, so the baseline is initialized
  from the PTB-XL Super5 linear-probe head;
- ECGFounder command passes `--ref_meta_json`, `--k 500`, `--init_head_path`,
  target-val count/seed, and a K500-compatible selector;
- ECGFounder VAE-LHAT command now expands one target-center command at a time,
  uses seed20260531 K500 latent anchors, K500 direct heads as frozen residual
  adapter base heads, hard-BCE anchor sampling, `--selection_source
  target_real_val`, and `--device cuda` so GPU id remains controlled by
  `CUDA_VISIBLE_DEVICES`;
- `run_manifest.json` now has `manifest_schema_version=2` and an
  `artifact_trace` block with metric views, mapping metadata, K500 refs,
  checkpoints/init heads, child output roots, and expected result artifacts;
- write-plan/execute now freezes the traced K500 references into a managed
  `k500_ref_ids.json` artifact with per-center record ids and source ref-meta
  hashes;
- write-plan/execute now writes managed `selection.json`, which validates the
  K500-internal/source-floor selection contract and explicitly records that
  held-out target labels and full target-center distributions are not used for
  model selection;
- VAE-LHAT configs now declare `logging.required_epoch_metrics`, and managed
  artifact verification checks `training_log.json` for the required attack
  success/ASR, validation-selection, and source-floor diagnostic fields;
- Managed child and postprocess commands now write per-command combined logs,
  per-command stdout/stderr logs, and run-level aggregate `stdout.log` /
  `stderr.log`; manifest run records include all log paths even when a command
  fails;
- Managed runner and postprocess output options are now required to include the
  resolved `runtime.run_id`, preventing unscoped output roots from overwriting
  earlier managed runs;
- experiment config schema now requires the `logging` lifecycle artifact
  section, and YAML-managed postprocess commands must declare structured
  expected artifacts with `role` and `path`;
- `configs/local/.gitignore` is locked by test to allow only `.gitignore` and
  `*.example.yaml`, so real local YAML remains private by default;
- generated legacy commands now pass a protocol audit for required options,
  K500/seed alignment, ref-meta paths, VAE direct-init checkpoints, and
  target-real validation selection, and refuses YAML-level
  `CUDA_VISIBLE_DEVICES` overrides;
- `--execute` now verifies required inputs before child scripts run, then checks
  launch/child artifacts after legacy commands return successfully and rejects
  runs whose eval JSON lacks the expected mapping version/hash;
- `command.sh` includes environment prefixes;
- metrics exporter writes drop-all-zero sample counts from drop-specific fields;
- metrics exporter writes both raw `view` and `canonical_view`, so ECGFounder
  target ref-excluded rows can be compared with PN2021 ref-excluded rows
  without merging all-zero-kept and drop-all-zero semantics;
- metrics exporter now supports method-level `--run-id` override and
  target-center filtering, so per-target-center artifacts with all-center
  eval JSONs can be safely collapsed into one method row without leaking
  non-target center rows;
- `scripts/merge_metrics_long.py` merges method-level `metrics_long.csv` files
  while rejecting mixed mappings and duplicate metric keys;
- paper table exporter rejects missing center metrics and filters by canonical
  view while preserving `raw_views` in the output.
- YAML-managed `postprocess.commands` are now supported for reporting/export
  steps. They are allowlisted to reporting scripts, run after child artifact
  verification, and their expected outputs are checked before a managed run is
  marked `succeeded`.
- Managed execute manifests now record child execution history in top-level
  `command_runs` and managed reporting execution history in top-level
  `postprocess_runs`, with command index, name, return code, status, log path,
  and UTC start/end timestamps for both success and failure paths. Completed
  managed runs also record `execution_started_at_utc`, `finished_at_utc`, and
  `duration_seconds`.
- Tracked YAML source files under `configs/defaults/`,
  `configs/experiments/`, and `configs/active_scripts.yaml` now have a
  CPU-only guard that rejects machine-local absolute paths,
  `CUDA_VISIBLE_DEVICES`, Python executable fields, and obvious secret-key
  fragments. Host paths remain isolated to `configs/local/*.example.yaml` or
  gitignored local YAML.
- `ecg_adv_gen.evaluation.selection` now owns the paper-safe model-selection
  contract. Config validation requires `allowed_data` to be exactly K500 train,
  K500 internal validation, and PTB-XL source floor; runner command audit also
  rejects held-out target selection references such as `pn2021_heldout`.
- `scripts/audit_managed_configs.py` now gives a CPU-only all-active-config
  preflight. It resolves every `active_wrapped` config from
  `configs/active_scripts.yaml`, runs command/protocol audit, builds manifest
  traces, and can verify required input artifacts without invoking legacy child
  scripts.
- `ecg_adv_gen.models.ecgfounder` now owns ECGFounder linear-probe feature
  cache names, K-shot head run dirs, VAE-LHAT run dirs, and K500 base-head
  lookup rules. The launcher manifest code uses this contract instead of
  carrying ECGFounder path patterns inline.
- `ecg_adv_gen.models.ecgfounder_heads` now owns the ECGFounder VAE-LHAT
  residual/feature adapter heads and frozen linear-head clone helper. The module
  is kept separate from the no-torch path-contract module.
- `ecg_adv_gen.models.ecgfounder_heads` now also owns ECGFounder full-FT dense
  initialization from the frozen-feature Super5 linear head checkpoint,
  including state-dict shape checks and dense-layer validation.
- `ecg_adv_gen.models.ecgfounder_inference` now owns cached-feature ECGFounder
  head prediction/evaluation helpers and full signal-model prediction/indexed
  split evaluation helpers. It takes metric/view callbacks from the runners
  rather than importing dated paper metrics directly.
- `ecg_adv_gen.evaluation.selection` now owns the ECGFounder full-FT
  source/internal-target validation score helper for `source_auprc`,
  `target_val_auprc`, and `source_plus_target_val_auprc`, alongside the
  paper-safe selection policy checks.
- `ecg_adv_gen.models.ecgfounder_torch` now owns ECGFounder full-FT torch
  input conversion from 100Hz channel-time ECG to 500Hz ECGFounder input plus
  per-sample global z-score normalization. It is intentionally not re-exported
  from `ecg_adv_gen.models.__init__`, keeping config/audit imports lightweight.
- `ecg_adv_gen.adaptation.anchor_sampling` now owns hard/uncertain anchor
  difficulty weighting and seeded hard-anchor sampling for ECGFounder VAE-LHAT.
  It stays as an explicit torch submodule instead of being re-exported from the
  pure adaptation facade.
- `ecg_adv_gen.adaptation.anchor_sampling` now also owns ECGFounder full-FT
  anchor class-weight parsing, weighted class quota allocation with repeat
  caps, and weighted index sampling. The full-FT runner keeps compatibility
  wrappers for the legacy function names.
- `ecg_adv_gen.training.stream_sampling` now owns weighted source/target/adversarial
  feature-stream DataLoader construction for ECGFounder VAE-LHAT, including
  class-weighted multi-label row sampling and stream-id tracking.
- `ecg_adv_gen.training.signal_streams` now owns ECGFounder full-FT
  signal-level cached/memory dataset wrappers, source/target/adversarial
  stream ids, teacher logits, and weighted DataLoader construction. The
  full-FT runner keeps only source fold/limit selection before calling the
  package helper.
- `ecg_adv_gen.adaptation.latent_hull_torch` now owns the shared
  `initial_hull_latent` deterministic start diagnostic used by ECGFounder
  VAE-LHAT and full-FT.
- `ecg_adv_gen.training.losses` now owns the shared masked BCE, clipped
  `pos_weight`, ordinary multi-label per-sample BCE, masked per-sample BCE,
  stream-weighted masked BCE for source/target/adv stream ids, ECGFounder
  VAE-LHAT attack-success diagnostics/merge logic, and selected-class pairwise
  ranking loss. `train_ptbxl.py` re-exports the shared BCE helpers for legacy
  callers, and active ECGFounder and EfficientNet K500/LHAT scripts import
  package helpers directly.
- `ecg_adv_gen.training.splits` now owns the deterministic random train/val
  split helper shared by the direct EfficientNet K500 and ECGFounder K-shot
  head fine-tune runners, while preserving their different zero-validation
  legacy policies through thin compatibility wrappers.
- `ecg_adv_gen.training.torch_utils` now owns `set_module_requires_grad`, used
  by the full-FT runner for freeze/unfreeze operations instead of a local helper.
- `ecg_adv_gen.labels.super5` now also exposes the canonical
  `label_mapping.pn2021_super5` artifact payload, and ECGFounder direct K500
  eval JSONs write that payload so launcher artifact verification can enforce
  v6 mapping metadata.
- `ecg_adv_gen.data.real_anchors` now owns ECGFounder target real-anchor latent
  base lookup, PN2021 cache label/signal matching, legacy min-one proportional
  K-shot anchor selection, and explicit selected-id pool loading. ECGFounder
  VAE-LHAT keeps compatibility wrappers for `real_anchor_base` and
  `load_anchor_pool`, while ECGFounder full-FT no longer imports the VAE-LHAT
  script only to locate anchors.
- `ecg_adv_gen.evaluation.target_splits` now owns the ECGFounder full-FT
  target K-shot train/validation split, including the legacy `split_seed+1701`
  random split and rarity/coverage-biased stratified split used for
  K500-internal validation.
- `ecg_adv_gen.data.kshot` now also owns selected ref-id parsing for legacy
  K-shot ref-meta JSON variants (`record_ids`, `ref_record_ids`,
  `selected_ref_record_ids`, top-level row lists, and `items`), plus the
  ECGFounder direct-head exact-K/source-K ref-meta fallback and min-one
  primary-class K-shot row selection. ECGFounder full-FT and direct-head
  runners keep compatibility wrappers around the package helpers.
- `ecg_adv_gen.evaluation.pn2021_metric_views` now owns the ECGFounder
  target-ref-excluded PN2021 view assembly path used by the linear-probe,
  direct-head, and VAE-LHAT runners through the legacy `evaluate_pn2021_views`
  wrapper. It also now owns the pure per-center row summary used by
  `eval_crosscenter.py` after labels/scores/record IDs exist, preserving the
  legacy PN2021 fields `n_scanned`, cache metadata, drop-all-zero metrics, and
  zero-valued `n_include_kept`.

CPU-only verification:

```text
pytest util/tests/test_config_loader.py util/tests/test_active_script_index.py \
       util/tests/test_adaptation_lhat.py util/tests/test_data_contracts.py \
       util/tests/test_anchor_sampling.py \
       util/tests/test_evaluation_metrics.py \
       util/tests/test_evaluation_views.py util/tests/test_labels_super5.py \
       util/tests/test_latent_hull_torch.py \
       util/tests/test_metrics_export.py util/tests/test_paper_tables.py \
       util/tests/test_evaluation_selection.py \
       util/tests/test_models_ecgfounder.py \
       util/tests/test_pn2021_index_kshot.py \
       util/tests/test_pn2021_metric_views.py \
       util/tests/test_preprocessing_signals.py \
       util/tests/test_real_anchors.py \
       util/tests/test_run_naming.py \
       util/tests/test_training_losses.py \
       util/tests/test_signal_streams.py \
       util/tests/test_signal_cache.py \
       util/tests/test_stream_sampling.py \
       util/tests/test_training_splits.py \
       util/tests/test_training_torch_utils.py -q

229 passed
```

No GPU training was launched during this fact check.

GPU execute smoke:

```text
CUDA_VISIBLE_DEVICES=0 scripts/run_experiment.py \
  --config configs/experiments/ecgfounder_direct_k500_v6.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id refactor_smoke_ecgfounder_direct_ep2_20260528 \
  --execute \
  --set training.epochs=2 \
  --set training.batch_size=256 \
  --set training.eval_batch_size=8192

manifest:
  /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_ecgfounder_direct_ep2_20260528/run_manifest.json

result:
  status=succeeded
  artifact_verification.passed=true
  checked_artifacts=18
  missing=0
  mapping_errors=0
```

The first ep1 smoke intentionally caught a real protocol gap: the legacy
ECGFounder direct K500 child artifacts existed, but their `eval_result.json`
files lacked `label_mapping.pn2021_super5`, so launcher verification rejected
them. The ep2 smoke passed after adding the package-level label-mapping payload
and writing it into ECGFounder direct K500 results.

## 2026-05-28 Reporting Backfill

Recent v6 artifacts were backfilled under:

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/metrics_export/refactor_v6_20260528/
```

Generated method-level exports:

- `effnet_direct_k500_v6/metrics_long.csv`
- `effnet_vae_lhat_k500_v6/metrics_long.csv`
- `ecgfounder_inithead_fullft_k500_v6/metrics_long.csv`
- `combined/metrics_long.csv`
- `combined_effnet/metrics_long.csv`

Generated table views:

- `paper_table_all_zero_kept_no_delta/paper_table.csv`: all three methods,
  no cross-backbone delta.
- `paper_table_effnet_all_zero_kept_delta/paper_table.csv`: EfficientNet
  direct vs VAE-LHAT same-backbone delta.
- `paper_table_drop_all_zero/paper_table.csv`: VAE-LHAT and ECGFounder
  drop-all-zero view. Direct EfficientNet v6 artifacts did not contain
  drop-all-zero metrics, so no direct baseline delta is reported for this view.

Four-center center-mean snapshot:

```text
all-zero-kept:
  effnet_direct_k500_v6                 AUROC 0.854321  AUPRC 0.487896
  effnet_vae_lhat_k500_v6               AUROC 0.872177  AUPRC 0.514767
  ecgfounder_inithead_fullft_k500_v6     AUROC 0.913539  AUPRC 0.629738

EfficientNet same-backbone delta:
  VAE-LHAT vs direct                    +0.017856 AUROC, +0.026872 AUPRC

drop-all-zero:
  effnet_vae_lhat_k500_v6               AUROC 0.901780  AUPRC 0.677552
  ecgfounder_inithead_fullft_k500_v6     AUROC 0.931305  AUPRC 0.740574
```

Additional managed-eval evidence:

- Full PN2021 eval wrapper execute run:
  `refactor_full_pn2021_eval_runscoped_20260528`
- Manifest:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_execute/refactor_full_pn2021_eval_runscoped_20260528/run_manifest.json`
- Result: `status=succeeded`, `artifact_verification.passed=true`,
  `missing=0`, `mapping_errors=0`.
- Metrics export:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/metrics_export/refactor_full_pn2021_eval_runscoped_20260528/metrics_long.csv`
- Postprocess dry-plan smoke:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/dry_runs/refactor_postprocess_plan_smoke_20260528/`
- Integrated postprocess execute run:
  `refactor_execute_postprocess_pn2021_eval_20260528`
- Integrated manifest:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_execute/refactor_execute_postprocess_pn2021_eval_20260528/run_manifest.json`
- Integrated result: `status=succeeded`,
  `artifact_verification.passed=true`, `n_checked=16`,
  six verified artifacts marked `postprocess=true`, `missing=0`,
  `mapping_errors=0`, and `input_verification.n_checked=36`.
- Integrated postprocess artifacts:
  `metrics_export/refactor_execute_postprocess_pn2021_eval_20260528/metrics_long.csv`
  with 84 metric rows, plus all-zero-kept and drop-all-zero paper tables.
- Integrated table center means:
  all-zero-kept `0.854321 / 0.487896` AUROC/AUPRC; drop-all-zero
  `0.881637 / 0.634631` AUROC/AUPRC.

## 2026-05-28 All-Mainline Postprocess Verification

Managed reporting postprocess is no longer limited to PN2021 eval. The six
active paper/evaluation configs now all declare YAML-managed postprocess
commands:

- `effnet_direct_k500_v6`: metrics export, all-zero-kept table,
  drop-all-zero table.
- `effnet_vae_lhat_k500_v6`: metrics export, all-zero-kept table,
  drop-all-zero table.
- `ecgfounder_inithead_fullft_k500_v6`: metrics export, all-zero-kept table,
  drop-all-zero table.
- `ecgfounder_direct_k500_v6`: metrics export and all-zero-kept table. This
  legacy direct-head script does not emit a drop-all-zero view.
- `ecgfounder_vae_lhat_k500_v6`: metrics export, all-zero-kept table,
  drop-all-zero table.
- `pn2021_eval_v6_refexcluded`: metrics export, all-zero-kept table,
  drop-all-zero table.

Fresh active-config audit with existing-input verification:

```text
output=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_all_mainline_postprocess_verify_20260528/
managed_experiment_count=6
passed_count=6
failed_count=0
```

Fresh dry-run/write-plan manifests were generated for all six active configs:

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/dry_runs/refactor_all_postprocess_plan_*_20260528/
```

Structured manifest check:

```text
effnet_direct_k500_v6: commands=1, postprocess=3
effnet_vae_lhat_k500_v6: commands=4, postprocess=3
ecgfounder_inithead_fullft_k500_v6: commands=4, postprocess=3
ecgfounder_direct_k500_v6: commands=1, postprocess=2
ecgfounder_vae_lhat_k500_v6: commands=4, postprocess=3
pn2021_eval_v6_refexcluded: commands=4, postprocess=3
ALL_DRY_RUN_MANIFESTS_OK=True
```

Current CPU regression after the later shared child-run contract, macro metric
helper extraction, and `eval_crosscenter.py` PN2021 per-center row helper
migration:

```text
229 passed
compileall passed
git diff --check passed
no production package/reporting entrypoint imports scripts.*
config loader import: torch_loaded False
```

Post-smoke active-config audit remains green:

```text
output=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_post_training_smoke_verify_20260528/
managed_experiment_count=6
passed_count=6
failed_count=0
```

## 2026-05-28 ECGFounder Direct Training Postprocess Smoke

A real non-PN2021 training wrapper smoke was run for the cached-feature
ECGFounder direct K500 baseline:

```text
config=configs/experiments/ecgfounder_direct_k500_v6.yaml
run_id=refactor_smoke_ecgfounder_direct_postprocess_ep2_20260528
output=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_ecgfounder_direct_postprocess_ep2_20260528/
CUDA_VISIBLE_DEVICES=0
overrides=training.epochs=2, training.batch_size=256, training.eval_batch_size=8192
```

The first execute attempt deliberately exposed a reporting bug after successful
child artifact verification: `export_metrics_long.py` recursively collected
`training_log.json`, whose root is a list, and failed before metrics export.
The fix was to stop treating `training_log.json` as a metrics artifact during
directory scans and to add explicit parsing support for the ECGFounder direct
`target_view` result shape.

The resumed integrated launcher run then passed:

```text
status=succeeded
input_verification.passed=true, n_checked=10
artifact_verification.passed=true, n_checked=22
missing=0, mapping_errors=0, content_errors=0
postprocess_artifacts_verified=4
metrics_long.csv rows=42
paper_table.csv center mean AUROC/AUPRC=0.8632266722 / 0.5105771844
```

This smoke verifies a real training wrapper plus managed postprocess; it is a
2-epoch engineering smoke, not a paper result.

## 2026-05-28 EfficientNet Direct Training Postprocess Smoke

A real EfficientNet1DV2 direct-K500 wrapper smoke was run after adding a
smoke-only config and nested PN2021 eval limits:

```text
config=configs/experiments/effnet_direct_k500_v6_smoke.yaml
run_id=refactor_smoke_effnet_direct_limit64_ep1_20260528
output=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_effnet_direct_limit64_ep1_20260528/
CUDA_VISIBLE_DEVICES=0
training=1 epoch, batch_size=64, eval_batch_size=64, num_workers=0
nested_eval=--eval_min_pos 2, --eval_pn2021_limit 64
```

The integrated launcher run passed:

```text
status=succeeded
input_verification.passed=true, n_checked=9
artifact_verification.passed=true, n_checked=32
missing=0, mapping_errors=0, content_errors=0
postprocess_artifacts_verified=6
metrics_long.csv rows=58
all-zero-kept paper table center mean AUROC/AUPRC=0.8563106854 / 0.6585915833
drop-all-zero paper table center mean AUROC/AUPRC=nan / nan
```

The drop-all-zero table is allowed to be `nan` in this smoke because
`--eval_pn2021_limit 64` leaves too few positive classes for at least one
target-center drop-all-zero subset. The smoke verifies launcher execution,
child training artifacts, nested PN2021 eval, v6 mapping metadata checks, and
managed postprocess; it is not a paper metric result.

## 2026-05-28 EfficientNet VAE-LHAT Training Postprocess Smoke

A real EfficientNet1DV2 VAE-LHAT wrapper smoke was added and executed:

```text
config=configs/experiments/effnet_vae_lhat_k500_v6_smoke.yaml
run_id=refactor_smoke_effnet_vae_lhat_width2_limit64_ep1_20260528
output=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_effnet_vae_lhat_width2_limit64_ep1_20260528/
CUDA_VISIBLE_DEVICES=0
training=1 epoch, batch_size=32, eval_batch_size=64, num_workers=0
latent_hull=hull_M=4, hull_steps=1, k_anchor=32, pgd_batch=8
latent_augmix=width=2, depth=1, severity=1
nested_eval=--eval_min_pos 2, --eval_pn2021_limit 64
```

The first attempt intentionally exposed a real smoke-config bug:
`latent_augmix_width=1` is invalid because branch 0 is reserved for the VAE
adversarial sample. The smoke config was fixed to the minimum legal
`latent_augmix_width=2`.

The integrated launcher run then passed:

```text
status=succeeded
input_verification.passed=true, n_checked=17
artifact_verification.passed=true, n_checked=32
missing=0, mapping_errors=0, content_errors=0
postprocess_artifacts_verified=6
metrics_long.csv rows=58
all-zero-kept paper table center mean AUROC/AUPRC=0.8787944647 / 0.6913762713
drop-all-zero paper table center mean AUROC/AUPRC=nan / nan
```

This smoke verifies the full managed VAE-LHAT runtime path for all four target
centers: ECGTwin VAE load/decode, latent-hull PGD, latent AugMix branch,
target K500 internal validation, final ref-excluded PN2021 eval, v6 mapping
metadata checks, and managed metrics/table postprocess. It is not a paper
metric result.

Post-smoke verification after moving active child-run naming/path contracts,
ECG macro AUROC/AUPRC helpers, and PN2021 per-center row assembly onto shared
lightweight package helpers:

```text
pytest refactor suite: 229 passed
compileall passed
git diff --check passed
no production package/reporting entrypoint imports scripts.*
config loader import: torch_loaded False
active-config audit:
  output=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_phase12_closeout_final_20260528/
  managed_experiment_count=6
  passed_count=6
  failed_count=0
```

## 2026-05-28 ECGFounder VAE-LHAT Postprocess Smoke

The ECGFounder VAE-LHAT wrapper now has a real four-center execute smoke:

```text
config=configs/experiments/ecgfounder_vae_lhat_k500_v6_smoke.yaml
run_id=refactor_smoke_ecgfounder_vae_lhat_limit64src_ep1_20260528
CUDA_VISIBLE_DEVICES=0
training=1 epoch, batch_size=64, eval_batch_size=512, source_train_limit=64
latent_hull=hull_M=4, hull_steps=1, k_anchor=16, pgd_batch=8
```

The first postprocess attempt exposed a real reporting gap: ECGFounder
VAE-LHAT stores final PN2021 results under `final_pn2021_views`, which the
metrics exporter did not parse. `ecg_adv_gen.reporting.metrics_export` now
extracts `final_ptbxl_fold10` and `final_pn2021_views`, with a regression test.

Final smoke evidence:

```text
status=succeeded
input_verification.passed=true, n_checked=21
artifact_verification.passed=true, n_checked=28
missing=0, mapping_errors=0, content_errors=0
metrics_long.csv rows=84
all-zero-kept center mean AUROC/AUPRC=0.9045766643 / 0.5976623929
drop-all-zero center mean AUROC/AUPRC=0.9221097550 / 0.7048084381
```

## 2026-05-28 ECGFounder Init-Head Full-FT Smoke

The ECGFounder init-head full fine-tune wrapper now has a real four-center
execute smoke:

```text
config=configs/experiments/ecgfounder_inithead_fullft_k500_v6_smoke.yaml
run_id=refactor_smoke_ecgfounder_inithead_cacheunion_ep1_20260528
CUDA_VISIBLE_DEVICES=0
training=1 epoch, batch_size=32, eval_batch_size=64, source_train_limit=64
cache_dir=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/paper_ecgfounder_fullft_signal_cache_union_20260528/cache
```

The wrapper reuses a user-owned signal-cache union built from existing cached
PTB-XL and target-center signal files, avoiding large WFDB cache rebuilds
during smoke verification.

Final smoke evidence:

```text
status=succeeded
input_verification.passed=true, n_checked=14
artifact_verification.passed=true, n_checked=24
missing=0, mapping_errors=0, content_errors=0
metrics_long.csv rows=126
all-zero-kept center mean AUROC/AUPRC=0.9092978816 / 0.6049917842
drop-all-zero center mean AUROC/AUPRC=0.9256955173 / 0.7112885361
```

## Next Plan

1. Keep legacy training/evaluation scripts in place while real wrapper runs are
   exercised.
2. Continue Phase 4 extraction behind compatibility shims: data loading,
   preprocessing, evaluation, and adaptation interfaces first; full training
   loop moves only after wrapper evidence is stable.
3. Add a small real execute smoke for one non-PN2021 training wrapper whenever
   GPU availability is safe, then decide whether the first-stage refactor is
   ready to commit.

## 2026-05-28 Phase 4 Start

The first package-extraction slice is deliberately small:

- Added `ecg_adv_gen.labels.super5` for stable Super5 protocol metadata.
- Switched config validation and metrics export to the labels facade instead
  of importing `scripts.triple_labels.label_schemes` directly.
- Kept full mapping/conversion logic in `scripts/triple_labels/label_schemes.py`
  for now, because moving that before PN2021/PTB-XL data contracts are extracted
  would risk breaking legacy experiment entrypoints.
- Added CPU-only tests to assert the facade remains synchronized with the
  legacy label scheme.

The second package-extraction slice keeps the same conservative boundary:

- Added `ecg_adv_gen.data.contracts` for the PTB-XL -> PN2021 data/preprocess
  protocol: target/eval centers, leak-excluded centers, source/target dataset
  names, classifier 100 Hz x 1000 input, PTB-XL lead order, and ECGTwin
  1024 -> 1000 lead-reorder metadata.
- Switched config validation to enforce this contract, so YAML changes that
  silently alter centers, sampling rate, length, or lead order fail CPU-only
  validation.
- Added `ecg_adv_gen.data.manifest` and `scripts/export_data_manifest.py` to
  write lightweight `data_manifest.json` files. The manifest records configured
  PTB-XL/PN2021 roots, cache dirs, PN2021 center dirs, and write-boundary state
  without loading ECG waveforms or recursively scanning large datasets. The
  exporter now also supports an explicit `--include-counts` mode that counts
  PN2021 `*.hea` files at center level and one grouping level below it for
  metadata-only size auditing.
- `scripts/run_experiment.py --write-plan` now emits `data_manifest.json` as a
  launch artifact next to `k500_ref_ids.json`.
- Added `ecg_adv_gen.preprocessing.signals` for pure NumPy waveform axis
  inference, ECGTwin -> PTB-XL lead reorder, deterministic linear resampling,
  and ECGTwin decoded-to-classifier channel-last conversion.
- Added `ecg_adv_gen.evaluation.views` for canonical PN2021/PTB-XL view names
  and macro metric aliases. Reporting now imports view semantics from this
  evaluation facade instead of defining them inside exporter/table modules.
- Kept actual dataset loading and waveform preprocessing in legacy scripts for
  now; this slice only freezes protocol metadata.

The latest package-extraction slice keeps training behavior stable while
reducing cross-script coupling:

- Added `ecg_adv_gen.training.losses` for masked BCE, masked per-sample BCE,
  stream-weighted masked BCE, and `pos_weight`.
- Added `ecg_adv_gen.models.ecgfounder_torch` for ECGFounder full-FT torch
  input conversion and per-sample global z-score normalization.
- Added `ecg_adv_gen.training.signal_streams` for ECGFounder full-FT
  signal-level cached/memory stream datasets and weighted source/target/adv
  DataLoader construction.
- Added `ecg_adv_gen.training.torch_utils` for small freeze/unfreeze utilities.
- Added `ecg_adv_gen.training.splits` for deterministic random train/val split
  handling shared by active K500 fine-tuning runners.
- `scripts/triple_labels/train_ptbxl.py` now imports and re-exports these
  helpers, preserving existing legacy imports.
- Active ECGFounder linear-probe, K-shot head, full fine-tune, VAE-LHAT,
  EfficientNet direct-K500, and legacy VAE-LHAT orchestration scripts now
  import the helpers from `ecg_adv_gen.training`.
- Added a real single-GPU `--execute` smoke for the ECGFounder cached-feature
  direct K500 wrapper. It verified CUDA safety gates, input verification,
  child-script invocation, launch artifacts, K500 refs, child artifacts, and
  v6 mapping metadata enforcement.

## Not Done Yet

- A real full-length training `--execute` run for EfficientNet VAE-LHAT,
  ECGFounder full fine-tune, or ECGFounder VAE-LHAT has not been launched under
  the new wrappers; the current training execute evidence is limited to short
  engineering smokes for all active training wrappers.
- PN2021 eval wrapper execute is now complete for both smoke and full
  ref-excluded evaluation, and the integrated managed postprocess path has also
  succeeded. The earlier multiprocessing `AF_UNIX path too long` and missing
  output-parent failures are retained here only as historical bugs that the
  current short-`TMPDIR` and parent-directory creation fixes address.
- Legacy scripts still contain substantial training/evaluation logic.
- `AGENTS.md` still contains older historical v3/v5 mapping notes; use
  `scripts/triple_labels/label_schemes.py` and `configs/defaults/common_v6_super5.yaml`
  as the current v6 source of truth.

## 2026-05-28 Continuation Verification

After the short-`TMPDIR` and output-parent fixes, the current worktree passed
the CPU-only refactor gates:

```text
pytest util/tests/test_config_loader.py util/tests/test_active_script_index.py \
       util/tests/test_adaptation_lhat.py util/tests/test_data_contracts.py \
       util/tests/test_anchor_sampling.py \
       util/tests/test_evaluation_selection.py \
       util/tests/test_evaluation_views.py util/tests/test_labels_super5.py \
       util/tests/test_latent_hull_torch.py \
       util/tests/test_metrics_export.py util/tests/test_models_ecgfounder.py \
       util/tests/test_paper_tables.py \
       util/tests/test_pn2021_index_kshot.py \
       util/tests/test_pn2021_metric_views.py \
       util/tests/test_preprocessing_signals.py \
       util/tests/test_real_anchors.py \
       util/tests/test_run_naming.py \
       util/tests/test_training_losses.py \
       util/tests/test_signal_streams.py \
       util/tests/test_signal_cache.py \
       util/tests/test_stream_sampling.py \
       util/tests/test_training_splits.py \
       util/tests/test_training_torch_utils.py -q

229 passed
```

Additional CPU checks passed:

- `compileall` for `ecg_adv_gen`, launcher/reporting scripts, active
  ECGFounder wrappers, `train_ptbxl.py`, and `eval_crosscenter.py`;
- `git diff --check`;
- no `from scripts.*` / `import scripts.*` imports from the production
  `ecg_adv_gen` package or wrapper/reporting entrypoints checked here.

The active-config audit also passed with existing input verification:

```text
managed_experiment_count=6
passed_count=6
failed_count=0
output=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_active_configs_runscoped_20260528/
```

The latest shared child-run contract, EfficientNet/ECGFounder macro metric
helper, and eval-crosscenter view helper audit supersedes this older run-scoped
audit:

```text
output=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_phase12_closeout_final_20260528/
managed_experiment_count=6
passed_count=6
failed_count=0
```

A PN2021 eval dry-run/write-plan now resolves to the short runtime tmp dir and
writes traceable plan files:

```text
run_id=refactor_dryrun_pn2021_eval_runscoped_20260528
TMPDIR=/home/linbinhao/tmp_ecg
output=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_dryrun/refactor_dryrun_pn2021_eval_runscoped_20260528/
```

This proves the wrapper plan and input contracts, not the full PN2021 eval
runtime. A real PN2021 eval smoke remains the next execution-level check.

## 2026-05-28 PN2021 Eval Smoke

To avoid making every launcher runtime check pay the full PN2021 evaluation
cost, a separate smoke-only config was added:

```text
configs/experiments/pn2021_eval_v6_refexcluded_smoke.yaml
```

It preserves the same v6 mapping, K500 ref-exclusion, 4 target-center matrix,
100 Hz x 1000 preprocessing, and all-zero/drop-all-zero views, but passes:

```text
--pn2021_limit 64
--num_workers 0
--min_pos 2
```

The real execute smoke succeeded on GPU 0:

```text
run_id=refactor_smoke_pn2021_eval_runscoped_limit64_20260528
manifest=/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_pn2021_eval_runscoped_limit64_20260528/run_manifest.json
status=succeeded
artifact_verification.passed=true
n_checked=10
missing=0
mapping_errors=0
content_errors=0
eval_count=4
all_eval_artifacts_under_run_id=true
eval_mapping=(v6_super5_clinician_review_20260524, 3adc673a60ad)
TMPDIR=/home/linbinhao/tmp_ecg
```

This confirms the short `TMPDIR`, output-parent creation, launch artifacts,
child eval artifacts, and v6 mapping-metadata verification under actual child
runtime. It is only an engineering smoke, not a paper metric result.

The managed configs now use `runtime.run_id` interpolation for child output
roots, for example
`${paths.output_root}/${experiment.name}/${runtime.run_id}/...`. This fixes the
previous overwrite risk where repeated PN2021 eval wrapper runs would target
the same `${paths.output_root}/${experiment.name}/${center}` location.
