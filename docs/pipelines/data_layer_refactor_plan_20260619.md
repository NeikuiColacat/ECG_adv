# Data Layer Refactor Plan - 2026-06-19

Status: active data-module starting point after the 2026-06-19 parallel
read-only scan.

This document consolidates the prior temporary HTML plan
`docs/tmp_html/modular_refactor_plan_20260614.html`, the durable workspace
refactor plan, and the 2026-06-19 subagent findings. It is a refactor guide,
not a source of experiment metrics.

## Authority

Use these current files before older reports:

- `configs/active_scripts.yaml`: managed launch surfaces, package boundaries,
  legacy entrypoints, inactive configs, and artifact-risk notes.
- `configs/README.md`: YAML runner contract, typed adapters, local-config
  boundary, dry-run and execute flow.
- `docs/pipelines/refactor_preflight_20260614.md`: latest refactor preflight
  pattern and CPU gate style.
- `docs/pipelines/ai_agent_workspace_refactor_plan_20260604.md`: broad
  AI-agent-friendly modularization direction.
- `docs/pipelines/dataset_preprocessing_pipeline.md`: signal shape, sampling,
  lead order, and normalization rules.

Treat older `/root/...` commands as historical on this migrated host. Current
host paths are `/home/linbinhao/ECG_adv_Gen` and
`/home/linbinhao/ECG_adv_data`.

## Non-Move Rule

Do not move active legacy entrypoints directly. Keep their paths stable until
all of these are updated together:

- package helper with CPU tests;
- legacy wrapper or import alias;
- `configs/active_scripts.yaml` if a declared source-of-truth path changes;
- YAML dry-run parity for managed configs;
- data/preprocess smoke tests.

High-risk entrypoints that must remain callable:

- `scripts/triple_labels/train_ptbxl.py`
- `scripts/triple_labels/eval_crosscenter.py`
- `scripts/triple_labels/eval_pn2021_corruptions.py`
- `scripts/pgd_cross_center/synth_online_at_super5.py`
- active `scripts/paper/*` wrappers declared in `configs/active_scripts.yaml`

## Data Contracts To Preserve

The first data-module refactor is not generic cleanup. It must preserve these
behavioral contracts:

- Super5 class order: `CD,HYP,MI,NORM,STTC`.
- PN2021 mapping: `v7_super5_sjr_rgq_review_20260528`,
  hash `555ec85d5b51`.
- Target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- Eval centers: the seven PN2021 centers, excluding `ptb-xl` / `ptbxl` leak
  shards.
- Classifier signal contract: 100 Hz, length 1000, `(N,1000,12)`,
  PTB-XL lead order.
- Dataset item contract: training datasets may emit `(12,crop_len)` tensors,
  but caches and NPZ artifacts stay channel-last unless explicitly documented.
- ECGTwin decoded signals: `(B,1024,12)` in ECGTwin/MIMIC lead order, converted
  to `(B,1000,12)` PTB-XL order with reorder indices
  `(0,1,2,3,5,4,6,7,8,9,10,11)`.
- ECGFounder input contract: 500 Hz, `(12,5000)`,
  `official_ptbxl_eval`.
- K500 reference records used for adaptation must be excluded from target
  validation/evaluation views.
- PN2021 all-zero-kept and drop-all-zero views must remain separate.
- Cache names, cache metadata, mapping metadata, and PN2021-C cache versions
  must not silently change.

## Current Data Surface

Package-owned helpers already exist:

- `ecg_adv_gen/data/contracts.py`: paper protocol and preprocess contract.
- `ecg_adv_gen/data/synthetic_npz.py`: synthetic classifier NPZ loading and
  `(N,1000,12)` normalization.
- `ecg_adv_gen/data/latent_pools.py`: `.latent.npz` loading and shape checks.
- `ecg_adv_gen/data/pn2021_records.py`: PN2021 metadata, primary class,
  hash fold, and prompt-token cache records.
- `ecg_adv_gen/data/pn2021_waveforms.py`: CPU-safe materialization helper.
- `ecg_adv_gen/data/pn2021_raw_kshot.py`: selected raw1000 K-shot materializer.
- `ecg_adv_gen/data/kshot_artifacts.py`: K-shot path and ref-meta helpers.
- `ecg_adv_gen/evaluation/pn2021_eval_cache.py`: clean PN2021 cache naming and
  metadata helpers.
- `ecg_adv_gen/preprocessing/signals.py`: shape, lead-axis, ECGTwin-to-PTBXL,
  and decoded-to-classifier helpers.
- `ecg_adv_gen/preprocessing/classifier.py`: package-owned classifier
  preprocessing facade for legacy `crosscenter_v2` behavior.

Still brittle or duplicated:

- `scripts/triple_labels/train_ptbxl.py` is still imported by evaluation and
  adaptation scripts for PTB-XL preprocessing, labels, and dataset classes.
- `scripts/triple_labels/eval_crosscenter.py` still owns part of PN2021 clean
  cache loading, include/exclude slicing, and dataset construction.
- Target-real NPZ, synthetic NPZ, decoded ECGTwin, PN2021 clean caches, and
  PN2021-C paths each perform shape checks in different places.
- ECGTwin author reproduction uses `model/ECGTwin/utils/data_utils.py` directly;
  keep it as a separate compatibility boundary in this phase.

## Phase 0 - Baseline

Run this before broad data-layer edits:

```bash
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
micromamba run -n ECGTwin python -m pytest -q \
  util/tests/test_data_contracts.py \
  util/tests/test_preprocessing_signals.py \
  util/tests/test_synthetic_npz_contract.py \
  util/tests/test_latent_pools.py \
  util/tests/test_kshot_artifacts.py \
  util/tests/test_pn2021_index_kshot.py \
  util/tests/test_pn2021_waveforms.py \
  util/tests/test_ref_exclusion.py \
  util/tests/test_pn2021_eval_cache.py \
  util/tests/test_labels_super5.py \
  util/tests/test_pn2021_metric_views.py \
  util/tests/test_evaluation_metrics.py
micromamba run -n ECGTwin python -m pytest -q \
  util/tests/test_active_script_index.py \
  util/tests/test_config_loader.py
git diff --check
```

Resolved 2026-06-19 inventory drift: the read-only review found
`configs/experiments/pn2021c_effnet_threechain_locked_official_s5.yaml`
tracked but missing from `configs/active_scripts.yaml`. It is now registered as
an active wrapped EfficientNet1DV2 three-chain locked PN2021-C official-s5 eval
surface.

## Phase 1 - Classifier Preprocessing Ownership

Goal: package-own the shared classifier preprocessing behavior while keeping
legacy import paths stable.

Implemented first slice:

- Added `ecg_adv_gen/preprocessing/classifier.py` as the package-owned home for
  `unified_preprocess_to_1000`, `crop_signal_tc`, `reorder_leads_tc`,
  `resample_tc`, `per_sample_zscore`, and filter/pad helpers.
- Re-exported those helpers from `ecg_adv_gen.preprocessing`.
- Kept `scripts/crosscenter_v2/preprocess_utils.py` as a compatibility wrapper.
- Updated package code in `ecg_adv_gen/data/pn2021_raw_kshot.py` to import from
  `ecg_adv_gen.preprocessing` instead of from a legacy script path.
- Added a CPU test proving the helpers are package-owned and keep the
  `(1000,12)` classifier contract.

Do not yet rewrite all script imports. Script-level imports can move in small
batches once the wrapper is tested and YAML dry-run parity is stable.

## Phase 2 - PTB-XL Package Helpers

Next narrow slice:

- Add `ecg_adv_gen/data/ptbxl.py` or `ecg_adv_gen/training/ptbxl_super5.py`.
- Move pure helpers out of `scripts/triple_labels/train_ptbxl.py`:
  `preprocess_ptbxl_all`, label loading facade, fold split helpers, and cache
  metadata naming.
- Keep `train_ptbxl.py` as the CLI owner and compatibility wrapper.
- Add tests using tiny synthetic arrays and CSV rows; do not scan real PTB-XL.

Acceptance:

- Existing `train_ptbxl.py` imports still work.
- Managed dry-run for source training still expands to the same argv shape.
- No cache version changes unless an explicit rebuild plan is written.

## Phase 3 - Dataset Classes And NPZ Contracts

After PTB-XL pure helpers:

- Move dataset classes that are currently used as data APIs into a package
  module, for example `ecg_adv_gen/data/waveform_datasets.py`.
- Keep Torch-heavy code importable only when needed; config loading must stay
  CPU-light.
- Consolidate target-real NPZ validation with synthetic NPZ and raw1000 K-shot
  shape checks.

Candidate classes/functions:

- `PTBXLDatasetScheme`
- `SynthNPZDataset`
- `PN2021CachedCenterDataset`
- target-real NPZ signal/label/record-id validators

## Phase 4 - PN2021 Cache Facade

Only after Phase 1 and Phase 2 are stable:

- Move model-free PN2021 clean cache include/exclude/limit slicing into a
  package helper.
- Preserve cache reuse: K500 ref exclusion happens after loading complete
  center caches.
- Keep all-zero-kept and drop-all-zero metric views as evaluation-layer
  concepts, not data-loader defaults.

## Acceptance Checklist

A first data-layer PR is acceptable when:

- No GPU jobs were required.
- No generated data, checkpoints, cache arrays, or model handles are staged.
- Existing public import paths still work.
- `git diff --check` passes.
- The narrow data/eval CPU suite passes.
- Active-script/config-loader suite passes, or the pre-existing YAML inventory
  drift is explicitly separated.
- Mapping version/hash, cache versions, K500 ref-exclusion policy, signal shape,
  sampling rate, lead order, and normalization semantics are unchanged.
