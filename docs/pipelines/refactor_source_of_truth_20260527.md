# Refactor Source Of Truth 2026-05-27

This document records the current source-of-truth boundaries for the YAML
configuration refactor. It is not a replacement for experiment result files; it
explains which files are authoritative while the project is still transitioning
from dated scripts to managed configs.

## Current Refactor State

The refactor starts with wrappers and manifests, not by moving training code.

Completed traceability layers:

- YAML defaults, experiment configs, schemas, and local path example live under
  `configs/`.
- Local machine paths are isolated in `configs/local/*.yaml`; real local files
  are gitignored. `configs/local/.gitignore` is test-locked to allow only
  `.gitignore` and `*.example.yaml` into git.
- `scripts/run_experiment.py` resolves YAML, validates shared-server path
  boundaries, writes resolved config/manifest files, and can execute legacy
  scripts only after explicit GPU safety gates. It also supports audited
  whitelist `--set key=value` overrides for resource/training/method
  hyperparameters while rejecting paper protocol, path, K-shot, center, env,
  checkpoint, and raw runner argv changes.
- The launcher binds `runtime.run_id` before interpolation. Managed child
  outputs should be scoped under
  `${paths.output_root}/${experiment.name}/${runtime.run_id}/...` so repeated
  smoke/full runs do not overwrite previous child artifacts. The protocol
  audit now rejects managed child or postprocess output options that do not
  include the resolved `runtime.run_id`.
- The launcher now also supports YAML-managed `postprocess.commands`. These are
  restricted to reporting/export entrypoints, run only after child artifacts
  pass verification, and are included in final artifact verification before the
  manifest is marked `succeeded`.
- Managed execute manifests now also keep top-level `command_runs` and
  `postprocess_runs` execution histories. Each record contains command index,
  name, return code, status, log path, and UTC start/end timestamps, including
  failed child commands. Completed managed runs also keep
  `execution_started_at_utc`, final `finished_at_utc`, and `duration_seconds`.
- `util/tests/test_active_script_index.py` now scans tracked YAML sources under
  `configs/defaults/`, `configs/experiments/`, and
  `configs/active_scripts.yaml` to enforce the tracked/local boundary: no
  machine-local absolute paths, `CUDA_VISIBLE_DEVICES`, Python executable
  fields, or obvious secret-key fragments in paper-protocol YAML. It also
  checks the local-config `.gitignore` allowlist.
- `scripts/export_metrics_long.py` backfills existing result JSONs into
  `metrics_long.csv` and `artifact_manifest.json`.
- `scripts/merge_metrics_long.py` merges method-level `metrics_long.csv` files
  into one paper-table input while rejecting mixed mappings and duplicate
  metric keys.
- `scripts/export_paper_table.py` builds compact tables from a selected
  `metrics_long.csv` view.
- All six active paper/evaluation configs now declare YAML-managed reporting
  postprocess commands. Every active mainline can write a run plan whose
  `command.sh` contains the legacy child command(s) plus managed
  `export_metrics_long.py` and paper-table export steps. The one legacy
  ECGFounder direct-head path currently exports only the all-zero-kept table
  because that script does not emit a drop-all-zero view.
- `scripts/audit_managed_configs.py` performs a CPU-only audit across every
  `active_wrapped` experiment in `configs/active_scripts.yaml`, using the same
  config loader, command audit, manifest trace, and optional required-input
  checks as the launcher without invoking legacy child scripts.
- `ecg_adv_gen.runner.launch_plan` now owns durable run-plan rendering and
  materialization for YAML-managed launches: command text, resolved config
  files, data manifest, K-shot ref IDs, selection record, launch artifact
  status, and the initial agent-readable run record. `scripts/run_experiment.py`
  remains the CLI wrapper and safety gate rather than the owner of plan-file
  layout.
- `ecg_adv_gen.labels.super5` now holds stable Super5 protocol metadata used
  by config validation and reporting, and
  `ecg_adv_gen.labels.super5_mapping` owns the Super5 PTB-XL, PN2021, and MIMIC
  conversion policy. Legacy `scripts/triple_labels/label_schemes.py` re-exports
  that Super5 API for old entrypoints.
- `ecg_adv_gen.data.contracts` now holds the current PTB-XL -> PN2021 data and
  preprocessing contract used by YAML validation: target/eval centers,
  PTB-XL/PN2021 dataset names, leakage exclusions, 100 Hz x 1000 classifier
  input, and ECGTwin 1024 -> 1000 lead-reorder metadata.
- `ecg_adv_gen.data.manifest` and `scripts/export_data_manifest.py` build a
  lightweight `data_manifest.json` for configured PTB-XL/PN2021 roots, cache
  dirs, and PN2021 center dirs without loading waveforms or recursively scanning
  large datasets. The exporter also has an explicit `--include-counts` mode
  that counts PN2021 `*.hea` files at center level and one grouping level below
  it for a metadata-only size snapshot. `scripts/run_experiment.py
  --write-plan` now writes this
  manifest next to `k500_ref_ids.json` and the managed `selection.json`
  policy-safety record.
- `ecg_adv_gen.data.pn2021_index`, `ecg_adv_gen.data.pn2021_records`, and
  `ecg_adv_gen.data.kshot` now hold the legacy-compatible metadata helpers for
  PN2021 header parsing, record-id basename handling, age/sex metadata, Super5
  primary-class/SNOMED policy, deterministic hash folds, hybrid
  floor-plus-natural K-shot sampling, leak-center guards, ref/include meta
  loading, K-shot artifact schema checks, selected ref-id JSON variant parsing,
  and proportional primary-label K-shot selection. They also own the ECGFounder
  direct-head exact-K/source-K ref-meta fallback and min-one primary-class
  target selection used by frozen-feature K-shot head fine-tuning. The helpers
  do not load waveforms or build caches.
- `ecg_adv_gen.data.real_anchors` now owns target-center real-anchor latent
  base path resolution, PN2021 cache record-id/center matching, legacy
  min-one primary-class K-shot anchor selection, and explicit selected-id
  anchor pool loading. ECGFounder VAE-LHAT and ECGFounder full-FT now delegate
  their real-anchor pool assembly to this package helper while keeping the
  training loops in the dated runners.
- `ecg_adv_gen.preprocessing.signals` contains pure NumPy helpers for waveform
  axis inference, ECGTwin -> PTB-XL lead reorder, deterministic linear
  resampling, and ECGTwin decoded `(1024, 12)` / `(12, 1024)` conversion to
  classifier `(1000, 12)` channel-last format.
- `ecg_adv_gen.evaluation.views` contains canonical view names and macro metric
  aliases used by metrics export and paper tables. It keeps ECGFounder
  `target_*_refexcluded` raw views comparable with PN2021 ref-excluded views
  without merging all-zero-kept and drop-all-zero semantics.
- `ecg_adv_gen.evaluation.pn2021_metric_views` contains the first pure NumPy
  PN2021 metric-view assembly helpers for ref-excluded, all-zero-kept, and
  drop-all-zero views after labels/scores/record IDs already exist. It does not
  run inference, load PN2021 waveforms, or build caches. ECGFounder
  linear-probe/direct-head/VAE-LHAT runners now reach their target-ref-excluded
  PN2021 views through this shared assembly path, and `eval_crosscenter.py`
  uses `summarize_center_view` for its PN2021 per-center row shape while
  preserving legacy fields such as `n_scanned`, cache metadata, and
  `n_include_kept`.
- `ecg_adv_gen.evaluation.pn2021_eval_cache` owns clean PN2021 eval cache path
  naming, preprocess/cache metadata construction, legacy metadata
  compatibility, NPZ metadata decoding, and mmap cache read/write helpers.
  `eval_crosscenter.py` keeps its stable private helper names as wrappers while
  delegating cache metadata and mmap/NPZ loading to the package helper.
- `ecg_adv_gen.evaluation.metrics` contains the shared ECG macro AUROC/AUPRC
  helper with legacy ECGFounder and EfficientNet/triple-label class gates,
  optional unknown-label masking, configurable empty-metric values, and the
  per-class `n_valid` schema. ECGFounder zero-shot/linear-probe and
  EfficientNet `train_ptbxl.py` keep thin compatibility wrappers but no longer
  own the metric implementation.
- `ecg_adv_gen.evaluation.selection` contains the current paper-safe selection
  contract: allowed data is exactly K500 train split, K500 internal validation,
  and PTB-XL source floor; held-out PN2021 target references such as
  `pn2021_heldout` are rejected by config validation and runner command audit.
- `ecg_adv_gen.evaluation.target_splits` contains the ECGFounder full-FT
  target-center K-shot train/validation split helper. It preserves the legacy
  random and rarity/coverage-biased stratified behavior while keeping the
  selection split in the package-level evaluation policy surface.
- `ecg_adv_gen.adaptation.lhat` contains the first pure Latent-Hull online AT
  helpers for K500 internal validation masks, source/class weight parsing,
  anchor quota allocation, K-shot inverse-frequency weights, anchor-preserving
  soft labels, class trust maps, global z-score, capped Dirichlet branch
  weights, no-revisit stratified pool walking, same-label latent candidate
  indexing, the array-level latent AugMix mixer, anchor sampling mode parsing,
  and adversarial stream linear warmup. The legacy
  `scripts/pgd_cross_center/synth_online_at_super5.py` keeps the training loop,
  PGD generator, DataLoader, ECGTwin decode logic, and torch AugMix operator
  wrapper while re-exporting the imported helper names for compatibility.
  ECGFounder full-FT and VAE-LHAT scripts now import these shared helpers
  directly from the package rather than reaching through that legacy script.
- `configs/experiments/ecgtwin_prompt_token_online_at_minimal.yaml` is the
  first managed prompt-token gated-pool online-AT launch surface for
  `synth_online_at_super5.py`. It is launch-only evidence: the legacy script
  still owns training/decode logic, while config validation owns the same-run
  gated-pool, direct-K500 checkpoint, K500 target-real, and quick-eval
  selection contracts.
- `configs/experiments/ecgtwin_prompt_token_online_at_minimal_pn2021_eval.yaml`
  is the matching evaluation-only PN2021 ref-excluded launch surface for that
  prompt-token online-AT slice. It consumes the same-run online-AT model
  directory, excludes all four target-center K500 ref-meta files, and records
  metrics/table postprocess artifacts without promoting the result to trusted
  paper evidence.
- `ecg_adv_gen.models.ecgfounder` contains CPU-only ECGFounder filesystem and
  run-layout contracts for linear-probe feature caches, direct K-shot head
  run dirs, VAE-LHAT run dirs, and K500 base-head lookup. The config manifest
  code now uses these helpers instead of duplicating ECGFounder path patterns.
- `ecg_adv_gen.models.ecgfounder_heads` contains the torch-only ECGFounder
  linear-head adapters shared by VAE-LHAT: `ResidualAdapterHead`,
  `FeatureAdapterHead`, linear-head unwrapping, frozen linear-head cloning, and
  ECGFounder full-FT dense initialization from a simple Super5 head checkpoint.
  It is separate from `ecgfounder.py` so CPU-only config/path validation keeps
  its no-torch path-contract surface.
- `ecg_adv_gen.models.ecgfounder_inference` contains torch-based ECGFounder
  inference helpers: cached-feature head inference, clipped sigmoid, generic
  feature-head metric callbacks, PTB-XL fold evaluation, PN2021 feature-head
  view evaluation, plus full signal-model prediction and indexed split
  evaluation for full-FT runners. It accepts metric/view callbacks instead of
  importing dated paper scripts.
- `ecg_adv_gen.evaluation.selection` contains the paper-safe selection-policy
  contract and the ECGFounder full-FT source/internal-target validation score
  helper used by active K500 model selection.
- `ecg_adv_gen.adaptation.anchor_sampling` contains torch-based hard/uncertain
  anchor difficulty scoring, normalized sampling weights, seeded hard-anchor
  draws, ESS/weight diagnostics, and ECGFounder full-FT anchor class-weight
  parsing, quota allocation with repeat caps, and weighted index sampling. It is
  intentionally not re-exported from `ecg_adv_gen.adaptation.__init__`, so pure
  config/audit imports do not load torch.
- `ecg_adv_gen.training.losses` now owns the shared masked BCE-with-logits,
  ordinary multi-label per-sample BCE, clipped per-class `pos_weight`,
  ECGFounder VAE-LHAT attack-success diagnostics/merge logic, and selected-class
  pairwise ranking loss. The legacy `scripts/triple_labels/train_ptbxl.py`
  re-exports the shared BCE helpers for backward compatibility, while active
  ECGFounder and EfficientNet K500/LHAT scripts import package helpers directly.
- `ecg_adv_gen.training.stream_sampling` contains the ECGFounder VAE-LHAT
  source/target-real/adversarial weighted feature-stream DataLoader helper,
  multi-label row sample weights, class-weight multipliers, and stream-id
  tracking.
- `ecg_adv_gen.adaptation.latent_hull_torch` contains the shared torch
  deterministic latent-hull start diagnostic used by ECGFounder VAE-LHAT and
  ECGFounder full-FT. It supports optimized, one-hot, uniform, and stochastic
  fallback modes without being re-exported from the pure adaptation facade.
- `ecg_adv_gen.run_naming` contains lightweight legacy-compatible run-leaf
  builders for EfficientNet direct K500, EfficientNet VAE-LHAT/AugMix, and
  ECGFounder full-FT outputs. Together with the ECGFounder path contracts in
  `ecg_adv_gen.models.ecgfounder`, the config loader and dated runners now
  share child-run directory rules so manifests, summaries, and real outputs
  stay aligned without importing torch in config/audit paths.
- `configs/active_scripts.yaml` is the machine-readable index of active managed
  wrappers and legacy scripts that must not be moved yet.
- `docs/pipelines/refactor_fact_check_decision_20260527.md` records the
  2026-05-27 multi-agent fact check and the current wrapper-first decision.

## Source-Of-Truth Files

| Topic | Authoritative File |
|---|---|
| Shared server safety and host constraints | `AGENTS.md` |
| YAML refactor plan | `docs/reports/archive/20260527/project_refactor_yaml_config_plan_20260527.md` |
| YAML usage and tracked/local boundary | `configs/README.md` |
| Active script and config index | `configs/active_scripts.yaml` |
| PN2021 Super5 mapping logic | `ecg_adv_gen/labels/super5_mapping.py` |
| PN2021 Super5 protocol metadata facade | `ecg_adv_gen/labels/super5.py` |
| PTB-XL -> PN2021 data/preprocess contract | `ecg_adv_gen/data/contracts.py` |
| Lightweight data path manifest | `ecg_adv_gen/data/manifest.py`, `scripts/export_data_manifest.py` |
| PN2021/K-shot metadata helpers | `ecg_adv_gen/data/pn2021_index.py`, `ecg_adv_gen/data/pn2021_records.py`, `ecg_adv_gen/data/kshot.py` |
| Real-anchor latent pool loading | `ecg_adv_gen/data/real_anchors.py` |
| Waveform shape/resample/lead-order helpers | `ecg_adv_gen/preprocessing/signals.py` |
| Evaluation view semantics | `ecg_adv_gen/evaluation/views.py` |
| PN2021 metric view assembly | `ecg_adv_gen/evaluation/pn2021_metric_views.py` |
| PN2021 clean eval cache helpers | `ecg_adv_gen/evaluation/pn2021_eval_cache.py` |
| Shared macro AUROC/AUPRC helpers | `ecg_adv_gen/evaluation/metrics.py` |
| Paper-safe selection policy and full-FT selection score | `ecg_adv_gen/evaluation/selection.py` |
| Target K-shot internal train/val split | `ecg_adv_gen/evaluation/target_splits.py` |
| Latent-Hull online AT pure helpers and CPU-testable samplers | `ecg_adv_gen/adaptation/lhat.py` |
| ECGFounder model/run path contracts | `ecg_adv_gen/models/ecgfounder.py` |
| ECGFounder head adapters | `ecg_adv_gen/models/ecgfounder_heads.py` |
| ECGFounder feature-head and signal-model inference helpers | `ecg_adv_gen/models/ecgfounder_inference.py` |
| ECGFounder torch preprocessing helpers | `ecg_adv_gen/models/ecgfounder_torch.py` |
| Torch hard-anchor sampling helpers | `ecg_adv_gen/adaptation/anchor_sampling.py` |
| Shared training loss, pos-weight, masked per-sample BCE, stream-weighted BCE, and full-FT adversarial diagnostics helpers | `ecg_adv_gen/training/losses.py` |
| Legacy run naming helpers | `ecg_adv_gen/run_naming.py` |
| Deterministic training split helpers | `ecg_adv_gen/training/splits.py` |
| Weighted feature stream sampling | `ecg_adv_gen/training/stream_sampling.py` |
| Signal-level stream datasets and weighted loaders | `ecg_adv_gen/training/signal_streams.py` |
| Small torch training utilities | `ecg_adv_gen/training/torch_utils.py` |
| Torch latent-hull start helpers | `ecg_adv_gen/adaptation/latent_hull_torch.py` |
| Experiment config schemas | `configs/schemas/*.json` |
| Launcher config/path/runtime code | `ecg_adv_gen/config/` |
| Launch plan rendering and materialization | `ecg_adv_gen/runner/launch_plan.py` |
| Reporting/export code | `ecg_adv_gen/reporting/` |
| Active managed-config audit | `ecg_adv_gen/config/audit.py`, `scripts/audit_managed_configs.py` |

The experiment schema currently requires the top-level `logging` lifecycle
section. YAML-managed postprocess commands must declare structured expected
artifacts, including at least `role` and `path`, so reporting outputs can be
verified by the launcher rather than only described in prose.

## Managed Mainlines

The active YAML-managed mainlines are:

| Experiment | Config | Legacy Entrypoint | Status |
|---|---|---|---|
| `effnet_direct_k500_v6` | `configs/experiments/effnet_direct_k500_v6.yaml` | `scripts/paper/run_direct_finetune_k500_20260516.py` | wrapped |
| `effnet_vae_lhat_k500_v6` | `configs/experiments/effnet_vae_lhat_k500_v6.yaml` | `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py` | wrapped |
| `ecgfounder_direct_k500_v6` | `configs/experiments/ecgfounder_direct_k500_v6.yaml` | `scripts/paper/run_ecgfounder_kshot_head_ft_20260517.py` | wrapped |
| `ecgfounder_inithead_fullft_k500_v6` | `configs/experiments/ecgfounder_inithead_fullft_k500_v6.yaml` | `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py` | wrapped |
| `ecgfounder_vae_lhat_k500_v6` | `configs/experiments/ecgfounder_vae_lhat_k500_v6.yaml` | `scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py` | wrapped |
| `pn2021_eval_v6_refexcluded` | `configs/experiments/pn2021_eval_v6_refexcluded.yaml` | `scripts/triple_labels/eval_crosscenter.py` + managed reporting postprocess | wrapped |

These configs currently generate legacy commands. They do not yet replace the
training/evaluation logic inside the legacy scripts.

As of the 2026-05-27 fact check, these wrappers are expected to be
protocol-equivalent at the command level: K500 seed20260531 refs are explicit,
VAE-LHAT uses K500-internal target validation for quick selection, ECGFounder
frozen-feature direct fine-tuning traces the v6 linear-probe head/features and
four K500 ref-meta files, ECGFounder full fine-tuning passes `--ref_meta_json`,
`--k 500`, and `--init_head_path`,
ECGFounder VAE-LHAT traces the ECGFounder checkpoint, linear-probe feature
caches, K500 direct heads, and K500 latent anchors, and direct EfficientNet
writes under the configured managed output root.

First real execute smoke:

- Config: `configs/experiments/ecgfounder_direct_k500_v6.yaml`
- Run id: `refactor_smoke_ecgfounder_direct_ep2_20260528`
- Mode: single-GPU cached-feature smoke, `CUDA_VISIBLE_DEVICES=0`,
  `training.epochs=2`, `training.batch_size=256`,
  `training.eval_batch_size=8192`
- Manifest:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_ecgfounder_direct_ep2_20260528/run_manifest.json`
- Evidence: `status=succeeded`, `artifact_verification.passed=true`,
  `missing=0`, `mapping_errors=0`, 18 artifacts checked.
- Current VAE-LHAT configs also declare `logging.required_epoch_metrics`; future
  managed runs will reject `training_log.json` artifacts that omit required
  attack-success/ASR, validation-selection, or source-floor diagnostic fields.
- Managed execute logs now include per-command combined logs, split
  stdout/stderr logs, and run-level aggregate `stdout.log` / `stderr.log`; the
  manifest records all three paths for child and postprocess commands,
  including failures.

Current CPU gate evidence:

- Refactor-focused pytest suite: 229 passed.
- `compileall` passed for the package, launcher/reporting scripts, active
  ECGFounder wrappers, changed ECGFounder zero-shot/linear-probe scripts,
  `scripts/triple_labels/train_ptbxl.py`, and
  `scripts/triple_labels/eval_crosscenter.py`.
- `git diff --check` passed.
- Checked production package/wrapper/reporting entrypoints have no direct
  `from scripts.*` / `import scripts.*` imports.
- Active managed-config audit passed for all 6 `active_wrapped` experiments
  after `eval_crosscenter.py` PN2021 per-center row assembly was routed through
  the shared package helper:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_phase12_closeout_final_20260528/`.
- Six active-config dry-run/write-plan manifests were regenerated under
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/dry_runs/refactor_all_postprocess_plan_*_20260528/`.
  Each manifest has `status=dry_run`, non-empty `postprocess_commands`, and a
  `command.sh` with the `# Managed postprocess commands` section.

Current PN2021 eval wrapper state:

- `configs/experiments/pn2021_eval_v6_refexcluded.yaml` now resolves runtime
  `TMPDIR` to `paths.short_tmp_root` (`/home/linbinhao/tmp_ecg`) to avoid the
  long-path multiprocessing `AF_UNIX` failure seen during execute smoke.
- `scripts/run_experiment.py --write-plan` produces a dry-run plan under
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_dryrun/refactor_dryrun_pn2021_eval_runscoped_20260528/`
  with four ref-excluded PN2021 eval commands, the short runtime env, and
  run-id-scoped child output paths.
- The launcher and `eval_crosscenter.py` now create output parent directories
  before child artifact/result writes.
- A separate smoke-only config,
  `configs/experiments/pn2021_eval_v6_refexcluded_smoke.yaml`, passes
  `--pn2021_limit 64`, `num_workers=0`, and `min_pos=2` while preserving the
  same v6 mapping and K500 ref-exclusion protocol.
- The run-id-scoped smoke execute run
  `refactor_smoke_pn2021_eval_runscoped_limit64_20260528` succeeded on GPU 0 with
  `artifact_verification.passed=true`, `n_checked=10`, `missing=0`,
  `mapping_errors=0`, short `TMPDIR=/home/linbinhao/tmp_ecg`, and all 4 eval
  artifacts under the run-id-scoped child output root.
- The full run-id-scoped PN2021 eval execute run
  `refactor_full_pn2021_eval_runscoped_20260528` also succeeded with
  `artifact_verification.passed=true`, `n_checked=10`, `missing=0`,
  `mapping_errors=0`, and all four eval JSONs under the scoped child output
  root.
- Its reporting backfill produced
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/metrics_export/refactor_full_pn2021_eval_runscoped_20260528/metrics_long.csv`
  with 84 metric rows and two paper-table views. Center means were
  all-zero-kept AUROC/AUPRC `0.854321 / 0.487896` and drop-all-zero
  AUROC/AUPRC `0.881637 / 0.634631`.
- `pn2021_eval_v6_refexcluded.yaml` now declares managed postprocess commands
  to export this same `metrics_long.csv` plus all-zero-kept/drop-all-zero paper
  tables automatically after a future managed execute run.
- The integrated postprocess execute run
  `refactor_execute_postprocess_pn2021_eval_20260528` has now verified that
  future path end-to-end:
  `status=succeeded`, `artifact_verification.passed=true`, `n_checked=16`,
  six verified artifacts marked `postprocess=true`, `missing=0`,
  `mapping_errors=0`, and `input_verification.n_checked=36`.
- Integrated execute outputs:
  `metrics_export/refactor_execute_postprocess_pn2021_eval_20260528/metrics_long.csv`,
  `paper_table/refactor_execute_postprocess_pn2021_eval_20260528_all_zero_kept/paper_table.csv`,
  and
  `paper_table/refactor_execute_postprocess_pn2021_eval_20260528_drop_all_zero/paper_table.csv`.
- Integrated execute center means match the earlier manual backfill:
  all-zero-kept AUROC/AUPRC `0.854321 / 0.487896`; drop-all-zero AUROC/AUPRC
  `0.881637 / 0.634631`.

Current ECGFounder direct training wrapper smoke:

- Config: `configs/experiments/ecgfounder_direct_k500_v6.yaml`
- Run id: `refactor_smoke_ecgfounder_direct_postprocess_ep2_20260528`
- Mode: cached-feature direct-head training smoke,
  `CUDA_VISIBLE_DEVICES=0`, `training.epochs=2`,
  `training.batch_size=256`, `training.eval_batch_size=8192`
- Manifest:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_ecgfounder_direct_postprocess_ep2_20260528/run_manifest.json`
- Evidence: `status=succeeded`, `input_verification.passed=true`,
  `artifact_verification.passed=true`, `n_checked=22`, `missing=0`,
  `mapping_errors=0`, `content_errors=0`, and 4 verified postprocess
  artifacts.
- Reporting evidence:
  `metrics_export/refactor_smoke_ecgfounder_direct_postprocess_ep2_20260528_ecgfounder_direct_k500_v6/metrics_long.csv`
  has 42 metric rows, and the all-zero-kept paper table center mean is
  AUROC/AUPRC `0.8632266722 / 0.5105771844`.
- This smoke exposed and fixed a metrics-exporter compatibility gap:
  `training_log.json` is not a metrics artifact, and ECGFounder direct
  `target_view` JSONs now export rows correctly.

Current EfficientNet direct training wrapper smoke:

- Config: `configs/experiments/effnet_direct_k500_v6_smoke.yaml`
- Run id: `refactor_smoke_effnet_direct_limit64_ep1_20260528`
- Mode: one-epoch EfficientNet1DV2 direct-K500 smoke with nested limited
  PN2021 eval, `CUDA_VISIBLE_DEVICES=0`, `training.batch_size=64`,
  `training.eval_batch_size=64`, `resources.default_num_workers=0`,
  `evaluation.min_pos=2`, and `evaluation.pn2021_limit=64`.
- Manifest:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_effnet_direct_limit64_ep1_20260528/run_manifest.json`
- Evidence: `status=succeeded`, `input_verification.passed=true`,
  `artifact_verification.passed=true`, `n_checked=32`, `missing=0`,
  `mapping_errors=0`, `content_errors=0`, and 6 verified postprocess
  artifacts.
- Reporting evidence:
  `metrics_export/refactor_smoke_effnet_direct_limit64_ep1_20260528_effnet_direct_k500_v6_smoke/metrics_long.csv`
  has 58 metric rows. The all-zero-kept paper table center mean is
  AUROC/AUPRC `0.8563106854 / 0.6585915833`. The drop-all-zero smoke table is
  `nan / nan` because the 64-record PN2021 limit can remove all valid positive
  classes for one drop-all-zero target subset.
- This smoke verifies a real EfficientNet training wrapper, nested
  `eval_crosscenter.py`, v6 mapping metadata checks, run-id-scoped child
  outputs, and YAML-managed postprocess. It is an engineering smoke, not a
  paper result.

Current EfficientNet VAE-LHAT training wrapper smoke:

- Config: `configs/experiments/effnet_vae_lhat_k500_v6_smoke.yaml`
- Run id: `refactor_smoke_effnet_vae_lhat_width2_limit64_ep1_20260528`
- Mode: one-epoch four-center EfficientNet1DV2 VAE-LHAT smoke with
  `CUDA_VISIBLE_DEVICES=0`, `training.batch_size=32`,
  `training.eval_batch_size=64`, `resources.default_num_workers=0`,
  `adaptation.anchors.k_anchor=32`, `adaptation.hull.M=4`,
  `adaptation.hull.steps=1`, `adaptation.attack.pgd_batch=8`,
  `adaptation.latent_augmix.width=2`, `adaptation.latent_augmix.depth=1`,
  `evaluation.min_pos=2`, and `evaluation.pn2021_limit=64`.
- Manifest:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_effnet_vae_lhat_width2_limit64_ep1_20260528/run_manifest.json`
- Evidence: `status=succeeded`, `input_verification.passed=true`,
  `artifact_verification.passed=true`, `n_checked=32`, `missing=0`,
  `mapping_errors=0`, `content_errors=0`, and 6 verified postprocess
  artifacts.
- Reporting evidence:
  `metrics_export/refactor_smoke_effnet_vae_lhat_width2_limit64_ep1_20260528_effnet_vae_lhat_k500_v6_smoke/metrics_long.csv`
  has 58 metric rows. The all-zero-kept paper table center mean is
  AUROC/AUPRC `0.8787944647 / 0.6913762713`. The drop-all-zero smoke table is
  `nan / nan` because the 64-record PN2021 limit can remove all valid positive
  classes for one drop-all-zero target subset.
- This smoke verifies a real four-center VAE-LHAT managed runtime path:
  ECGTwin VAE load/decode, latent-hull PGD, latent AugMix, target K500 internal
  validation, final ref-excluded PN2021 eval, v6 mapping metadata checks,
  run-id-scoped child outputs, and YAML-managed postprocess.

Current ECGFounder VAE-LHAT training wrapper smoke:

- Config: `configs/experiments/ecgfounder_vae_lhat_k500_v6_smoke.yaml`
- Run id: `refactor_smoke_ecgfounder_vae_lhat_limit64src_ep1_20260528`
- Mode: one-epoch four-center ECGFounder VAE-LHAT smoke with
  `CUDA_VISIBLE_DEVICES=0`, `training.batch_size=64`,
  `training.eval_batch_size=512`, `training.source_train_limit=64`,
  `adaptation.anchors.k_anchor=16`, `adaptation.hull.M=4`,
  `adaptation.hull.steps=1`, and `adaptation.attack.pgd_batch=8`.
- Manifest:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_ecgfounder_vae_lhat_limit64src_ep1_20260528/run_manifest.json`
- Evidence: `status=succeeded`, `input_verification.passed=true`,
  `artifact_verification.passed=true`, `n_checked=28`, `missing=0`,
  `mapping_errors=0`, `content_errors=0`, and 6 verified postprocess
  artifacts.
- Reporting evidence:
  `metrics_export/refactor_smoke_ecgfounder_vae_lhat_limit64src_ep1_20260528_ecgfounder_vae_lhat_k500_v6_smoke/metrics_long.csv`
  has 84 metric rows. The all-zero-kept paper table center mean is
  AUROC/AUPRC `0.9045766643 / 0.5976623929`; the drop-all-zero center mean is
  `0.9221097550 / 0.7048084381`.
- This smoke exposed and fixed a metrics-exporter compatibility gap:
  ECGFounder VAE-LHAT final metrics are stored under `final_pn2021_views`, not
  the older single-run `target_*` keys.

Current ECGFounder init-head full fine-tune wrapper smoke:

- Config: `configs/experiments/ecgfounder_inithead_fullft_k500_v6_smoke.yaml`
- Run id: `refactor_smoke_ecgfounder_inithead_cacheunion_ep1_20260528`
- Mode: one-epoch four-center ECGFounder full fine-tune smoke with
  `CUDA_VISIBLE_DEVICES=0`, `training.batch_size=32`,
  `training.eval_batch_size=64`, `training.source_train_limit=64`, and
  signal-cache reuse from
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/paper_ecgfounder_fullft_signal_cache_union_20260528/cache`.
- Manifest:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/launcher_smoke/refactor_smoke_ecgfounder_inithead_cacheunion_ep1_20260528/run_manifest.json`
- Evidence: `status=succeeded`, `input_verification.passed=true`,
  `artifact_verification.passed=true`, `n_checked=24`, `missing=0`,
  `mapping_errors=0`, `content_errors=0`, and 6 verified postprocess
  artifacts.
- Reporting evidence:
  `metrics_export/refactor_smoke_ecgfounder_inithead_cacheunion_ep1_20260528_ecgfounder_inithead_fullft_k500_v6_smoke/metrics_long.csv`
  has 126 metric rows. The all-zero-kept paper table center mean is
  AUROC/AUPRC `0.9092978816 / 0.6049917842`; the drop-all-zero center mean is
  `0.9256955173 / 0.7112885361`.

## Non-Move Rule

Do not move these areas until the wrapper, manifest, metrics, and table layers
are stable across real runs:

- `scripts/paper/`
- `scripts/pgd_cross_center/synth_online_at_super5.py`
- `scripts/triple_labels/train_ptbxl.py`
- `scripts/triple_labels/eval_crosscenter.py`
- `scripts/ecgtwin_author_repro/`
- `model/*` external repo link handles

Moving them too early would break reproducibility because many current results
and dated scripts encode behavior in CLI defaults and path assumptions.

## Reporting Rules

Paper tables should be produced in this order:

1. Export one or more result artifacts to `metrics_long.csv` with
   `scripts/export_metrics_long.py`. For per-target-center runs whose JSON
   contains all PN2021 centers, use explicit `--run-id` and
   `--filter-to-target-center` so each artifact contributes only its own target
   center to the method-level row.
2. Merge method-level `metrics_long.csv` files with
   `scripts/merge_metrics_long.py` when comparing multiple methods.
3. Confirm `artifact_manifest.json` or `merge_manifest.json` contains exactly one PN2021 Super5 mapping
   pair unless the analysis intentionally opts into mixed mapping.
4. Build tables with `scripts/export_paper_table.py` using an explicit `--view`.
5. Use `--baseline-run-id` only when the baseline row has the same
   dataset/view/scope/center group.

The all-zero-kept and drop-all-zero PN2021 views must remain separate rows and
must not be averaged together.

## Next Extraction Targets

After wrappers are exercised on real runs, extract in this order:

1. `ecg_adv_gen.labels`: Super5 metadata and conversion policy are package-owned;
   next extraction is broader non-Super5 scheme cleanup after data loaders move
   out of `scripts/`.
2. `ecg_adv_gen.data`: paper-protocol contract, lightweight path manifest,
   optional PN2021 header-count helper, PN2021 metadata index helpers, K-shot
   artifact helpers, real-anchor pool loading, and full-FT signal-cache
   metadata/schema helpers are in place; next extraction is migrating more
   legacy callers to these shims without moving WFDB/cache/preprocess logic.
3. `ecg_adv_gen.preprocessing`: first pure shape/resample/lead-order helpers
   are in place; next extraction is adapting legacy callers behind thin wrappers.
4. `ecg_adv_gen.evaluation`: first view/metric semantics and pure PN2021
   metric-view assembly helpers are in place; next extraction is adapting
   legacy EfficientNet/ECGFounder evaluators behind compatibility shims.
5. `ecg_adv_gen.adaptation`: pure Latent-Hull sampling/soft-label/trust
   helpers, `StratifiedPoolWalker`, `SameLabelLatentIndex`, the array-level
   latent AugMix mixer, and signal-model anchor difficulty / positive-boundary
   scoring helpers are in place; next extraction is separating GPU PGD/decode
   interfaces from the legacy script behind compatibility shims after wrapper
   real-run evidence is stable.
6. `ecg_adv_gen.models`: ECGFounder path/run-layout contracts are in place;
   next extraction is model-adapter interfaces only after DeepECG/ECGFounder
   smoke runs are stable.
7. `ecg_adv_gen.training`: shared masked BCE, class `pos_weight`, stream-weighted
   BCE, signal-stream loaders, ECGFounder full-FT adversarial diagnostics, and
   full-FT legacy run naming helpers are in place. Active ECGFounder scripts and
   the YAML config loader now use the shared lightweight top-level run naming
   helper directly; the remaining training loop code should stay in legacy
   entrypoints until real-run wrapper evidence is stable.

Until then, legacy scripts remain the execution layer and YAML configs remain
the experiment definition layer.
