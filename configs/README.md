# ECG_adv_Gen Configuration

This directory holds tracked experiment definitions. Machine-specific paths
belong in `configs/local/*.yaml`, which is ignored except for example files.

`configs/active_scripts.yaml` is the machine-readable index of which tracked
configs wrap which legacy scripts. Keep it synchronized whenever a new YAML
mainline or reporting CLI is added.

`configs/label_mappings/` holds structured label-mapping evidence, such as
PN2021 to PTB-XL Super5 clinician-review JSONL records. Human-readable review
documents belong in `docs/labeling/`.

## Boundary

Tracked YAML may contain:

- paper protocol facts: mapping version/hash, class order, centers, K-shot policy;
- model and method names;
- hyperparameters;
- legacy script entrypoints and arguments;
- required artifacts and evaluation views.

Tracked YAML must not contain:

- `/home/linbinhao` or `/root` absolute paths;
- GPU ids;
- Python executable paths;
- private temporary directories;
- symlink targets for external model repos;
- credentials.

`util/tests/test_active_script_index.py` enforces this boundary for
`configs/defaults/`, `configs/experiments/`, and `configs/active_scripts.yaml`.
Machine-local paths belong only in `configs/local/*.yaml`, with example files
kept for host setup documentation. `configs/local/.gitignore` is intentionally
restricted to:

```gitignore
*
!.gitignore
!*.example.yaml
```

The active-script tests lock this pattern so real local YAML, private paths,
and credentials cannot be accidentally tracked.

Local YAML is a host overlay only. Its top-level keys are restricted to
`host`, `paths`, `python`, `resources`, and `safety`. It must not override
tracked experiment protocol or launch semantics such as `paper_protocol`,
`data`, `preprocess`, `model`, `evaluation`, `runner`, `postprocess`, or
`logging`; those sections belong in tracked YAML where audit and review can
see them.

`configs/schemas/experiment_config.schema.json` now treats `logging` as a
required lifecycle section. Experiment YAML must declare launch, child, and
postprocess artifact buckets, and YAML-managed postprocess commands must list
structured expected artifacts with at least `role` and `path`.

## Launcher

The supported entrypoint can resolve configs as a CPU-only dry-run:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq_matrix.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id dryrun_effnet_direct_k500_v7 \
  --dry-run
```

Dry-run validates the YAML, checks path boundaries, compares Super5 mapping
metadata and data/preprocess protocol fields against code, and prints the
legacy commands without invoking child scripts.

The authoritative active/smoke/superseded inventory is
`configs/active_scripts.yaml`. Current agent-facing active surfaces are v7
SJR/RGQ configs covering EfficientNet Direct/VAE-LHAT K500 and percent-shot
matrices, benchmark backbones, ECGFounder Direct/VAE-LHAT K500, ECGTwin author
reproduction, minimal prompt-token train/generate/gate, PN2021 ref-excluded
eval, and PN2021-C eval. Older v6 configs are retained only when
`configs/active_scripts.yaml` classifies them as smoke or superseded.

The smoke configs are not paper result configs. The PN2021 eval smoke passes
`--pn2021_limit`; the EfficientNet direct smoke uses one training epoch,
`num_workers=0`, and passes `--eval_pn2021_limit` to the nested PN2021 eval.
The ECGFounder VAE-LHAT smoke keeps K=500/ref-exclusion semantics but limits
the source feature training rows, online anchors, hull size, and epoch count.
The ECGFounder init-head full-FT smoke keeps the same four-target-center
matrix and K=500/ref-exclusion semantics, but limits source training rows and
reuses an existing signal-cache union under `${paths.data_root}` to avoid
rebuilding raw WFDB caches during engineering verification.
They write under their own output roots so launcher execution, short-TMPDIR
runtime env, output directory creation, child artifact verification, and
mapping-metadata checks can be exercised without running full experiments.

Small launch-time overrides are supported only through an audited whitelist:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq_matrix.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id smoke_epochs2_workers1 \
  --dry-run \
  --set training.epochs=2 \
  --set resources.default_num_workers=1
```

Allowed `--set` keys are limited to resource, training, and method
hyperparameters such as `training.epochs`, `training.batch_size`,
`training.optimizer.lr`, `resources.default_num_workers`,
`adaptation.hull.lambda`, and `adaptation.latent_augmix.severity`. The
launcher rejects overrides for paper protocol, mapping, centers, K-shot, paths,
checkpoints, environment, and raw runner argv. Every accepted override is saved
in the resolved config and `run_manifest.json`.

Managed child output paths are scoped by the launcher `run_id` through the
`runtime.run_id` interpolation key. New experiment YAML should write child
artifacts under paths like:

```text
${paths.output_root}/${experiment.name}/${runtime.run_id}/...
```

This keeps repeated dry-runs, smoke runs, and full runs from overwriting each
other while preserving a stable method-level grouping under
`${paths.output_root}/${experiment.name}`.
The protocol audit also rejects managed child or postprocess output options
(`--out_root`, `--out_dir`, `--output_path`, `--output-dir`, and
`--output_dir`) that do not include the resolved `runtime.run_id`.

## Managed Execution

Execution is intentionally stricter than dry-run:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=3 micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq_matrix.yaml \
  --local-config configs/local/linbinhao_server.yaml \
  --run-id effnet_direct_k500_v7_seed20260531 \
  --execute
```

Run `--execute` only after the selected GPU is actually free. Before invoking
any legacy child script, the launcher requires explicit `CUDA_VISIBLE_DEVICES`,
runs its own `nvidia-smi` snapshot, writes `run_config.resolved.yaml`,
`run_config.resolved.json`, `run_manifest.json`, `command.sh`,
`data_manifest.json`, `k500_ref_ids.json`, `selection.json`, `run_card.json`,
`run_file_index.json`, and `summary.md`, and refuses
non-empty output directories unless `--resume` or a matching scoped `--force`
is supplied. The
manifest includes an
`artifact_trace` section with mapping metadata, metric views, K500 ref-meta
paths, frozen K500 record-id hashes, model-selection policy, input
checkpoints/heads, child output roots, expected result artifacts, and any
managed reporting postprocess artifacts. Before child scripts run, required
inputs are checked. After child scripts return zero, the launcher verifies
required launch and child artifacts and rejects the run if expected eval JSON
files are missing, carry the wrong PN2021 Super5 mapping metadata, or the
selection record fails the K500-internal/source-floor safety contract. If
`logging.required_epoch_metrics` is declared, `training_log.json` artifacts are
also checked for those epoch-level diagnostic fields, so VAE-LHAT runs cannot
silently drop attack-success or validation-selection signals. If
`postprocess.commands` are declared, they run only after child artifacts pass
this check; their expected outputs are then verified before the manifest is
marked `succeeded`. Executed child commands are
recorded in top-level `command_runs`; managed reporting commands are recorded
in top-level `postprocess_runs`. Each record includes command index, name,
return code, status, combined log path, split stdout/stderr log paths, and UTC
start/end timestamps, including failed commands. The run directory also keeps
aggregate `stdout.log` and `stderr.log` files with section headers across child
and postprocess commands. The manifest records execute-level
`execution_started_at_utc`, final `finished_at_utc`, and `duration_seconds`
when a managed run finishes.

Each managed run is also finalized into an agent-readable layout:

```text
<run_dir>/
  run_card.json             # purpose, outcome, result summary, protocol, metric summary
  run_file_index.json       # logical file categories for the run
  summary.md                # short human/agent handoff note
  configs/                  # resolved config and command snapshot
  manifests/                # manifest snapshots and launch records
  logs/                     # command stdout/stderr logs
  checkpoints/              # checkpoint handles and checkpoint indexes
  eval/                     # metrics_long, paper tables, per-center/class outputs
  diagnostics/              # diagnostics_epoch, agent_decision, training logs
  reports/                  # rendered summaries
  artifacts/                # other small generated files
```

If an older run predates this layout, backfill the handoff files without
rerunning training:

```bash
micromamba run -n ECGTwin python scripts/agent/finalize_run.py \
  --run-dir /path/to/run_dir \
  --purpose "Why this experiment was run" \
  --result-summary "What the run showed" \
  --outcome provisional
```

Important runs can then be registered in `configs/active_evidence_registry.yaml`
using placeholder paths under `${paths.output_root}`:

```bash
micromamba run -n ECGTwin python scripts/agent/register_run.py \
  --run-dir /path/to/run_dir \
  --status provisional
```

Registration expects an already finalized run. `register_run.py` will not
auto-create `run_card.json`; `finalize_run.py` and registration both reject runs
that lack explicit purpose/result text, manifest schema v2, git commit, resolved
config hash, command argv records, K-shot ref traces, `selection.json`, declared
eval artifacts, and succeeded-run metric center coverage.

Active paper/evaluation configs should declare managed postprocess commands.
The common pattern is:

- `scripts/export_metrics_long.py`
- `scripts/export_paper_table.py` for `pn2021_all_zero_kept_refexcluded`
- `scripts/export_paper_table.py` for `pn2021_drop_all_zero_refexcluded`

Older smoke/superseded configs may carry narrower postprocess coverage when
the legacy result script does not emit every view. Check
`configs/active_scripts.yaml` before treating any such config as a current
paper surface.

The generated `command.sh` includes both the legacy child commands and these
postprocess commands, so a managed run produces paper-table inputs without a
separate hand-written shell step after child artifacts pass verification.

## Data Manifest

The lightweight data manifest records path and protocol facts without loading
waveforms or recursively scanning large datasets:

```bash
micromamba run -n ECGTwin python scripts/export_data_manifest.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq_matrix.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --output-dir /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/data_manifest/<name>
```

Add `--include-counts` when you want a metadata-only PN2021 center-size
snapshot. It counts `*.hea` files directly under each configured center
directory and one PN2021 grouping level below it, does not recurse further, and
does not load waveform files.

Use `--require-existing` as a preflight gate when you want missing PTB-XL or
PN2021 center directories to fail before a run starts.

## Metrics Export

Existing result JSONs can be backfilled into a paper-safe long table without
rerunning models:

```bash
micromamba run -n ECGTwin python scripts/export_metrics_long.py \
  --input /path/to/run_dir_or_eval_result.json \
  --output-dir /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/metrics_export/<name>
```

The exporter writes `metrics_long.csv` and `artifact_manifest.json`, records
artifact SHA-256 hashes, refuses mixed PN2021 Super5 mapping versions by
default, and keeps all-zero-kept and drop-all-zero rows as separate `view`
values. It also writes `canonical_view` so ECGFounder single-center
`target_*_refexcluded` rows can be compared with PN2021 ref-excluded rows
without losing the original raw `view`.

For per-target-center runs whose `eval_result.json` contains all PN2021 centers,
export with an explicit method `--run-id` and filter each artifact to its
inferred target center:

```bash
micromamba run -n ECGTwin python scripts/export_metrics_long.py \
  --input /path/to/ningbo_run/eval_result.json /path/to/georgia_run/eval_result.json \
  --output-dir /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/metrics_export/<method> \
  --run-id <method_run_id> \
  --filter-to-target-center \
  --target-centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --expected-mapping-version v7_super5_sjr_rgq_review_20260528 \
  --expected-mapping-hash 555ec85d5b51
```

Multiple method-level exports can then be merged into one table input:

```bash
micromamba run -n ECGTwin python scripts/merge_metrics_long.py \
  --input /path/to/direct/metrics_long.csv /path/to/lhat/metrics_long.csv \
  --output-dir /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/metrics_export/<combined>
```

Compact paper tables are built from `metrics_long.csv` with an explicit view:

```bash
micromamba run -n ECGTwin python scripts/export_paper_table.py \
  --metrics-long /path/to/metrics_long.csv \
  --output-dir /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/paper_table/<name> \
  --view pn2021_all_zero_kept_refexcluded \
  --dataset pn2021 \
  --centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --no-dataset-rows
```

Use `--baseline-run-id` only when the baseline row is in the same
`metrics_long.csv`; deltas are computed only for matching
dataset/canonical-view/scope and center group. The output table keeps `raw_views`
for auditability.

## Managed Config Audit

Use the active-config auditor when you want a quick CPU-only preflight across
every `active_wrapped` experiment listed in `configs/active_scripts.yaml`:

```bash
micromamba run -n ECGTwin python scripts/audit_managed_configs.py \
  --local-config configs/local/linbinhao_server.example.yaml \
  --require-existing-inputs \
  --output-dir /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/<name>
```

It does not invoke legacy child scripts or use GPU. It resolves each tracked
config, runs the same command/protocol audit as the launcher, builds the same
artifact trace, and optionally verifies required input checkpoints, heads,
feature caches, and K500 ref artifacts exist.

Current verification snapshot:

```text
micromamba run -n ECGTwin python scripts/audit_managed_configs.py \
  --local-config configs/local/linbinhao_server.example.yaml

managed_experiment_count=23
passed_count=23
failed_count=0
```

Historical 2026-05 execute-smoke evidence:

- `pn2021_eval_v6_refexcluded`: `refactor_execute_postprocess_pn2021_eval_20260528`
  succeeded with managed postprocess and two paper-table views.
- `effnet_direct_k500_v6_smoke`: `refactor_smoke_effnet_direct_limit64_ep1_20260528`
  succeeded with nested limited PN2021 eval and managed postprocess.
- `effnet_vae_lhat_k500_v6_smoke`:
  `refactor_smoke_effnet_vae_lhat_width2_limit64_ep1_20260528` succeeded with
  ECGTwin VAE decode, latent-hull PGD, latent AugMix, and managed postprocess.
- `ecgfounder_direct_k500_v6`:
  `refactor_smoke_ecgfounder_direct_postprocess_ep2_20260528` succeeded with
  cached ECGFounder features and managed postprocess.
- `ecgfounder_vae_lhat_k500_v6_smoke`:
  `refactor_smoke_ecgfounder_vae_lhat_limit64src_ep1_20260528` succeeded with
  84 metrics rows; center means were all-zero-kept AUROC/AUPRC
  `0.9045766643 / 0.5976623929` and drop-all-zero
  `0.9221097550 / 0.7048084381`.
- `ecgfounder_inithead_fullft_k500_v6_smoke`:
  `refactor_smoke_ecgfounder_inithead_cacheunion_ep1_20260528` succeeded with
  126 metrics rows; center means were all-zero-kept AUROC/AUPRC
  `0.9092978816 / 0.6049917842` and drop-all-zero
  `0.9256955173 / 0.7112885361`.

Dry-run/write-plan manifests were also regenerated for all six active configs
under:

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/dry_runs/refactor_all_postprocess_plan_*_20260528/
```

Additional historical 2026-05 execute-smoke evidence:

- `ecgfounder_direct_k500_v6` has a cached-feature 2-epoch training smoke:
  `refactor_smoke_ecgfounder_direct_postprocess_ep2_20260528`, with
  `status=succeeded`, input/artifact verification passed, 22 checked
  artifacts, and managed metrics/table postprocess verified.
- `effnet_direct_k500_v6_smoke` has a one-epoch EfficientNet1DV2 direct-K500
  smoke:
  `refactor_smoke_effnet_direct_limit64_ep1_20260528`, with
  `status=succeeded`, input/artifact verification passed, 32 checked
  artifacts, 6 managed postprocess artifacts verified, and nested PN2021 eval
  limited by `--eval_pn2021_limit 64`.
- `effnet_vae_lhat_k500_v6_smoke` has a one-epoch four-center
  EfficientNet1DV2 VAE-LHAT smoke:
  `refactor_smoke_effnet_vae_lhat_width2_limit64_ep1_20260528`, with
  `status=succeeded`, input/artifact verification passed, 32 checked
  artifacts, 6 managed postprocess artifacts verified, and nested PN2021 eval
  limited by `--eval_pn2021_limit 64`. This smoke also exercises ECGTwin VAE
  decode, latent-hull PGD, latent AugMix, target K500 internal validation, and
  final ref-excluded PN2021 eval.

Both are engineering smokes for the launcher/wrapper/postprocess path, not
paper result runs.

## Agent Operating Layer

The current agent-facing evidence entry point is:

```text
configs/active_evidence_registry.yaml
```

Use it when a new agent needs to know the current trusted Direct/VAE mainline,
mapping version/hash, fixed K500 protocol, ref-exclusion paths, metrics exports,
paper tables, comparison bundle, and no-commit artifact policy.

CPU-only audit:

```bash
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
```

Backfill the legacy VAE-LHAT run manifest when local artifacts exist:

```bash
micromamba run -n ECGTwin python scripts/agent/backfill_vae_lhat_manifest.py
```

Managed Direct vs VAE comparison bundle:

```bash
micromamba run -n ECGTwin python scripts/agent/build_comparison_bundle.py --force
```

The full rationale and checkpoint contract are documented in:

```text
docs/pipelines/agent_operating_layer_20260529.md
```
