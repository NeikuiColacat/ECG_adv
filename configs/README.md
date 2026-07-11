# ECG_adv_Gen Configuration

This directory holds tracked experiment definitions. Machine-specific paths
belong in `configs/local/*.yaml`, which is ignored except for example files.

`configs/active_scripts.yaml` is the machine-readable index of which tracked
configs map to managed package runners. Keep it synchronized whenever a new
YAML mainline or reporting CLI is added.

`configs/label_mappings/` holds structured label-mapping evidence, such as
PN2021 to PTB-XL Super5 clinician-review JSONL records. Human-readable review
documents belong in `docs/labeling/`.

## Boundary

Tracked YAML may contain:

- paper protocol facts: mapping version/hash, class order, centers, K-shot policy;
- model and method names;
- hyperparameters;
- managed runner entrypoints and arguments;
- required artifacts and evaluation views.

Tracked YAML must not contain:

- `/home/linbinhao` or `/root` absolute paths;
- GPU ids;
- Python executable paths;
- private temporary directories;
- symlink targets for external model repos;
- credentials.

`util/tests/test_active_script_index.py` enforces this boundary for
`configs/defaults/`, `configs/experiments/`, `configs/replications/`, and
`configs/active_scripts.yaml`.
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
`host`, `paths`, `python`, `resources`, `safety`, and `external_models`. It must not override
tracked experiment protocol or launch semantics such as `paper_protocol`,
`data`, `preprocess`, `model`, `evaluation`, `runner`, `postprocess`, or
`logging`; those sections belong in tracked YAML where audit and review can
see them.

External model repository targets also belong in local YAML or environment
variables, not tracked experiment configs or long launch arguments. The current
host example declares them under `external_models`:

```bash
micromamba run -n ECGTwin python scripts/agent/check_external_models.py \
  --local-config configs/local/linbinhao_server.example.yaml
bash scripts/bootstrap_model_repos.sh --check-only \
  --local-config configs/local/linbinhao_server.example.yaml
```

`scripts/agent/audit_agent_workspace.py` uses the same contract: verified dirty
`model/*` symlink handles are local-only and ignored for handoff readiness, but
staged guarded paths or unverified model handles still require action.

`configs/schemas/experiment_config.schema.json` now treats `logging` as a
required lifecycle section. Experiment YAML must declare launch, child, and
postprocess artifact buckets, and YAML-managed postprocess commands must list
structured expected artifacts with at least `role` and `path`.

## Launcher

The supported entrypoint can resolve configs as a CPU-only dry-run:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id dryrun_effnet_direct_k500_v7 \
  --dry-run
```

Dry-run validates the YAML, checks path boundaries, compares Super5 mapping
metadata and data/preprocess protocol fields against code, and prints the
managed commands without invoking child scripts.

The authoritative public inventory is `configs/active_scripts.yaml`. For the
current reproducible paper path, use its `latest_mainline` block first: it declares the
`vae_lhat_threechain_augmix_pn2021c` VAE-LHAT + three-chain AugMix method as
10 stages, all launched through `scripts/run_experiment.py`, covering EfficientNet
and ECGFounder baselines, training, PN2021 clean eval, and PN2021-C evaluation.
Public `configs/experiments/*.yaml` is intentionally limited to those 10
latest-mainline configs.

## Non-canonical replications

`configs/replications/` contains thin, indexed wrappers for reruns that must not
expand the canonical 10-stage latest-mainline snapshot or its golden contract.
The matched EfficientNet three-seed surface is recorded under
`configs/active_scripts.yaml:replication_surfaces`. Seed `20260601` keeps using
the four canonical configs; seeds `20260531` and `20260611` each use four thin
wrappers that only change `paper_protocol.kshot.seed`,
`paper_protocol.kshot.subset_seed`, and, for PN2021-C evaluation, the matching
`model.eval_seed`.

Within one seed, launch train, clean, S5, and depth23 with exactly the same
`--run-id`. Their launcher plan directories must still be different, for
example `--output-dir "$PLAN_ROOT/train"`, `.../clean`, `.../s5`, and
`.../depth23`; otherwise the four launcher manifests would collide. Do not use
`--set` to change K500 or paper-protocol seeds. The corruption RNG seed remains
fixed at `20260501` across all selection/training seeds.

The `20260611` K500 inputs are intentionally marked missing in the index. They
must be materialized and identity-checked before GPU execution. Never copy or
rename another seed's K500 artifacts to satisfy that preflight: these wrappers
bind the K500 draw and the training seed to the same declared value.

Small launch-time overrides are supported only through an audited whitelist:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id dryrun_epochs2_workers1 \
  --dry-run \
  --set training.epochs=2 \
  --set resources.default_num_workers=1
```

Allowed `--set` keys are limited to resource, training, and method
hyperparameters such as `training.epochs`, `training.batch_size`,
`training.optimizer.lr`, `resources.default_num_workers`,
`adaptation.hull.lambda`, and `adaptation.latent_augmix.severity`. The
launcher rejects overrides for paper protocol, mapping, centers, K-shot, paths,
checkpoints, environment, and runner launch semantics. Every accepted override is saved
in the resolved config and `run_manifest.json`.

Managed child output paths are scoped by the launcher `run_id` through the
`runtime.run_id` interpolation key. New experiment YAML should write child
artifacts under paths like:

```text
${paths.output_root}/${experiment.name}/${runtime.run_id}/...
```

This keeps repeated dry-runs and full runs from overwriting each other while
preserving a stable method-level grouping under
`${paths.output_root}/${experiment.name}`.
The protocol audit also rejects managed child or postprocess output options
(`--out_root`, `--out_dir`, `--output_path`, `--output-dir`, and
`--output_dir`) that do not include the resolved `runtime.run_id`.

Optional `stages[]` records can describe future multi-step orchestration with
`name`, `requires`, `produces`, and `skip_if_exists`. The launcher records them
in dry-run `pipeline_stages` manifests for handoff and planning; it does not
yet turn them into a multi-stage scheduler.

Latest replay configs use typed `runner.adapter` entries instead of tracked
argument lists. The adapter registry lives in
`ecg_adv_gen/config/adapters/registry.py` and the active replay surface uses
`direct_finetune`, `effnet_vae_lhat`, `ecgfounder_fullft`,
`ecgfounder_pn2021c_eval`, `pn2021_eval`, and `pn2021c_eval`; each builds the
managed commands from typed YAML fields, so active configs do not carry full
argument lists.
PN2021 eval adapter always emits `--eval_protocol paper_refexcluded` and a
K500 `--min_target_ref_excluded` gate. The PN2021-C adapter requires a clean
eval JSON, `v7_refexcluded_100hz1000` cache metadata, and explicit K500 ref
exclusion. The ECGFounder PN2021-C adapter consumes locked full-FT run
directories, evaluates the `bottleneck5000` order, and rejects stabilizer
options for the main method. ECGFounder locked training uses the full-FT
adapter; residual-adapter VAE-LHAT recipes are provenance only, not active
replay adapters.
`runner.argv` is not a supported launch surface; configs must use typed
`runner.adapter` fields.

All active configs share `preprocess.contract_id` for PTB-XL/PN2021/ECGTwin
decode contracts. ECGFounder configs additionally declare
`preprocess.ecgfounder` and `model.preprocess_policy=official_ptbxl_eval` so
500Hz/5000-point feature-cache runs cannot be mixed into EfficientNet 100Hz
classifier contracts by accident.

## Managed Execution

Execution is intentionally stricter than dry-run:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=3 micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml \
  --local-config configs/local/linbinhao_server.yaml \
  --run-id effnet_direct_k500_v7_seed20260601 \
  --execute
```

Run `--execute` only after the selected GPU is actually free. Before invoking
any managed child command, the launcher requires explicit `CUDA_VISIBLE_DEVICES`,
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
  --status auto
```

Registration expects an already finalized run. `register_run.py` will not
auto-create `run_card.json`; `finalize_run.py` and registration both reject runs
that lack explicit purpose/result text, manifest schema v2, git commit, resolved
config hash, command argv records, K-shot ref traces, `selection.json`, declared
eval artifacts, and succeeded-run metric center coverage.
With `--status auto`, registration reads the finalized run-record status when
available and falls back to `provisional`.

Active paper/evaluation configs should declare managed postprocess commands.
The common pattern is:

- `scripts/export_metrics_long.py`
- `scripts/export_paper_table.py` for `pn2021_all_zero_kept_refexcluded`
- `scripts/export_paper_table.py` for `pn2021_drop_all_zero_refexcluded`

Latest-mainline paper/evaluation configs should emit both PN2021 metric views.
Check `configs/active_scripts.yaml:latest_mainline` before treating any config
as a current paper surface.

The generated `command.sh` includes both the managed child commands and these
postprocess commands, so a managed run produces paper-table inputs without a
separate hand-written shell step after child artifacts pass verification.

## Data Manifest

The lightweight data manifest records path and protocol facts without loading
waveforms or recursively scanning large datasets:

```bash
micromamba run -n ECGTwin python scripts/export_data_manifest.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml \
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
every `active_managed` experiment listed in `configs/active_scripts.yaml`:

```bash
micromamba run -n ECGTwin python scripts/audit_managed_configs.py \
  --local-config configs/local/linbinhao_server.example.yaml \
  --require-existing-inputs \
  --output-dir /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/<name>
```

It does not invoke managed child commands or use GPU. It resolves each tracked
config, runs the same command/protocol audit as the launcher, builds the same
artifact trace, and optionally verifies required input checkpoints, heads,
feature caches, and K500 ref artifacts exist.

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

Historical v7 VAE-LHAT manifest backfill is provenance-only and remains listed
under `historical_reporting_tools`; it is not a latest mainline replay command.

Managed Direct vs VAE comparison bundle:

```bash
micromamba run -n ECGTwin python scripts/agent/build_comparison_bundle.py --force
```

The full rationale and checkpoint contract are documented in:

```text
docs/pipelines/agent_operating_layer_20260529.md
```
