# AI Agent Workspace Refactor Plan 2026-06-04

This document turns the 2026-06-04 parallel agent scan into an execution plan.
It is a refactor guide, not experiment evidence. It should help future agents
avoid re-scanning the same dirty workspace and avoid creating more long
argument launch scripts.

## Current Verdict

The workspace is dirty, but the architecture direction is sound.

- `scripts/run_experiment.py` and `ecg_adv_gen/config/` already provide the
  managed launch surface.
- `configs/active_evidence_registry.yaml` already records the trusted current
  claim and artifact policy.
- `configs/active_scripts.yaml` already records which legacy scripts are
  active wrappers and should not be moved without tests.
- `ecg_adv_gen/` already contains several CPU-testable helpers, especially for
  labels, data contracts, metric views, losses, ECGFounder path contracts, and
  LHAT pure helpers.
- The main problem is unfinished convergence: new v7 matrix scripts and some
  PN2021-C / LHAT entrypoints still encode long CLI argument lists and runtime
  orchestration that should be YAML-managed or package-owned.

## 2026-06-04 Progress

- Added CPU-tested package helpers for real-all-present class trust JSON and
  LHAT diagnostics/attack decisions.
- Added `configs/experiments/pn2021c_effnet_v7_augmix_vs_noaug.yaml`, which
  expands four centers times VAE noAug/LHAT into eight run-id-scoped
  `eval_pn2021_corruptions.py` commands without tracked GPU ids.
- Added PN2021-C command auditing and artifact tracing in
  `ecg_adv_gen/config/loader.py`, then registered the config in
  `configs/active_scripts.yaml`.
- Verified a PN2021-C dry-run through `scripts/run_experiment.py` with
  `configs/experiments/pn2021c_effnet_v7_augmix_vs_noaug.yaml`.
- Verified `util/tests/test_config_loader.py`, `util/tests/test_active_script_index.py`,
  py_compile, and `scripts/agent/audit_agent_workspace.py --skip-existing-artifacts`.
- Added `ecg_adv_gen/runner/process.py` with CPU tests for command rendering,
  process env construction, dry-run logging, append logging, streaming output,
  and failure propagation.
- Added `ecg_adv_gen/runner/launch_plan.py` with CPU tests for stable launch
  command rendering and YAML-managed plan materialization. `scripts/run_experiment.py`
  now delegates resolved-config files, `command.sh`, data manifest, K-shot ref
  IDs, selection record, launch artifact status, and initial run record writing
  to this package helper.
- Rewired `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py` and the
  v7 ECGFounder matrix wrapper to use the shared process helper instead of
  local `subprocess.Popen` copies.
- Verified `util/tests/test_runner_process.py`, low-risk helper tests,
  config/index tests, py_compile, retrospective input inventory, and
  `scripts/agent/audit_agent_workspace.py --skip-existing-artifacts`.
- Added `ecg_adv_gen/adaptation/buffer_labels.py` with CPU tests for every
  adversarial buffer label mode used by LHAT, then rewired
  `push_adv_to_buffer` to call the package helper while preserving the legacy
  function boundary.
- Added `ecg_adv_gen/training/checkpoint_state.py` with CPU tests for atomic
  torch save, JSONL append, resume path resolution, quality buffer state
  round-trip, and RNG state capture/restore, then rewired LHAT checkpoint
  wrappers to delegate to the package helper.
- Added `ecg_adv_gen/evaluation/pn2021_corruptions.py` with CPU tests for
  PN2021-C cache path naming, metadata decode, stable corruption seeds,
  ref-id filtering, clean-metric lookup, and aggregate summaries, then rewired
  `eval_pn2021_corruptions.py` compatibility wrappers to delegate to it.
- Tightened `ecg_adv_gen/runner/process.py` so dry-run logs render only argv
  and `env` is treated as the final child process environment, then added
  regression tests for sensitive-env leakage and parent-env inheritance.
- Added a CPU integration test for `push_adv_to_buffer` legacy label modes so
  the package helper extraction is covered across the old script boundary.
- Updated `configs/active_scripts.yaml` to point the active EfficientNet and
  PN2021 ref-excluded entries at v7 SJR/RGQ configs, and removed a duplicate
  trusted/provisional VAE-LHAT catalog entry from the evidence registry.
- Added `configs/experiments/effnet_direct_k500_v7_sjr_rgq_matrix.yaml`, a
  per-center Direct K500 v7 matrix config that replaces the Direct K500 command
  expansion from `run_effnet_v7_multiseed_kshot_matrix_20260530.py` without
  tracking GPU ids.
- Added `configs/experiments/effnet_v7_sjr_rgq_subset_export.yaml`, a managed
  prep config for exporting K500, 10 percent, and 20 percent v7 ECGTwin-anchor
  subsets. `audit_runner_commands` now rejects unsafe `--force`, invalid
  percents, wrong centers, and non-run-scoped summary paths for this legacy
  prep script.
- Added `configs/experiments/effnet_direct_percent_v7_sjr_rgq_matrix.yaml`, a
  Direct p10/p20 v7 matrix config that consumes the exported seed20260531
  percent-shot subset K values and expands to one run-id-scoped command per
  center/protocol case.
- Added `configs/experiments/effnet_vae_lhat_percent_v7_sjr_rgq.yaml`, a
  VAE-LHAT p10/p20 v7 matrix config that uses protocol-matched Direct percent
  checkpoints from the same `runtime.run_id` and anchors each command to the
  same center/protocol/K subset case.
- Added `configs/defaults/benchmark_backbone_v7_sjr_rgq_defaults.yaml` and
  `configs/experiments/benchmark_resnet1d_direct_k500_v7_sjr_rgq_matrix.yaml`
  to expose benchmark ResNet1D Direct K500 as a managed per-center launch
  surface with an explicit source-checkpoint prerequisite instead of hidden
  source training inside the dated benchmark matrix script.
- Added
  `configs/experiments/benchmark_resnet1d_direct_percent_v7_sjr_rgq_matrix.yaml`
  to expose benchmark ResNet1D Direct p10/p20 as a managed per-center/protocol
  launch surface using the exported seed20260531 percent-shot subset K values.
- Added `configs/experiments/benchmark_source_v7_sjr_rgq_matrix.yaml` to expose
  benchmark PTB-XL source pretraining for ResNet1D/Inception1D/FCN as a managed
  source launch surface instead of a hidden prerequisite inside the dated
  benchmark matrix script. `ecg_adv_gen/config/loader.py` now audits
  `train_ptbxl.py` source-pretrain commands and records source child-run
  artifacts in the dry-run manifest.
- Added
  `configs/experiments/benchmark_inception1d_direct_k500_v7_sjr_rgq_matrix.yaml`
  and `configs/experiments/benchmark_fcn_wang_direct_k500_v7_sjr_rgq_matrix.yaml`
  so all three default benchmark backbones now have managed Direct K500
  per-center launch surfaces.
- Added `configs/defaults/ecgfounder_v7_sjr_rgq_defaults.yaml` and
  `configs/experiments/ecgfounder_direct_k500_v7_sjr_rgq_matrix.yaml` to
  expose ECGFounder frozen-feature Direct K500 v7 as a managed per-center
  launch surface with the v7 linear-probe cache, seed20260531 SJR/RGQ K500
  subset root, and no tracked GPU ids.
- Added `configs/experiments/ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml` to
  expose ECGFounder residual-adapter VAE-LHAT K500 v7 as a managed per-center
  launch surface that consumes v7 Direct K500 heads from the same
  `runtime.run_id`, uses the v7 linear-probe cache, and records v7 SJR/RGQ
  K500 latent/ref artifacts.
- Updated `ecg_adv_gen/config/loader.py` so ECGFounder direct head fine-tune
  commands support `runner.matrix.center`, require `--ref_root` for matrix
  launches, treat `--ref_root` as a path-safety-audited option, and record v7
  K500 ref-meta/signals from the configured subset root in artifact traces.
- Updated `ecg_adv_gen/config/loader.py` so ECGFounder VAE-LHAT v7 matrix
  commands require `--ref_root`, accept managed v7 Direct-head roots instead of
  warning as if they were v6-only, and trace K500 refs through the configured
  v7 subset root while preserving v6 compatibility.
- Added `evidence_scope` to `configs/active_scripts.yaml` so future agents can
  distinguish `trusted_registry_linked`, `launch_only`, `prep_only`, and
  `evaluation_only` managed surfaces instead of treating every runnable config
  as trusted paper evidence.
- Added `inactive_experiment_configs` to `configs/active_scripts.yaml` so every
  tracked `configs/experiments/*.yaml` is explicitly classified as active
  managed, smoke-only, or superseded by v7. The active index tests now enforce
  full experiment-config coverage.
- Ran read-only subagent scans over the large LHAT script and benchmark matrix
  surfaces. The safe next extractions are adapter/model-surgery helpers,
  freeze-aware epoch helpers, latent-pool loading, adversarial buffer insertion,
  and attack BCE diagnostics; the risky GPU/VAE/PGD orchestration should stay
  in legacy scripts until smaller tested protocol objects exist.
- Ran a second read-only subagent scan over module boundaries and YAML launch
  gaps. The highest-value package targets are PN2021 clean cache/eval helpers,
  PTB-XL preprocessing contracts, EfficientNet adapter/model-surgery helpers,
  prompt-token bank utilities, PN2021 Super5 ref selection, prompt-token pool
  selection, and ECGTwin author-repro artifact helpers. The highest-value YAML
  gaps are ECGTwin author IBE/DiT stages, prompt-token train/generate/gate,
  prompt-token online AT plus PN2021 eval, and generalized PN2021-C queues.
- Added `ecg_adv_gen/data/latent_pools.py` with CPU tests for `.npz` to
  `.latent.npz` sidecar resolution, latent/label shape validation,
  source-id/source-name metadata, record-id metadata, and clear package errors.
  The legacy `synth_online_at_super5.py::load_synth_pool` entrypoint now
  delegates to the package helper while preserving script-level `SystemExit`
  behavior for invalid pool artifacts.
- Added `configs/experiments/ecgtwin_author_ibe_repro.yaml` and
  `configs/experiments/ecgtwin_author_dit_repro.yaml` to replace the first two
  stages of `scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh` with
  managed run-id-scoped YAML launch surfaces. `ecg_adv_gen/config/loader.py`
  now audits these author-repro commands for output scoping, non-root
  train/val paths, managed DiT `--ibe_path` coupling, no
  `--use_pretrained_author_ibe` shortcut, and expected author artifact traces.
  These configs are `launch_only`; they are not current classifier metric
  evidence by themselves.
- Added `configs/experiments/ecgtwin_prompt_token_train_minimal.yaml`,
  `configs/experiments/ecgtwin_prompt_token_generate_minimal.yaml`, and
  `configs/experiments/ecgtwin_prompt_token_gate_minimal.yaml` to replace the
  first small prompt-token train -> generate -> gate slice with managed
  run-id-scoped launch surfaces. `ecg_adv_gen/config/loader.py` now audits
  prompt-token train/generate/gate commands for non-root cache/prompt/token
  paths, no tracked GPU ids, same-run token-bank and generated-pool coupling,
  target-token arm safety, NORM/MI/STTC class scope, and expected token/gated
  artifact traces.
- Added `configs/experiments/ecgtwin_prompt_token_online_at_minimal.yaml` to
  wrap the first prompt-token gated-pool online-AT slice around
  `scripts/pgd_cross_center/synth_online_at_super5.py`. The config consumes
  same-run gated latent/class-trust/ref-meta artifacts, v7 K500 target-real
  signals, and the same-run direct-K500 checkpoint; the loader audit rejects
  PN2021 quick-eval selection, unsafe class scopes, unscoped outputs, and
  unmanaged gated-pool paths.
- Added `configs/experiments/ecgtwin_prompt_token_online_at_minimal_pn2021_eval.yaml`
  to wrap the matching v7 PN2021 ref-excluded evaluation for that minimal
  prompt-token online-AT slice. The eval config consumes the same-run online-AT
  model directory, excludes all four target-center K500 ref-meta files, and
  declares managed metrics/table postprocess artifacts without making the eval
  output trusted paper evidence by itself.
- Ran read-only subagent scans over long bash launch artifacts and current YAML
  audit gaps. The highest-priority remaining YAML surfaces are the full
  prompt-token v42/effectiveness pilot, PTB-XL prompt-token boundary AT,
  prompt-token v32-v36 pool build and online AT, module-ablation real-anchor
  PGD, and selected PN2021-C legacy queues. The highest-priority package
  backlog is K500 ref-artifact resolution, PN2021 result/cache normalization,
  prompt-token compilation, gated-pool writing, managed artifact IO, and
  data-driven command contracts.
- Added `ecg_adv_gen/data/kshot_artifacts.py` as the canonical K-shot
  artifact resolver for center/K/seed base paths, signal/latent/ref-meta/trust
  path groups, and ref-meta record-id validation. `ecg_adv_gen/config/loader.py`
  now uses it for K500 ref artifact traces while preserving the existing
  manifest shape.
- Verified buffer-label, checkpoint-state, PN2021-C helper, LHAT helper,
  runner-process, EffNet Direct K500 matrix dry-run, EffNet Direct p10/p20
  matrix dry-run, EffNet VAE-LHAT p10/p20 matrix dry-run, EffNet subset-export
  dry-run, benchmark ResNet1D Direct K500 matrix dry-run, benchmark ResNet1D
  Direct p10/p20 matrix tests, benchmark source-pretrain matrix tests,
  benchmark Inception1D/FCN Direct K500 matrix tests, ECGFounder Direct K500 v7
  matrix dry-run, ECGFounder VAE-LHAT K500 v7 matrix dry-run,
  config/index tests, run-naming tests, managed config audit, py_compile, and
  `scripts/agent/audit_agent_workspace.py --skip-existing-artifacts`.
- Ran a fresh three-agent read-only scan over package extraction candidates,
  remaining YAML launch gaps, and the latest gated-pool helper refactor. The
  scan promoted Super5 mapping, PN2021 clean eval/cache, PN2021-C reporting,
  K-shot/ref-selection, ECGTwin signal transforms, prompt-token generation
  artifacts, quick PN2021 eval, and legacy run finalization into the next
  package backlog; it also identified full prompt-token v42/v36 online-AT,
  selected PN2021-C queues, multiseed suites, and run lifecycle finalization as
  the main remaining YAML gaps.
- Hardened `ecg_adv_gen/data/gated_pools.py` and
  `select_quality_prompt_token_pool.py` after code review: sample and latent NPZ
  metadata must agree on center/class order, merged pools must match the expected
  Super5 class order, quality selection now rejects mixed-center gated dirs, and
  config command auditing treats `--input_dirs` and `--gated_dirs` as path-safe
  multi-value inputs.
- Added managed YAML git-state reporting to `audit_active_managed_configs`.
  The audit now exposes row-level active config state plus a full
  `active_scripts.config_git_inventory` over `configs/defaults/` and
  `configs/experiments/`, summarized by `active_scripts.config_git_summary`.
  Future agents can see which YAML launch configs still need review before
  claiming the long-CLI launch surface is clean.
- Added full dirty-layer path reporting to the CPU-only workspace audit.
  `git.dirty_summary.by_layer` now includes complete `paths`, staged/unstaged
  and untracked path lists, guarded path lists, and raw `status_entries` so an
  agent can review package/script/test/docs layers without re-parsing
  `git status`. All remaining non-model untracked source, test, script, config,
  and archived-doc paths were moved to `git add -N` intent-to-add state after
  artifact checks; no file content was staged by that operation.
- Added source-of-truth staged-vs-unstaged diff numstats to
  `handoff_contract.source_of_truth_status`. Each source-of-truth item now
  reports `staged_diff` and `unstaged_diff`, which makes mixed index/worktree
  files such as `AGENTS.md`, `configs/active_scripts.yaml`, and
  `configs/active_evidence_registry.yaml` reviewable without manually running
  separate `git diff --cached` and `git diff` commands first.
- Added `handoff_contract.source_of_truth_review_queue`, a compact ordered
  queue for dirty source-of-truth paths. It prioritizes mixed index/worktree
  paths before staged content, intent-to-add paths, and unstaged-only paths,
  and attaches the appropriate next action plus staged/unstaged diff stats for
  each item.
- Added `ecg_adv_gen/data/synthetic_npz.py` with CPU tests for synthetic
  classifier NPZ contracts. The helper owns flat `signals` plus `labels` or
  `labels5`, per-center `<center>__signals` plus `<center>__labels5`, stable
  per-center concatenation, `(N,12,1000)` to `(N,1000,12)` normalization, and
  length/shape errors. `scripts/triple_labels/train_ptbxl.py::SynthNPZDataset`
  now delegates NPZ contract handling to the package helper while preserving
  the legacy Dataset boundary.
- Ran a fresh three-subagent read-only scan on 2026-06-04 over training/eval
  entrypoints, data/label/cache APIs, and YAML launch boundaries. The scan
  confirmed the next module targets as PTB-XL source training/preprocessing,
  PN2021 clean/ref-excluded eval, PN2021-C eval, VAE/latent-hull online AT,
  prompt-token train/generate/gate, ECGTwin author IBE/DiT reproduction,
  PN2021 index/waveform/cache, Super5 mapping, synthetic NPZ, K-shot
  artifacts, gated pools, and run-record finalization.
- Hardened the YAML launch boundary after that scan. `configs/local/*.yaml`
  may now define only host/path/python/resource/safety overlays and cannot
  override paper protocol, model, evaluation, runner argv, postprocess, or
  logging sections. `audit_runner_commands` now rejects unknown child runner
  entrypoints unless they are in the managed runner audit allowlist, and
  `scripts/run_experiment.py --execute` imports `finalize_run_record` directly
  so successful child/postprocess commands cannot fall through to a finalizer
  `NameError`.
- Added `ecg_adv_gen/config/entrypoints.py` as the managed legacy entrypoint
  registry. `ecg_adv_gen/config/loader.py` now derives
  `MANAGED_RUNNER_SCRIPT_NAMES` from that registry, and the active-script tests
  require every managed experiment `legacy_entrypoint` to have a registered
  script profile. This creates the seam for future adapter extraction without
  changing legacy CLI behavior.
- Extended runner command audit output with `managed_entrypoints`, a
  JSON-friendly list of script profile records (`script_name`, `relative_path`,
  `wrapper_root`, and `family`). Managed dry-run manifests now expose which
  legacy wrapper families they invoke, making future adapter extraction and
  review possible without re-parsing raw `runner.argv`.
- Started the adapter split for `loader.py` by extracting the YAML-managed
  `train_ptbxl.py` source-pretraining command audit into
  `ecg_adv_gen/config/adapters/source_training.py`. The loader now delegates
  that branch to `audit_train_ptbxl_command`, and focused tests cover both the
  adapter API and the full managed config path.
- Continued the adapter split by extracting the managed
  `run_direct_finetune_k500_20260516.py` command audit into
  `ecg_adv_gen/config/adapters/direct.py` and centralizing adapter argv helpers
  in `ecg_adv_gen/config/adapters/common.py`. The loader now delegates the
  Direct K-shot fine-tune branch to `audit_direct_finetune_command`, preserving
  existing YAML audit behavior while shrinking script-specific logic in
  `loader.py`.
- After staging the source-training and Direct command-audit adapters, the full
  CPU-only `util/tests` run passed with 427 tests. The compact workspace audit
  reported `passed=true`, `hard_reason_count=0`, zero
  unstaged/untracked/mixed source-of-truth paths, zero guarded staged paths, and
  only the expected local-only `model/*` warnings plus staged-content review
  attention before handoff or commit.
- Cleared the managed YAML intent-to-add risk: the 21 new v7/default
  `configs/defaults/*.yaml` and `configs/experiments/*.yaml` files are now
  staged with real blob contents after artifact-git-guard checks. The
  `active_scripts.config_git_summary.intent_to_add_count` audit value is now
  `0`; those YAML files still require staged-content review before commit.
- Added `handoff_contract.source_of_truth_status` to the unified agent audit so
  every path in `configs/active_scripts.yaml:source_of_truth` reports
  `exists`, `tracked_by_git`, `git_status`, `layer`, and
  `required_for_handoff`. Added `handoff_contract.source_of_truth_summary` so
  agents can inspect clean, dirty, untracked, and missing source-of-truth counts
  before reading the full list. Added `git.blocking_artifact_risks` so staged
  guarded paths, blocked staged/tracked artifacts, and dirty local-only guarded
  paths are visible as a compact risk checklist before staging or handoff.
  The unified audit now also emits `source_of_truth_missing` as an error and
  `source_of_truth_untracked` / `source_of_truth_dirty` as warnings so these
  handoff risks are machine-searchable in `issues`.
- Added `handoff_contract.handoff_readiness` to distinguish CPU audit pass/fail
  from dirty-worktree handoff readiness. A report may still have `passed=true`
  while `ready_for_handoff=false` when source-of-truth files are dirty or
  untracked, local-only `model/*` handles are dirty, or the dirty handoff gate
  requires review.
- Added `handoff_contract.handoff_readiness.source_control_ready` so agents can
  tell when source-of-truth paths, managed YAML configs, the index, and hard
  artifact gates are clean even if strict `ready_for_handoff` remains false
  because expected local-only `model/*` handles keep the dirty-worktree
  attention gate active.
- Extended the same handoff contract so source-of-truth paths now expose raw
  git index/worktree columns and an `intent_to_add` flag. The summary, issues,
  and readiness reasons now separate `staged_content_paths`,
  `unstaged_content_paths`, `mixed_index_worktree_paths`, and
  `intent_to_add_paths` from normal dirty paths, making partial staging and
  `git add -N` review candidates visible without implying their contents are
  fully reviewed. The same summary now includes `by_layer_status` so agents can
  review configs, docs, package helpers, scripts, and tests separately before
  staging or handing off.
- Promoted `docs/codex-handoffs/current_workspace_handoff.md` into
  `configs/active_scripts.yaml:source_of_truth` and marked it with
  `git add -N` after artifact-guard checks. The handoff note is now a
  machine-readable source-of-truth path while remaining unstaged content until
  final review.
- Added `ecg_adv_gen/labels/super5_mapping.py` as the package-owned Super5
  conversion policy for PTB-XL SCP codes, PN2021 SNOMED projection, and MIMIC
  report regex labels. `scripts/triple_labels/label_schemes.py` now re-exports
  the Super5 API for legacy callers while keeping sub23/pn26 logic local.
- Ran a three-agent read-only scan over launch/config surfaces, package
  extraction candidates, and agent handoff hygiene. The scan confirmed
  `scripts/run_experiment.py` plus tracked YAML as the only default launch
  surface, flagged `model/*` symlink handles as local-only dirty state, and
  promoted PN2021 records/cache, preprocessing, remaining label registry,
  quality buffers, score fusion, model-zoo construction, and gated-pool
  selection into the next CPU-testable package backlog.
- Added `dirty_summary` to the CPU-only active evidence registry audit so a
  handoff agent can see staged/unstaged/untracked counts by layer
  (`configs`, `docs`, `package`, `scripts`, `tests`, `external_models`,
  `agent_instructions`, `codex_skills`, and related buckets) instead of
  re-parsing the full dirty worktree from scratch.
- Extended `scripts/agent/audit_agent_workspace.py` so the same CPU-only
  handoff command also embeds the active managed-config audit from
  `configs/active_scripts.yaml`. Future agents can inspect one JSON for active
  evidence status, managed YAML launch coverage, and dirty workspace layers.
- Added a tested `launch_surface_policy` block to `configs/active_scripts.yaml`
  so the active index explicitly rejects new long bash launchers and dated
  matrix scripts as the default experiment startup model. New launch surfaces
  should use `scripts/run_experiment.py` with tracked YAML, keep GPU selection
  outside YAML, and update the active index for any human-approved exception.
  The managed-config audit now returns this policy at top level so the unified
  handoff audit exposes it without requiring a separate YAML lookup.
- Added a tested `documentation_surface_policy` block to
  `configs/active_scripts.yaml` so the active index also captures the document
  boundary: durable process docs live under `docs/pipelines/`, historical
  reports live under `docs/reports/archive/YYYYMMDD/`, handoff notes live under
  `docs/codex-handoffs/`, and archived reports are not active evidence unless
  linked through the registry or a registered run record. The managed-config
  audit returns this policy at top level for handoff agents.
- Added a tested `implementation_surface_policy` block to
  `configs/active_scripts.yaml` so the active index captures the package/helper
  and legacy-wrapper boundary: new shared logic belongs in `ecg_adv_gen/` with
  CPU tests under `util/tests/`; active legacy entrypoints stay as thin wrappers
  until the index, YAML configs, and tests move with them. The managed-config
  audit also returns this policy at top level for handoff agents.
- Added per-layer `handoff_action` values to `git.dirty_summary.by_layer` in the
  CPU-only workspace audit so dirty handoffs are not just counted; each layer
  now carries an immediate next action such as keeping `model/*` local-only,
  verifying YAML boundaries, or running focused CPU tests for package changes.
- Added `git.dirty_summary.handoff_gate` so the same audit exposes a compact
  dirty-worktree checklist: whether attention is required, how many dirty
  layers exist, which guarded paths must remain local-only, and the per-layer
  handoff actions to follow before staging or continuing edits.
- Updated the first-100-line `AGENTS.md` startup block to point future agents at
  `git.dirty_summary.handoff_gate` first and `git.dirty_summary.by_layer` for
  detailed per-layer actions.
- Added the stable handoff note
  `docs/codex-handoffs/current_workspace_handoff.md` and tested that both the
  note and the first-100-line `AGENTS.md` startup block point future agents at
  the current audit, dirty gate, active registry, active scripts index, YAML
  launcher, package boundary, archive boundary, and local-only `model/*` guard.
- Added a hard staged-path guard for registry `artifact_policy.do_not_commit`
  entries: `model/*` handles, blocked output/cache directories, and matching
  artifact globs now surface through `git.guarded_staged_paths` and emit
  `guarded_path_staged` errors when staged.
- Added `handoff_contract.current_handoff_note` to the unified audit JSON so a
  future agent can see the stable handoff note path, whether it exists, and
  whether git already tracks it.
- Added top-level `handoff_contract` to `scripts/agent/audit_agent_workspace.py`
  output so a new agent can read one JSON object for startup order,
  source-of-truth paths, the dirty gate path, and launch/docs/implementation
  policies sourced from `configs/active_scripts.yaml`.
- Added `evidence_surface_policy` to `configs/active_evidence_registry.yaml`
  and the registry audit output so the handoff contract also captures what can
  be cited as active paper evidence: trusted claims and registered artifacts,
  not archived reports or launch-only YAML by default.
- Ran a follow-up three-agent read-only scan over the YAML launch surface,
  module extraction candidates, and handoff/audit boundary. The current
  highest-value remaining YAML conversions are the full prompt-token
  effectiveness and boundary-AT shell matrices, offline synth PGD buffer/train
  surfaces, non-minimal `synth_online_at_super5.py` recipes, and self-distill
  synthetic-selection flows. The current highest-value package extractions are
  PTB-XL/synthetic dataset contracts, PN2021 waveform cache loading,
  ref/include filtering, ECGTwin prompt-token PN2021 record preparation, gated
  pool quality/selection, selector score blending, K-shot runner datasets, and
  ECGTwin author repro logging utilities.
- Extracted the PN2021 center-local include/ref-exclusion array filter into
  `ecg_adv_gen.evaluation.filter_pn2021_center_records` and made
  `scripts/triple_labels/eval_crosscenter.py` call the package helper. The
  helper preserves the paper-safe order of applying explicit include IDs before
  K-shot ref exclusion, returns `n_include_kept` and `n_excluded_ref`, and is
  indexed as `pn2021_record_filtering` in `configs/active_scripts.yaml`.
- Extracted PN2021 clean-eval cache selection into
  `ecg_adv_gen.evaluation.load_existing_pn2021_eval_cache`. The package helper
  now owns the mmap-first, valid-NPZ fallback, and NPZ-to-mmap upgrade path;
  `scripts/triple_labels/eval_crosscenter.py` keeps tuple compatibility for
  the legacy evaluator.
- Extracted build-time PN2021 waveform materialization into
  `ecg_adv_gen.data.materialize_pn2021_center_records`. The package helper now
  owns the WFDB-read failure accounting, invalid-signal handling, preprocessing
  callback, label stacking, basename `record_ids`, and empty-array contract;
  the legacy evaluator supplies `wfdb.rdrecord`,
  `unified_preprocess_to_1000`, and the active label mapper.

## Non-Negotiable Boundaries

- Do not move active legacy entrypoints until `configs/active_scripts.yaml`,
  tests, and managed configs are updated in the same change.
- Do not add new long bash launchers or dated matrix scripts as the default
  startup path. Use `scripts/run_experiment.py` plus tracked
  `configs/experiments/*.yaml`, or document the exception in
  `configs/active_scripts.yaml`.
- Do not treat archived Markdown/HTML reports as active evidence by default.
  Cite them through `configs/active_evidence_registry.yaml`, registered run
  records, or durable `docs/pipelines/` process docs.
- Do not put new shared abstractions in legacy wrappers. Extract small,
  CPU-tested helpers into `ecg_adv_gen/`, keep script-level compatibility
  exports where other legacy scripts import them, and avoid moving full GPU
  orchestration loops in one step.
- Do not commit host-local `model/*` symlink/path changes. External model
  handles are runtime setup, not paper evidence.
- Do not add `/root/...`, `/home/linbinhao/...`, Python executable paths,
  credentials, or GPU ids to tracked experiment YAML.
- Do not set `CUDA_VISIBLE_DEVICES` inside tracked YAML. Select GPUs externally
  after `nvidia-smi`.
- Do not replace the managed launcher. Reuse `load_experiment_config`,
  `runner.matrix`, `build_runner_commands`, `build_postprocess_commands`,
  `audit_runner_commands`, and `validate_local_paths`.
- Do not move the full LHAT or ECGFounder training loops in one step. Extract
  small helpers with focused CPU tests first.

## Target Startup Model

Future agents should not write new long bash commands or dated matrix launchers
for active experiments. The standard dry-run surface is:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/<experiment>.yaml \
  --local-config configs/local/linbinhao_server.yaml \
  --run-id <run_id> \
  --dry-run
```

Execution is allowed only after shared-server checks and an explicit external
GPU selection:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/<experiment>.yaml \
  --local-config configs/local/linbinhao_server.yaml \
  --run-id <run_id> \
  --execute
```

Tracked YAML owns paper protocol, method knobs, `runner.argv`,
`runner.matrix`, and `postprocess.commands`. Gitignored local YAML owns host
paths and Python executable resolution.

## Latest Subagent Scan Triage

Do not make these long-argument scripts agent-facing launch surfaces:

1. `scripts/pgd_cross_center/synth_online_at_super5.py`
   - Very broad online-AT argparse surface. Keep as legacy backend; wrap
     prompt-token online-AT and eval through YAML with typed artifacts.
2. `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py`
   - Already wrapped for current EffNet VAE-LHAT configs. Keep new runs under
     YAML, not hand-written bash invocations.
3. `scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py`
   - Already wrapped for ECGFounder VAE-LHAT K500 v7. Do not expand by direct
     CLI; add YAML matrices when percent-shot variants are needed.
4. `scripts/ecgtwin_gen/train_center_prompt_tokens.py`
   - Minimal YAML train/generate/gate configs exist. Expand prompt-token
     families by config, preserving run-id coupling between token bank,
     generated pool, gated pool, and downstream online-AT.
5. `scripts/paper/run_effnet_v7_multiseed_kshot_matrix_20260530.py`
   - Treat as historical orchestration. Current Direct/VAE-LHAT v7 child
     surfaces are split into explicit managed configs.

Next package extraction candidates from the latest scan:

1. `ecg_adv_gen.data.pn2021_records`: PN2021 header age/sex/Dx parsing,
   primary Super5 class selection, deterministic folds, and hybrid K-shot
   sampling.
2. `ecg_adv_gen.preprocessing.ecg_pipeline`: named 100Hz/1000 preprocessing,
   lead reorder, crop, NaN cleanup, and z-score rules currently imported from
   legacy script modules.
3. `ecg_adv_gen.evaluation.pn2021_eval_cache`: clean PN2021 cache
   path/metadata, mmap/NPZ loading, and missing-cache behavior. Status:
   implemented for cache path naming, metadata compatibility, NPZ metadata
   decode, mmap read/write, and `eval_crosscenter.py` compatibility wrappers.
   Ref/include/limit filtering remains in the eval loop and
   `pn2021_metric_views.py`.
4. `ecg_adv_gen.labels.schemes`: package the remaining sub23/pn26 registry
   while leaving `scripts/triple_labels/label_schemes.py` as a compatibility
   wrapper.
5. `ecg_adv_gen.adaptation.quality_buffer`: quality-aware buffer, masked BCE
   epoch helpers, rescoring, and checkpoint-compatible buffer state.
6. `ecg_adv_gen.evaluation.score_fusion`: EffNet/ECGFounder selector blending,
   classwise tie-breaks, min-gain gates, and source-weight ranking.
7. `ecg_adv_gen.models.super5_zoo`: architecture aliasing, checkpoint prefix
   stripping, and brittle external-model imports isolated behind package tests.
8. `ecg_adv_gen.data.gated_pool_selection`: quality/boundary/diversity ranking
   and selected gated-pool artifact round-trips on top of
   `ecg_adv_gen.data.gated_pools`.

Status on 2026-06-04: `ecg_adv_gen.data.pn2021_records` is implemented with
CPU tests and legacy wrappers in `build_prompt_token_latent_cache.py` and
`prep_center_dataset_super5.py`. It owns PN2021 header age/sex metadata,
Super5 primary-class/SNOMED selection, deterministic hash folds, and hybrid
floor-plus-natural record selection without loading WFDB records.

Handoff hygiene fixes from the scan:

- `model/DeepECG`, `model/ECGTwin`, `model/advdiff`,
  `model/ecg_ptbxl_benchmarking`, and `model/ecgfounder` may be dirty on this
  migrated host, but must stay unstaged and uncommitted.
- `AGENTS.md` and `configs/README.md` should point agents to v7
  `active_scripts.yaml` and `active_evidence_registry.yaml`; older v3/v5/v6
  names are historical unless explicitly indexed as smoke or superseded.
- Direct legacy defaults that still mention `/root/...`, `cuda`, or v3 eval
  JSON paths are safe only when managed configs override them. Future tests
  should guard those overrides before any direct script is promoted.
- Fresh three-agent scan refresh on 2026-06-04:
  - Highest remaining YAML targets are full prompt-token effectiveness and
    boundary-AT pipelines, `synth_online_at_super5.py` recipe matrices,
    offline synth PGD buffer/train split, real-anchor online-AT ablations,
    synth-anchor K400 ablations, self-distill synth prep/train, selected
    PN2021-C queues, and lifecycle/finalization config blocks.
  - Highest package targets are PTB-XL dataset/preprocess contracts, generic
    inference/evaluation loops, checkpoint discovery/loading normalization,
    synthetic quality gating, ECGTwin signal-format conversion, quick PN2021
    eval subset construction, prompt-token record-selection payloads, and
    synthetic classifier NPZ contracts.
  - Agent-operating-layer gaps are source-of-truth tracked/dirty status in the
    audit, explicit replay status for legacy/backfilled evidence, and keeping
    `model/*` guarded paths unstaged while many new source-of-truth files are
    still untracked.

### 2026-06-04 Multi-Agent Refresh: YAML And Module Queue

This refresh used three read-only repo-mapper agents plus local active-index
inspection. It should be treated as the current backlog ordering for avoiding
new long bash or ad-hoc argparse launches.

Highest-priority YAML launch conversions:

1. Full prompt-token effectiveness and boundary-AT flows:
   `scripts/ecgtwin_gen/run_center_token_effectiveness_pilot_20260503.sh`,
   `scripts/ecgtwin_gen/run_ptbxl_prompt_token_boundary_at_v1.sh`, and
   `scripts/ecgtwin_gen/run_center_token_v42_noleak_style_semantic_20260503.sh`.
   Split them into existing stage families instead of one new monolithic
   runner: prompt-token train, generate, gate, online-AT, PN2021 eval, and
   optional style/C2ST/no-leak validation configs.
2. Multi-center prompt-token online-AT:
   `scripts/ecgtwin_gen/run_multicenter_prompt_token_v36_online_at.sh`.
   Promote the existing single-center minimal online-AT config into a
   `runner.matrix.center` surface that consumes same-run gated pool artifacts
   and writes run-id-scoped per-center child runs.
3. Non-minimal LHAT and offline adversarial recipes:
   `scripts/pgd_cross_center/synth_online_at_super5.py`,
   `scripts/pgd_cross_center/offline_synth_pgd_at_super5.py`, and
   `scripts/pgd_cross_center/run_module_ablation_1_real_anchored.sh`. Keep the
   Python scripts as compatibility backends; add YAML recipes for attack,
   anchor, loss, class-scope, and evaluation knobs only after the corresponding
   package contracts are tested.
4. PN2021-C selected queues:
   `scripts/triple_labels/run_selected_pn2021_c_queue.sh` and watcher variants.
   Replace hardcoded model-dir arrays with tracked matrix configs and managed
   postprocess exporters for summary and candidate-coverage artifacts.
5. Lifecycle/finalization configs:
   move run purpose, status, and result-summary inputs for `finalize_run.py`,
   `register_run.py`, and `build_comparison_bundle.py` into a small managed
   run-record section so handoff metadata and evidence registry entries do not
   drift.

Highest-priority package extractions:

1. `ecg_adv_gen/data/ptbxl.py` and `ecg_adv_gen/training/ptbxl_super5.py`:
   PTB-XL label loading, split handling, preprocessing, synthetic mixing,
   DataLoader construction, and checkpoint/artifact naming currently centered
   in `scripts/triple_labels/train_ptbxl.py`.
2. `ecg_adv_gen/evaluation/pn2021_clean.py` plus
   `ecg_adv_gen/evaluation/ref_filters.py`: clean PN2021 cache loading/building,
   `PN2021CachedCenterDataset`, include-before-exclude ref filtering, and
   effective-record counts currently split across `eval_crosscenter.py`,
   PN2021-C eval, and selector scripts.
3. `ecg_adv_gen/preprocessing/ecgtwin_waveforms.py`: explicit ECGTwin
   1024-sample raw-mV conversion, PTB-XL/ECGTwin lead reordering, decoded
   1024-to-classifier-1000 conversion, and shape/NaN validation.
4. `ecg_adv_gen/data/latent_pool_exports.py`: canonical `.latent.npz`,
   class-trust JSON, ref-meta JSON, source-id/source-name, and duplicate-ref
   checks for real-anchor and prompt-token pools.
5. `ecg_adv_gen/generation/prompt_token_cache.py` and prompt-token
   sampling/gating modules: PN2021 record scanning, deterministic K/ref
   selection payloads, token-bank loading, generation summaries, and canonical
   gated-pool writes on top of `ecg_adv_gen/data/gated_pools.py`.
6. `ecg_adv_gen/evaluation/target_validation.py` and
   `ecg_adv_gen/evaluation/score_fusion.py`: K500-internal masks, EffNet and
   ECGFounder selector blending, classwise tie-breaks, and min-positive
   fallbacks without heldout-target oracle drift.
7. `ecg_adv_gen/reporting/pn2021_c_summaries.py` and
   `ecg_adv_gen/evidence/legacy_outputs.py`: normalize legacy
   `training_log.json`, `train_result.json`, `eval_result*.json`, PN2021-C
   JSON, and prompt-token summaries into finalizable run records.

Keep these entrypoints stable as legacy wrappers until the active index,
managed configs, and tests move with them: `scripts/triple_labels/train_ptbxl.py`,
`scripts/triple_labels/eval_crosscenter.py`,
`scripts/ecgtwin_author_repro/train_ibe_repro.py`,
`scripts/ecgtwin_author_repro/train_dit_repro.py`,
`scripts/ecgtwin_gen/train_center_prompt_tokens.py`,
`scripts/ecgtwin_gen/generate_center_prompt_token_synth.py`,
`scripts/ecgtwin_gen/gate_prompt_token_synth.py`,
`scripts/pgd_cross_center/synth_online_at_super5.py`, and
`scripts/pgd_cross_center/offline_synth_pgd_at_super5.py`.

Future `configs/active_scripts.yaml` entries for these surfaces should keep
the existing fields (`status`, `evidence_scope`, `config`, `legacy_entrypoint`,
`launcher`, `method_family`, `mapping_version`, `owns_training_logic`) and add
stage/lineage notes when a config consumes artifacts from another config in the
same `runtime.run_id`. The next launcher abstraction should be typed
`runner.adapter` or staged `stages[].requires` / `stages[].produces`, not a new
bash pipeline.

## YAML Management Priorities

1. `scripts/paper/run_effnet_v7_multiseed_kshot_matrix_20260530.py`
   - Replace as a default launch surface with tracked configs for direct and
     VAE-LHAT v7 matrix runs.
   - Start from existing `effnet_direct_k500_v7_sjr_rgq.yaml` and
     `effnet_vae_lhat_k500_v7_sjr_rgq.yaml`.
   - Add matrix dimensions only after the single-seed/single-protocol dry-run
     remains traceable.
   - Status: Direct K500 per-center matrix is managed by
     `configs/experiments/effnet_direct_k500_v7_sjr_rgq_matrix.yaml`; K500,
     p10, and p20 subset export prep is managed by
     `configs/experiments/effnet_v7_sjr_rgq_subset_export.yaml`;
     percent-shot Direct p10/p20 variants are managed by
     `configs/experiments/effnet_direct_percent_v7_sjr_rgq_matrix.yaml`;
     percent-shot VAE-LHAT p10/p20 variants are managed by
     `configs/experiments/effnet_vae_lhat_percent_v7_sjr_rgq.yaml` with
     protocol-matched Direct checkpoint coupling. Remaining matrix-script
     work is now ECGFounder and benchmark-backbone surfaces.

2. `scripts/paper/eval_effnet_v7_pn2021c_augmix_vs_noaug_20260530.py`
   - Add a managed PN2021-C eval config with corruptions, severities,
     severity profile, ref-exclusion metadata, clean eval JSON, and output
     paths.
   - Keep the heavy eval as a legacy child until CPU-only command generation
     and artifact naming are tested.
   - Status: managed by
     `configs/experiments/pn2021c_effnet_v7_augmix_vs_noaug.yaml`; the legacy
     eval script now delegates CPU-safe cache naming, severity metadata,
     ref-id filtering, and aggregation helpers to
     `ecg_adv_gen/evaluation/pn2021_corruptions.py`.

3. `scripts/paper/run_ecgfounder_v7_multiseed_kshot_matrix_20260530.py`
   - Split into prep, direct, and VAE-LHAT managed configs.
   - Keep ECGFounder internals in legacy scripts until equivalent package
     helpers have tests.
   - Status: Direct K500 per-center matrix is managed by
     `configs/experiments/ecgfounder_direct_k500_v7_sjr_rgq_matrix.yaml`;
     VAE-LHAT K500 per-center matrix is managed by
     `configs/experiments/ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml`.
     Remaining ECGFounder work is percent-shot protocols and any cache
     prep/relabeling surfaces that must be rerun rather than treated as
     existing prerequisites.

4. `scripts/paper/run_benchmark_backbone_v7_kshot_matrix_20260530.py`
   - Add a benchmark defaults file and one small managed config before
     expanding the model matrix.
   - Do not hide multi-GPU scheduling inside tracked YAML defaults.
   - Status: benchmark source pretraining, benchmark defaults, Direct K500
     per-center matrices for ResNet1D/Inception1D/FCN, and ResNet1D Direct
     p10/p20 per-center/protocol matrix are managed by
     `configs/experiments/benchmark_source_v7_sjr_rgq_matrix.yaml`,
     `configs/experiments/benchmark_resnet1d_direct_k500_v7_sjr_rgq_matrix.yaml`,
     `configs/experiments/benchmark_inception1d_direct_k500_v7_sjr_rgq_matrix.yaml`,
     `configs/experiments/benchmark_fcn_wang_direct_k500_v7_sjr_rgq_matrix.yaml`,
     and
     `configs/experiments/benchmark_resnet1d_direct_percent_v7_sjr_rgq_matrix.yaml`.
     Benchmark VAE-LHAT and broader percent-shot/multi-model expansions still
     need separate managed surfaces.

5. `scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh`
   - Replace the hand-written stage-1/stage-2 author reproduction shell with
     managed run-id-scoped configs.
   - Keep the legacy training scripts in place for artifact-layout stability;
     extract shared logging/artifact helpers only after package tests exist.
   - Status: stage 1 IBE is managed by
     `configs/experiments/ecgtwin_author_ibe_repro.yaml`; stage 2 DiT is
     managed by `configs/experiments/ecgtwin_author_dit_repro.yaml`. DiT
     consumes `IBE_best.pth` from the same `runtime.run_id`. Persistent
     DataLoader workers are intentionally omitted from these managed configs
     because the prior interrupted author run had worker cleanup issues.

6. `scripts/paper/export_percent_kshot_v7_sjr_rgq_20260530.py`
   - Treat as a managed prep candidate.
   - A true pre-child stage may be needed before this can replace current
     matrix-script behavior. Until then, keep it explicit and dry-run first.

### Next YAMLization Queue From Read-Only Scan

1. Wrap one minimal prompt-token train/generate/gate pipeline:
   `train_center_prompt_tokens.py`, `generate_center_prompt_token_synth.py`,
   and `gate_prompt_token_synth.py` should start as one small target-center
   config family before replacing the older 100+ line shell flows.
   Status: managed by
   `configs/experiments/ecgtwin_prompt_token_train_minimal.yaml`,
   `configs/experiments/ecgtwin_prompt_token_generate_minimal.yaml`, and
   `configs/experiments/ecgtwin_prompt_token_gate_minimal.yaml`.
2. Add full prompt-token v42/no-leak style-semantic YAML:
   `scripts/ecgtwin_gen/run_center_token_v42_noleak_style_semantic_20260503.sh`
   still encodes the style checkpoint, semantic checkpoint, auxiliary losses,
   no-leak validation, downstream generation/gating, and output roots in shell.
   Make this a typed prompt-token config family before treating it as a default
   launch route.
3. Wrap prompt-token online AT plus PN2021 eval:
   `scripts/pgd_cross_center/synth_online_at_super5.py` should consume
   managed `gated_samples.latent.npz`, class-trust JSON, and ref-meta JSON
   artifacts, with `eval_crosscenter.py` as managed postprocess. This replaces
   the long `run_multicenter_prompt_token_v36_online_at.sh` style launch surface
   before adding more prompt-token pool variants.
   Status: the first single-center online-AT launch surface is managed by
   `configs/experiments/ecgtwin_prompt_token_online_at_minimal.yaml`, and its
   matching single-center v7 PN2021 ref-excluded eval/reporting surface is
   managed by
   `configs/experiments/ecgtwin_prompt_token_online_at_minimal_pn2021_eval.yaml`.
   Full v36/v42 matrices remain pending.
4. Generalize the PN2021-C selected queue:
   replace hardcoded model-dir arrays in `run_selected_pn2021_c_queue.sh` with
   a matrix config plus postprocess exports for PN2021-C summaries and
   candidate coverage.
5. Add multiseed suite orchestration only after child configs are stable:
   dated EffNet/ECGFounder suite runners should become explicit prerequisite
   plus child-command references instead of reintroducing `--gpus` and hidden
   subprocess scheduling.
6. Add a lifecycle/finalization block:
   `finalize_run.py`, `register_run.py`, and `build_comparison_bundle.py` should
   consume experiment description/status/result summary from YAML or a small
   run-record section so handoff metadata does not drift from the registry.
7. Add a staged-pipeline config abstraction for author/prompt-token flows once
   at least two YAML-managed staged examples are stable under dry-run audit.
8. Add package-level ECGTwin author artifact/logging helpers so
   `train_ibe_repro.py` and `train_dit_repro.py` can share JSONL, curve, and
   run-config writing without moving the stable legacy entrypoints.

Longer-term config-loader work should add either `runner.kind` or
`runner.adapter` and a staged pipeline schema with `stages[].requires`,
`stages[].produces`, and `stages[].skip_if_exists`. Until that exists, prefer
explicit `runner.argv` YAML over new bash scripts.

## Package Extraction Priorities

Prefer small modules that can be tested without GPU, WFDB scans, or large
artifacts.

| Priority | Target module | Source area | Purpose |
|---:|---|---|---|
| 1 | `ecg_adv_gen/runner/process.py` | v7 matrix scripts and stage-3 runner | shared `run_stream`, base env, dry-run/log helpers |
| 2 | `ecg_adv_gen/data/class_trust.py` | EfficientNet stage-3 runner | write real-all-present class trust JSON from `.signals.npz` labels |
| 3 | `ecg_adv_gen/adaptation/diagnostics.py` | LHAT script | decoded invalid stats and attack decision helpers |
| 4 | `ecg_adv_gen/adaptation/buffer_labels.py` | LHAT script | adversarial buffer label construction |
| 5 | `ecg_adv_gen/training/checkpoint_state.py` | LHAT script | checkpoint path resolution, index append, resumable state helpers |
| 6 | `ecg_adv_gen/evaluation/pn2021_corruptions.py` | PN2021-C eval script | stable seed, cache path naming, aggregation, pure ref-id filtering helpers |
| 7 | `ecg_adv_gen/evaluation/blending.py` | ECGFounder/EffNet selector scripts | classwise/global score blending and K500-internal selector logic without heldout oracle drift |
| 8 | `ecg_adv_gen/models/ecgfounder_runtime.py` | ECGFounder VAE-LHAT script | torch runtime wrappers only, kept out of CPU config imports |
| 9 | `ecg_adv_gen/training/run_artifacts.py` | ECGTwin author repro scripts | shared metrics JSONL, loss curves, run_config, and checkpoint artifact layout |
| 10 | `ecg_adv_gen/models/effnet_adapters.py` | LHAT script | Foldable LoRA classifier adapters, classifier-only adaptation, last-block unfreezing, checkpoint-compatible state dicts |
| 11 | `ecg_adv_gen/training/epochs.py` | LHAT script | freeze-aware masked BCE, source-logit anchoring epoch helpers, EWA updates, max-batch limits |
| 12 | `ecg_adv_gen/adaptation/buffers.py` | LHAT script | adversarial buffer insertion around class trust, boundary windows, crops, and per-sample scores |
| 13 | `ecg_adv_gen/data/latent_pools.py` | LHAT script | `.latent.npz` pool loading, shape validation, source-id/source-name/record-id metadata contracts |
| 14 | `ecg_adv_gen/training/consistency.py` | LHAT script | raw corruption consistency epochs after corruption view construction has package tests |

Status on 2026-06-04: priorities 1-6 and 13 are implemented with CPU tests and
legacy compatibility wrappers. The prompt-token compiler sidecar and synthetic
classifier NPZ contract are also implemented as CPU-only helpers. Priorities
7-12 and 14 remain candidates; keep them small and behavior-locked before
moving any larger training loop.

Additional sidecar-scan backlog:

- `ecg_adv_gen/labels/super5_mapping.py`: Super5 conversion policy has been
  moved out of `scripts/triple_labels/label_schemes.py` without changing class
  order, mapping version, or mapping hash. Legacy `label_schemes.py` re-exports
  the package functions for old entrypoints.
- `ecg_adv_gen/data/pn2021_records.py`: PN2021 record metadata policy has been
  moved out of prompt-token data scripts for age/sex parsing, primary Super5
  class, primary SNOMED, hash fold, and hybrid record selection. Legacy
  prompt-token cache/dataset scripts keep their function names as wrappers.
- `ecg_adv_gen/evaluation/pn2021_eval_cache.py`: clean PN2021 cache path,
  metadata, NPZ metadata decode, and mmap read/write helpers have been moved
  out of `eval_crosscenter.py`. Remaining clean-eval package work is the
  model-free include/exclude/limit slicing facade around loaded caches.
- `ecg_adv_gen/training/ptbxl_super5.py` and
  `ecg_adv_gen/training/artifacts.py`: extract PTB-XL Super5 dataset, split
  handling, and training artifact contracts from `train_ptbxl.py` while keeping
  the legacy CLI path stable. The synthetic NPZ loading/normalization contract
  is already package-owned by `ecg_adv_gen/data/synthetic_npz.py`.
- `ecg_adv_gen/data/pn2021_refs.py`: package the remaining K-shot/ref-selection
  and center-latent-cache policy around ref-meta IDs used for downstream
  exclusion, building on `ecg_adv_gen/data/pn2021_records.py`.
- `ecg_adv_gen/preprocessing/signals.py`: finish the ECGTwin raw-to-latent and
  decoded-to-classifier transform facade so lead order, raw mV vs normalized
  signals, 1024-to-1000 resampling, and aVL/aVF swaps are not reimplemented in
  scripts.
- `ecg_adv_gen/generation/prompt_token_artifacts.py`: package prompt-token
  generation summaries, ref trace records, and sampling policy metadata before
  adding more prompt-token pool selectors.
- `ecg_adv_gen/evaluation/pn2021_quick.py`: move quick PN2021 eval subset logic
  out of online-AT scripts after the clean PN2021 eval/cache package exists.
- `ecg_adv_gen/evidence/legacy_outputs.py`: normalize legacy
  `training_log.json`, `train_result.json`, `eval_result*.json`, PN2021-C JSON,
  and prompt-token summaries into finalizable run records.
- `ecg_adv_gen/data/kshot_artifacts.py`: canonical K500 signals/latent/ref-meta
  path construction, ref-id extraction, and K/seed validation. Status:
  implemented for path grouping and ref-meta ID validation; ref-union writing
  can be layered on top when the launcher/finalizer code path is next touched.
- `ecg_adv_gen/protocols/pn2021_super5.py`: target/eval centers, leak-excluded
  centers, mapping version/hash, and K500 seed validation against YAML.
- `ecg_adv_gen/evaluation/pn2021_results.py`: clean PN2021, PN2021-C, and
  metrics-long result normalization with all-zero kept/drop-all-zero views.
- `ecg_adv_gen/data/pn2021_cache.py`: clean/corrupt PN2021 cache lookup,
  metadata matching, and mmap/NPZ naming without changing existing cache names.
- `ecg_adv_gen/generation/prompt_tokens.py`: prompt bank lookup, direct vs
  factorized token-bank decoding, token scaling, and nonzero text-mask
  construction. Status: implemented for prompt-bank SNOMED/class fallback,
  direct/factorized bank decoding, initialization-aware token scaling, repeated
  token append, and nonzero mask construction; `generate_center_prompt_token_synth.py`
  now imports this helper instead of carrying local token-bank functions or
  importing the trainer private lookup helper.
- `ecg_adv_gen/data/gated_pools.py`: prompt-token gated pool write/validate
  helpers for `gated_samples.npz`, `.latent.npz`, class-trust, ref-meta, and
  gate-report artifacts. Status: implemented for canonical artifact paths,
  matching sample/latent NPZ writes and validation, HYP/CD-cautious class trust,
  and ref-meta JSON writing; `gate_prompt_token_synth.py` and YAML child-run
  artifact discovery now use this helper for the canonical gated-pool layout.
  The helper also owns gated-pool merge semantics: center consistency checks,
  sample/latent agreement, source-trust-aware class-trust merge, ref-id union,
  and `merge_report.json`; `merge_gated_prompt_token_pools.py` is now a thin
  CLI wrapper over that package helper. Selector-style gated-pool writes are
  also centralized through `write_selected_gated_pool_artifacts`, with
  `select_quality_prompt_token_pool.py` now using it for canonical artifacts
  and `quality_report.json`.
- `ecg_adv_gen/artifacts/io.py`: shared JSON/YAML write helpers, SHA records,
  and small-file artifact layout helpers for launch/finalize/report tools.
- `ecg_adv_gen/experiments/commands.py`: data-driven command contract specs
  for required options, forbidden options, path roles, output roles, and
  lineage edges, introduced first as validation/dry-run helpers.

YAML knobs still worth moving out of LHAT long argv once the corresponding
package contracts are tested: source sampling weights/floors, anchor-class
weight policy, `classes_in_scope`, boundary probability windows,
`adv_soft_target_floor`, raw-corruption consistency options, source-logit
anchor options, classifier adapter/freezing options, and quality-gate
thresholds. Resume policy should stay CLI/local-run owned.

Keep script-level compatibility exports for functions imported by other legacy
scripts, especially `run_pgd_on_synth_pool` and `push_adv_to_buffer` from
`scripts/pgd_cross_center/synth_online_at_super5.py`.

## Agent Structure Responsibilities

| Layer | Owns | Should not own |
|---|---|---|
| `AGENTS.md` | shared-server safety, current host override, startup commands, repository navigation | long historical commands, old metrics, full dated reports |
| `configs/active_evidence_registry.yaml` | current trusted claims, mapping hash, run lineage, artifact policy | how-to prose or exploratory notes |
| `configs/active_scripts.yaml` | active managed configs, inactive/smoke config inventory, evidence_scope tier, legacy wrapper index, do-not-move paths | detailed experiment narratives or metric claims not backed by the registry |
| `configs/experiments/*.yaml` | paper protocol, method knobs, runner argv/matrix, postprocess commands | host-specific paths or GPU ids |
| `configs/local/*.yaml` | host paths, Python executable, output roots, write boundary | paper protocol or method changes |
| `ecg_adv_gen/` | reusable tested helpers and contracts | dated one-off experiment story |
| `scripts/paper/`, `scripts/pgd_cross_center/`, `scripts/triple_labels/` | legacy entrypoints and thin wrappers | new shared abstractions where package helpers are available |
| `docs/pipelines/` | durable process and source-of-truth docs | raw temporary reports |
| `docs/reports/archive/` | historical Markdown/HTML reports | canonical active evidence unless explicitly linked |

## Execution Batches

### Batch 0: Worktree Safety

- Record current `git status --short`.
- Keep staged documentation/archive work separate from unstaged code work.
- Confirm no guarded `model/*` paths are staged before any commit.
- Inspect `git.dirty_summary.by_layer` from `audit_agent_workspace.py` before
  taking over a dirty handoff; it is the fastest split between source-of-truth
  edits, package/helper edits, docs/archive moves, tests, and local-only model
  handles. Follow each layer's `handoff_action` before staging or continuing
  edits in that layer.
- Inspect `git.dirty_summary.handoff_gate` first when you only need the compact
  dirty-worktree checklist.
- Inspect `handoff_contract.source_of_truth_summary` first, then
  `handoff_contract.source_of_truth_status`, to see which declared
  source-of-truth files are clean, dirty, or untracked before deciding what can
  be handed off safely.
- Inspect `handoff_contract.handoff_readiness` before treating a green audit as
  handoff-ready. `passed=true` only means no CPU audit hard errors; readiness
  remains false while dirty/untracked source-of-truth or local-only artifact
  risks still require review.
- For small project source files that are declared as source-of-truth but still
  untracked, run artifact-git-guard checks and use `git add -N <paths>` before
  full staging. This moves them from `source_of_truth_untracked` to reviewable
  dirty source files without putting content into the staged diff.
- Inspect `git.blocking_artifact_risks` before any staging/commit operation;
  guarded staged paths and blocked staged artifacts are hard errors, while dirty
  guarded `model/*` handles must remain local-only.
- Inspect `active_scripts` in the same audit JSON before launching or editing
  managed experiments; it is the fastest check that tracked YAML still expands
  and protocol-audits cleanly.

Validation:

```bash
git status --short
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
```

### Batch 1: Low-Risk Package Helpers

Create focused modules and tests for:

- `ecg_adv_gen/runner/process.py`
- `ecg_adv_gen/data/class_trust.py`
- `ecg_adv_gen/adaptation/diagnostics.py`

Status on 2026-06-04: implemented and CPU-tested. Stage-3 EfficientNet and v7
ECGFounder matrix wrappers now use `ecg_adv_gen.runner.process` for shared
streamed child-process execution.

Validation:

```bash
micromamba run -n ECGTwin pytest \
  util/tests/test_runner_process.py \
  util/tests/test_class_trust.py \
  util/tests/test_adaptation_diagnostics.py -q
```

### Batch 2: YAML-Managed Launch Expansion

- Add one EffNet v7 matrix config variant with `runner.matrix` instead of a
  dated matrix script.
- Add one PN2021-C eval config with run-id-scoped output paths.
- Update `configs/active_scripts.yaml` and config-loader tests.

Status on 2026-06-04: EffNet Direct K500 v7 per-center matrix, EffNet Direct
p10/p20 v7 matrix, EffNet VAE-LHAT p10/p20 v7 matrix, EffNet v7
subset-export prep, benchmark ResNet1D Direct K500 v7 matrix, benchmark
ResNet1D Direct p10/p20 v7 matrix, ECGFounder Direct K500 v7 matrix,
ECGFounder VAE-LHAT K500 v7 matrix, and PN2021-C EfficientNet v7 noAug versus
LHAT eval are YAML-managed and indexed in
`configs/active_scripts.yaml`. PN2021-C also has package-owned CPU-safe helper
logic in `ecg_adv_gen/evaluation/pn2021_corruptions.py`. Every tracked
`configs/experiments/*.yaml` is now covered by either `managed_experiments` or
`inactive_experiment_configs`, so future agents can distinguish active launch
surfaces from smoke-only and superseded v6 files before scheduling work.

Validation:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/<new_effnet_config>.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id dryrun_agent_refactor_20260604 \
  --dry-run

micromamba run -n ECGTwin pytest \
  util/tests/test_pn2021_corruptions.py \
  util/tests/test_config_loader.py \
  util/tests/test_active_script_index.py -q

micromamba run -n ECGTwin python scripts/audit_managed_configs.py \
  --local-config configs/local/linbinhao_server.example.yaml
```

### Batch 3: Sensitive LHAT Extraction

- Extract buffer label construction.
- Extract checkpoint/resume helpers.
- Preserve legacy wrapper imports.

Status on 2026-06-04: buffer label construction, generic checkpoint state
helpers, and latent pool loading have been extracted and CPU-tested in
`ecg_adv_gen/adaptation/buffer_labels.py`,
`ecg_adv_gen/training/checkpoint_state.py`, and
`ecg_adv_gen/data/latent_pools.py`. LHAT walker state remains local and can be
extracted later if another script needs it.

Validation:

```bash
micromamba run -n ECGTwin pytest \
  util/tests/test_checkpoint_state.py \
  util/tests/test_latent_pools.py \
  util/tests/test_buffer_labels.py \
  util/tests/test_adaptation_lhat.py -q
```

### Batch 4: Final Agent Audit

Run only CPU-safe checks unless the user explicitly requests GPU execution.

```bash
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
micromamba run -n ECGTwin python scripts/agent/audit_retrospective_inputs.py --format markdown
git grep -n "docs/tmp_md\\|docs/tmp_html\\|docs/tmp_jsonl" -- . || true
```

## Completion Criteria

The refactor is not complete until current-state evidence proves all of these:

- active launch surfaces have YAML-managed dry-run plans for the selected v7
  and PN2021-C workflows;
- new reusable helpers live under `ecg_adv_gen/` with CPU tests;
- active legacy scripts still import/run through compatible wrappers;
- `AGENTS.md`, `active_evidence_registry.yaml`, `active_scripts.yaml`, and
  `docs/pipelines/` have non-overlapping responsibilities;
- `configs/active_scripts.yaml` references tracked configs, covers every
  tracked experiment YAML as active or inactive/smoke/superseded, and avoids
  stale v6 names for the current v7 EfficientNet mainline where v7 configs
  exist;
- no host-local `model/*`, large artifacts, datasets, checkpoints, or generated
  samples are staged;
- CPU-only audit and relevant tests pass.
