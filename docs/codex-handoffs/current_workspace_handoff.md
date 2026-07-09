# Current Workspace Handoff

Updated: 2026-07-10

This is the stable human-readable entrypoint for the current AI-agent refactor
handoff. It complements the machine-readable audit JSON; it does not replace
`AGENTS.md`, `configs/active_evidence_registry.yaml`, or
`configs/active_scripts.yaml`.

Current cleanup state: `configs/experiments/` contains exactly the 10 YAMLs
referenced by `configs/active_scripts.yaml:latest_mainline`; there is no second
public SOTA-replay YAML set. Old executable scripts/configs and legacy
provenance directories were removed from the public tree; use
`docs/refactor_cleanup/cleanup_manifest_summary_20260705.md` and
`docs/refactor_cleanup/latest_mainline_dry_run_20260705.md` for the compact
deletion manifest and dry-run verification record.

## Startup Order

1. Read the first 100 lines of `AGENTS.md` before writes, GPU work, or
   environment changes.
2. Run the CPU-only workspace audit:

```bash
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
```

3. Inspect `handoff_contract` first for the expected startup sequence and
   handoff readiness. Its `handoff_contract.handoff_readiness` entry separates
   CPU audit pass/fail from whether the current dirty workspace is ready for
   handoff or commit. Use
   `handoff_contract.handoff_readiness.source_control_ready` to distinguish a
   clean source/config/index checkpoint from a strict dirty-worktree gate that
   may still be false because guarded `model/*` handles are intentionally
   local-only. Its `handoff_contract.source_of_truth_summary` entry
   reports clean, dirty, untracked, missing, staged-content, unstaged-content,
   mixed index/worktree, and intent-to-add source-of-truth counts. Its
   `staged_content_paths` entry shows declared source-of-truth files whose
   content is already in the index. Its `unstaged_content_paths` entry shows
   paths with worktree-only changes. Its `mixed_index_worktree_paths` entry
   shows paths that have both staged and unstaged changes. Its
   `intent_to_add_paths` entry separates `git add -N` review candidates from
   fully staged content. Its `by_layer_status` entry provides the same counts
   grouped by agent layer, so configs/docs/package/script/test source-of-truth
   risk can be reviewed without scanning every path. Its
   `handoff_contract.source_of_truth_status` list reports whether each
   source-of-truth path exists, is tracked by git, is clean/dirty/untracked,
   and carries raw `index_status`, `worktree_status`, and `intent_to_add`
   fields. Each source-of-truth item also carries `staged_diff` and
   `unstaged_diff` numstat summaries so mixed index/worktree files can be
   reviewed without guessing which side contains the larger change.
   Its `handoff_contract.source_of_truth_review_queue` entry is the compact
   ordered review list for dirty source-of-truth paths; start there before
   manually scanning individual diffs.
   Its `handoff_contract.latest_mainline` entry is the current replay contract:
   `latest_mainline`, method `vae_lhat_threechain_augmix_pn2021c`, launcher
   `scripts/run_experiment.py`, and the `inspect_active_scripts_latest_mainline`
   startup check all come from `configs/active_scripts.yaml`.
   Also inspect `claims[0].evaluation_views` and
   `claims[0].required_reporting_views` before writing results. The active
   registry keeps PN2021 and PN2021-C all-zero-kept/drop-all-zero views as four
   distinct reporting contracts; do not merge those views when quoting a
   result.
   Its `current_handoff_note` entry reports whether this file exists, is
   tracked by git, and what its current git/index status is.
   Also inspect the top-level `active_scripts.config_git_inventory` and
   `active_scripts.config_git_summary` entries; they report whether YAML-managed
   launch configs under `configs/defaults/` and `configs/experiments/` are
   tracked, clean, untracked, staged, unstaged, or only intent-to-add. Inspect
   `config_git_summary.untracked_paths` and `config_git_summary.intent_to_add_paths`
   before treating the YAML launch surface as handoff-ready.
   The audit `issues` list also emits `source_of_truth_missing`,
   `source_of_truth_untracked`, `source_of_truth_dirty`,
   `source_of_truth_intent_to_add`, and `source_of_truth_staged_content` for
   machine-searchable handoff risk triage.
4. Inspect `external_models` and
   `handoff_contract.handoff_readiness.ignored_verified_external_model_dirty_paths`.
   Verified dirty `model/*` handles are host-local symlink targets and do not
   by themselves block handoff/commit readiness. Staged guarded paths or
   unverified dirty model handles still require action.
5. Inspect `git.blocking_artifact_risks` for staged guarded paths, staged
   large/generated artifacts, and unverified dirty local-only guarded paths.
6. Inspect `git.dirty_summary.handoff_gate` for the compact dirty-worktree
   gate.
7. Inspect `git.dirty_summary.by_layer` for per-layer actions before editing,
   staging, or handing off dirty work. Each layer exposes full `paths`,
   `staged_paths`, `unstaged_paths`, `untracked_paths`, and `status_entries`;
   use `git.dirty_summary.by_layer.*.paths` to review exact files rather than
   relying on sample paths.
8. Inspect `git.guarded_staged_paths` before any commit; it must stay empty.

## Source Of Truth

- Active evidence and paper-claim facts live in
  `configs/active_evidence_registry.yaml`.
- Active launch/config policy and managed runner inventory live in
  `configs/active_scripts.yaml`.
- The current paper replay path is the `latest_mainline` block in
  `configs/active_scripts.yaml`: method `vae_lhat_threechain_augmix_pn2021c`,
  launched only through `scripts/run_experiment.py`.
- YAML-managed launch materialization is owned by `scripts/run_experiment.py`
  and package helpers under `ecg_adv_gen/runner/`.
- Managed runner registration lives in
  `ecg_adv_gen/config/entrypoints.py`; `ecg_adv_gen/config/loader.py` derives
  its runner allowlist from that registry. Dry-run manifests expose invoked
  managed runner families under
  `artifact_trace.protocol_audit.command_audit.managed_entrypoints`.
- Direct K-shot fine-tune command expansion and its managed command audit live
  together in `ecg_adv_gen/config/adapters/direct_finetune.py`; shared argv
  helpers remain in `ecg_adv_gen/config/adapters/common.py`. The deleted
  `ecg_adv_gen/config/adapters/direct.py` compatibility shim is not a live
  source of truth.
- Typed VAE-LHAT command expansion for the current EfficientNet mainline lives
  in `ecg_adv_gen/config/adapters/effnet_vae_lhat.py`. ECGFounder locked
  source/K500/three-chain training now routes through the full fine-tune
  adapter (`ecg_adv_gen/config/adapters/ecgfounder_fullft.py`) rather than a
  residual-adapter VAE-LHAT compatibility adapter.
- Current replay surfaces use typed runner adapters:
  `direct_finetune`, `effnet_vae_lhat`, `ecgfounder_fullft`,
  `ecgfounder_pn2021c_eval`, `pn2021_eval`, and `pn2021c_eval`.
  The PN2021 eval adapter emits the paper-safe
  `paper_refexcluded` protocol with K500 minimum ref exclusion; the PN2021-C
  adapter requires clean eval metadata, `v7_refexcluded_100hz1000`, and
  same-center K500 ref exclusion.
- Pure online-AT orchestration helpers live in
  `ecg_adv_gen/training/online_at.py`; the package runner
  `ecg_adv_gen/runner/synth_online_at_super5.py` owns the managed training
  loop.
- Historical v6, prompt-token, nonlocked matrix, and superseded ECGFounder
  launch recipes are no longer live public launch surfaces. Inactive configs
  should not point back into deleted paths or reintroduce `runner.argv`.
- Optional YAML `stages[]` records are now normalized by
  `ecg_adv_gen/config/loader.py` and recorded in dry-run `pipeline_stages`
  manifests without changing single-run command behavior when absent.
- The data/preprocess contract now explicitly records the shared
  PTB-XL/PN2021/ECGTwin decode rules plus ECGFounder
  `official_ptbxl_eval` 500Hz/5000-point feature-cache policy.
- Run finalization and registration consume YAML-derived `run_record` metadata.
  New trusted/provisional registrations require a schema-v2
  `run_file_index.json`, SHA-256 coverage of every reproduction-critical input
  and output, and a registry-pinned index digest. Schema v1 remains readable
  only for exploratory/historical records. `register_run.py --status auto`
  resolves status from finalized metadata before falling back to `provisional`.
- Reusable shared logic belongs under `ecg_adv_gen/`, with focused CPU tests
  under `util/tests/`.
- Historical reports belong under `docs/reports/archive/` and are not active
  evidence unless linked through the registry or a registered run record.

## Dirty Layer Rules

- `model/*` changes are local-only external model handles. Keep their audit
  action as `keep_local_only_do_not_stage`. If a guarded model handle appears
  in `git.guarded_staged_paths`, unstage it before continuing.
- Config changes must keep host paths and GPU ids out of tracked YAML.
- If `source_of_truth_untracked` appears for small project source files that
  should become tracked, run artifact-git-guard checks first and prefer
  `git add -N <paths>` to make them reviewable without staging content.
- If `source_of_truth_intent_to_add` appears, those paths are visible to git
  and should be reviewed, but their contents are not staged yet. Fully stage
  them only after artifact guards and source-of-truth diff review pass, or
  declassify them from `configs/active_scripts.yaml:source_of_truth`.
- If `source_of_truth_staged_content` appears, those source-of-truth paths have
  real content in the index. Review `staged_content_paths`,
  `unstaged_content_paths`, and `mixed_index_worktree_paths` together before
  any commit so partially staged files are not treated as fully reviewed.
- Script changes should archive old entrypoints as provenance and keep callable
  launch behavior behind managed YAML plus package runners.
- Package helper changes require focused CPU tests before handoff.
- Documentation changes should separate durable pipeline docs from archived
  historical reports.

## 2026-07-10 Current Refactor Closeout

The implementation snapshot before this documentation closeout is
`6e4e4cf13810f1ce986b5d4475a1bd7fe432019a`. The authoritative completion
details and final post-documentation SHA live in
`docs/refactor_cleanup/refactor_goal_completion_20260710.md`; use that report
instead of copying counts from the historical sections below.

- ECGFounder method commit `b57ee16` remains in branch history. Evidence commit
  `02f7cef` was integrated exactly once as equivalent local commit `b799e13`,
  touching only the registry and archived depth23 report.
- The 10 latest-mainline stages resolve through the managed launcher and typed
  adapters; the golden contract passes with SHA-256
  `df5d6597167e3eb27b63054ed0e62e823b43c5a48f037c048e8bb6c307b404e1`.
- Startup P0 issues, preprocessing behavior, K500/ref-exclusion wiring,
  ECGFounder copies/JSD preflight, PN2021-C metadata, and all-zero reporting
  views have focused CPU contracts.
- Ten behavior-preserving compression commits remove a raw net 821 LOC. The
  15-file algorithm review surface fell from 11,609 to 11,410 whole-file LOC;
  the one-day review plan covers 3,008 LOC deeply plus 2,877 LOC through guided
  invariant review.
- The active Registry contains six managed records. The direct run is the only
  replay-ready record (45 critical files verified); five incomplete historical
  records remain explicitly non-ready rather than being promoted from their
  surviving evaluation outputs.
- ECGFounder depth2+3 evidence (`0.8161 / 0.5163`, matched-direct gains
  `+4.48 / +7.53 pp`) is registered as single-seed provisional evidence. It is
  not a multi-seed paper conclusion, and its provenance-gap record remains
  active.
- The complete CPU suite passed with 414 tests and 13 third-party deprecation
  warnings. The live workspace audit passed with Registry `error_count=0`, six
  structurally valid managed records, and active configs 10/10.
- The refreshed upstream base is `afec2b883106b0e05339f62cb62476b967301e45`.
  Nothing was pushed, no GPU work was run, and guarded `model/*` handles remain
  local-only and unstaged.

## Historical snapshots (2026-07-01 through 2026-07-05)

The staged counts, test totals, YAML counts, and readiness statements below are
dated provenance. They do not describe the 2026-07-10 worktree and must not
override the current closeout section or the final report.

The current branch is an active, dirty refactor workspace, not a clean handoff
checkpoint. The latest-mainline replay contract itself passes audit: all 10
`latest_mainline` stages resolve through package runners, use typed adapters,
and expose no `runner.argv`. Source control is not ready yet.

Current latest-mainline source-control blockers reported by
`active_scripts.latest_mainline`:

- `source_control_passed=false`
- `untracked_stage_count=0`
- `intent_to_add_stage_count=0`
- `dirty_stage_count=10`
- `intent_to_add_stage_configs=[]`

The current EffNet latest eval configs explicitly use K500
`seed=20260601` / `subset_seed=20260601` for both model paths and ref-exclusion
metadata. This keeps `pn2021_eval_v7_sjr_rgq_refexcluded` and
`pn2021c_effnet_threechain_locked_official_s5` aligned with the locked
EfficientNet direct and three-chain training configs.

Latest PN2021-C configs declare corrupted all-zero-kept and drop-all-zero views
separately as `pn2021c_all_zero_kept_corrupted_refexcluded` and
`pn2021c_drop_all_zero_corrupted_refexcluded`. The recovery CSVs keep the
legacy local `view` values (`macro`, `drop_all_zero`) and also write
`canonical_view` for agent-readable filtering.

The repo-tracked and active runtime `ecg-adv-gen` Codex skill now start with a
2026-07-01 current-mainline override, because the older body of the skill still
contains historical v3/v5 and legacy script notes. The PN2021-C locked default
`model.run_leaf_stem` no longer advertises the stale `augmix_s2` run family; it
uses the locked S5 three-chain naming.

The same skill now treats old prompt-token/script paths and PN2021 v3 notes as
historical rather than active source-of-truth guidance. The legacy script-root
families for paper runners, triple-label runners, PGD cross-center runners,
ECGTwin generation, and ECGTwin author reproduction currently have zero live
files in the public tree; their removal is documented under
`docs/refactor_cleanup/`.

Latest package code is guarded against reintroducing live `scripts.*` imports by
`test_package_code_does_not_import_scripts_modules`, and managed runner profiles
are guarded to match the active managed entrypoints only. Latest-facing package
source also has a focused guard against describing active code with old wrapper
or shim wording. Package source now also has a broad guard against old wrapper
wording and individual archived Python file paths. Startup docs now also guard
against showing legacy dotted import paths or individual archived Python file
paths as examples; the MIMIC Super5 label example in `AGENTS.md` points to the
package-owned label API.

The staged refactor is still source-control dirty. Do not rely on count
snapshots copied into this handoff; rerun the CPU audit and inspect
`handoff_contract.source_of_truth_review_queue`,
`handoff_contract.source_of_truth_summary`, and
`active_scripts.config_git_summary` for the current counts before any handoff
or commit decision. Do not treat `passed=true` from the CPU audit as
commit/handoff readiness. The guarded `model/*` handles remain local-only and
must not be staged.

Latest documentation closeout: `docs/pipelines/README.md` now points readers to
the `latest_mainline` replay contract, and
`test_pipeline_readme_points_to_latest_mainline_replay_contract` guards that
entrypoint. The small archived diagnostic report
`docs/reports/archive/20260701/native_rawfirst_ours_effectiveness_20260701.md`
records the native raw-first PN2021-C check as provenance and explicitly leaves
same-position Direct K500 comparison as not yet claimed.

Latest README evidence closeout: `README.md` now mirrors
`configs/active_evidence_registry.yaml:active_claims[0].summary_metrics` for
PN2021 and PN2021-C, with all-zero-kept and drop-all-zero views shown
separately. `test_readme_current_evidence_matches_active_registry_summary_metrics`
guards those headline values against registry drift.

Fresh verification used on this dirty workspace:

```bash
micromamba run -n ECGTwin python -m pytest util/tests/test_config_loader.py::test_latest_mainline_configs_validate_and_expand_commands util/tests/test_config_loader.py::test_effnet_latest_eval_refs_use_locked_k500_seed util/tests/test_config_loader.py::test_latest_mainline_configs_dry_run_through_run_experiment_cli util/tests/test_active_script_index.py::test_active_script_index_declares_latest_mainline_scope util/tests/test_active_script_index.py::test_active_managed_config_audit_passes_and_writes_reports util/tests/test_agent_operating_layer.py::test_agent_workspace_cli_combines_registry_and_active_script_audits -q
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py::test_package_code_does_not_import_scripts_modules util/tests/test_active_script_index.py::test_package_code_does_not_import_scripts_crosscenter_modules util/tests/test_active_script_index.py::test_active_runner_entrypoints_are_registered_as_managed_runner_profiles util/tests/test_active_script_index.py::test_managed_runner_profiles_match_active_managed_entrypoints_only -q
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py::test_scripts_ecgtwin_gen_has_no_live_python_entrypoints util/tests/test_active_script_index.py::test_live_python_roots_do_not_import_archived_script_modules util/tests/test_active_script_index.py::test_startup_docs_do_not_point_agents_at_legacy_script_roots util/tests/test_active_script_index.py::test_active_source_docs_do_not_point_agents_at_legacy_script_roots util/tests/test_active_script_index.py::test_active_data_layer_plan_does_not_reintroduce_legacy_script_roots -q
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py::test_latest_package_source_does_not_describe_active_code_as_legacy_wrapper util/tests/test_active_script_index.py::test_package_facing_docstrings_do_not_describe_current_code_as_legacy_wrappers -q
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py::test_package_source_does_not_expose_old_wrapper_or_archived_script_paths util/tests/test_active_script_index.py::test_latest_package_source_does_not_describe_active_code_as_legacy_wrapper util/tests/test_active_script_index.py::test_package_facing_docstrings_do_not_describe_current_code_as_legacy_wrappers -q
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py::test_startup_docs_do_not_show_legacy_dotted_import_paths util/tests/test_active_script_index.py::test_startup_docs_do_not_point_agents_at_legacy_script_roots util/tests/test_active_script_index.py::test_agent_startup_docs_point_to_latest_mainline_contract -q
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py::test_startup_docs_do_not_point_to_individual_archived_script_files util/tests/test_active_script_index.py::test_startup_docs_do_not_show_legacy_dotted_import_paths util/tests/test_active_script_index.py::test_startup_docs_do_not_point_agents_at_legacy_script_roots -q
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py::test_pipeline_readme_points_to_latest_mainline_replay_contract util/tests/test_active_script_index.py::test_current_workspace_handoff_records_latest_doc_and_report_closeout util/tests/test_active_script_index.py::test_current_workspace_handoff_avoids_stale_hardcoded_audit_counts -q
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py -q
micromamba run -n ECGTwin python -m pytest util/tests/test_config_loader.py::test_latest_mainline_configs_dry_run_through_run_experiment_cli -q
git diff --check -- .
git diff --cached --check
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
```

The latest focused config/dry-run test run passed with 17 tests, and the
latest legacy-import/profile guard runs passed with 4 tests and 5 tests. The
latest active-code wording guard passed with 2 tests, and the broad package
old-wrapper/path guard passed with 3 tests. The latest startup-doc
old-import guard passed with 3 tests, and the individual archived-script-file
startup-doc guard passed with 3 tests. The latest pipeline README/handoff
closeout guard passed with 3 tests, the full active-script index test passed
with 103 tests, and the latest-mainline dry-run test passed with 1 test. The
CPU-only audit reported `passed=true`, `latest_passed=true`,
`latest_source_control_passed=false`,
`latest_untracked_stage_count=0`, `latest_intent_to_add_stage_count=0`, and
`handoff_contract.handoff_readiness.source_control_ready=false`.

Additional 2026-07-02 verification after README evidence sync:

```bash
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py::test_readme_current_evidence_matches_active_registry_summary_metrics util/tests/test_active_script_index.py::test_readme_points_reproduction_at_latest_mainline_index util/tests/test_active_script_index.py::test_readme_kshot_seed_matches_active_evidence_registry -q
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py -q
micromamba run -n ECGTwin python -m pytest util/tests/test_config_loader.py -q
micromamba run -n ECGTwin python -m pytest util/tests/test_metrics_export.py util/tests/test_paper_tables.py util/tests/test_agent_operating_layer.py -q
micromamba run -n ECGTwin python -m pytest util/tests -q
git diff --check
git diff --cached --check
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
```

Those checks passed with 3, 107, 113, and 38 pytest tests respectively.
The latest full CPU suite passed with 759 tests and 13 third-party deprecation
warnings.
`audit_agent_workspace.py` returned `passed=true`, `warning_count=9`,
`mixed_index_worktree_paths=[]`, and `unstaged_content_paths=[]`; the remaining
warnings are staged source-of-truth review plus guarded local `model/*` handles
that must stay unstaged.

Follow-up completion audit: a read-only subagent confirmed the EffNet
latest-mainline replay contract is covered by registered artifacts, and flagged
two evidence-scope boundaries. ECGFounder locked PN2021-C official S5 now has
supporting metrics/table artifacts registered under
`active_claims[0].supporting_reporting_artifacts.ecgfounder_locked_threechain_pn2021c_s5`
with no direct-delta claim. The depth2/depth3 PN2021-C stages remain managed
evaluation contracts only until their locked latest-mainline metrics are run
and registered; do not cite them as trusted evidence from the current registry.
`test_latest_mainline_evidence_scopes_do_not_overclaim_unregistered_evals` and
`test_supporting_ecgfounder_pn2021c_reporting_artifacts_are_registered_and_exist`
guard those boundaries.

Cleanup audit on 2026-07-05: `configs/experiments/` contains only 18 public
YAMLs: 10 latest-mainline configs plus 8 SOTA replay references. `util/tests/`
contains 11 core tests, the legacy archive directory has zero files,
latest-mainline dry-run passed 10/10, and the core CPU pytest set passed with
189 tests. Source-control readiness remains false until final review and an
explicit commit decision; do not auto-commit.

Final cleanup verification on 2026-07-05:

```bash
micromamba run -n ECGTwin python -m pytest util/tests/test_active_script_index.py::test_current_workspace_handoff_records_latest_doc_and_report_closeout util/tests/test_active_script_index.py::test_current_workspace_handoff_avoids_stale_hardcoded_audit_counts util/tests/test_active_script_index.py::test_latest_mainline_evidence_scopes_do_not_overclaim_unregistered_evals -q
git diff --check
git diff --cached --check
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
micromamba run -n ECGTwin python -m pytest util/tests -q
```

The handoff/evidence-scope guard passed with 3 tests. Both diff whitespace
checks passed. After the subagent remediation, `test_active_script_index.py`
passed with 103 tests, `test_config_loader.py` passed with 113 tests, the
latest-mainline config-loader dry-run pair passed with 11 tests, and the fresh
full CPU suite passed with 754 tests and 13 third-party deprecation warnings in
105.53 seconds. The final audit summary returned `passed=true`,
`error_count=0`, `warning_count=9`, `staged_count=464`,
`guarded_staged_count=0`, `blocked_staged_count=0`, and
`latest_mainline.passed=true`; readiness stays `source_control_ready=false`
because the staged source-of-truth diff still needs final review/commit
approval.

Pasted-goal completion audit on 2026-07-02:

| Requirement | Current evidence |
|---|---|
| Latest method only | `configs/active_scripts.yaml:latest_mainline` declares `vae_lhat_threechain_augmix_pn2021c` with 10 stages. |
| Fixed mapping/order | Latest-mainline audit reports `v7_super5_sjr_rgq_review_20260528`, hash `555ec85d5b51`, class order `CD,HYP,MI,NORM,STTC`. |
| YAML-managed launch | All latest-mainline stages resolve through `scripts/run_experiment.py`, package runner entrypoints, and typed adapters; resolved configs have no `command` override. |
| Mainline code location | Latest runner/config/evaluation/label code is under `ecg_adv_gen/`; live `scripts/` contains only launcher, audit, export, bootstrap, and agent utilities. |
| No old import/CLI compatibility | `ecg_adv_gen/` and live `scripts/` have no deleted legacy package imports. Active `configs/experiments/*.yaml` has no deleted legacy entrypoints or `runner.argv`. |
| Historical work removed | Historical scripts and old executable YAML configs are absent from the public tree; cleanup provenance is recorded under `docs/refactor_cleanup/`. |
| Excluded non-main methods | Latest-mainline excludes stabilizer frontend, raw-supervised branch, single-chain shortcut, operator oracle, and post-hoc selector; evidence tests guard against overclaiming unregistered scopes. |
| PN2021/PN2021-C metric views | Active registry and README report PN2021 and PN2021-C all-zero-kept and drop-all-zero views separately. |
| Ref exclusion | Latest configs/adapters use K500 ref metadata and ref-excluded PN2021/PN2021-C evaluation. |
| Artifacts and external models | Staged model paths, checkpoint/data/cache artifacts, generated samples, and secrets are absent; local `model/*` handle changes remain unstaged. |
| No automatic commit | Branch remains staged/dirty by design; `source_control_ready=false` until user approves final review/commit. |

This historical section originally treated depth23 as unregistered. Since
2026-07-09, its outputs are registered only as single-seed provisional evidence;
the current closeout rules above supersede that older boundary.

## Launch Policy

Use tracked YAML in `configs/defaults/` and `configs/experiments/` plus
`scripts/run_experiment.py` for new managed starts. Do not add new long bash
launchers or dated matrix scripts by default. GPU selection remains external to
tracked YAML and must happen only after checking shared-server GPU state.
Local YAML is host-only. It may define `host`, `paths`, `python`, `resources`,
and `safety`, but it must not override paper protocol, model, evaluation,
runner, postprocess, or logging sections.
If `active_scripts.config_git_summary.requires_attention` is true, review the
reported YAML config paths before starting more launch-surface refactors.
If `config_git_summary.intent_to_add_paths` is non-empty, those YAML contents
are not staged yet; after artifact-git-guard checks, convert intended files to
real staged content before any commit or handoff claim.

## Verification

Minimum CPU-only handoff checks for this layer:

```bash
CUDA_VISIBLE_DEVICES='' micromamba run -n ECGTwin python -m pytest -q util/tests methods/augmix/tests
micromamba run -n ECGTwin python scripts/agent/build_latest_mainline_golden.py --check
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
git diff --check
git diff --cached --check
git diff --cached --name-only -- model
```
