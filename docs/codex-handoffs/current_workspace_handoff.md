# Current Workspace Handoff

Updated: 2026-06-04

This is the stable human-readable entrypoint for the current AI-agent refactor
handoff. It complements the machine-readable audit JSON; it does not replace
`AGENTS.md`, `configs/active_evidence_registry.yaml`, or
`configs/active_scripts.yaml`.

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
   Its `current_handoff_note` entry reports whether this file exists, is
   tracked by git, and is currently only an intent-to-add review candidate.
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
4. Inspect `git.blocking_artifact_risks` for staged guarded paths, staged
   large/generated artifacts, and dirty local-only guarded paths.
5. Inspect `git.dirty_summary.handoff_gate` for the compact dirty-worktree
   gate.
6. Inspect `git.dirty_summary.by_layer` for per-layer actions before editing,
   staging, or handing off dirty work. Each layer exposes full `paths`,
   `staged_paths`, `unstaged_paths`, `untracked_paths`, and `status_entries`;
   use `git.dirty_summary.by_layer.*.paths` to review exact files rather than
   relying on sample paths.
7. Inspect `git.guarded_staged_paths` before any commit; it must stay empty.

## Source Of Truth

- Active evidence and paper-claim facts live in
  `configs/active_evidence_registry.yaml`.
- Active launch/config policy and legacy wrapper inventory live in
  `configs/active_scripts.yaml`.
- YAML-managed launch materialization is owned by `scripts/run_experiment.py`
  and package helpers under `ecg_adv_gen/runner/`.
- Managed legacy entrypoint registration lives in
  `ecg_adv_gen/config/entrypoints.py`; `ecg_adv_gen/config/loader.py` derives
  its runner allowlist from that registry. Dry-run manifests expose invoked
  legacy wrapper families under
  `artifact_trace.protocol_audit.command_audit.managed_entrypoints`.
- The first command-audit adapter is
  `ecg_adv_gen/config/adapters/source_training.py`, which owns the managed
  `train_ptbxl.py` source-pretraining command checks.
- Direct K-shot fine-tune command checks live in
  `ecg_adv_gen/config/adapters/direct.py`, with shared argv helpers in
  `ecg_adv_gen/config/adapters/common.py`.
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
- Script changes should keep legacy entrypoints as thin wrappers when package
  helpers already exist.
- Package helper changes require focused CPU tests before handoff.
- Documentation changes should separate durable pipeline docs from archived
  historical reports.

## 2026-06-04 Commit State

As of the local `Refactor agent workspace handoff layer` checkpoint commit, all
non-`model/*` source, config, script, test, and documentation changes from this
agent-workspace refactor are committed. The index is expected to be empty, and
`git diff --name-only -- . ':(exclude)model/**'` should return no paths. The
only dirty paths expected to remain are the five guarded external model handles
under `model/`. Do not stage them unless the user explicitly changes the
external-model policy.

Fresh verification used:

```bash
micromamba run -n ECGTwin python -m pytest util/tests -q
git diff --cached --check
git diff --cached --name-only -- model
git diff --name-only -- . ':(exclude)model/**'
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
```

The full `util/tests` run passed with 427 tests and only third-party
matplotlib/pyparsing deprecation warnings. The latest CPU-only workspace audit
reported `passed=true`, `hard_reason_count=0`, no dirty/staged/untracked/missing
source-of-truth paths, no dirty/staged/untracked managed YAML config paths, no
guarded staged paths, and no blocking artifact errors. It still reports
`ready_for_handoff=false` and `ready_for_commit=false` only because the five
guarded `model/*` handles remain intentionally local-only and therefore keep
the dirty-workspace attention gate active. In that state,
`handoff_contract.handoff_readiness.source_control_ready=true` is the field that
confirms source-of-truth and YAML launch surfaces are clean despite the local
model-handle warnings.

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
micromamba run -n ECGTwin python -m pytest util/tests/test_agent_operating_layer.py util/tests/test_active_script_index.py -q
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
git diff --cached --name-only -- model
```
