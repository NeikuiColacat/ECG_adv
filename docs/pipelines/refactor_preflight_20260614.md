# Refactor Preflight Baseline - 2026-06-14

Status: **PASSED - preflight gates passed after config-inventory fix**

This preflight freezes the current state before the planned repository-wide
modular refactor. It did not start GPU work, did not execute training, did not
move legacy entrypoints, and did not delete old code.

## Environment

- Repository: `/home/linbinhao/ECG_adv_Gen`
- Branch: `main`
- HEAD: `849b30d21f41d2fe85cee74d2115eea9244ef3be`
- Python: `/home/linbinhao/micromamba/envs/ECGTwin/bin/python`
- Remote: `origin git@github.com:NeikuiColacat/ECG_adv.git`

Required startup guardrail was followed: read the first 100 lines of
`AGENTS.md` before any write or environment-changing action.

## Mainline Frozen For Refactor

- Main claim boundary: EfficientNet1DV2 Direct K500 vs ECGTwin VAE-LHAT.
- Mapping version: `v7_super5_sjr_rgq_review_20260528`
- Mapping hash: `555ec85d5b51`
- Class order: `CD,HYP,MI,NORM,STTC`
- Target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`
- K-shot protocol: fixed `K=500`, seed/subset seed `20260531`
- Final eval rule: K500 reference records must be excluded.
- Selection policy: `k500_internal_val_plus_source_floor`; held-out target labels
  and full target distribution tuning are forbidden.

## Initial Dirty State

The worktree was already dirty before this preflight. Existing changes were not
reverted or staged.

Key dirty categories observed:

- Source-of-truth or package paths dirty:
  - `AGENTS.md`
  - `configs/active_scripts.yaml`
  - `ecg_adv_gen/config/adapters/ecgfounder_vae_lhat.py`
  - `ecg_adv_gen/config/adapters/effnet_vae_lhat.py`
  - `ecg_adv_gen/config/adapters/pn2021c_eval.py`
  - `ecg_adv_gen/runner/effnet_vae_lhat.py`
  - `ecg_adv_gen/evaluation/pn2021_corruptions.py`
- Guarded external model paths dirty and must not be staged:
  - `model/DeepECG`
  - `model/ECGTwin`
  - `model/advdiff`
  - `model/ecg_ptbxl_benchmarking`
  - `model/ecgfounder`
- Numerous untracked experiment YAMLs and report files are present. Only the
  YAMLs required to make `configs/active_scripts.yaml` coherent are candidates
  for this preflight commit; unrelated reports and generated files remain
  unstaged.

## Read-Only Audits

### Agent workspace audit

Command:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python \
  scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
```

Result:

- Exit code: `0`
- Overall `passed`: `true`
- Warning count: `8`

Warnings:

- Five guarded external-model paths are dirty and must not be staged.
- Two external model targets exist but are not git checkouts:
  - `/home/linbinhao/ECG_adv_data/models/DeepECG`
  - `/home/linbinhao/ECG_adv_data/models/advdiff`
- Declared source-of-truth paths are dirty and need review before handoff.

### Managed config audit

Command:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python \
  scripts/audit_managed_configs.py \
  --local-config configs/local/linbinhao_server.example.yaml
```

Result:

- Exit code: `0`
- Hard failure: none observed
- The audit reports active wrapped configs with mapping version
  `v7_super5_sjr_rgq_review_20260528` and mapping hash `555ec85d5b51`.

## Mainline Dry-Run Baselines

### Direct K500

Command:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python \
  scripts/run_experiment.py \
  --config configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id refactor_baseline_direct_dryrun \
  --dry-run
```

Result:

- Exit code: `0`
- Status: `dry_run`
- Legacy child scripts invoked: `false`
- Generated one legacy command using
  `scripts/paper/run_direct_finetune_k500_20260516.py`.
- Command covers all four target centers in one `--centers` argument.
- Managed postprocess commands generated:
  - `scripts/export_metrics_long.py`
  - `scripts/export_paper_table.py` for `pn2021_all_zero_kept_refexcluded`
  - `scripts/export_paper_table.py` for `pn2021_drop_all_zero_refexcluded`
- Protocol audit passed.

### VAE-LHAT K500

Command:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python \
  scripts/run_experiment.py \
  --config configs/experiments/effnet_vae_lhat_k500_v7_sjr_rgq.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --run-id refactor_baseline_vae_lhat_dryrun \
  --dry-run
```

Result:

- Exit code: `0`
- Status: `dry_run`
- Legacy child scripts invoked: `false`
- Generated four legacy commands using
  `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py`, one per target
  center.
- Managed postprocess commands generated:
  - `scripts/export_metrics_long.py`
  - `scripts/export_paper_table.py` for `pn2021_all_zero_kept_refexcluded`
  - `scripts/export_paper_table.py` for `pn2021_drop_all_zero_refexcluded`
- Required epoch metrics include `asr_overall`, `attack_vs_anchor`,
  `loss_gain`, `decoded_invalid_rate`, `latent_augmix_stats`, and `quick_eval`.
- Note: with this synthetic dry-run id, Direct init checkpoints are expected
  under the same new run id and currently do not exist. This is expected for a
  dry-run baseline and confirms the derived dependency path.

## Inventory Fix

The first CPU gate failed because `configs/active_scripts.yaml` did not cover
18 experiment YAML files already present under `configs/experiments/`.

Fix applied:

- Registered the 18 previously unindexed YAMLs under
  `inactive_experiment_configs`.
- Marked smoke configs ending in `_smoke.yaml` as `smoke_only`.
- Marked provisional raw-supervised, mask/shift, raw-JSD, and PN2021-C draft
  configs as `historical_unmanaged`.
- Did not promote any of these draft configs to active evidence.

The worktree also contains five already-declared active launch-surface YAMLs.
They are intentionally left out of this preflight commit because the staged-only
clean-checkout verification showed they depend on separate PN2021-C/adapter
code changes already present in the dirty worktree:

```text
configs/experiments/benchmark_fcn_wang_vae_noaug_k500_v7_sjr_rgq.yaml
configs/experiments/benchmark_inception1d_vae_noaug_k500_v7_sjr_rgq.yaml
configs/experiments/benchmark_resnet1d_vae_noaug_k500_v7_sjr_rgq.yaml
configs/experiments/ecgfounder_vae_lhat_augmix_k500_v7_sjr_rgq.yaml
configs/experiments/pn2021c_effnet_v7_strong_10to20pp.yaml
```

## CPU Test Gate

Command:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_active_script_index.py \
  util/tests/test_config_loader.py \
  util/tests/test_labels_super5.py \
  util/tests/test_data_contracts.py \
  util/tests/test_preprocessing_signals.py \
  util/tests/test_evaluation_protocols.py \
  util/tests/test_ref_exclusion.py \
  util/tests/test_pn2021_metric_views.py \
  util/tests/test_metrics_export.py \
  util/tests/test_pn2021c_metadata.py \
  util/tests/test_adaptation_lhat.py \
  util/tests/test_buffer_labels.py \
  methods/augmix/tests
```

Initial result:

- Exit code: `1`
- Passed: `326`
- Failed: `1`
- Failure:
  `util/tests/test_active_script_index.py::test_experiment_config_inventory_covers_every_tracked_yaml`

Re-run after the inventory fix:

- `util/tests/test_active_script_index.py`: `22 passed in 6.81s`
- Full CPU gate: `327 passed in 27.23s`

No GPU work, training, cache build, or long inference was started.

## Staged-Only Verification

To avoid committing unrelated workspace changes, the candidate patch was tested
in a detached temporary worktree at
`/home/linbinhao/tmp_ecg_refactor_preflight_verify`.

Verification details:

- Applied only the staged patch from this preflight commit.
- Re-pointed `model/*` symlinks inside the temporary worktree to the current
  host-local external model targets under `/home/linbinhao/ECG_adv_data`.
- Did not stage or commit any `model/*` changes.
- Result:
  `util/tests/test_active_script_index.py`: `22 passed in 6.24s`

The first staged-only attempt failed when the full active launch-surface YAML
set was staged, because `pn2021c_effnet_v7_strong_10to20pp.yaml` depends on
separate PN2021-C/adapter code changes. Those active YAMLs and their code
changes remain intentionally unstaged.

Final agent audit before commit:

- Exit code: `0`
- Overall `passed`: `true`
- Warning count: `9`
- Expected staged-content warning: `configs/active_scripts.yaml`
- Guarded `model/*` paths remain dirty but unstaged.

## Commit / Push Decision

Commit and push are allowed only after artifact-git-guard and staged-only
verification. The candidate commit scope is intentionally narrow:

- `docs/pipelines/refactor_preflight_20260614.md`
- the `inactive_experiment_configs` inventory hunk in
  `configs/active_scripts.yaml`
- the 18 provisional/smoke experiment YAMLs registered as inactive inventory

Do not stage guarded external model paths, local environment files, generated
reports, or unrelated source changes already present in the dirty worktree.

## Recommended Fix Order

1. Keep the mainline boundary frozen: Direct K500, VAE-LHAT K500, Super5 v7
   SJR/RGQ mapping, K500 ref exclusion, and non-oracle selection policy.
2. Extract modules incrementally from script-level helpers into `ecg_adv_gen/`.
3. For each extraction, run dry-run or smoke tests before moving the next
   module.
4. Keep legacy entrypoints as wrappers until their YAML launch surfaces and
   tests are updated in the same change.
5. Commit this preflight baseline with:

   ```text
   docs: add refactor preflight baseline
   ```

6. Push the current branch to `origin`.
