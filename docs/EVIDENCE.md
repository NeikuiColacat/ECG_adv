# Evidence, limits and historical recovery

Current YAML defines a replayable recipe; it does not prove performance.
These recipes were tuned using heldout target development feedback.
Existing centers and repeated runs are development evidence, not independent
paper-final testing. Final claims need a frozen protocol, untouched test
population, independent repeats and registered run/checkpoint identities.

## Source and direct baselines

`configs/baselines/ptbxl_source_v1.yaml` owns the selected source checkpoint
paths and hashes. EfficientNet source SHA256:
`fff9bd293677be54a606e28775c66d410d55b0ad7440a4e7fbde55357ec13e68`;
ECGFounder source SHA256:
`c23ec4361b097b74a4efed427488c4dd6a3753adfc75f7d1fc281e4df0989948`.
The source runs are under
`/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_effnet_source_v1`
and `managed_ptbxl_ecgfounder_source_v1` in the same parent directory.
PTB-XL folds 1–8 train, fold 9 selects best AP, fold 10 is assessed once after restore.
New prospective source training does not prove exact replay of those historical runs.

The historical `configs/baselines/pn2021_direct_v1.yaml` registry is recoverable
from the pre-cleanup head below. Current source-transfer evaluation uses the
retained source registry. Distinguish source transfer, clean-only K500 fine-tuning,
direct corruption fine-tuning and the matched no-VAE control.
Do not edit a locked checkpoint identity in place.

## Historical experiment pointers

- Historical multi-seed comparison:
  `/home/linbinhao/ECG_adv_data/runs/jsd_simclr_width_repeat_20260912_v1/dual_jsd_workflow/training/width_comparison.json`.
  This includes historical NumPy-cache results; retain the recorded backend
  and cohort. A child result is not proof that its historical parent workflow
  finished archival. No current GPU validation is claimed here.
- Historical Founder width comparison:
  `/home/linbinhao/ECG_adv_data/runs/founder_jsd_width_20260911`.
  This is the auxiliary two-stage JSD-12 experiment, not single-stage R18.
  Recover its original configs and protocol from Git if needed; this pointer
  alone does not certify task completion or performance.
- Historical aligned target-only summaries were tracked as
  `agent_workspace/performance_summary_20260727/aligned_targetonly_augmix_lhat_lock_20260804.json`
  and `aligned_targetonly_augmix_lhat_report_20260804.html` in the same directory.
  Their pre-cleanup bytes remain in Git and the baseline archive below.

## Recovery and artifact boundaries

The pre-cleanup head of this branch is `ce752e84a8f6f6eaf32ff37dbe2aa60e17dc35f2`.
The source archive is
`/home/linbinhao/ECG_adv_data/reports/traditional_slim_20261106T090435Z/baseline/traditional-jsd`.
The adjacent `docs_deletions.json` lists exact redundant tracked documentation
removed during the authorized simplification.
External runs, checkpoints, data, local source snapshots and other worktrees
are preserved; a file being reproducible does not make it disposable.

Read a removed protocol or config without changing this checkout:

```bash
git show ce752e84a8f6f6eaf32ff37dbe2aa60e17dc35f2:docs/baselines/ptbxl_source_v1.md
git show ce752e84a8f6f6eaf32ff37dbe2aa60e17dc35f2:configs/active_evidence_registry.yaml
```

Old queues, one-off grids, joint/width protocols and navigation registries are
historical recovery material in Git. Recover the appropriate pre-head config
and its immutable run snapshots when reproducing an old experiment.
Do not import historical code into the active runtime or assume an old queue
is still authorized work. The source baseline's older config is recorded at
`254c0c80318840430171317c0051497f62232a6e`; the older clean-room provenance
snapshot is `3a39a516420b52c219a782a7b7440f82746f4b90`.

For every reported comparison retain model, source SHA, recipe/config closure,
seed and K500 identity, optimizer-step budget, mapping hash, backend, cohort,
metric view and checkpoint/run hashes. A report is not independent validation;
new code requires its own relevant CPU and scoped GPU checks.
