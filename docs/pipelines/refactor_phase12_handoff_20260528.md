# Refactor Phase 1/2 Handoff 2026-05-28

This is the one-hour closeout point for the YAML/config refactor. It records
what is safe to treat as landed, what remains legacy by design, and which
checks prove the current state.

## Completed Scope

The refactor is closed for the current wrapper-first phase:

- YAML-managed experiment definitions live under `configs/`.
- Machine-local paths are isolated under `configs/local/*.yaml`; only
  `.example.yaml` files are intended for git.
- `scripts/run_experiment.py` resolves configs, writes manifests and command
  plans, validates shared-server path boundaries, enforces run-id-scoped child
  outputs, supports limited audited `--set` overrides, runs child commands only
  through explicit execute mode, and verifies expected artifacts.
- `scripts/audit_managed_configs.py` provides a CPU-only preflight for all
  `active_wrapped` configs.
- Reporting is centralized through `scripts/export_metrics_long.py`,
  `scripts/merge_metrics_long.py`, and `scripts/export_paper_table.py`.
- Active configs declare managed postprocess commands for metrics/table export.
- Core pure logic and path contracts now live under `ecg_adv_gen/`, with legacy
  runners keeping compatibility wrappers.

## Active Managed Experiments

The current active wrapped configs are:

- `effnet_direct_k500_v6`
- `effnet_vae_lhat_k500_v6`
- `ecgfounder_inithead_fullft_k500_v6`
- `ecgfounder_direct_k500_v6`
- `ecgfounder_vae_lhat_k500_v6`
- `pn2021_eval_v6_refexcluded`

Smoke variants exist for fast runtime checks, but they are not paper metrics.

## Package Boundaries Now In Place

The current `ecg_adv_gen` package owns:

- Super5 metadata facade: `ecg_adv_gen.labels.super5`
- PTB-XL/PN2021 data contracts, K-shot metadata, real-anchor loading, signal
  cache schema, and lightweight manifests: `ecg_adv_gen.data`
- ECGTwin/PN2021 signal shape, resample, and lead-order helpers:
  `ecg_adv_gen.preprocessing`
- View names, macro metric helpers, PN2021 per-center/target-view assembly,
  selection policy, and target split helpers: `ecg_adv_gen.evaluation`
- Latent-Hull/anchor sampling helpers: `ecg_adv_gen.adaptation`
- ECGFounder run-path contracts, heads, torch preprocessing, and inference
  helpers: `ecg_adv_gen.models`
- Shared loss, stream sampling, signal stream datasets, deterministic splits,
  and small torch utilities: `ecg_adv_gen.training`
- Legacy-compatible run naming: `ecg_adv_gen.run_naming`

Recent closeout changes specifically moved ECG macro metric behavior and
`eval_crosscenter.py` PN2021 per-center row assembly into package helpers while
preserving legacy output schemas.

## Verification Evidence

Latest CPU-only verification:

```text
refactor-focused pytest: 229 passed
compileall: passed
git diff --check: passed
production package / launcher / reporting entrypoints: no scripts.* imports
config loader import: torch_loaded False
active managed-config audit: 6/6 passed
```

Latest audit artifacts:

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/refactor_phase12_closeout_final_20260528/
```

This audit uses `configs/local/linbinhao_server.example.yaml` and does not run
GPU training or child evaluation.

## Stop Line

Do not continue moving large runtime code as part of this closeout. These areas
remain legacy execution surfaces by design:

- WFDB loading and PN2021/PTB-XL/MIMIC cache runtime in
  `scripts/triple_labels/eval_crosscenter.py`
- EfficientNet training loop in `scripts/triple_labels/train_ptbxl.py`
- GPU PGD/decode loops in `scripts/pgd_cross_center/synth_online_at_super5.py`
- dated paper runners under `scripts/paper/`
- ECGTwin author reproduction scripts
- external model repo link handles under `model/`

The current safe contract is: YAML/config/reporting/package helper layers are
the experiment-control surface; legacy scripts are still the execution layer.

## Next Phase

The next phase should be treated as a separate refactor:

1. Add narrow compatibility shims before moving any runtime-heavy code.
2. Split WFDB/cache loading only after a smoke run proves the wrapper path.
3. Keep paper result equivalence checks tied to mapping hash, view, run id, and
   artifact manifests.
4. Run GPU smoke tests only with explicit `nvidia-smi` inspection and
   `CUDA_VISIBLE_DEVICES`.

The current phase is ready for handoff and for continuing experiments through
the managed YAML wrappers.
