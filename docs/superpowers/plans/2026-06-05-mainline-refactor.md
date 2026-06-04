# Mainline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the current ECG_adv_Gen experimental mainline toward stable handoff by reducing launch drift, blocking paper-unsafe checkpoint selection, and extracting CPU-testable training helpers.

**Architecture:** Keep legacy script entrypoints in place and move deterministic command construction, safety policy, and run-record structure into package modules. Favor typed YAML adapters and pure helper functions that can be tested without GPU or training.

**Tech Stack:** Python, PyYAML config loader, pytest CPU smoke tests, legacy ECGFounder/EfficientNet scripts.

---

### Task 1: ECGFounder VAE-LHAT Typed Adapter And Selection Gate

**Files:**
- Create: `ecg_adv_gen/config/adapters/ecgfounder_vae_lhat.py`
- Modify: `ecg_adv_gen/config/adapters/registry.py`
- Modify: `configs/experiments/ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml`
- Modify: `ecg_adv_gen/evaluation/selection.py`
- Modify: `ecg_adv_gen/evaluation/__init__.py`
- Modify: `scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py`
- Test: `util/tests/test_config_loader.py`
- Test: `util/tests/test_evaluation_selection.py`
- Test: `util/tests/test_real_anchors.py`

- [x] Write failing tests proving the v7 config uses `runner.adapter=ecgfounder_vae_lhat`, includes standardized/local-random latent-neighbor options, records `checkpoint_index`, and rejects `pn2021_heldout` unless explicitly allowed.
- [x] Implement the ECGFounder adapter by mapping existing typed YAML fields to the legacy argv expected by `run_ecgfounder_vae_only_lhat_head_ft_20260523.py`.
- [x] Register the adapter and convert the v7 ECGFounder VAE-LHAT config from long `runner.argv` to typed `runner.adapter`.
- [x] Add a runtime selection gate in the legacy runner and default it to `target_real_val`.
- [x] Add a lightweight child checkpoint-selection index JSONL written beside each ECGFounder VAE-LHAT child run.

### Task 2: synth_online_at CPU Helper Cut

**Files:**
- Create or modify: `ecg_adv_gen/training/online_at.py`
- Modify: `scripts/pgd_cross_center/synth_online_at_super5.py`
- Test: `util/tests/test_online_at_helpers.py`

- [x] Write failing tests for a pure helper that resolves quick-eval selection source and validates `target_real_val` prerequisites without building datasets.
- [x] Move the branch decision out of the legacy script while preserving legacy CLI names and printed semantics.
- [x] Keep GPU/training behavior unchanged; only route existing variables through the helper.

### Task 3: Staged Pipeline Config Sketch

**Files:**
- Modify: `configs/schemas/experiment_config.schema.json`
- Modify: `scripts/run_experiment.py`
- Test: `util/tests/test_config_loader.py`

- [x] Add schema coverage for optional `stages[]` records with `name`, `requires`, `produces`, and `skip_if_exists`.
- [x] Keep current single-run command behavior intact when `stages` is absent.
- [x] Add a dry-run manifest field that records staged dependencies when present.

### Task 4: Preprocessing Contract Audit

**Files:**
- Modify: `ecg_adv_gen/data/`
- Modify: `util/tests/test_data_contracts.py`

- [x] Add CPU-only assertions that EfficientNet, ECGTwin, and ECGFounder configs declare incompatible sampling/lead contracts explicitly.
- [x] Fail config validation if a model-family config omits its preprocessing policy.

### Task 5: Metadata Closure

**Files:**
- Modify: `scripts/agent/finalize_run.py`
- Modify: `scripts/agent/register_run.py`
- Modify: `scripts/agent/build_comparison_bundle.py`
- Test: `util/tests/test_run_record_management.py`

- [x] Pull experiment purpose, status, mapping, and result summary from resolved YAML/run records where available.
- [x] Keep manual summary fields only as explicit overrides.
- [x] Verify finalizer/register/comparison commands still pass CPU smoke tests.

### Smoke Test Plan

- [x] `pytest -q util/tests/test_config_loader.py util/tests/test_evaluation_selection.py util/tests/test_real_anchors.py`
- [x] `pytest -q util/tests/test_online_at_helpers.py`
- [x] `pytest -q util/tests/test_data_contracts.py -k 'contract or ecgfounder'`
- [x] `pytest -q util/tests/test_run_record_management.py::test_finalize_run_record_uses_manifest_run_record_defaults util/tests/test_run_record_management.py::test_register_run_in_registry_can_auto_resolve_status_from_run_record`
- [x] `pytest -q util/tests/test_agent_operating_layer.py::test_build_comparison_bundle_from_registry`
- [x] `python -m py_compile ecg_adv_gen/config/adapters/ecgfounder_vae_lhat.py ecg_adv_gen/evaluation/selection.py scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py scripts/pgd_cross_center/synth_online_at_super5.py`
- [x] `python scripts/run_experiment.py --config configs/experiments/ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml --local-config configs/local/linbinhao_server.example.yaml --run-id smoke_ecgfounder_lhat_refactor --dry-run`
- [x] `micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts`
