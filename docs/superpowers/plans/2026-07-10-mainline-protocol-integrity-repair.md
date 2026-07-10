# Mainline Protocol Integrity Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:subagent-driven-development` and follow TDD task-by-task.

**Goal:** Repair managed launch integrity, implement the locked two-clean plus
one-uncorrupted-VAE topology for EfficientNet and ECGFounder, and prevent K500
lineage mismatches from remaining active evidence.

**Architecture:** Reuse the existing typed adapters, `clean_clean_third` core,
raw ECG decode fields, ref metadata, and registry audit. Add no new framework or
dependency. Delete the unused latent-weight-cap surface.

**Design:** `docs/superpowers/specs/2026-07-10-mainline-protocol-integrity-repair-design.md`

## Global Constraints

- Work only in `/home/linbinhao/ECG_rebuild` on
  `refactor/data-module-20260620`.
- Preserve all pre-existing dirty and untracked work; never stage `model/*`.
- Use `apply_patch` for manual edits; do not reset, checkout, clean, or stash.
- Do not run GPU work, install dependencies, or push.
- Put pytest caches and transient reports under `/dev/shm`.
- Preserve class order `CD,HYP,MI,NORM,STTC`, mapping
  `v7_super5_sjr_rgq_review_20260528`, hash `555ec85d5b51`.
- A repaired mainline checkpoint uses a new run ID; old signal-space checkpoints
  must not resume.
- Follow red -> green -> focused refactor for each task.

## Baseline

Already verified before implementation:

```text
232 passed, 13 warnings
```

Command:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest -q \
  util/tests/test_config_loader.py \
  util/tests/test_lhat_augmix_ablation.py \
  util/tests/test_pn2021c_metadata.py \
  util/tests/test_evidence_registry_audit.py
```

### Task 1: Close the managed launch and checkout boundaries

**Production files:**

- `ecg_adv_gen/runner/effnet_vae_lhat_augmix.py`
- `ecg_adv_gen/runner/effnet_vae_lhat.py`
- `ecg_adv_gen/config/loader.py`
- `ecg_adv_gen/config/paths.py`
- `configs/local/linbinhao_server.example.yaml`

**Tests:** `util/tests/test_config_loader.py`

- [ ] Add failing tests proving the wrapper accepts and forwards
  `vae_adv_consistency_weight` and `latent_augmix_adv_base_mix`.
- [ ] Add a failing test proving an explicit sibling-checkout `project_root` is
  rejected and commands/manifest stay in the config's checkout.
- [ ] Implement the two missing parser/forwarding links.
- [ ] Derive project root from the experiment config checkout; retain explicit
  mismatch validation and remove `project_root` from the tracked local example.
- [ ] Run the focused test file and a managed CPU dry-run.

### Task 2: Repair EfficientNet locked topology and delete the dead knob

**Production files:**

- `ecg_adv_gen/runner/effnet_vae_lhat.py`
- `ecg_adv_gen/runner/effnet_vae_lhat_augmix.py`
- `ecg_adv_gen/adaptation/lhat.py`
- `ecg_adv_gen/config/adapters/effnet_vae_lhat.py`
- `configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml`

**Tests:**

- `util/tests/test_lhat_augmix_ablation.py`
- `util/tests/test_config_loader.py`
- the narrow EfficientNet runner tests discovered by `rg` before editing

- [ ] Add deterministic failing coverage for two corrupted clean chains and one
  uncorrupted adversarial chain.
- [ ] Add failing runner coverage proving AugMix receives raw decoded tensors and
  the locked path enqueues only the mixed views.
- [ ] Change the main YAML to `clean_clean_third` with matching chain roles.
- [ ] Request raw decode storage while keeping normalized attack diagnostics.
- [ ] Skip the independent VAE buffer push only in the locked mixed-view path;
  preserve existing ablation modes.
- [ ] Remove `latent_weight_cap` from code, configuration, logs/names/resume
  surfaces, and delete its unused helper/export.
- [ ] Record/reject the old signal-space resume contract.
- [ ] Run focused tests.

### Task 3: Repair ECGFounder signal ordering

**Production files:**

- `ecg_adv_gen/runner/ecgfounder_vae_lhat_fullft.py`
- `configs/experiments/ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml`

**Tests:**

- `util/tests/test_config_loader.py`
- `util/tests/test_ecgfounder_vae_lhat_fullft.py` or the existing equivalent
- `util/tests/test_ecgfounder_pn2021c_evaluator.py` as an unchanged evaluator
  regression

- [ ] Add failing coverage that corruption receives `(12, 5000)` raw input and
  global z-score happens afterward.
- [ ] Change the main YAML to `clean_clean_third` with matching chain roles.
- [ ] Reuse the existing 1000-to-5000 converter without early z-score.
- [ ] Make source/target/adversarial training batches shape-compatible at 5000;
  do not change the evaluator's already-correct bottleneck path.
- [ ] Reject old normalized-first resume metadata and run focused tests.

### Task 4: Downgrade invalid evidence and close K500 lineage

**Production/evidence files:**

- `ecg_adv_gen/evaluation/pn2021c_metadata.py`
- `ecg_adv_gen/evidence/registry.py`
- the existing ECGFounder evaluator/adapter lineage boundary found by `rg`
- `configs/active_evidence_registry.yaml`
- `docs/reports/archive/20260709/ecgfounder_decoupled_fourcenter_depth23_summary.md`

**Tests:**

- `util/tests/test_pn2021c_metadata.py`
- `util/tests/test_evidence_registry_audit.py`
- `util/tests/test_ecgfounder_pn2021c_evaluator.py`
- `util/tests/test_pn2021c_eval_adapter.py`
- `util/tests/test_run_record_integrity.py`

- [ ] Add failing tests for current-training/init-checkpoint K500 mismatch and
  training/evaluation-exclusion mismatch.
- [ ] Reuse existing ref seed/hash fields to reject a different target K500
  initialization; allow an explicitly source-only initialization.
- [ ] Audit active claims against recorded evaluation artifact identity, not
  only the current YAML/ref metadata.
- [ ] Mark the affected EfficientNet claim/method/bundle and ECGFounder support
  entry `deprecated`; set paper use to `prohibited_protocol_invalid`.
- [ ] Preserve absolute metrics as descriptive data, rename/remove matched
  deltas, and document the observed per-center leakage counts.
- [ ] Run registry audit and focused tests.

### Task 5: Integrated CPU verification and handoff

- [ ] Run both managed mainline dry-runs and prove every command path/cwd belongs
  to `/home/linbinhao/ECG_rebuild`.
- [ ] Run all focused tests from Tasks 1-4, then the broader relevant suite.
- [ ] Run registry/config/workspace audits with output under `/dev/shm`.
- [ ] Run `git diff --check`, artifact guard, and inspect every staged path.
- [ ] Obtain an independent whole-diff code review and fix all Critical/Important
  findings.
- [ ] Commit only intended code/config/test/evidence/docs files; leave external
  models and unrelated `docs/superpowers/` files unstaged.
- [ ] Report commits, exact verification evidence, remaining dirty files, and
  ahead/behind. Do not push.
