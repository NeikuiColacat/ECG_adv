# Mainline Review Repair and Rerun Plan

> Execute continuously on the `58d25a9` refactor baseline. Use TDD, task-scoped review, and fresh run directories. Do not claim experimental closure from code-only changes.

## Goal

Make the EfficientNet and ECGFounder mainline mechanically honest and reviewable, then rerun the smallest experiment set that can resolve the blocking findings in `docs/refactor_cleanup/current_mainline_review_findings_20260710.md`.

## Global constraints

- Preserve the dirty `/home/linbinhao/ECG_rebuild` worktree and all `model/*` state. Do not stage unrelated paths and do not push.
- Never use GPU 2 or 3. Use at most four GPUs chosen from 0, 1, 4, 5, 6, 7 after a fresh occupancy check; never use a GPU already occupied by another user.
- Keep transient logs and review packages under `/dev/shm`; persist only reproduction-critical run records and final evidence under `/home/linbinhao/ECG_adv_data`.
- Check CPU idle, available memory, `/dev/shm`, SSD space, and GPU occupancy before every heavy launch. Aggressive use is allowed only when current shared-server headroom supports it.
- Preserve the v7 mapping/hash, K500 ref exclusion, raw-first 500 Hz corruption protocol, lead order, seeds, and final JSON schemas.
- Use the smallest root-cause change. No new dependency or speculative abstraction.
- Mechanical completion and experimental evidence are separate states. F-001/F-002/F-004/F-005 remain `EVIDENCE_PENDING` until paired artifacts satisfy their acceptance criteria.
- The primary claim is `known_family_corruption_robustness`; depth2/3 combinations of the same five operators are not unseen-family evidence.

## Task 1: Make managed protocol fields operational

Files:

- `configs/defaults/vae_lhat_defaults.yaml`
- `configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml`
- `ecg_adv_gen/config/adapters/effnet_vae_lhat.py`
- `ecg_adv_gen/runner/effnet_vae_lhat.py`
- `ecg_adv_gen/config/audit.py` and resume-contract code only where required
- `util/tests/test_config_loader.py`
- `util/tests/test_resume_contract.py`

TDD requirements:

1. Add failing parameterized tests proving that a non-default `model.init_checkpoint`, selection contract, `include_anchor=false`, `init_logit_gap=0`, `pgd_eps`, ASR bounds, and `latent_augmix.consistency.enabled=false` reach the final child command exactly.
2. Add a failing resume-drift test for each newly operational field.
3. Implement the minimal adapter/wrapper wiring. Emit boolean flags conditionally; do not encode hidden Python defaults as paper protocol.
4. Keep legacy CLI defaults for unmanaged callers, but require managed YAML to resolve every paper-critical field.
5. Run only the targeted config/resume tests, then commit.

Acceptance:

- Mutating any listed YAML field changes the child command or explicitly disables the feature.
- Command audit rejects YAML/argv disagreement.
- Existing ECGFounder adapters remain unchanged.

## Task 2: Establish a matched EfficientNet comparison contract

Files:

- `ecg_adv_gen/runner/effnet_direct_finetune.py`
- `ecg_adv_gen/runner/effnet_vae_lhat.py`
- `ecg_adv_gen/runner/synth_online_at_super5.py`
- existing K500 split/selection helpers
- run-record/preflight code
- focused tests in `util/tests/`

TDD requirements:

1. With 500 synthetic record IDs, prove Direct and method arms receive identical train/validation IDs and that validation IDs never enter latent candidates.
2. Prove both arms initialize from the same YAML-declared source checkpoint and accept a verifiable `ptbxl_source` lineage without pretending it is target-adapted K500 evidence.
3. Add a counting-optimizer test proving identical epoch and optimizer-step budgets. Fold AugMix/JSD into the matched budget or give the clean control the same number of clean-control steps; log the realized step counts.
4. Make `paper_protocol.selection` the shared source for validation fraction, seed, metric, source-floor rule, and best-checkpoint evaluation.

Acceptance:

- A0 and A5 differ only in declared method components, not init checkpoint, K500 split, source exposure, scheduler horizon, selection data, or optimizer-step budget.
- Run records contain source checkpoint identity, train/val ID hashes, selection metric, source-floor result, and realized optimizer steps.
- This closes the mechanical part of F-001 only; status remains `EVIDENCE_PENDING` until rerun.

## Task 3: Replace nominal adversarial weight with a measurable target objective

Files:

- `ecg_adv_gen/training/signal_streams.py`
- `ecg_adv_gen/training/losses.py`
- `ecg_adv_gen/runner/synth_online_at_super5.py`
- adapter/default/resume files touched in Task 1
- focused training tests

TDD requirements:

1. Add a CPU toy test with two clean-target and two adversarial samples. At `target_adv_fraction=0.5`, assert exact 50/50 target-domain loss mass independent of source sample count.
2. Assert per-stream actual counts, unweighted loss, weighted loss, and realized contribution fractions are logged.
3. Implement `L_target=(1-rho)*L_clean+rho*L_adv`; do not increase sampler weights and call that an objective fix.
4. Expose managed ablation values `rho={0,0.25,0.5}` while retaining a single implementation.

Acceptance:

- `target_adv_fraction` is an objective coefficient with measurable realized contribution, not a `WeightedRandomSampler` alias.
- F-004 remains `EVIDENCE_PENDING` until matched rho runs pass source and clean floors.

## Task 4: Remove latent-hull anchor collapse and restore diagnostics

Files:

- `adversarial/latent_hull_pgd.py`
- `ecg_adv_gen/adaptation/latent_hull_torch.py`
- `ecg_adv_gen/runner/synth_online_at_super5.py`
- managed defaults/config and focused tests

TDD requirements:

1. For 20 candidates with `include_anchor=false`, `init_logit_gap=0`, and `hull_lambda=0.6`, assert initial top-1 weight is 0.05 and the pre-PGD effective original-anchor share is 0.4.
2. Assert `atk_init` is non-null and is computed from clean, initial-hull, and final adversarial logits using existing diagnostic helpers.
3. Log initial/final weights, top-1 candidate identity, pre/post projection norm, projection scale, effective lambda, candidate-anchor weight, and effective original-anchor share.
4. Use the balanced recipe only as a candidate configuration: `include_anchor=false`, `init_logit_gap=0`, `hull_lambda=0.6`, `steps=5`, explicit `pgd_eps`. Do not register it as trusted before the real-run gates pass.

Acceptance:

- CPU tests demonstrate non-degenerate initialization and complete diagnostics.
- Real evidence must later show median effective original-anchor share below 0.5, non-null `atk_init`, meaningful loss gain, valid decodes, and source floor.

## Task 5: Add interruption-safe PN2021-C evaluation

Files:

- `ecg_adv_gen/runner/ecgfounder_pn2021c_eval.py`
- evaluator-focused tests

TDD requirements:

1. Simulate failure after the third corruption unit; resume must call only unfinished units and produce the same final JSON as an uninterrupted run.
2. Sidecar identity must include resolved protocol/config, model checkpoint SHA256, seed, ref-ID hash, center, corruption, severity, mapping, and input mode. Any mismatch must reject reuse.
3. Write each completed unit atomically; keep the final public JSON schema unchanged.
4. Verify workers 0 and 2 preserve record order, labels, deterministic per-record seeds, metadata, and outputs on a small CPU fixture.

Acceptance:

- A killed depth2+3 run no longer loses completed combinations.
- Default workers remain configurable; launch policy chooses 2 or 4 only after checking actual CPU headroom.

## Task 6: Managed ablation surface and claim scope

Files:

- managed experiment config/index/golden files
- protocol manifest builders
- `util/tests/test_active_script_index.py`
- `util/tests/test_latest_mainline_golden_contract.py`
- pipeline/table wording

Steps:

1. Add explicit `claim_scope: known_family_corruption_robustness` plus resolved train/eval operator sets to configs and run manifests.
2. Add one compact managed matrix for A0/A2/A3/A4/A5; do not duplicate four near-identical YAMLs. Skip A1 for the source-init primary design.
3. Keep ECGFounder A0/A3/A5 optional until backbone-independence is claimed.
4. Regenerate, rather than hand-edit, versioned command/golden contracts. Update active inventory atomically.

Acceptance:

- A test rejects overlapping train/eval operator sets unless claim scope is known-family.
- Active roles expose A0/A2/A3/A4/A5 and a source reference without weakening K500 leakage guards.
- F-003 is mechanically closed for the narrowed claim; unseen-family stays out of all conclusions.

## Task 7: Update review ledger and verify the branch

Files:

- `docs/refactor_cleanup/current_mainline_review_findings_20260710.md`
- a dated final report under `docs/reports/archive/20260711/`

Steps:

1. Replace stale queue entries for the already implemented third chain and four all-zero reporting views with `MECHANICALLY_IMPLEMENTED / EVIDENCE_PENDING`.
2. For every finding, record separate code, test, and evidence states plus commit/run IDs. Do not mark F-001/F-002/F-004/F-005 resolved from unit tests.
3. Run the three targeted pytest groups, registry audit, active-script/golden checks, and workspace artifact audit with caches disabled and transient output under `/dev/shm`.
4. Request task-scoped review after each commit and a whole-branch review before integration.

## Task 8: Rerun the minimal sufficient evidence matrix

Preconditions:

- Tasks 1–7 pass review and CPU verification.
- Fresh resource check shows safe GPU/CPU headroom.

Run order:

1. CPU-only dry-runs and preflight for every arm/seed.
2. One-center smoke on a free allowed GPU; require non-null diagnostics, valid anchor share, target objective contribution, and source-floor machinery.
3. Paired EfficientNet A0/A2/A3/A4/A5 on seeds `20260531`, `20260601`, `20260611`, using up to four free allowed GPUs.
4. Ref-excluded clean, official S5, and depth2+3 known-family evaluation for eligible checkpoints.
5. ECGFounder A0/A3/A5 only if the EfficientNet mechanism passes and resources remain safe.

Final acceptance:

- Report paired per-seed deltas, mean, standard deviation, and bootstrap confidence intervals against the matched A0.
- F-001 resolves only if comparison contracts and artifacts match.
- F-002 resolves only if all required arms and three paired seeds complete.
- F-004 resolves only if realized target adversarial contribution is correct and source/clean floors hold.
- F-005 resolves only if non-collapse and attack-diagnostic gates hold.
- Otherwise report the exact failed gate and retain `EVIDENCE_PENDING` without overclaiming.
