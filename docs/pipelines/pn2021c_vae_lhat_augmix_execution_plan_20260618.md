# PN2021-C VAE-LHAT AugMix Execution Plan, 2026-06-18

> For agentic workers: execute this plan task-by-task. Use subagent-driven
> development for independent code slices, or inline execution with checkpoints
> when keeping context in one session is safer.

**Goal:** Implement and run the locked exploratory protocol in
`docs/pipelines/pn2021c_vae_lhat_augmix_locked_protocol_20260618.md`.

**Architecture:** First make the protocol enforceable in CPU-only config/tests,
then add the new waveform-order PN2021-C evaluation path, then build the full-FT
ECGFounder and three-chain VAE-LHAT AugMix training surfaces. GPU experiments
start only after CPU smoke tests and command dry-runs prove the run commands use
the locked protocol.

**Tech Stack:** Python, PyTorch, micromamba `ECGTwin`, managed YAML launchers,
`scripts/run_experiment.py`, `util/tests`, ECGTwin VAE, EfficientNet1DV2,
ECGFounder.

---

## Non-Negotiable Protocol

- K500 uses all 500 target samples for training.
- Evaluate only the last checkpoint for method comparison.
- Exclude K500 record ids from target-center PN2021 evaluation.
- PN2021 four centers are an exploratory development scoreboard.
- Main PN2021-C order is waveform-first and z-score-last:
  - EfficientNet1DV2: `raw -> 100Hz -> corruption/augment -> z-score -> model`.
  - ECGFounder: `raw -> 100Hz -> 500Hz -> corruption/augment -> z-score -> model`.
- Official severity `5` is accepted if either EfficientNet1DV2 or ECGFounder
  reaches the roughly 10 pp band on the four-center, five-operator mean.
- Main AugMix has three chains:
  - chain 1: official corruption chain.
  - chain 2: official corruption chain.
  - chain 3: VAE-LHAT adversarial waveform.
- Chain 3 does not receive extra corruption.
- Candidate SOTA recipes must be checked on both EfficientNet1DV2 and
  ECGFounder before any model-agnostic claim is made.
- No stabilizer35, raw-supervised branch, head-only ECGFounder path,
  residual-adapter ECGFounder path, or oracle selection is allowed in the main
  method.

## Phase 0: Preflight And Guardrails

**Files:**

- Read: `AGENTS.md`
- Read: `docs/pipelines/pn2021c_vae_lhat_augmix_locked_protocol_20260618.md`
- Read: `configs/active_scripts.yaml`
- Read: `configs/active_evidence_registry.yaml`

**Steps:**

- [ ] Run repository status:

```bash
git status --short
```

Expected: dirty paths are understood before editing. Existing unrelated model
links or report artifacts are ignored unless they block this task.

- [ ] Run CPU-only agent audit:

```bash
micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py
```

Expected: audit completes. If it reports existing unrelated dirty paths, record
them in the execution notes and continue only if no artifact-risk gate blocks
the planned files.

- [ ] Before any GPU run, inspect GPU occupancy:

```bash
nvidia-smi
```

Expected: chosen GPU ids are explicit through `CUDA_VISIBLE_DEVICES=...`.

## Phase 1: Protocol Constants And CPU Tests

**Files:**

- Modify or create: `ecg_adv_gen/evaluation/pn2021c_protocol.py`
- Modify: `scripts/triple_labels/eval_pn2021_corruptions.py`
- Modify: `scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py`
- Test: `util/tests/test_pn2021c_metadata.py`
- Test: `util/tests/test_ecgfounder_pn2021c_evaluator.py`
- Test: `util/tests/test_pn2021c_eval_adapter.py`

**Steps:**

- [ ] Add a protocol constant for the locked main PN2021-C order:

```text
waveform_bottleneck_then_corrupt_then_zscore
```

Expected behavior: every main PN2021-C result records this id in metadata.

- [ ] Add CPU tests that reject pre-z-scored corruption for the main protocol.

Expected test assertions:

- EfficientNet path corrupts waveform before final z-score.
- ECGFounder path resamples to 5000, corrupts waveform, then z-scores.
- Metadata records input order and z-score timing.

- [ ] Run focused tests:

```bash
micromamba run -n ECGTwin pytest -q \
  util/tests/test_pn2021c_metadata.py \
  util/tests/test_ecgfounder_pn2021c_evaluator.py \
  util/tests/test_pn2021c_eval_adapter.py
```

Expected: all selected tests pass.

## Phase 2: Official Severity-5 Baseline Evaluation Surface

**Files:**

- Modify: `ecg_adv_gen/config/adapters/pn2021c_eval.py`
- Modify or create: `configs/experiments/pn2021c_official_s5_locked_protocol.yaml`
- Test: `util/tests/test_pn2021c_eval_adapter.py`
- Test: `util/tests/test_config_loader.py`

**Steps:**

- [ ] Add a managed config that evaluates official severity `5` under the
      locked waveform-order protocol.

Required config semantics:

- severity profile: `standard`.
- severity: `5`.
- centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- operators: five PN2021-C operators listed in the locked protocol.
- output name includes `official_s5_locked`.

- [ ] Adapter dry-run must show the locked protocol arguments and output name:

```bash
micromamba run -n ECGTwin python scripts/run_experiment.py \
  --config configs/experiments/pn2021c_official_s5_locked_protocol.yaml \
  --run-id dryrun_protocol_check \
  --dry-run
```

Expected: generated commands use severity `5`, profile `standard`, four target
centers, five operators, K500 ref exclusion, and locked protocol metadata.

- [ ] Run focused config tests:

```bash
micromamba run -n ECGTwin pytest -q \
  util/tests/test_pn2021c_eval_adapter.py \
  util/tests/test_config_loader.py
```

Expected: tests pass.

## Phase 3: ECGFounder Full-FT Mainline Surface

**Files:**

- Modify or create: `ecg_adv_gen/config/adapters/ecgfounder_fullft.py`
- Modify: `ecg_adv_gen/config/adapters/registry.py`
- Modify: `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py`
- Create: `configs/experiments/ecgfounder_ptbxl_super5_fullft_locked.yaml`
- Create: `configs/experiments/ecgfounder_k500_fullft_locked.yaml`
- Test: `util/tests/test_config_loader.py`
- Test: `util/tests/test_run_naming.py`

**Steps:**

- [ ] Add a managed ECGFounder full-FT adapter or extend the existing full-FT
      pilot launcher so it exposes the staged mainline:

```text
official ECGFounder checkpoint
-> PTB-XL Super5 full-FT
-> K500 full-FT
-> VAE-LHAT / VAE-LHAT+AugMix full-FT
```

Expected: the managed surface produces `best_model.pt` or last full-model
checkpoint paths, never `best_head.pt`.

- [ ] Force the mainline path to avoid frozen-feature, head-only,
      residual-adapter, and cached feature routes.

Expected: dry-run commands do not include `best_head.pt`, `residual_adapter`,
`freeze_base_head`, or frozen-feature cache roots.

- [ ] Run dry-runs for PTB-XL full-FT and K500 full-FT configs.

Expected: commands resolve without launching GPU work and record the locked
input order in resolved config or run metadata.

- [ ] Run focused tests:

```bash
micromamba run -n ECGTwin pytest -q \
  util/tests/test_config_loader.py \
  util/tests/test_run_naming.py
```

Expected: tests pass.

## Phase 4: Three-Chain VAE-LHAT AugMix

**Files:**

- Modify: `ecg_adv_gen/adaptation/lhat.py`
- Modify: `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py`
- Modify: `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py`
- Modify: `ecg_adv_gen/runner/effnet_vae_lhat.py`
- Test: `util/tests/test_adaptation_lhat.py`
- Test: `util/tests/test_latent_augmix_consistency.py`
- Test: `util/tests/test_effnet_vae_lhat_runner.py`
- Test: `util/tests/test_ecgfounder_fullft_raw_augmix.py`

**Steps:**

- [ ] Implement or expose a three-chain AugMix builder with this fixed
      topology:

```text
chain 1 = official corruption chain
chain 2 = official corruption chain
chain 3 = VAE-LHAT adversarial waveform
```

Expected: chain 3 bypasses additional corruption.

- [ ] Add config fields for bounded self-directed exploration:

```text
augmix_view_bce_weight
augmix_consistency_weight
augmix_clean_bce_weight
vae_pgd_steps
vae_latent_eps
vae_hull_lambda
vae_adv_branch_frequency
```

Expected: all fields are recorded in run metadata and resolved config.

- [ ] Add tests proving the topology cannot silently collapse to a single
      chain in mainline configs.

Expected: main configs reject `width=1`, `depth=1`, or fixed single-chain
mixing unless the run is explicitly marked diagnostic.

- [ ] Run focused tests:

```bash
micromamba run -n ECGTwin pytest -q \
  util/tests/test_adaptation_lhat.py \
  util/tests/test_latent_augmix_consistency.py \
  util/tests/test_effnet_vae_lhat_runner.py \
  util/tests/test_ecgfounder_fullft_raw_augmix.py
```

Expected: tests pass.

## Phase 5: Smoke Runs

**Files:**

- Create: `configs/experiments/effnet_vae_lhat_augmix_threechain_locked_smoke.yaml`
- Create: `configs/experiments/ecgfounder_vae_lhat_augmix_threechain_locked_smoke.yaml`

**Steps:**

- [ ] Check shared-server GPU usage:

```bash
nvidia-smi
```

Expected: selected GPUs are free enough for smoke runs.

- [ ] Run one-center, short-epoch EfficientNet smoke with fixed last checkpoint.

Expected artifacts:

- `run_card.json`
- `summary.md`
- resolved config
- last checkpoint
- training log with three-chain AugMix stats

- [ ] Run one-center, short-epoch ECGFounder smoke with fixed last checkpoint.

Expected artifacts match the EfficientNet smoke, with full-model checkpoint
metadata.

- [ ] Finalize smoke runs:

```bash
micromamba run -n ECGTwin python scripts/agent/finalize_run.py --run-dir <run_dir>
```

Expected: run file index and summary are generated.

## Phase 6: Official Severity-5 Exploratory Matrix

**Files:**

- Create: `configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml`
- Create: `configs/experiments/ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml`
- Create or update: `docs/reports/archive/20260618/` summaries after results.

**Steps:**

- [ ] Train direct K500, VAE-LHAT noAug, and VAE-LHAT+three-chain AugMix for
      EfficientNet1DV2 with all K500 samples and last checkpoint evaluation.

- [ ] Train direct K500, VAE-LHAT noAug, and VAE-LHAT+three-chain AugMix for
      ECGFounder full-FT with all K500 samples and last checkpoint evaluation.

- [ ] Evaluate PN2021 clean ref-excluded four-center metrics for all last
      checkpoints.

- [ ] Evaluate PN2021-C official severity `5` with the locked waveform-order
      protocol for all last checkpoints.

- [ ] Decide whether official severity is sufficient:

```text
If either backbone reaches the 8-12 pp mean drop band in AUROC or AUPRC:
  keep official severity 5 as the primary profile.
If neither backbone reaches the band:
  design one global custom profile, then rerun PN2021-C.
```

## Phase 7: Bounded Self-Directed Tuning

**Files:**

- Create: config variants under `configs/experiments/` with names containing
  `threechain_locked`.
- Update: run summaries under the corresponding run directories.

**Allowed tuning:**

- AugMix view BCE weight.
- AugMix consistency weight.
- Clean BCE versus AugMix-view loss weight.
- VAE PGD steps.
- VAE latent epsilon.
- Hull lambda.
- Adversarial branch frequency.
- AugMix alpha and mixture distribution parameters.

**Disallowed tuning:**

- Adding raw-supervised branches outside AugMix.
- Adding stabilizer35 or repair frontends.
- Changing severity per backbone or per center.
- Selecting non-last checkpoints.
- Reintroducing ECGFounder frozen/head-only/residual-adapter mainline rows.

**Acceptance gate:**

- A candidate is worth expanding only if it improves corrupted absolute AUPRC
  or AUROC without severe clean PN2021/PTB-XL collapse.
- Mark clean PN2021 AUPRC drops greater than about 2 pp and PTB-XL source AUPRC
  drops greater than about 2 pp as warnings.

## Phase 8: Cross-Backbone Universality And Trainable-Scope Ablations

**Trigger:** Run this phase whenever an EfficientNet1DV2 tuned recipe is
promoted as the current candidate SOTA.

**Required rows:**

- EfficientNet1DV2 direct K500 baseline.
- EfficientNet1DV2 VAE-LHAT only.
- EfficientNet1DV2 VAE-LHAT plus locked three-chain AugMix.
- ECGFounder direct K500 fullFT baseline.
- ECGFounder VAE-LHAT only.
- ECGFounder VAE-LHAT plus locked three-chain AugMix with the matched recipe.
- ECGFounder trainable-scope ablations when fullFT hurts robustness:
  dense-only, last-N-stage partial unfreeze, and LoRA if available.

**Acceptance gate:**

- Do not call the method model-universal from EfficientNet evidence alone.
- If ECGFounder fails while EfficientNet succeeds, report the result as a
  backbone-sensitive method and analyze the failure by trainable scope,
  corruption operator, center, clean performance, and adversarial diagnostics.
- Partial-unfreeze or LoRA gains are valid ablation evidence about ECGFounder
  adaptation sensitivity, but they do not replace the fullFT mainline unless
  the protocol is explicitly revised before final reruns.

## Phase 9: Report Bundle

**Files:**

- Create: `docs/reports/archive/20260618/pn2021c_threechain_locked_summary.md`
- Update only if needed: HTML report under `docs/tmp_html/` or report archive.

**Required tables:**

- PN2021 clean ref-excluded absolute AUROC/AUPRC.
- PN2021-C official severity `5` absolute corrupted AUROC/AUPRC.
- Clean-to-corrupt drops in pp.
- Per-center table.
- Per-operator table.
- Per-class table.
- PTB-XL/source sanity metrics.
- Method rows: direct K500, VAE-LHAT noAug, VAE-LHAT+three-chain AugMix.

**Required labels:**

- `exploratory`
- `last_checkpoint`
- `all_k500_train`
- `k500_ref_excluded`
- `waveform_bottleneck_then_corrupt_then_zscore`
- `three_chain_augmix_chain3_vae_lhat`

## Stop Conditions

Stop and report instead of launching more runs if any of these occur:

- The dry-run command includes `best_head.pt`, `residual_adapter`,
  `freeze_base_head`, stabilizer35, raw-supervised branches, or pre-z-scored
  corruption for a mainline run.
- The PN2021-C metadata does not record the locked input order.
- The run cannot trace K500 ref ids.
- The last checkpoint policy is not traceable.
- GPU or disk pressure makes a run unsafe for the shared server.
- A result directory lacks run-card or summary artifacts after finalization.
