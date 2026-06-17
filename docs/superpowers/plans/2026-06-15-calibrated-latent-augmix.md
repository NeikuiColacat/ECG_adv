# Calibrated Latent AugMix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and test an EfficientNet1DV2 VAE-LHAT + latent AugMix variant whose non-latent AugMix corruption chains use the calibrated PN2021-C 10-20 pp profile.

**Architecture:** Keep the existing VAE latent-hull online AT runner and latent AugMix branch. Add a `latent_augmix.severity_profile` config/CLI argument that defaults to `standard`, and route chain ops through the PN2021-C profile-aware op builder only when requested. Do not enable or reuse the raw ECG corruption supervised/JSD phase.

**Tech Stack:** Python, PyTorch training runner `scripts/pgd_cross_center/synth_online_at_super5.py`, managed YAML configs, `pytest`, PN2021-C evaluation scripts.

---

### Task 1: Add Test Coverage For Profile-Aware Latent AugMix

**Files:**
- Modify: `util/tests/test_lhat_augmix.py`
- Modify: `ecg_adv_gen/adaptation/lhat.py`

- [ ] **Step 1: Write failing tests**

Add tests that call `build_latent_augmix_branch_signals` with `severity_profile="standard"` and `severity_profile="calibrated_10to20pp"`. The tests should use a custom `op_apply_fn(sample, op_name, severity, severity_profile)` to prove the profile is passed into every non-latent chain op.

- [ ] **Step 2: Run RED test**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_lhat_augmix.py -q
```

Expected: FAIL because `build_latent_augmix_branch_signals` does not yet accept or forward `severity_profile`.

- [ ] **Step 3: Implement minimal profile forwarding**

Update `build_latent_augmix_branch_signals` to accept `severity_profile: str = "standard"` and call `op_apply_fn(sig, op_name, int(severity), severity_profile)`.

- [ ] **Step 4: Run GREEN test**

Run the same pytest command. Expected: PASS.

### Task 2: Wire CLI And Runner Adapter

**Files:**
- Modify: `scripts/pgd_cross_center/synth_online_at_super5.py`
- Modify: `ecg_adv_gen/config/adapters/effnet_vae_lhat.py`
- Modify: `ecg_adv_gen/runner/effnet_vae_lhat.py`

- [ ] **Step 1: Add failing CLI/config tests if existing adapter tests cover this path**

Search for existing adapter tests. If present, add assertions that `latent_augmix.severity_profile: calibrated_10to20pp` expands to `--latent_augmix_severity_profile calibrated_10to20pp`.

- [ ] **Step 2: Add CLI option**

Add `--latent_augmix_severity_profile` with choices matching PN2021-C stress profiles and default `standard`.

- [ ] **Step 3: Route op application**

Update the latent AugMix wrapper so `standard` uses existing AugMix op behavior and `calibrated_10to20pp` uses the profile-aware PN2021-C op construction.

- [ ] **Step 4: Forward adapter/runner args**

Pass `latent_augmix.severity_profile` from YAML through managed adapter and runner CLI.

### Task 3: Add Managed Experiment Configs

**Files:**
- Create: `configs/experiments/effnet_vae_lhat_calibrated_latent_augmix_k500_v7_sjr_rgq.yaml`
- Create: `configs/experiments/effnet_vae_lhat_calibrated_latent_augmix_k500_v7_sjr_rgq_cpsc_2018.yaml`

- [ ] **Step 1: Create four-center family config**

Extend `effnet_vae_lhat_k500_v7_sjr_rgq.yaml`, keep raw corruption consistency disabled, and set:

```yaml
adaptation:
  run_tag_extra: k500_callatentaugmix
  latent_augmix:
    latent_weight_cap: 0.25
    width: 3
    depth: -1
    alpha: 1.0
    severity: 5
    severity_profile: calibrated_10to20pp
    ops:
      - powerline_noise
      - emg_noise
      - baseline_wander
      - baseline_shift
      - random_leads_masking
```

- [ ] **Step 2: Create CPSC-only config**

Extend the family config and restrict `runner.matrix.center` to `[cpsc_2018]`.

### Task 4: Run Local Verification

**Files:**
- No new files unless test fixtures are required.

- [ ] **Step 1: Run focused tests**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_lhat_augmix.py util/tests/test_effnet_vae_lhat_runner.py -q
```

- [ ] **Step 2: Run config expansion smoke**

Run a dry config expansion for the new CPSC config through the existing managed launcher or config loader, and verify the generated command includes:

```text
--latent_augmix_severity 5
--latent_augmix_severity_profile calibrated_10to20pp
--latent_augmix_ops powerline_noise emg_noise baseline_wander baseline_shift random_leads_masking
```

### Task 5: Run CPSC Training And PN2021-C Evaluation

**Files:**
- Output under `/home/linbinhao/ECG_adv_data/runs/`

- [ ] **Step 1: Shared-server preflight**

Run `nvidia-smi`, pick up to four free GPUs, and set explicit `CUDA_VISIBLE_DEVICES`.

- [ ] **Step 2: Launch CPSC training**

Run the new CPSC-only config with a date/config-named run id that does not overwrite prior outputs.

- [ ] **Step 3: Evaluate clean PN2021**

Run ref-excluded v7 clean eval for the produced checkpoint.

- [ ] **Step 4: Evaluate calibrated PN2021-C**

Run `calibrated_10to20pp`, severity 5, five operators, same K500 ref-excluded protocol.

### Task 6: Aggregate And Report

**Files:**
- Create report under `docs/reports/archive/20260615/`

- [ ] **Step 1: Compare against matched references**

Compare the new run against matched `vae_noaug`, standard `vae_augmix`, and `calibrated raw-supervised stressor`.

- [ ] **Step 2: Write concise report**

Report clean delta, corrupted delta, drop reduction, and whether the method approaches the raw-supervised +6 pp recovery result.
