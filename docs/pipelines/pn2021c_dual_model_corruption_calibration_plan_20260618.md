# PN2021-C Dual-Model Corruption Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a new PN2021-C calibrated corruption profile whose five single-operator corruptions make both EfficientNet1DV2 and ECGFounder direct-K500/fullFT baselines drop 10-15 pp AUROC, while keeping the inter-model AUROC-drop gap small.

**Architecture:** Add a config-driven corruption-parameter profile surface shared by the EfficientNet and ECGFounder PN2021-C evaluators, then run staged calibration from deterministic smoke subsets to two-center tuning and frozen four-center confirmation. Calibration uses direct baselines only; VAE-LH + AugMix is evaluated only after the profile is frozen.

**Tech Stack:** Python, PyTorch, PN2021 Super5 v7, `scripts/triple_labels/eval_pn2021_corruptions.py`, `scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py`, YAML configs, run artifacts under `/home/linbinhao/ECG_adv_data/runs`.

---

## Why This New Plan Exists

The current locked official severity-5 protocol is accepted for a weaker goal: official s5 gives ECGFounder mean AUPRC drop 8.98 pp and EfficientNet1DV2 mean AUPRC drop 7.43 pp, with AUROC drops only 5.72 pp and 4.83 pp. The new user goal is stricter:

```text
For each of the five operators:
  EfficientNet1DV2 AUROC drop: 10-15 pp
  ECGFounder AUROC drop:       10-15 pp
  Inter-model AUROC-drop gap:  not too large
```

Therefore this must be a new explicitly labeled profile, not a reinterpretation of official severity 5.

Use the name:

```text
dual_model_10to15pp_v1
```

Keep official s5 as a separate benchmark unless the user explicitly says to replace it.

## Open User Decisions

These need user confirmation before full four-center calibration. Until changed, use the recommended default.

| Decision | Recommended default | Why |
|---|---|---|
| Primary target metric | AUROC only must be 10-15 pp; AUPRC is a guardrail | User specifically asked for AUROC drop; AUPRC is prevalence-sensitive across PN2021 centers. |
| Acceptable inter-model gap | <= 3 pp mean AUROC-drop gap per operator | Tight enough to avoid obvious model imbalance, loose enough to be feasible. |
| Calibration baseline | Direct-K500/fullFT baselines only | Avoid tuning the benchmark to VAE-LH + AugMix. |
| Confirmation scope | Tune on two centers, freeze, then confirm once on all four centers | Avoid presenting a four-center tuned profile as blind final evidence. |
| Official s5 vs calibrated stress | Keep official s5 primary; report calibrated stress as `dual_model_10to15pp_v1` | Official s5 remains comparable; calibrated stress tests the recovery claim. |
| `powerline_noise` input mode | Treat `native_raw_first` as a separate powerline diagnostic branch; do not mix it with `bottleneck5000` rows | Powerline is strongly path-dependent; mode-B official amp is weak after preprocessing. |
| `random_leads_masking` realism | Model plausible lead disconnection; avoid routinely zeroing most leads | `mask_leads_prob=0.57` is already near target; higher values can become unrealistic. |

Specific questions for the user:

1. Should the final inter-model gap threshold be 2 pp, 3 pp, or 5 pp?
2. Should calibrated stress replace official s5 in the main table, or remain a secondary stress benchmark?
3. For `powerline_noise`, do we prioritize physically faithful acquisition-side corruption (`native_raw_first`) or the existing locked model-facing path (`raw_first` for EfficientNet, `bottleneck5000` for ECGFounder)?
4. For `random_leads_masking`, is severe multi-lead blackout acceptable as a stress test, or should it remain a realistic lead-disconnection corruption?

## Evidence Baseline

### Current Official S5 Drops

Source: `docs/reports/archive/20260618/pn2021c_locked_official_s5_summary_20260618.md`

| Operator | ECGFounder AUROC/AUPRC drop | EfficientNet1DV2 AUROC/AUPRC drop | Status |
|---|---:|---:|---|
| `baseline_shift` | 3.49 / 5.24 pp | 5.92 / 8.41 pp | Too weak |
| `baseline_wander` | 1.53 / 2.24 pp | 2.98 / 4.60 pp | Far too weak |
| `emg_noise` | 5.98 / 9.84 pp | 4.86 / 8.12 pp | Needs stronger AUROC |
| `powerline_noise` | 8.65 / 12.92 pp | 1.01 / 1.47 pp | Biggest model-gap problem |
| `random_leads_masking` | 8.95 / 14.66 pp | 9.40 / 14.54 pp | Closest to target |

### Old EfficientNet-Only 10-20 pp Seed Parameters

Source: `/home/linbinhao/ECG_adv_data/runs/pn2021c_param_calibration_10to20pp_20260606/EfficientNet1DV2/vae_lhat/calibration_selected_10to20pp_params.csv`

These are useful seeds, not current final evidence, because they were calibrated on an older EfficientNet-only run family.

| Operator | Seed parameters | Old EfficientNet mean AUROC/AUPRC drop |
|---|---|---:|
| `baseline_shift` | `max_amplitude=2.4`, `shift_ratio=0.9`, `num_segment=6`, `p=1.0` | 10.22 / 13.76 pp |
| `baseline_wander` | `max_amplitude=2.5`, `k=6`, `max_freq=0.8`, `p=1.0` | 13.04 / 13.53 pp |
| `emg_noise` | `max_amplitude=2.3`, `p=1.0` | 10.77 / 14.23 pp |
| `powerline_noise` | `max_amplitude=8.0`, `p=1.0` | 10.72 / 12.96 pp |
| `random_leads_masking` | `mask_leads_prob=0.57`, `p=1.0` | 10.39 / 15.23 pp |

### Powerline Mode-B Smoke

Source: `/home/linbinhao/ECG_adv_data/runs/pn2021c_powerline_modeb_smoke_20260618/summary_powerline_modeb_smoke_limit2000.md`

This is CPSC 2018 only, `limit=2000`, diagnostic only.

| Model | Input mode | Powerline amp | AUROC/AUPRC drop |
|---|---|---:|---:|
| EfficientNet1DV2 | `raw_first` | 0.3 | 1.16 / 1.45 pp |
| EfficientNet1DV2 | `native_raw_first` | 0.3 | 0.30 / 0.76 pp |
| EfficientNet1DV2 | `native_raw_first` | 8.0 | 8.11 / 10.43 pp |
| EfficientNet1DV2 | `native_raw_first` | 16.0 | 9.62 / 12.16 pp |
| EfficientNet1DV2 | `native_raw_first` | 20.0 | 10.46 / 12.94 pp |
| ECGFounder | `bottleneck5000` | 0.3 | 10.80 / 21.99 pp |
| ECGFounder | `native_raw_first` | 0.3 | 2.30 / 3.86 pp |
| ECGFounder | `native_raw_first` | 8.0 | 10.10 / 13.58 pp |

Conclusion: powerline is dominated by input path. It needs a separately labeled protocol decision.

## Calibration Objective

For model `m`, center `c`, operator `o`, and operator parameter profile `theta_o`:

```text
D[m,c,o] = clean_AUROC[m,c] - corrupted_AUROC[m,c,o]
```

All drops must be same-subset drops: clean and corrupted metrics are recomputed on the exact same ref-excluded sample set.

Score each operator independently:

```text
score(theta_o) =
  band_penalty(mean_c D[EffNet,c,o], 0.10, 0.15)
+ band_penalty(mean_c D[ECGFounder,c,o], 0.10, 0.15)
+ 3.0 * max(0, abs(mean_c D[EffNet,c,o] - mean_c D[ECGFounder,c,o]) - 0.03)^2
+ 1.0 * center_instability(theta_o)
+ 2.0 * realism_violation(theta_o)
```

Use this band penalty:

```python
def band_penalty(x, lo=0.10, hi=0.15):
    if lo <= x <= hi:
        return 0.0
    if x < lo:
        return (lo - x) ** 2
    return (x - hi) ** 2
```

Use these acceptance rules:

```text
Mean AUROC drop for each model: 10-15 pp
Inter-model mean AUROC gap:     <= 3 pp
Per-center AUROC drop:          preferably 5-20 pp; outside this requires explicit reporting
AUPRC:                          tracked as guardrail; reject unexplained collapse
Realism:                        visual/SNR/PSD/physiology gates must pass
```

## Candidate Parameter Grid

The first sweep should use coarse-to-fine grids. Do not run a full cartesian explosion. Start with brackets from official s5, stress_v2, old EfficientNet calibration, and the mode-B powerline smoke.

### `baseline_shift`

Current official s5 is too weak for both models. Old EfficientNet seed is already near target.

Coarse candidates:

| Candidate | Parameters |
|---|---|
| `shift_amp1p8_r0p70_seg4` | `max_amplitude=1.8`, `shift_ratio=0.70`, `num_segment=4` |
| `shift_amp2p4_r0p90_seg6` | `max_amplitude=2.4`, `shift_ratio=0.90`, `num_segment=6` |
| `shift_amp2p8_r0p90_seg6` | `max_amplitude=2.8`, `shift_ratio=0.90`, `num_segment=6` |
| `shift_amp3p2_r0p90_seg8` | `max_amplitude=3.2`, `shift_ratio=0.90`, `num_segment=8` |

Refine around the first candidate that gives both models at least 8 pp AUROC drop on smoke.

### `baseline_wander`

Current official s5 is far too weak. This likely needs the largest sweep besides powerline.

Coarse candidates:

| Candidate | Parameters |
|---|---|
| `wander_amp2p0_k5_f0p6` | `max_amplitude=2.0`, `k=5`, `max_freq=0.6`, `min_freq=0.03` |
| `wander_amp2p5_k6_f0p8` | `max_amplitude=2.5`, `k=6`, `max_freq=0.8`, `min_freq=0.03` |
| `wander_amp3p0_k6_f0p8` | `max_amplitude=3.0`, `k=6`, `max_freq=0.8`, `min_freq=0.03` |
| `wander_amp3p5_k8_f1p0` | `max_amplitude=3.5`, `k=8`, `max_freq=1.0`, `min_freq=0.03` |

Reject candidates that visually look like non-ECG baseline flooding rather than plausible respiration/body-motion drift.

### `emg_noise`

Current official s5 is moderate but below AUROC target. Old EfficientNet seed is near target.

Coarse candidates:

| Candidate | Parameters |
|---|---|
| `emg_amp1p6` | `max_amplitude=1.6`, `min_amplitude=0.0` |
| `emg_amp2p0` | `max_amplitude=2.0`, `min_amplitude=0.0` |
| `emg_amp2p3` | `max_amplitude=2.3`, `min_amplitude=0.0` |
| `emg_amp2p8` | `max_amplitude=2.8`, `min_amplitude=0.0` |

Guardrail: PSD should show broadband high-frequency contamination, not only amplitude collapse.

### `powerline_noise`

This is path-dependent and must not be mixed across modes.

Evaluate two explicitly labeled branches:

| Branch | EfficientNet input | ECGFounder input | Purpose |
|---|---|---|---|
| `powerline_locked_path` | `raw_first` | `bottleneck5000` | Matches current locked protocol but currently has a huge inter-model gap. |
| `powerline_native_raw_first` | `native_raw_first` | `native_raw_first` | Simulates acquisition-side corruption and gives more balanced low/medium amplitudes, but needs stronger amplitude for EfficientNet. |

Start with `powerline_native_raw_first` because the locked path is already badly imbalanced.

Coarse candidates:

| Candidate | Parameters |
|---|---|
| `power_native_amp8` | `max_amplitude=8.0`, `min_amplitude=0.0`, native sample-rate aware |
| `power_native_amp12` | `max_amplitude=12.0`, `min_amplitude=0.0`, native sample-rate aware |
| `power_native_amp16` | `max_amplitude=16.0`, `min_amplitude=0.0`, native sample-rate aware |
| `power_native_amp20` | `max_amplitude=20.0`, `min_amplitude=0.0`, native sample-rate aware |

Run ECGFounder for `amp12/16/20` before accepting any candidate. The current smoke only measured ECGFounder native `amp8`.

### `random_leads_masking`

Current official s5 is already close. Do not jump aggressively.

Coarse candidates:

| Candidate | Parameters |
|---|---|
| `mask_p0p50` | `mask_leads_prob=0.50`, `mask_leads_selection=random` |
| `mask_p0p57` | `mask_leads_prob=0.57`, `mask_leads_selection=random` |
| `mask_p0p60` | `mask_leads_prob=0.60`, `mask_leads_selection=random` |
| `mask_p0p65` | `mask_leads_prob=0.65`, `mask_leads_selection=random` |

Reject candidates that routinely blank most of the 12 leads. This operator can overshoot abruptly.

## Required Implementation Surface

Current gap: evaluators only accept hard-coded `--severity_profile` values. Flexible sweeps need external parameter files.

### Task 1: Add Config-Driven Custom Corruption Profiles

**Files:**
- Modify: `scripts/triple_labels/eval_pn2021_corruptions.py`
- Modify: `scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py`
- Modify: `ecg_adv_gen/config/adapters/pn2021c_eval.py`
- Modify: `ecg_adv_gen/config/adapters/ecgfounder_pn2021c_eval.py`
- Modify: `configs/schemas/experiment_config.schema.json`
- Test: `util/tests/test_pn2021c_corruption_profiles.py`
- Test: `util/tests/test_pn2021c_eval_adapter.py`
- Test: `util/tests/test_ecgfounder_pn2021c_evaluator.py`

- [ ] Add CLI flags to both evaluators:

```text
--severity_profile custom
--severity_params_file /path/to/profile.yaml
--severity_params_name profile_name
```

- [ ] Use this YAML shape:

```yaml
profiles:
  emg_amp2p3:
    emg_noise:
      5:
        max_amplitude: 2.3
        min_amplitude: 0.0
        p: 1.0
        dependency: false
  mask_p0p57:
    random_leads_masking:
      5:
        mask_leads_prob: 0.57
        mask_leads_selection: random
        p: 1.0
```

- [ ] Preserve built-in `standard`, `stress_v2`, and `calibrated_10to20pp`.
- [ ] Record resolved custom profile name and exact params in every output JSON.
- [ ] Keep custom profiles stream-only; reject cache mode for custom params.

Validation:

```bash
micromamba run -n ECGTwin pytest -q \
  util/tests/test_pn2021c_corruption_profiles.py \
  util/tests/test_pn2021c_eval_adapter.py \
  util/tests/test_ecgfounder_pn2021c_evaluator.py
```

### Task 2: Create Candidate Profile File

**Files:**
- Create: `configs/corruption_profiles/pn2021c_dual_model_10to15pp_candidates_20260618.yaml`

- [ ] Encode every candidate listed in the Candidate Parameter Grid.
- [ ] Use one profile per candidate and one operator per profile for single-operator calibration.
- [ ] Do not encode final `dual_model_10to15pp_v1` until two-center calibration has selected winners.

Validation:

```bash
micromamba run -n ECGTwin python - <<'PY'
import yaml
from pathlib import Path
p = Path("configs/corruption_profiles/pn2021c_dual_model_10to15pp_candidates_20260618.yaml")
data = yaml.safe_load(p.read_text())
assert "profiles" in data
assert "mask_p0p57" in data["profiles"]
assert "power_native_amp20" in data["profiles"]
print("profile yaml ok", len(data["profiles"]))
PY
```

### Task 3: Freeze Calibration Baseline Inventory

**Files:**
- Create: `docs/reports/archive/20260618/pn2021c_dual_model_calibration_baseline_inventory_20260618.md`

- [ ] Identify direct-K500/fullFT baselines for both models and all four target centers.
- [ ] Use ECGFounder fullFT locked runs, not frozen/head-only runs:

```text
/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/{center}_k500_fullft_locked
```

- [ ] For EfficientNet1DV2, start from active evidence:

```text
configs/active_evidence_registry.yaml claim: effnet_direct_k500_v7_sjr_rgq
configs/active_scripts.yaml entry: effnet_direct_k500_v7_sjr_rgq_matrix
candidate root: /home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix
```

- [ ] If a direct baseline is missing, train it before calibration. Do not substitute VAE-LH + AugMix method rows.
- [ ] Record checkpoint path, clean eval JSON, K500 ref-meta path, PN2021 mapping version/hash, and all-zero policy.

Acceptance:

```text
Every model-center pair has:
  checkpoint path
  clean eval JSON
  selected K500 ref-meta path
  same class order CD,HYP,MI,NORM,STTC
  PN2021 Super5 v7 hash 555ec85d5b51
```

### Task 4: Smoke Sweep On Deterministic Subsets

**Files:**
- Create run outputs under:

```text
/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/smoke/
```

- [ ] Use `limit=2000` per selected center for smoke.
- [ ] Use the same limit and same ref-excluded sample order for clean and corrupted metrics.
- [ ] Run CPSC 2018 first because it already exposed the powerline mismatch.
- [ ] Use `CUDA_VISIBLE_DEVICES=0` or another free GPU after `nvidia-smi`.
- [ ] Use `TMPDIR=/home/linbinhao/tmp`.
- [ ] Run one operator at a time.

EfficientNet command template:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TMPDIR=/home/linbinhao/tmp ECG_ADV_GEN_DATA_ROOT=/home/linbinhao/ECG_adv_data \
micromamba run -n ECGTwin python -u scripts/triple_labels/eval_pn2021_corruptions.py \
  --mode stream \
  --scheme super5 \
  --model_dir <effnet_direct_model_dir> \
  --clean_eval_json <effnet_direct_clean_eval_json> \
  --checkpoint_name last_model.pt \
  --clean_mmap_cache_dir /home/linbinhao/ECG_adv_data/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal \
  --clean_cache_dir /home/linbinhao/ECG_adv_data/triple_labels/pn2021_eval_cache_minresample_perglobal \
  --required_cache_version v7_refexcluded_100hz1000 \
  --centers cpsc_2018 \
  --corruptions <operator> \
  --severities 5 \
  --severity_profile custom \
  --severity_params_file configs/corruption_profiles/pn2021c_dual_model_10to15pp_candidates_20260618.yaml \
  --severity_params_name <candidate_name> \
  --corruption_input raw_first \
  --pn2021_root /home/linbinhao/ECG_adv_data/physionet2021 \
  --exclude_ref_ids <cpsc_ref_meta_json> \
  --device cuda \
  --crop_len 1000 \
  --batch_size 192 \
  --num_workers 0 \
  --min_pos 10 \
  --seed 20260501 \
  --limit 2000 \
  --output_path <output_json>
```

ECGFounder command template:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TMPDIR=/home/linbinhao/tmp ECG_ADV_GEN_DATA_ROOT=/home/linbinhao/ECG_adv_data ECGFOUNDER_ROOT=/home/linbinhao/ECG_adv_data/ecgfounder \
micromamba run -n ECGTwin python -u scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py \
  --scheme super5 \
  --run_dir <ecgfounder_direct_run_dir> \
  --variant ecgfounder_k500_fullft_locked \
  --checkpoint /home/linbinhao/ECG_adv_data/ecgfounder/checkpoint/12_lead_ECGFounder.pth \
  --clean_mmap_cache_dir /home/linbinhao/ECG_adv_data/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal \
  --clean_cache_dir /home/linbinhao/ECG_adv_data/triple_labels/pn2021_eval_cache_minresample_perglobal \
  --required_cache_version v7_refexcluded_100hz1000 \
  --centers cpsc_2018 \
  --corruptions <operator> \
  --severities 5 \
  --severity_profile custom \
  --severity_params_file configs/corruption_profiles/pn2021c_dual_model_10to15pp_candidates_20260618.yaml \
  --severity_params_name <candidate_name> \
  --corruption_input bottleneck5000 \
  --pn2021_root /home/linbinhao/ECG_adv_data/physionet2021 \
  --device cuda \
  --crop_len 1000 \
  --batch_size 96 \
  --num_workers 0 \
  --min_pos 10 \
  --seed 20260501 \
  --limit 2000 \
  --output_path <output_json>
```

Powerline exception:

```text
Run both `powerline_locked_path` and `powerline_native_raw_first` branches.
For `powerline_native_raw_first`, set EfficientNet and ECGFounder `--corruption_input native_raw_first`.
Do not combine locked-path and native-first rows in one final table.
```

### Task 5: Two-Center Calibration

**Files:**
- Create run outputs under:

```text
/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/two_center_calibration/
```

- [ ] Use two calibration centers only:

```text
cpsc_2018
georgia
```

- [ ] Select candidates per operator using the objective in this document.
- [ ] Do not inspect Chapman/Ningbo full results while choosing candidate winners.
- [ ] Save a CSV:

```text
/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/two_center_calibration/candidate_scores.csv
```

CSV columns:

```text
operator,candidate,model,center,clean_auroc,corrupted_auroc,drop_auroc_pp,
clean_auprc,corrupted_auprc,drop_auprc_pp,inter_model_gap_pp,
score,accepted_on_two_center,output_json
```

### Task 6: Realism Diagnostics

**Files:**
- Create diagnostics under:

```text
/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/diagnostics/
```

- [ ] For each selected candidate, save fixed-sample clean/corrupted 12-lead galleries.
- [ ] Compute RMS ratio and max absolute diff:

```text
diff = corrupted_pre_zscore - clean_pre_zscore
rms_ratio = rms(diff) / max(rms(clean_pre_zscore), eps)
max_abs_diff = max(abs(diff))
```

- [ ] Compute PSD summaries:

```text
powerline_noise: narrow-band 50/60 Hz or native alias evidence
emg_noise: broadband high-frequency energy
baseline_wander: low-frequency energy
baseline_shift: DC/step shifts
random_leads_masking: explicit masked-lead count distribution
```

- [ ] Reject candidates with visually broken ECGs unless explicitly labeled as stress-only.

### Task 7: Freeze `dual_model_10to15pp_v1`

**Files:**
- Create: `configs/corruption_profiles/pn2021c_dual_model_10to15pp_v1.yaml`
- Create: `docs/reports/archive/20260618/pn2021c_dual_model_10to15pp_v1_selection_report.md`

- [ ] Copy only selected operator candidates into the frozen profile.
- [ ] Include exact params, input mode, seed, center split, and direct baseline paths.
- [ ] Mark whether `powerline_noise` uses locked path or native-first.
- [ ] Do not modify selected params after this point unless creating `v2`.

### Task 8: Four-Center Confirmation

**Files:**
- Create run outputs under:

```text
/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/four_center_confirmation/
```

- [ ] Run frozen `dual_model_10to15pp_v1` once on all four centers:

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

- [ ] Aggregate results into:

```text
docs/reports/archive/20260618/pn2021c_dual_model_10to15pp_v1_confirmation.md
docs/reports/archive/20260618/pn2021c_dual_model_10to15pp_v1_confirmation.csv
```

- [ ] If a selected operator misses the target on four-center confirmation, report it honestly:

```text
accepted_on_two_center = true
four_center_status = miss_low | in_band | miss_high
```

- [ ] If rerunning calibration using four-center feedback, create `dual_model_10to15pp_v2` and label it exploratory.

### Task 9: Evaluate VAE-LH + AugMix After Profile Freeze

**Files:**
- Create method-eval outputs under:

```text
/home/linbinhao/ECG_adv_data/runs/pn2021c_dual_model_10to15pp_sweep_20260618/method_eval/
```

- [ ] Evaluate direct-K500/fullFT baselines under the frozen profile.
- [ ] Evaluate VAE-LH noAug under the frozen profile.
- [ ] Evaluate VAE-LH + three-chain AugMix under the frozen profile.
- [ ] Report method gains relative to direct baselines on corrupted PN2021-C:

```text
gain_auroc_pp = method_corrupted_auroc - direct_corrupted_auroc
gain_auprc_pp = method_corrupted_auprc - direct_corrupted_auprc
```

- [ ] Do not use method results to alter the profile.

## Expected Sweep Outcome By Operator

| Operator | Expected difficulty | Starting recommendation |
|---|---|---|
| `baseline_shift` | Moderate | Start near old seed `amp2.4/r0.9/seg6`; increase only if ECGFounder stays low. |
| `baseline_wander` | High | Needs strong sweep; official s5 is far too weak. |
| `emg_noise` | Moderate | Old seed `amp2.3` likely close; check model gap. |
| `powerline_noise` | Highest | Decide locked path vs native-first before full sweep; native-first likely needs amp between 8 and 20. |
| `random_leads_masking` | Low to moderate | Narrow sweep around `0.57`; avoid high mask probabilities. |

## Failure Modes To Avoid

- Tuning separate parameters per model.
- Tuning separate parameters per center.
- Tuning using VAE-LH + AugMix method rows.
- Comparing corrupted subset metrics against full clean metrics.
- Mixing all-zero-kept with drop-all-zero metrics.
- Mixing `powerline_locked_path` and `powerline_native_raw_first` rows without labeling.
- Letting strong corruptions create visually non-ECG waveforms just to hit 10-15 pp.
- Reusing old `calibrated_10to20pp` as final current evidence.
- Presenting a four-center tuned profile as blind final evidence.

## Immediate Next Step

Before implementation, answer these two decisions:

```text
1. Inter-model gap threshold: 2 pp, 3 pp, or 5 pp?
2. Powerline final branch: locked path, native_raw_first, or run both and decide after two-center calibration?
```

Recommended answers:

```text
1. Use 3 pp.
2. Run both for calibration, but only freeze one branch in v1.
```
