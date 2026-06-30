# PN2021-C EfficientNet SOTA Tuning Handoff - 2026-06-22

## Scope

Goal: improve EfficientNet1DV2 recovery on PN2021-C `dual_model_10to15pp_v1`
from the direct K500 fullFT baseline by about 10 pp while preserving the main
VAE-LH online AT + locked three-chain AugMix paradigm.

Main-method constraints kept:

- VAE latent-hull online adversarial training is still active.
- AugMix remains locked three-chain, not collapsed to a single corruption view.
- No stabilizer35, preprocessing repair, operator oracle, heldout selector, or
  separate raw-supervised branch is counted as the main method.

## Completed Evidence

Current best completed run is cand94:

```text
CSV:
/home/linbinhao/ECG_adv_data/runs/pn2021c_sota_ablation_effnet_ecgfounder_20260621/cand94_robust4g_recovery_snapshot_20260622_0951.csv

Summary:
/home/linbinhao/ECG_adv_data/runs/pn2021c_sota_ablation_effnet_ecgfounder_20260621/cand94_robust4g_recovery_summary_20260622_0951.md
```

Four-center, five-operator mean versus direct K500 corrupted baseline:

| Candidate | Corrupted AUROC/AUPRC | Recovery AUROC/AUPRC |
|---|---:|---:|
| cand93 | 0.7521 / 0.4124 | +6.99 / +7.58 pp |
| cand94 | 0.7590 / 0.4202 | +7.67 / +8.36 pp |
| cand96 | 0.7407 / 0.4228 | +6.40 / +6.98 pp |

Cand94 recovery by center:

| Center | Recovery AUROC/AUPRC |
|---|---:|
| chapman_shaoxing | +6.83 / +8.70 pp |
| cpsc_2018 | +8.08 / +7.05 pp |
| georgia | +6.73 / +6.96 pp |
| ningbo | +9.06 / +10.72 pp |

Cand94 recovery by operator:

| Operator | Recovery AUROC/AUPRC |
|---|---:|
| baseline_shift | +7.66 / +8.68 pp |
| baseline_wander | +8.14 / +9.84 pp |
| emg_noise | +8.60 / +9.63 pp |
| powerline_noise | +9.93 / +9.39 pp |
| random_leads_masking | +4.03 / +4.24 pp |

Current completed evidence does not yet prove the +10 pp target. The main
bottleneck remains `random_leads_masking`.

## Running Follow-Up Experiments

As of 2026-06-22 16:16 CST, six GPUs are intentionally occupied:

| GPU | Run | Latest observed epoch |
|---:|---|---:|
| 0 | cand97 CPSC fixed-m=0.85 | 25/45 |
| 1 | cand97 Chapman fixed-m=0.85 | 25/45 |
| 2 | cand97 Georgia fixed-m=0.85 | 24/45 |
| 3 | cand97 Ningbo fixed-m=0.85 | 25/45 |
| 4 | cand98 Chapman fixed-m=1.0 accel6g resume | 10/45 |
| 5 | cand98 Georgia fixed-m=1.0 accel6g resume | 10/45 |

GPU6 and GPU7 were left idle. No cand97 or cand98 PN2021-C raw-first eval JSONs
existed at the 16:16 CST check.

Cand97 watcher:

```text
PID: 3452365
Log:
/home/linbinhao/ECG_adv_data/runs/pn2021c_sota_ablation_effnet_ecgfounder_20260621/watchdogs/cand97_fixedm085_4g_watch_20260622.log
```

Stopped intentionally:

```text
PID 3466277
scripts/agent/watch_effnet_cand98_fixedm100_after_cand97_4g_20260622.py
```

Reason: that watcher would have launched duplicate cand98 Chapman/Georgia jobs
after cand97. The active GPU4/GPU5 accel jobs should be reused instead.

## Code And Config Changes Made During This Segment

Resume/runtime fixes:

- `ecg_adv_gen/training/checkpoint_state.py`
  - Normalize restored torch CPU/CUDA RNG states to CPU `uint8` tensors.
  - Fixes resume failures from serialized or device-mapped RNG states.
- `scripts/agent/run_launch_config_child.py`
  - Added stage-specific argument overrides.
  - Added `--train-resume latest`.
  - Added per-stage log redirection via `--stage-log STAGE=PATH`.

Tests added/updated:

- `util/tests/test_checkpoint_state.py`
- `util/tests/test_launch_config_child_runner.py`

Verification:

```text
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest \
  util/tests/test_checkpoint_state.py \
  util/tests/test_launch_config_child_runner.py -q

Result: 10 passed in 1.78s
```

Config safety:

- Added raw-first eval configs for cand98 accel6g Chapman/Georgia:
  - `configs/experiments/pn2021c_effnet_dual3ch_cand98_fixedm100_accel6g_rawfirst_chapman_shaoxing.yaml`
  - `configs/experiments/pn2021c_effnet_dual3ch_cand98_fixedm100_accel6g_rawfirst_georgia.yaml`

Verification:

```text
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest \
  util/tests/test_config_loader.py -q

Result: 249 passed in 23.54s
```

## Next Actions

1. Let cand97 finish 45 epochs and wait for its four raw-first eval JSONs.
2. Compute cand97 four-center/operator recovery using the direct K500 baseline
   CSV.
3. If cand97 reaches both AUROC and AUPRC recovery around 9.5-10 pp, stop
   fallback expansion and write the final recovery summary.
4. If cand97 misses, continue cand98 only for missing centers:
   - reuse GPU4/GPU5 accel results for Chapman/Georgia;
   - launch only CPSC/Ningbo cand98 if still needed.
5. Do not claim goal completion until the current raw-first eval JSONs prove it.
