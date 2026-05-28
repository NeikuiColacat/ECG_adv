# Managed v6 Main Reproduction, 2026-05-28

This document records the first full managed-YAML rerun after the phase 1/2
refactor. It is intended as a compact reproducibility checkpoint before the
refactor changes were committed.

## Scope

The rerun used the v6 PN2021 Super5 mapping and the new managed experiment
launcher/config stack. Large run artifacts remain outside git under the migrated
data root:

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/20260528/
```

Preflight command:

```bash
micromamba run -n ECGTwin python scripts/audit_managed_configs.py \
  --index configs/active_scripts.yaml \
  --local-config configs/local/linbinhao_server.example.yaml \
  --output-dir /home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/config_audit/pre_main_rerun_20260528_1355 \
  --require-existing-inputs
```

Preflight result: 6/6 active managed configs passed.

## Executed Runs

| Managed config | Run id | Status | Duration |
|---|---:|---:|---:|
| `effnet_direct_k500_v6` | `main_repro_20260528_1355` | succeeded | 141.5 s |
| `effnet_vae_lhat_k500_v6` | `main_repro_20260528_1355_effnet_lhat` | succeeded | 2961.7 s |
| `ecgfounder_direct_k500_v6` | `main_repro_20260528_1355_ecgfounder_direct` | succeeded | 22.0 s |
| `ecgfounder_inithead_fullft_k500_v6` | `main_repro_20260528_1355_ecgfounder_inithead` | succeeded | 1665.2 s |
| `ecgfounder_vae_lhat_k500_v6` | `main_repro_20260528_1355_ecgfounder_lhat` | succeeded | 767.8 s |
| `pn2021_eval_v6_refexcluded` | `main_repro_20260528_1355_pn_eval` | succeeded | 78.6 s |

The first attempted shared run id was intentionally abandoned for all configs
except `effnet_direct_k500_v6`, because each managed launch directory refuses to
overwrite non-empty output directories. Subsequent configs used unique run ids.

## PN2021 Metrics

All metrics are macro averages over the PN2021 ref-excluded centers.

| Model / recipe | All-zero view | AUROC | AUPRC |
|---|---|---:|---:|
| EfficientNet1DV2 direct K500 | keep all-zero | 0.854320 | 0.487950 |
| EfficientNet1DV2 direct K500 | drop all-zero | 0.881636 | 0.634715 |
| EfficientNet1DV2 VAE L-HAT K500 | keep all-zero | 0.872050 | 0.514077 |
| EfficientNet1DV2 VAE L-HAT K500 | drop all-zero | 0.901647 | 0.677073 |
| ECGFounder direct K500 | keep all-zero | 0.904164 | 0.596816 |
| ECGFounder init-head full fine-tune K500 | keep all-zero | 0.912933 | 0.631507 |
| ECGFounder init-head full fine-tune K500 | drop all-zero | 0.930816 | 0.740829 |
| ECGFounder VAE L-HAT K500 | keep all-zero | 0.910242 | 0.610991 |
| ECGFounder VAE L-HAT K500 | drop all-zero | 0.927174 | 0.718523 |
| Independent PN2021 eval of EfficientNet direct | keep all-zero | 0.854321 | 0.487896 |
| Independent PN2021 eval of EfficientNet direct | drop all-zero | 0.881637 | 0.634631 |

## Main Takeaways

- The managed YAML refactor reproduced the intended main experiment surface:
  config audit, launch manifests, direct fine-tune, VAE L-HAT, ECGFounder
  wrappers, and independent PN2021 evaluation all ran end to end.
- EfficientNet1DV2 VAE L-HAT remained beneficial over direct K500 fine-tune:
  +1.77 pp AUROC / +2.61 pp AUPRC with all-zero samples kept, and
  +2.00 pp AUROC / +4.24 pp AUPRC with all-zero samples dropped.
- ECGFounder VAE L-HAT improved over direct K500 in the all-zero view
  (+0.61 pp AUROC / +1.42 pp AUPRC), but the ECGFounder init-head full
  fine-tune was stronger in this rerun.
- For ECGFounder L-HAT, `ningbo` and `chapman_shaoxing` selected epoch 0 in
  the center logs, which means no effective improvement there. `cpsc_2018`
  improved from about 0.8630/0.6219 to 0.8873/0.6814.
- EfficientNet1DV2 L-HAT attack success rates were in a useful range for
  `ningbo` and `georgia`; `cpsc_2018` became weaker later in training even
  though validation AUPRC kept increasing.

## Verification

Post-run checks:

```bash
micromamba run -n ECGTwin pytest util/tests
micromamba run -n ECGTwin python -m compileall -q \
  ecg_adv_gen scripts/run_experiment.py scripts/audit_managed_configs.py \
  scripts/export_data_manifest.py scripts/export_metrics_long.py \
  scripts/merge_metrics_long.py scripts/export_paper_table.py
git diff --check
git diff --cached --check
```

Result:

```text
244 passed, 13 warnings
compileall passed
git diff --check passed
git diff --cached --check passed
```

Generated Python bytecode and `.pytest_cache` were removed before staging.
External model repository handles under `model/` were left unstaged.
