# CPSC VAE-LH AugMix Chain-Base Debug Summary

Date: 2026-07-08

Scope: CPSC-only PN2021-C official severity-5 depth2+3 composite evaluation.
Metric: 20-combo mean macro AUROC / AUPRC, ref-excluded.

## Result

| Backbone | Variant | AUROC / AUPRC | Gain vs K500+AugMix |
|---|---:|---:|---:|
| EfficientNet1DV2 | K500 + clean three-chain AugMix | 0.7817 / 0.4800 | 0.00 / 0.00 pp |
| EfficientNet1DV2 | all VAE-adv AugMix base | 0.7875 / 0.4827 | +0.58 / +0.27 pp |
| EfficientNet1DV2 | one VAE-adv AugMix chain | 0.7863 / 0.4809 | +0.46 / +0.09 pp |
| EfficientNet1DV2 | clean AugMix + VAE stream scale 0.10 | 0.7800 / 0.4779 | -0.17 / -0.21 pp |
| EfficientNet1DV2 | clean AugMix + VAE stream scale 0.05 | 0.7809 / 0.4791 | -0.08 / -0.09 pp |
| ECGFounder | K500 + clean three-chain AugMix | 0.8631 / 0.6427 | 0.00 / 0.00 pp |
| ECGFounder | all VAE-adv AugMix base | 0.8229 / 0.5556 | -4.02 / -8.71 pp |
| ECGFounder | one VAE-adv AugMix chain | 0.8271 / 0.5627 | -3.60 / -8.00 pp |
| ECGFounder | clean AugMix + VAE stream scale 0.10 | 0.8695 / 0.6551 | +0.65 / +1.24 pp |

## Verdict

The failure is not that VAE-LH hard samples are invalid. The failure is the
routing of those samples into the AugMix base waveform.

EfficientNet benefits only when the VAE-LH adversarial waveform is used inside
the AugMix corruption base. ECGFounder is harmed by that same routing and only
benefits when clean-anchor AugMix stays clean-based while VAE-LH hard samples
enter as a separate supervised stream.

Current paper-safe CPSC fix is model-aware routing:

- EfficientNet1DV2: use `chain_base_mode=all_adv` for the VAE-LH AugMix branch.
- ECGFounder: use `chain_base_mode=all_clean_plus_vae_adv` with
  `vae_adv_stream_sample_scale=0.10`.

No single tested chain-base topology is positive for both backbones.

## Managed Config Wiring

- `configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml`
  now sets `adaptation.latent_augmix.chain_base_mode: all_adv`.
- `configs/experiments/ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml`
  now sets `adaptation.latent_augmix.chain_base_mode: all_clean_plus_vae_adv`
  and `adaptation.vae.adv_stream_sample_scale: 0.10`.

Verification:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest \
  util/tests/test_config_loader.py::test_effnet_vae_lhat_threechain_locked_k500_config_uses_official_s5_last_checkpoint \
  util/tests/test_config_loader.py::test_ecgfounder_locked_threechain_augmix_command_uses_fullft_last_checkpoint_mainline \
  util/tests/test_config_loader.py::test_synth_online_at_parser_accepts_decoupled_clean_augmix_vae_mode \
  util/tests/test_config_loader.py::test_ecgfounder_fullft_parser_accepts_decoupled_clean_augmix_vae_mode \
  util/tests/test_lhat_augmix_ablation.py -q
```

Result: `8 passed`.

Additional regression after wiring the managed configs:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_config_loader.py -q
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m py_compile \
  ecg_adv_gen/runner/ecgfounder_fullft.py \
  ecg_adv_gen/config/adapters/ecgfounder_fullft.py
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest \
  util/tests/test_lhat_augmix_ablation.py \
  util/tests/test_config_loader.py::test_effnet_vae_lhat_threechain_locked_k500_config_uses_official_s5_last_checkpoint \
  util/tests/test_config_loader.py::test_ecgfounder_locked_threechain_augmix_command_uses_fullft_last_checkpoint_mainline -q
```

Results: `140 passed`, `py_compile` passed, `6 passed`.

## Evidence

- EffNet scale0.05 train:
  `/home/linbinhao/ECG_adv_data/runs/20260708_cpsc_weightsplit_fix/effnet_decoupled_sampleweightfix_scale0p05_k500_ep15`
- EffNet scale0.05 PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/20260708_cpsc_weightsplit_fix/eval/effnet_decoupled_sampleweightfix_scale0p05_k500_ep15_cpsc_official_s5_depth23.json`
- EffNet all-adv PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/20260708_cpsc_chainbase_ablation/eval/effnet_all_adv_k500_ep15_cpsc_official_s5_depth23.json`
- EffNet all-clean PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/20260708_cpsc_chainbase_ablation/eval/effnet_all_clean_k500_ep15_cpsc_official_s5_depth23.json`
- ECGFounder all-adv PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/20260708_cpsc_chainbase_ablation/eval/ecgfounder_all_adv_k500_ep15_cpsc_official_s5_depth23.json`
- ECGFounder all-clean PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/20260708_cpsc_chainbase_ablation/eval/ecgfounder_all_clean_k500_ep15_cpsc_official_s5_depth23.json`
- ECGFounder one-adv PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/20260708_cpsc_chainbase_ablation/eval/ecgfounder_oneadv_k500_ep15_cpsc_official_s5_depth23.json`
- ECGFounder decoupled PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/20260708_cpsc_weightsplit_fix/eval/ecgfounder_decoupled_scale0p1_k500_ep15_cpsc_official_s5_depth23.json`
