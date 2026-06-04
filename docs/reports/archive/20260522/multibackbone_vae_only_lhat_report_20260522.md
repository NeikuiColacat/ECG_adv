# Multi-Backbone VAE-Only Online AT Report

Date: 2026-05-22

## Scope

This report covers the requested check: whether the current VAE-only real-anchor latent-hull online adversarial training method also works on five PTB-XL benchmark backbones beyond EfficientNet1DV2.

Completed backbones:

- `benchmark_fcn_wang`
- `benchmark_resnet1d_wang`
- `benchmark_inception1d`
- `benchmark_lstm`
- `benchmark_xresnet1d101`

## Protocol

Source training:

- Dataset: PTB-XL Super5, folds 1-8 train, fold 9 validation, fold 10 test.
- Input protocol: `minimal_resample`, `per_sample_global`, 100 Hz, 1000 samples = 10 seconds.
- Labels: Super5 order `CD, HYP, MI, NORM, STTC`.

Target-center adaptation:

- Centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- K: 500 target-center real ECG anchors per center.
- Evaluation excludes the K=500 adaptation records for the target center.
- Method: VAE-only real-anchor latent-hull online AT.
- No ECGTwin DiT synthetic samples and no center-token samples are used in this experiment.
- Main latent-hull settings: `M=20`, `lambda=0.15`, optimized convex weights, 10 online AT epochs, quality gate disabled.

Artifacts:

- Runner: `/root/autodl-tmp/ECG_adv_Gen/scripts/paper/run_multibackbone_vae_only_lhat_20260522.py`
- Summary CSV: `/root/autodl-tmp/paper_multibackbone_vae_only_lhat_20260522/summaries/multibackbone_vae_only_lhat.csv`
- Summary MD: `/root/autodl-tmp/paper_multibackbone_vae_only_lhat_20260522/summaries/multibackbone_vae_only_lhat.md`
- Logs: `/root/autodl-tmp/paper_multibackbone_vae_only_lhat_20260522/logs/`

## Aggregate Results

All 20 target-center comparisons improved both AUROC and AUPRC.

Overall mean across 5 models x 4 centers:

| baseline target | online AT target | delta |
|---:|---:|---:|
| 0.8388 / 0.5172 | 0.8681 / 0.5525 | +2.93pp / +3.54pp |

Mean by model:

| model | baseline target | online AT target | delta |
|---|---:|---:|---:|
| `benchmark_fcn_wang` | 0.8336 / 0.5132 | 0.8582 / 0.5394 | +2.47pp / +2.62pp |
| `benchmark_resnet1d_wang` | 0.8400 / 0.5146 | 0.8679 / 0.5480 | +2.79pp / +3.34pp |
| `benchmark_inception1d` | 0.8414 / 0.5220 | 0.8733 / 0.5599 | +3.19pp / +3.79pp |
| `benchmark_lstm` | 0.8407 / 0.5139 | 0.8736 / 0.5571 | +3.29pp / +4.32pp |
| `benchmark_xresnet1d101` | 0.8381 / 0.5221 | 0.8674 / 0.5583 | +2.94pp / +3.62pp |

Mean by target center:

| center | baseline target | online AT target | delta |
|---|---:|---:|---:|
| `ningbo` | 0.8641 / 0.4845 | 0.8865 / 0.5199 | +2.23pp / +3.53pp |
| `chapman_shaoxing` | 0.8733 / 0.4367 | 0.8941 / 0.4728 | +2.08pp / +3.61pp |
| `cpsc_2018` | 0.8018 / 0.5535 | 0.8621 / 0.6023 | +6.03pp / +4.88pp |
| `georgia` | 0.8158 / 0.5939 | 0.8297 / 0.6152 | +1.39pp / +2.13pp |

## Full Target-Center Table

| model | center | baseline target | online AT target | delta |
|---|---|---:|---:|---:|
| `benchmark_fcn_wang` | `ningbo` | 0.8608 / 0.4871 | 0.8835 / 0.5202 | +2.27pp / +3.32pp |
| `benchmark_fcn_wang` | `chapman_shaoxing` | 0.8714 / 0.4279 | 0.8885 / 0.4510 | +1.71pp / +2.32pp |
| `benchmark_fcn_wang` | `cpsc_2018` | 0.7921 / 0.5476 | 0.8395 / 0.5816 | +4.74pp / +3.39pp |
| `benchmark_fcn_wang` | `georgia` | 0.8100 / 0.5902 | 0.8214 / 0.6048 | +1.15pp / +1.46pp |
| `benchmark_resnet1d_wang` | `ningbo` | 0.8672 / 0.4903 | 0.8867 / 0.5221 | +1.96pp / +3.18pp |
| `benchmark_resnet1d_wang` | `chapman_shaoxing` | 0.8724 / 0.4227 | 0.8906 / 0.4561 | +1.82pp / +3.34pp |
| `benchmark_resnet1d_wang` | `cpsc_2018` | 0.8035 / 0.5520 | 0.8647 / 0.6009 | +6.12pp / +4.88pp |
| `benchmark_resnet1d_wang` | `georgia` | 0.8171 / 0.5935 | 0.8297 / 0.6130 | +1.26pp / +1.95pp |
| `benchmark_inception1d` | `ningbo` | 0.8677 / 0.4845 | 0.8920 / 0.5238 | +2.43pp / +3.92pp |
| `benchmark_inception1d` | `chapman_shaoxing` | 0.8783 / 0.4495 | 0.9004 / 0.4894 | +2.21pp / +3.99pp |
| `benchmark_inception1d` | `cpsc_2018` | 0.7988 / 0.5601 | 0.8666 / 0.6067 | +6.78pp / +4.67pp |
| `benchmark_inception1d` | `georgia` | 0.8208 / 0.5940 | 0.8343 / 0.6197 | +1.35pp / +2.57pp |
| `benchmark_lstm` | `ningbo` | 0.8639 / 0.4799 | 0.8874 / 0.5197 | +2.35pp / +3.98pp |
| `benchmark_lstm` | `chapman_shaoxing` | 0.8741 / 0.4224 | 0.8984 / 0.4674 | +2.43pp / +4.50pp |
| `benchmark_lstm` | `cpsc_2018` | 0.8065 / 0.5573 | 0.8767 / 0.6264 | +7.02pp / +6.91pp |
| `benchmark_lstm` | `georgia` | 0.8184 / 0.5961 | 0.8318 / 0.6150 | +1.34pp / +1.89pp |
| `benchmark_xresnet1d101` | `ningbo` | 0.8612 / 0.4810 | 0.8829 / 0.5136 | +2.16pp / +3.27pp |
| `benchmark_xresnet1d101` | `chapman_shaoxing` | 0.8704 / 0.4608 | 0.8928 / 0.4998 | +2.24pp / +3.90pp |
| `benchmark_xresnet1d101` | `cpsc_2018` | 0.8082 / 0.5506 | 0.8630 / 0.5961 | +5.49pp / +4.55pp |
| `benchmark_xresnet1d101` | `georgia` | 0.8125 / 0.5958 | 0.8311 / 0.6236 | +1.86pp / +2.78pp |

## Engineering Note

The LSTM backbone initially failed during latent-hull weight optimization with:

```text
cudnn RNN backward can only be called in training mode
```

The fix is in `adversarial/latent_hull_pgd.py`: during the inner latent-hull optimization loop, only `nn.RNNBase` child modules are temporarily switched to train mode and then restored. This enables cuDNN RNN input-gradient backward without changing BatchNorm/Dropout behavior for the rest of the victim model. A small LSTM smoke test passed before the full runner was resumed.

## Conclusion

The VAE-only real-anchor latent-hull online adversarial training route is not EfficientNet1DV2-specific. It improved target-center AUROC and AUPRC on all 20 model-center comparisons across five additional backbones from `model/ecg_ptbxl_benchmarking`.

The strongest average gains appear on `cpsc_2018`; `georgia` has smaller but still positive gains. This supports the paper claim that the main mechanism is target-center real ECG anchors in ECGTwin VAE latent space, not a single classifier architecture artifact.
