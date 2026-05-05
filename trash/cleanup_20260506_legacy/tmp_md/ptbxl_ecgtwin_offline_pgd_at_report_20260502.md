# PTB-XL ECGTwin Offline PGD-AT Report 2026-05-02

## Goal

Use PTB-XL folds 1-8 to build an ECGTwin synthetic Super5 pool, then test
whether VAE latent-space offline adversarial training improves EfficientNet1DV2
macro AUROC/AUPRC.

## Data And Artifacts

```text
refs:   /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/refs/ptbxl_train1000_balanced_seed42.pt
synth:  /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/synth/ptbxl_train1000_to_synth10000.npz
latent: /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/synth/ptbxl_train1000_to_synth10000.latent.npz
```

Synthetic pool:

```text
N = 10000
classes = CD/HYP/MI/NORM/STTC
per class = 2000
format = (N,12,1000), PTB-XL lead order, 100Hz, z-scored classifier scale
```

Quality summary:

| class | n | finite | victim top1 | target prob mean | Einthoven p95 mean | HR mean |
|---|---:|---:|---:|---:|---:|---:|
| CD | 2000 | 1.0000 | 0.7245 | 0.8098 | 0.1053 | 78.19 |
| HYP | 2000 | 1.0000 | 0.0265 | 0.5094 | 0.0950 | 77.82 |
| MI | 2000 | 1.0000 | 0.2780 | 0.6420 | 0.0923 | 85.09 |
| NORM | 2000 | 1.0000 | 0.8415 | 0.8334 | 0.1427 | 74.54 |
| STTC | 2000 | 1.0000 | 0.6440 | 0.7468 | 0.0915 | 81.95 |

HYP and MI class consistency are weak. The export is z-scored, so raw mV
digital validity still needs a raw decode export.

## Results

Same eval script and PN2021 v3 clean 7-center mapping.

| run | PTB-XL AUROC | PTB-XL AUPRC | PN2021 AUROC | PN2021 AUPRC | verdict |
|---|---:|---:|---:|---:|---|
| baseline `/root/autodl-tmp/triple_labels/super5` | 0.9064 | 0.7754 | 0.8344 | 0.5526 | reference |
| direct synth r0.25 | 0.9010 | 0.7605 | 0.8277 | 0.5401 | failed |
| free-PGD eps2 K10 aw0.50 | 0.8925 | 0.7498 | not run | not run | failed, too strong |
| free-PGD eps0.5 K5 aw0.25 | 0.9050 | 0.7736 | not run | not run | close, still below |
| free-PGD eps0.5 K5 aw0.10 | 0.9062 | 0.7756 | 0.8341 | 0.5535 | tiny AUPRC gain, AUROC slight drop |
| trusted3 eps0.5 K5 aw0.10 | 0.9062 | 0.7751 | 0.8337 | 0.5533 | did not beat all5 |
| latent-hull M5 lam0.25 n2000 aw0.10 | 0.9064 | 0.7754 | 0.8343 | 0.5533 | most stable, within noise |

## Conclusion

Direct ECGTwin waveform augmentation failed. Strong latent free-PGD also failed.
Conservative latent-space regularization gives only a tiny AUPRC signal and does
not yet justify a strong improvement claim. The most defensible current claim is
that VAE latent constraints can make ECGTwin-derived adversarial samples usable
as a weak regularizer, but the effect size is near the noise floor.

Next useful steps:

```text
1. Export raw ECGTwin decoded waveforms before z-score and run mV digital gates.
2. Scale latent-hull from n=2000 to n=10000 only if raw quality is acceptable.
3. Prefer target-center/center-token work over direct source-domain waveform augmentation.
```
