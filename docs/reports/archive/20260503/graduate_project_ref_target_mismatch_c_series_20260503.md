# Graduate Project C-Series Report: Ref-Target Mismatch Synthetic Pretrain

Date: 2026-05-03

## Question

Test whether ECGTwin prompt-token synthetic ECG can improve a 5-class
EfficientNet1DV2 trained from only 2000 random PTB-XL records.

The recommended route was:

```text
ECGTwin synthetic20k pretraining
-> real PTB-XL train2000 clean fine-tune
-> conservative real-anchor latent-hull AT
```

## Data And Generation

Real split:

```text
train = 2000 random PTB-XL records
val   = 2000 random PTB-XL records
test  = remaining 17799 PTB-XL records
preprocess = minimal_resample + per_sample_global
input = 10 seconds, 100 Hz, 12 leads, PTB-XL lead order
```

ECGTwin generation:

```text
reference ECG = random_any from train2000 ECGTwin latent cache
reference condition = translated/normalized English report prompt
target condition = super5 prompt + <ptbxl_CLASS> 4-vector prompt token
candidate pool = 4000/class = 20000 total
```

Gate result:

| class | candidates | accepted |
|---|---:|---:|
| NORM | 4000 | 2362 |
| MI | 4000 | 1884 |
| STTC | 4000 | 1483 |
| HYP | 4000 | 87 |
| CD | 4000 | 51 |
| total | 20000 | 5867 |

HYP/CD remain weak under strict digital/class gates.

## Results

| run | route | custom seed42 AUROC | custom seed42 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---|---:|---:|---:|---:|
| A-fair | real2000 only, AUPRC ckpt | 0.8357 | 0.6040 | 0.7339 | 0.4074 |
| C0 | synthetic20k only | 0.6649 | 0.4107 | not run | not run |
| C1 | synthetic20k -> real fine-tune | 0.8645 | 0.6723 | 0.7364 | 0.4195 |
| C3 | C1 -> latent-hull AT | 0.8654 | 0.6738 | 0.7381 | 0.4230 |
| C3-wide | C1 -> latent-hull AT, wider 0.45-0.65 gate | 0.8655 | 0.6733 | 0.7373 | 0.4198 |
| C4 | real-only -> latent-hull AT | 0.8356 | 0.6046 | 0.7314 | 0.4063 |

Official PTB-XL fold10 evaluation, using `eval_crosscenter.py`, moved in the
opposite direction:

| run | fold10 AUROC | fold10 AUPRC |
|---|---:|---:|
| A-fair | 0.8224 | 0.5922 |
| C1 | 0.7490 | 0.5106 |
| C3 | 0.7510 | 0.5118 |
| C3-wide | 0.7512 | 0.5095 |
| C4 | 0.8214 | 0.5925 |

## Interpretation

The method is not a total failure. On the requested random train2000/test-rest
protocol, synthetic pretraining gives a large AUPRC gain, and conservative
latent-hull AT adds a small further gain.

The result is also not yet thesis-proof. The official fold10 drop means the
benefit may depend on the random split. The PN2021 external gain is positive
but modest, so the next experiment should target robustness rather than only
raising the custom split score.

## Next Recommended Ablation

The wider boundary follow-up was already run:

```text
strict 0.50-0.60: 75/1000 adv samples, PN2021 AUPRC 0.4230
wide   0.45-0.65: 427/3000 adv samples, PN2021 AUPRC 0.4198
```

The wider buffer did not help. The next recommended ablation is not to make the
boundary wider again; instead, keep strict boundary filtering and test whether
multi-seed C1/C3 repeats preserve the custom-test and PN2021 gains.
