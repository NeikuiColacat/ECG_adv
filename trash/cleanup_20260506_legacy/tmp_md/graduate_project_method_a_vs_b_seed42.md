# Graduate Project Method A vs Method B Report

Date: 2026-05-03

## Experiment Setup

Task:

```text
PTB-XL super5 multi-label classification
class order = CD, HYP, MI, NORM, STTC
model = EfficientNet1DV2 s_v2
preprocess = minimal_resample + per_sample_global
input = 100Hz, 10s, 1000 samples, 12 leads
split = random record-level seed42
train_real = 2000
val_real = 2000
test_real = 17799
```

Split:

```text
/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json
```

Train class counts:

| class | positives |
|---|---:|
| CD | 477 |
| HYP | 255 |
| MI | 560 |
| NORM | 839 |
| STTC | 499 |

## Method A

Method A trains EfficientNet1DV2 from scratch using only the 2000 real PTB-XL
training ECGs.

Output:

```text
/root/autodl-tmp/graduate_project/method_a_real2000_seed42
```

Test result:

```text
macro AUROC = 0.8433
macro AUPRC = 0.6234
```

Per-class:

| class | AUROC | AUPRC |
|---|---:|---:|
| CD | 0.8336 | 0.6474 |
| HYP | 0.7634 | 0.3223 |
| MI | 0.8183 | 0.5685 |
| NORM | 0.9155 | 0.8762 |
| STTC | 0.8855 | 0.7028 |

## Method B

Method B uses the same 2000 real PTB-XL training ECGs plus ECGTwin prompt-token
synthetic ECG.

Prompt-token setting:

```text
center = ptbxl
token shape = 5 x 4 x 768
n_token_vectors = 4
token_repeat = 1
sample_strategy = class_balanced
token training steps = 2000
```

Token output:

```text
/root/autodl-tmp/graduate_project/method_b_ptbxl_tokens_seed42_mv4/prompt_token_bank.pt
```

Reference cache:

```text
/root/autodl-tmp/graduate_project/method_b_ptbxl_prompt_cache_v1
```

The train2000 split produced 1971 usable ECGTwin latent records with super5
primary classes:

| primary class | count |
|---|---:|
| NORM | 816 |
| MI | 421 |
| STTC | 304 |
| CD | 313 |
| HYP | 117 |

Synthetic candidate generation:

```text
candidate samples = 12500
2500 per class before gate
DDPM steps = 25
generation script = scripts/ecgtwin_gen/generate_center_prompt_token_synth_batched.py
```

Gated synthetic pool:

```text
/root/autodl-tmp/graduate_project/method_b_synth_candidates_mv4_seed42/ptbxl/gated_max1000/gated_samples.npz
```

Gate result:

| class | candidates | accepted |
|---|---:|---:|
| CD | 2500 | 118 |
| HYP | 2500 | 246 |
| MI | 2500 | 1000 |
| NORM | 2500 | 1000 |
| STTC | 2500 | 1000 |
| total | 12500 | 3364 |

Classifier training:

```text
real PTB-XL train = 2000
synthetic train = 3364
sampling = WeightedRandomSampler, target synth:real = 1:1
```

Output:

```text
/root/autodl-tmp/graduate_project/method_b_real2000_plus_synth3364_mv4_seed42
```

Test result:

```text
macro AUROC = 0.8477
macro AUPRC = 0.6419
```

Per-class:

| class | AUROC | AUPRC |
|---|---:|---:|
| CD | 0.8339 | 0.6823 |
| HYP | 0.7759 | 0.3432 |
| MI | 0.8435 | 0.6430 |
| NORM | 0.9146 | 0.8844 |
| STTC | 0.8707 | 0.6565 |

## Comparison

| run | train real | train synth | AUROC | AUPRC | delta AUROC | delta AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Method A | 2000 | 0 | 0.8433 | 0.6234 | baseline | baseline |
| Method B | 2000 | 3364 | 0.8477 | 0.6419 | +0.0044 | +0.0185 |

Per-class deltas:

| class | delta AUROC | delta AUPRC |
|---|---:|---:|
| CD | +0.0004 | +0.0349 |
| HYP | +0.0124 | +0.0209 |
| MI | +0.0252 | +0.0745 |
| NORM | -0.0008 | +0.0082 |
| STTC | -0.0148 | -0.0464 |

## Interpretation

This first graduate-project run is positive overall:

```text
macro AUROC +0.0044
macro AUPRC +0.0185
```

The strongest gains are on MI, CD, and HYP. NORM is essentially stable. STTC
regressed, despite 1000 gated STTC samples, so the synthetic STTC prompt/gate is
the main risk to investigate next.

The result supports the current thesis direction: ECGTwin prompt-token synthetic
data can improve a low-sample PTB-XL EfficientNet1DV2 classifier when the
experiment is controlled to the same real train split and the same preprocessing.

## Next Checks

1. Repeat with seeds 43 and 44 to see whether the AUPRC gain survives split
   randomness.
2. Run a trusted-classes ablation using only NORM/MI/CD/HYP or only NORM/MI,
   because STTC is the only class with a clear regression.
3. Inspect STTC generated ECG visually and digitally; adjust prompt from generic
   `st-t change|st segment abnormality|t wave abnormality` if it is drifting into
   MI-like or nonspecific morphology.
4. Compare `5 x 1 x 768` and `5 x 2 x 768` tokens to verify that the 4-vector
   token is actually beneficial and not merely adding synthetic volume.
