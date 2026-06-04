# Class-Fallback New-Generator Control

Date: 2026-05-03

## Question

The old Method B result was positive:

```text
Method A real2000:                 0.8433 / 0.6234
Old class-fallback Method B:       0.8477 / 0.6419
```

But after fixing ECGTwin's clinical-report path, actual-report Method B was not
positive:

```text
Actual-report Method B:            0.8412 / 0.6239
```

This control checks whether the old gain came from class-fallback text itself or
from the old generated pool / sampling / gate composition.

## Control Setup

Use the old class-fallback MV4 token bank, but generate with the new generator:

```text
token_bank = /root/autodl-tmp/graduate_project/method_b_ptbxl_tokens_seed42_mv4/prompt_token_bank.pt
ref_text_mode = class_fallback
ref_class_policy = same_class
per_ref_cap = 20
n_per_class = 2500
steps = 25
seed = 42
```

Output:

```text
candidate pool = /root/autodl-tmp/graduate_project/method_b_synth_candidates_classfallback_newgen_sameclass_cap20_seed42/
gated pool = /root/autodl-tmp/graduate_project/method_b_synth_candidates_classfallback_newgen_sameclass_cap20_seed42/ptbxl/gated_max1000/gated_samples.npz
downstream run = /root/autodl-tmp/graduate_project/method_b_real2000_plus_synth3391_classfallback_newgen_sameclass_cap20_seed42
```

## Gate Result

| pool | total accepted | NORM | MI | STTC | HYP | CD |
|---|---:|---:|---:|---:|---:|---:|
| old class-fallback | 3364 | 1000 | 1000 | 1000 | 246 | 118 |
| new class-fallback | 3391 | 1000 | 1000 | 1000 | 259 | 132 |
| actual-report | 3313 | 1000 | 1000 | 1000 | 255 | 58 |

The accepted count distribution is very close between old and new
class-fallback pools, so downstream differences are likely due to sample
identity/diversity/quality rather than simple class counts.

## Downstream Result

EfficientNet1DV2 from scratch, same seed42 PTB-XL split, same preprocessing,
same `synth_ratio=1.0`:

```text
best_model_auroc.pt: test AUROC/AUPRC = 0.8432 / 0.6065
best_model_auprc.pt: test AUROC/AUPRC = 0.8414 / 0.6042
```

Per-class AUPRC for the AUROC checkpoint:

| class | AUPRC |
|---|---:|
| CD | 0.5975 |
| HYP | 0.3331 |
| MI | 0.5142 |
| NORM | 0.8864 |
| STTC | 0.7015 |

## Interpretation

The new-generator class-fallback control did not reproduce the old positive
Method B result.

Compared with Method A:

```text
AUROC -0.0001
AUPRC -0.0169
```

Compared with old class-fallback Method B:

```text
AUROC -0.0045
AUPRC -0.0354
```

Conclusion:

```text
The old 0.6419 AUPRC should be treated as a pool-specific positive ablation,
not as reliable proof that class-fallback text by itself improves the
classifier.
```

Recommended next step:

```text
Compare old vs new class-fallback pools at sample level:
  1. reference reuse distribution;
  2. feature diversity / nearest-neighbor duplicate rate;
  3. per-class victim probability distributions;
  4. digital sanity metrics by accepted sample;
  5. whether old pool contains a lucky MI/CD/HYP subset that new per_ref_cap
     sampling removed.
```
