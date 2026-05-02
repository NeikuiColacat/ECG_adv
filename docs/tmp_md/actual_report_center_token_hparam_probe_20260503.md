# Actual-Report Center Token Fix And Hyperparameter Probe

Date: 2026-05-03

## What Was Fixed

The previous PTB-XL Method B run had clinical report text in the cache, but the
ECGTwin IBE/base-vector path used the class fallback prompt instead of each
reference ECG's actual report embedding.

The code now separates:

```text
ref ECG latent + ref actual report + ref HR/age/sex -> IBE -> base_vector
target class prompt + learned class token vectors -> DiT text path
```

Touched files:

```text
methods/ecgtwin_gen/prompt_token/trainer.py
scripts/ecgtwin_gen/train_center_prompt_tokens.py
scripts/ecgtwin_gen/generate_center_prompt_token_synth_batched.py
scripts/triple_labels/train_ptbxl.py
docs/pipelines/graduate_project.md
```

## Cache Check

```text
cache = /root/autodl-tmp/graduate_project/method_b_ptbxl_prompt_cache_v1/center_full_latents/ptbxl.pt
records = 1971
text coverage = 1971 / 1971
text_embed item shape = (1, 768)
latent shape = (1971, 4, 128)
```

## Hyperparameter Probe

Small probes generated 200 candidates per class and then applied the same
digital/teacher gate.

| probe | kept/1000 | NORM | MI | STTC | HYP | CD |
|---|---:|---:|---:|---:|---:|---:|
| MV4 step2000, actual report, same-class ref | 413 | 139 | 128 | 122 | 20 | 4 |
| MV4 step4000, actual report, same-class ref | 404 | 103 | 142 | 132 | 26 | 1 |
| MV4 step4000, actual report, normal ref | 258 | 103 | 70 | 83 | 1 | 1 |
| MV4 step4000, actual report, random-any ref | 273 | 74 | 89 | 104 | 5 | 1 |
| MV8 step2000, actual report, same-class ref | 357 | 110 | 151 | 80 | 12 | 4 |

Current best probe choice:

```text
ref_text_mode       = actual_report
ref_class_policy    = same_class
n_token_vectors     = 4
checkpoint          = step2000
per_ref_cap         = 20
num_inference_steps = 25
```

## Full Downstream Run

Full actual-report generation:

```text
candidate pool = /root/autodl-tmp/graduate_project/method_b_synth_candidates_actual_report_mv4_step2000_seed42/
accepted = 3313 / 12500
NORM = 1000 / 2500
MI   = 1000 / 2500
STTC = 1000 / 2500
HYP  = 255  / 2500
CD   = 58   / 2500
```

EfficientNet1DV2 from-scratch run:

```text
run = /root/autodl-tmp/graduate_project/method_b_real2000_plus_synth3313_actual_report_mv4_step2000_seed42
train = PTB-XL real2000 + synthetic3313
synth_ratio = 1.0
checkpoint_metric = val AUROC
test AUROC/AUPRC = 0.8412 / 0.6239
```

## Interpretation

Against Method A real-only:

```text
Method A:              0.8433 / 0.6234
Actual-report Method B: 0.8412 / 0.6239
Delta:                -0.0021 / +0.0005
```

Against the old class-fallback Method B:

```text
Old Method B:          0.8477 / 0.6419
Actual-report Method B: 0.8412 / 0.6239
Delta:                -0.0065 / -0.0180
```

So the actual-report fix makes the ECGTwin path more faithful to the author's
base-vector design, but it does not improve downstream EfficientNet1DV2
performance on this seed42 split.

Recommended next experiment:

```text
Do not blindly scale actual-report synthesis.
First rerun class-fallback with the new same-class/per_ref_cap generator to
isolate whether the old gain came from class-fallback text or from pool
sampling/gating. Add diversity/duplicate reporting before any larger pool.
```
