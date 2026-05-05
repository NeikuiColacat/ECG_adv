# Graduate Project Branch Mainline

This branch is organized around the final graduation-project evidence path, not
around every historical exploration in the repository.

## Active Claim

The low-sample thesis result compares three EfficientNet1DV2 super5 routes on
the same PTB-XL random record split:

| arm | role |
|---|---|
| `real2000` | 2000 real PTB-XL records, random-init baseline |
| `no-token hard -> real2000 FT` | ECGTwin synthetic pretrain without center token, then clean real2000 fine-tune |
| `center-token hard -> real2000 FT` | ECGTwin center-class prompt-token synthetic pretrain, then clean real2000 fine-tune |

The primary low-sample comparison metric is the custom split test set from:

```text
/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json
```

That split is `train=2000`, `val=2000`, and `test=17799` remaining PTB-XL
records. It is an internal record-level low-sample comparison split, not the
official PTB-XL fold10 protocol.

Current best low-sample center-token run:

```text
/root/autodl-tmp/graduate_project/self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc
custom AUROC/AUPRC ~= 0.8735 / 0.7013
```

Strict matched no-token hard-label control:

```text
/root/autodl-tmp/graduate_project/self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc
custom AUROC/AUPRC ~= 0.8669 / 0.6828
```

Thesis wording should stay scoped: center-token is strongest on the custom
low-sample split, while no-token controls can match or exceed it on some
fold10/PN2021 auxiliary evaluations.

## Active Entrypoints

| purpose | path |
|---|---|
| Three-arm low-sample reproduction | `scripts/final_round/run_low_sample_three_methods.sh` |
| Final medical-validity proxy ablation | `scripts/final_round/run_medical_validity_ablation.py` |
| Streamlit demonstration | `apps/streamlit_ecg_demo/app.py` |
| ONNX export / TensorRT scripts / backend benchmark | `scripts/deploy/` |
| EfficientNet1DV2 super5 train/eval | `scripts/triple_labels/` |
| ECGTwin textual-inversion prompt-token implementation | `methods/ecgtwin_gen/prompt_token/` |
| Prompt-token train/generate/gate scripts | `scripts/ecgtwin_gen/train_center_prompt_tokens.py`, `scripts/ecgtwin_gen/generate_center_prompt_token_synth.py`, `scripts/ecgtwin_gen/gate_prompt_token_synth.py` |
| Latent-Hull / online AT cross-center ablations | `scripts/pgd_cross_center/` |

Run the three-arm low-sample reproduction with:

```bash
cd /root/autodl-tmp/ECG_adv_Gen_graduate
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
RUN_TAG=20260506_final \
bash scripts/final_round/run_low_sample_three_methods.sh
```

## Archive Boundary

Historical branches that are not part of the final route were moved under:

```text
trash/cleanup_20260506_legacy/
```

This includes old 256-d AdaLN/base-vector center-token hooks, style-translator
experiments, old top-level runner scripts, old CT-v2 synth-anchor pilots, and
temporary markdown reports. Shared helper packages such as
`scripts/crosscenter_v2/`, `scripts/crosscenter_tierM/`, and `methods/augmix/`
remain in the active tree because current super5 preprocessing, PN2021-C
corruptions, and online-AT scripts still import them.

Do not restore archived code into the active tree unless a new experiment
explicitly revives that historical route.
