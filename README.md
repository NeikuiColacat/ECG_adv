# ECG_adv_Gen Graduate Project Branch

This worktree is the `graduate-project` branch for the final graduation-design
pipeline.

## Mainline

The active thesis route is:

```text
PTB-XL super5 low-sample split
-> EfficientNet1DV2 real2000 baseline
-> ECGTwin no-token hard-label pretrain + real2000 fine-tune
-> ECGTwin textual-inversion center-token hard-label pretrain + real2000 fine-tune
-> custom split / fold10 subset / PN2021 evaluation
-> final ablation + Streamlit demo
```

The low-sample custom split is:

```text
/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json
```

It uses `train=2000`, `val=2000`, and the remaining `17799` PTB-XL records as
the internal custom test set.

## Active Entrypoints

| purpose | path |
|---|---|
| Three-method low-sample reproduction | `scripts/final_round/run_low_sample_three_methods.sh` |
| Final medical-validity proxy ablation | `scripts/final_round/run_medical_validity_ablation.py` |
| Streamlit demo | `apps/streamlit_ecg_demo/app.py` |
| Inference export/benchmark | `scripts/deploy/` |
| EfficientNet1DV2 super5 training/eval | `scripts/triple_labels/` |
| ECGTwin prompt-token code | `methods/ecgtwin_gen/prompt_token/` |
| Prompt-token train/generate/gate | `scripts/ecgtwin_gen/train_center_prompt_tokens.py`, `scripts/ecgtwin_gen/generate_center_prompt_token_synth.py`, `scripts/ecgtwin_gen/gate_prompt_token_synth.py` |
| Latent-Hull / online AT ablations | `scripts/pgd_cross_center/` |

## Quick Commands

Use the project Python directly:

```bash
PY=/root/miniforge3/envs/ECGTwin/bin/python
cd /root/autodl-tmp/ECG_adv_Gen_graduate
```

Run the three-method low-sample reproduction:

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
bash scripts/final_round/run_low_sample_three_methods.sh
```

Run the Streamlit demo:

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
$PY -m streamlit run apps/streamlit_ecg_demo/app.py \
  --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

## Archive Boundary

Historical code outside the final route is kept under:

```text
trash/cleanup_20260506_legacy/
```

Shared helper directories such as `scripts/crosscenter_v2/`,
`scripts/crosscenter_tierM/`, and `methods/augmix/` are still present because
current super5 preprocessing, PN2021-C corruptions, and online-AT code import
them.

For the durable project memory, see `AGENTS.md`. For the branch-specific
mainline, see `docs/pipelines/graduate_project_branch_mainline.md`.
