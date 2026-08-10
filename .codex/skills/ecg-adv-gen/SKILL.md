---
name: ecg-adv-gen
description: Work on the ECG_manual_refactor clean-room PTB-XL Super5, PN2021/PN2021-C, EfficientNet1DV2, ECGFounder, AugMix, and VAE-LHAT pipeline. Use for implementation, protocol, evaluation, or evidence work in this graduation-project repository; treat legacy ECG_adv_Gen material as read-only provenance.
---

# ECG Manual Refactor Project Skill

Read `AGENTS.md` first. Use
`docs/refactor_cleanup/manual_refactor_keep_manifest.md` as the code and config
boundary, then use `configs/active_scripts.yaml` and
`configs/active_evidence_registry.yaml` for executable and evidentiary truth.

## Current Mainline Override

As of 2026-08-06, prefer this block over all historical notes below when they
conflict.

```text
repo root: /home/linbinhao/ECG_manual_refactor_clean
launcher: boot_scripts/run_experiment.py + configs/experiments/*.yaml
active index: configs/active_scripts.yaml:latest_mainline
method: target-only two-chain AugMix-SimCLR -> rotating depth2/3 -> contracted VAE-LHAT
mapping: v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51
class order: CD,HYP,MI,NORM,STTC
primary metric view: drop_all_zero
evidence: heldout-tuned single-seed development, not paper-final
```

Hard boundaries:

- New runtime code may use only keep-manifest files or artifacts produced by
  them.
- Do not import `ecg_adv_gen`, `methods`, `scripts`, or exploratory
  `agent_workspace` code.
- Adaptation is single-center K500 only. Exclude those identities from target
  evaluation.
- Keep CPSC 2018 and Extra as one logical center.
- Report v7 mapping/hash, source checkpoint, seed identity, config closure,
  optimizer-step budget, and drop-all-zero metrics.
- Do not promote development evidence without a frozen recipe and independent
  repeats.

## Environment

- Repo: `/home/linbinhao/ECG_manual_refactor_clean`
- Data/output root: `/home/linbinhao/ECG_adv_data`
- Python: `/home/linbinhao/miniforge3/envs/ECGTwin/bin/python`
- Shared hardware: check live CPU, memory, disk, port, process, and GPU state
  before resource-heavy work.
- Keep datasets, caches, checkpoints, generated signals, TensorBoard events,
  and run outputs outside Git.

## Historical ECG_adv_Gen Guidance

The remaining sections preserve old research context only. Their `/root`,
`ECG_adv_Gen`, removed-script, old-mapping, and old-result references are not
active commands or current evidence. Consult them only when the user explicitly
requests historical recovery, and recover code from Git history rather than
reintroducing it into the clean tree.

### Historical Graduation Project Story

Mainline method:

```text
PTB-XL super5 real ECG
-> EfficientNet1DV2 super5 baseline
-> ECGTwin author training reproduction (IBE + DiT) with thesis-grade logs
-> ECGTwin VAE latent manifold for target-center augmentation
-> TA-OMAT / synth-anchor ablations
-> PN2021 7-center AUROC/AUPRC evaluation
```

Current preferred thesis route:

```text
reproduce ECGTwin author pipeline as generative-framework evidence
-> use target-center real ECG anchors in ECGTwin VAE latent space
-> run same-label / real_all_present Latent-Hull online adversarial training
-> evaluate with PN2021 held-out ref-excluded and PN2021-C robustness tests
-> keep center prompt tokens and direct ECGTwin synthetic augmentation as
   generation-control / ablation evidence unless matched controls prove a
   downstream causal win
```

Do not make no-IBE ECGTwin self-training mandatory for the thesis mainline unless the user explicitly revives it. Do not make MIMIC mandatory; PTB-XL + PN2021 gives the clean closed loop.

Latest 2026-05-23 state:

- The strongest EfficientNet1DV2-side conclusion is still target-center
  real-anchor VAE Latent-Hull / TA-OMAT style adaptation, with center tokens as
  an ablation rather than the primary effect.
- The latest ECGFounder branch is a promising new mainline candidate:
  residual-adapter + VAE-only Latent-Hull online AT with source-logit anchor
  improves four target centers while preserving PTB-XL source performance.
  It needs multi-seed replication and fair comparison to EfficientNet1DV2.
- Recent ECGFounder/paper docs use PN2021 Super5 v5 strict mapping:
  `v5_super5_strict_voltage_pacing_suppress_20260522`. Freeze and report the
  mapping version/hash before final tables.
- Prefer the newest `docs/pipelines/*.md` facts over older command snippets in
  this skill when they conflict.

Current 2026-05 execution priorities:

1. rerun PN2021 v3 super5 clean evaluation for every cited EfficientNet1DV2 model;
2. implement ECGTwin center-class prompt tokens as learnable 768-d text embeddings, not tokenizer vocabulary hacks;
3. implement Latent-Hull TA-OMAT using constrained same-label VAE latent convex combinations;
4. build a PN2021-C corruption benchmark for `ningbo`, `chapman_shaoxing`, `cpsc_2018`, and `georgia`.

Current long run:

```text
/root/autodl-tmp/ecgtwin_author_repro/full_dit_main_accel_bf16_mbs512
Stage 1 IBE partial run saved through epoch 9; full author pipeline did not complete.
Config: effective batch 65536, micro batch 512, num_workers 8,
pin_memory + persistent_workers + prefetch_factor 2, bf16 AMP.
Metrics:
  epoch 1 train_loss=3.052135 eval_score=0.718906
  epoch 2 train_loss=1.694837 eval_score=0.657278
  epoch 3 train_loss=1.213244 eval_score=0.627823
  epoch 4 train_loss=0.919663 eval_score=0.625451
  epoch 5 train_loss=0.751632 eval_score=0.628861
  epoch 6 train_loss=0.634500 eval_score=0.611151
  epoch 7 train_loss=0.557742 eval_score=0.611722
  epoch 8 train_loss=0.498903 eval_score=0.616669
  epoch 9 train_loss=0.455224 eval_score=0.611132
Failure: after epoch 9 the main process was killed; orphan DataLoader workers
held stdout pipe/GPU memory until manually terminated. Do not rerun same worker
config blindly. Add resume support or retry with fewer workers/no persistent workers.
Saved: IBE_best.pth, best.pt, latest.pt, metrics.jsonl, loss_curve.csv/png.
```

Current queued 1-2h GPU task:

```text
No active GPU task at the moment.
Last completed GPU sequence:
  1. PN2021-C self-clean fill-in for nin_M5, extra_M5, geo_M20.
  2. Multi-center prompt-token v32 generation/gating for
     chapman_shaoxing, cpsc_2018, georgia.
  3. v33/v35 targeted boosts and v36 merged gated pools.
  4. v36 online-AT pilots for chapman, cpsc, georgia.
  5. Optional PN2021-C for synth K400 nin and prompt-token v14.
Result:
  chapman_v36 -> PN2021 0.8344 / 0.5554
  cpsc_v36    -> PN2021 0.8340 / 0.5553
  georgia_v36 -> PN2021 0.8343 / 0.5541
None beats prompt-token v4 (0.8344 / 0.5558) or real-anchor nin_M10
(0.8341 / 0.5560); skip PN2021-C for v36 pilots.
PN2021-C now includes 12 JSONs. Best absolute corrupted mean AUPRC:
  prompt-token v14 = 0.492808
  synth K400 nin   = 0.492707
  real-anchor nin_M10 = 0.492684
Baseline still has the lowest self-clean AUPRC drop, so enhanced models should
be reported as improving absolute corrupted performance, not relative drop.
```

Confirmed current choices:

- first target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`;
- use K=500 per center directly for center-token training/testing;
- K=500 ref ids must be excluded from downstream fine-tune validation/eval;
- Latent-Hull TA-OMAT first version uses same-label combination only;
- auto-scan existing `best_model.pt` runs before rerunning historical model evals.

## Key Paths

- Current summary: `docs/experiment_summary_for_advisor.md`
- Current ablation: `docs/module_ablation_1_real_vs_synth.md`
- Pipeline index: `docs/pipelines/README.md`
- Current auto plan: `docs/pipelines/next_auto_execution_plan.md`
- PN2021 v3 re-eval plan: `docs/pipelines/pn2021_v3_reevaluation_pipeline.md`
- ECGTwin center prompt-token plan: `docs/pipelines/ecgtwin_center_prompt_token_pipeline.md`
- Latent-Hull online AT plan: `docs/pipelines/latent_hull_online_at_pipeline.md`
- PN2021-C benchmark plan: `docs/pipelines/pn2021_c_corruption_benchmark_pipeline.md`
- Cleanup provenance: `docs/refactor_cleanup/final_cleanup_summary_20260705.md`
- PTB-XL raw/preprocessed: `/root/ECG_adv_Gen/datasets/PTBXL/`, `/root/autodl-tmp/ptbxl/`
- PTB-XL VAE/nomic cache: `/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt`
- PN2021: `/root/autodl-tmp/physionet2021/training/<center>/`
- MIMIC: `/root/autodl-tmp/MIMIC/`
- ECGTwin repo: `model/ECGTwin/`
- Latest managed launcher: `boot_scripts/run_experiment.py`
- Latest retained runtime roots: `boot_scripts/`, `core/`,
  `data_preprocess/`, `models/`, and `util/`

## ECGTwin Facts

- Original ECGTwin VAE input/output is `(B, 1024, 12)` channels-last, raw mV, MIMIC lead order.
- VAE latent is `(B, 4, 128)` and PTB-XL latent cache is already scaled by `0.18215`.
- ECGTwin MIMIC lead order swaps PTB-XL aVL/aVF:
  `ECGTWIN_TO_PTBXL_INDICES = [0,1,2,3,5,4,6,7,8,9,10,11]`.
- Before classifier training/eval: decode, reorder leads, resample 1024 to 1000 at 100 Hz, then use the existing classifier preprocessing.
- Original DiT text path uses `text_embed` `(B,L,768)`, `text_embed_mask` `(B,L)`, and patient info `(hr, age, sex)` through `text_projector + CrossAttention`.
- Original IBE path supplies `base_vector` for AdaLN modulation. Preserve this for ECGTwin author reproduction; no-IBE replacements are historical unless explicitly revived.
- Never set `text_embed_mask` all zeros; original cross-attention can softmax all `-inf` and produce NaN. Use a null text embedding with mask `1`.
- Official inference supports open-vocabulary diagnostic text through pretrained `bert-base-uncased` + `nomic-ai/nomic-embed-text-v1.5`; it does not natively train new tokenizer tokens.
- A center token written in the prompt should be implemented by a prompt compiler that appends/inserts a learnable 768-d embedding into `text_embed`, with mask value `1`.
- The removed center-token implementation was a 256-d AdaLN/base-vector hook,
  not a textual-inversion prompt token. Treat it as historical or ablation
  unless explicitly requested.
- Prompt-token code is historical/provenance for the current refactor. Its old
  script entrypoints are not latest-mainline launch surfaces.
- Historical prompt-token cache root was `/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/`; training outputs went under `/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/`.
- Historical prompt-token bank supports legacy single-vector, direct multi-vector,
  and factorized center+class+residual schemas. Direct MV4 is currently the best
  new schema; factorized MV4 and target-center-only training underperformed.
- The old prompt-token reference selector is not part of the public tree. The
  quality-ref seed1042 selection improved some class counts but did not beat
  old seed42 references downstream.
- Current prompt-token gated pool:
  `/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v2/ningbo/gated/`
  with 97/150 samples passed (NORM 41, MI 41, STTC 15).
- Current balanced prompt-token pool:
  `/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced/ningbo/gated/`
  with 131 samples (NORM 41, MI 41, STTC 49).
- Current best prompt-token downstream pilot:
  `/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20`
  with PN2021 v3 `0.8344 / 0.5558`.
- Other prompt-token attempts:
  v3 balanced repeat2 reached `0.8345 / 0.5556`;
  v4 M5 adv_weight=0.25 reached `0.8343 / 0.5555`;
  hybrid real200+prompt131 reached `0.8345 / 0.5549`;
  quality top40 reached `0.8343 / 0.5551`;
  source-aware hybrid reached `0.8342 / 0.5552`;
  mixed high/boundary selectors reached at best `0.8345 / 0.5547`;
  v4 reruns/adv_weight sweeps reached at best `0.8342 / 0.5554`;
  fresh v1-token expansion plus top50/class selection reached `0.8343 / 0.5548`;
  quality-ref direct/factorized MV4 reached at best `0.8340 / 0.5546`;
  seed42 direct MV4 with targeted MI/STTC boosts reached `0.8342 / 0.5558`;
  class-specific checkpoint composition improved gate to `119/240` but downstream
  reached only `0.8342 / 0.5546`;
  v31 mixed-diverse MV4 selected NORM/MI/STTC 50 each and reached
  `0.8340 / 0.5550`, with self-clean PN2021-C mean drop
  `0.022597 / 0.034711`.
- Conclusion: center-token samples are useful and reach the same gain band, but
  current best overall AUPRC remains real-anchor `nin_M10` at `0.8341 / 0.5560`.
  High-confidence-only prompt selection, mixed boundary selection, hybrid
  concatenation, source-aware hybrid sampling, and downstream adv_weight/seed
  sweeps all underperformed or only matched the original v4 balanced pool. Next
  center-token improvement should use gate-aware token checkpoint/probe selection
  plus diversity-preserving candidate selection before downstream training, not
  blind token_repeat, target-center-only training, M, adv_weight, or post-hoc
  pool expansion sweeps.
- `base_vector` should still come from the ECGTwin reference ECG/text/patient info path. Do not pollute `base_vector` with target-center style; put target-center style in the text token path.
- Best currently validated direct prompts are NORM, MI, and STTC. HYP and CD should be trained/validated as tokens, but kept out of first production synthetic augmentation unless digital gates pass.
- For HYP/CD, first check whether the ECGTwin author inference path can run the official-style prompts and generate nonbroken ECG. Treat "can generate" and "passes medical digital gates" as separate claims.

Current prompt fragments:

| class | fragment | status |
|---|---|---|
| NORM | `sinus rhythm|normal ecg.` | main |
| MI | `stemi|st elevation myocardial infarction|acute` | main, may express Q-wave MI |
| STTC | `nstemi|non st elevation|t wave inversion` | main |
| HYP | `left ventricular hypertrophy|high voltage` | token/ablation until voltage gates pass |
| CD | `left bundle branch block|lbbb` | token/ablation until conduction gates pass |

## Super5 Labels

Class order is:

```text
CD, HYP, MI, NORM, STTC
```

Use `ecg_adv_gen/labels/super5.py` as the source of truth.

Hard rules:

- PTB-XL super5 uses `diagnostic_class` from `scp_statements.csv`, class order `CD,HYP,MI,NORM,STTC`.
- Current PTB-XL implementation includes diagnostic SCP keys with likelihood/confidence `0` because it checks `conf >= 0.0`; do not change this threshold without invalidating old label caches and rerunning label distribution sanity checks.
- If label mapping, confidence threshold, `scp_statements.csv`, or class order changes, delete/regenerate `ptbxl_labels.C5.all.npy` and record new per-class positives.
- PN2021 super5 uses a project-defined semantic projection, not an official PN2021->PTB-XL crosswalk. The current latest-mainline mapping is `v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`.
- Current managed PN2021 eval cache metadata must include the super5 mapping version/hash and must be rebuilt if mapping, parser, class order, or preprocessing changes.
- MIMIC super5 is weak regex labeling. Treat it as noisy external reference or optional pretraining/ablation, not as main supervised thesis evidence.

Historical Scheme B prompt table:

| class | prompt |
|---|---|
| NORM | `normal ecg|sinus rhythm` |
| MI | `myocardial infarction|pathological q wave` |
| STTC | `st-t change|st segment abnormality|t wave abnormality` |
| CD | `conduction disturbance|bundle branch block` |
| HYP | `ventricular hypertrophy|left ventricular hypertrophy` |

For multi-label samples, concatenate prompt fragments with `|`.

## PN2021 Evaluation Rules

- Main metrics: macro AUROC and macro AUPRC.
- Use PN2021 7 centers only:
  `chapman_shaoxing`, `cpsc_2018`, `cpsc_2018_extra`, `georgia`, `ningbo`, `ptb`, `st_petersburg_incart`.
- Hard-exclude `ptb-xl` / `ptbxl` shard because it leaks PTB-XL training data.
- Old PN2021 JSONs are historical unless they report the current v7 mapping metadata. Any cited latest-mainline result must carry `v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`.
- Current managed dry-run entry:

```bash
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml \
  --run-dir /home/linbinhao/ECG_adv_data/runs/dryrun_new_unique_name \
  --dry-run
```

Choose a `--run-dir` path that does not already exist. A dry-run validates the
resolved plan without creating that directory or loading data, a model, or a
GPU.

## Latent-Hull Online AT Rules

Use same-label, same-center or same-center-cluster ECGTwin latents only:

```text
z_mix = sum_i softmax(a_i) * z_i
z_adv = (1 - lambda) * z0 + lambda * z_mix
```

Do not assume arbitrary latent linear combinations preserve medical semantics. For the first version, keep labels fixed only when all mixed anchors share the same super5 vector or primary class and digital/teacher gates pass. Avoid cross-class mixtures unless a later explicit soft-label or union-label experiment is requested.

Current latest-mainline launch path:

```text
configs/active_scripts.yaml:latest_mainline
boot_scripts/run_experiment.py
boot_scripts/, core/, data_preprocess/, models/, util/
```

Default first-round settings:

```text
M in [5, 10, 20], main M=10
lambda=0.25
hull_steps=5
hull_lr=0.3
pgd_eps=2.0 L2 cap after convex-hull move
label_mode=primary first, exact as stricter ablation
```

Current best 2026-05 Latent-Hull AUPRC run:

```text
/root/autodl-tmp/latent_hull_super5_pilot/nin_M10_lambda025_ep20
PN2021 v3 = 0.8341 / 0.5560
PN2021-C stream mean corrupted AUPRC = 0.492684
PN2021-C self-clean mean drop = 0.022671 / 0.035135
```

Completed coefficient/lambda controls did not beat the main run:

```text
C0 one-hot  = 0.8341 / 0.5547
C1 uniform  = 0.8341 / 0.5549
C2 Dirichlet= 0.8344 / 0.5550, self-clean PN2021-C drop 0.022652 / 0.034737
lambda 0.10 = 0.8340 / 0.5546
lambda 0.40 = 0.8339 / 0.5552, self-clean PN2021-C drop 0.022677 / 0.035201
Keep C3 optimized softmax weights and lambda=0.25 as default.
```

Current prompt-token Latent-Hull pilot:

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20
PN2021 v3 = 0.8344 / 0.5558
PN2021-C stream eval done at eval_pn2021_c_stream_v2_all4.json; clean AUPRC improves,
but self-clean PN2021-C drop does not improve vs baseline/prompt v31.
Conclusion: usable generation->gate->online-AT loop; balanced gated pool improves
clean AUPRC but is not yet a relative robustness result.
```

## PN2021-C Rules

Use single-corruption benchmark copies, not AugMix training mixtures, for robustness reporting. First corruption set:

```text
powerline_noise, emg_noise, baseline_wander, baseline_shift, random_leads_masking
```

Target centers:

```text
ningbo, chapman_shaoxing, cpsc_2018, georgia
```

Recommended output root:

```text
/root/autodl-tmp/triple_labels/pn2021_c_cache/
```

Current PN2021-C CSV exports:

```text
/root/autodl-tmp/triple_labels/pn2021_c_run_summary.csv
/root/autodl-tmp/triple_labels/pn2021_c_corruption_summary.csv
/root/autodl-tmp/triple_labels/pn2021_c_severity_summary.csv
/root/autodl-tmp/triple_labels/pn2021_c_center_summary.csv
/root/autodl-tmp/triple_labels/pn2021_c_per_class_summary.csv
```

PN2021-C current self-clean ranking by mean AUPRC drop: clean baseline is lowest
drop (`0.034103`), prompt v31 is second (`0.034711`), and real-anchor
`nin_M10_lambda025` is fifth (`0.035135`) but has the best absolute corrupted
mean AUPRC (`0.492684`). Report Latent-Hull as improved corrupted absolute
performance, not reduced relative corruption sensitivity.

PN2021-C candidate coverage table:

```text
/root/autodl-tmp/triple_labels/pn2021_c_candidate_coverage.csv
```

High-priority missing PN2021-C when GPU is free:
`latent_hull_super5_pilot/nin_M5_lambda025_ep20`,
`latent_hull_super5_pilot/extra_M5_lambda025_ep20`,
`latent_hull_super5_pilot/geo_M20_lambda025_ep20`.

## Implementation Guidance

Keep original `model/ECGTwin/` intact. Historical managed business logic lived
in the following legacy package roots; they are provenance only and must not be
restored as current clean-room runtime dependencies:

```text
ecg_adv_gen/runner/       # historical managed package runners
ecg_adv_gen/config/       # historical YAML loading and launch audit
ecg_adv_gen/evaluation/   # historical PN2021/PN2021-C reports
ecg_adv_gen/labels/       # historical Super5 mapping metadata
```

Archived/historical no-IBE files should not be restored into the mainline without an explicit user request.

If no-IBE Scheme B is explicitly revived, the old expected layout was:

```text
model.py              # fork DiT_ECGTwin, remove IBE, keep text path
dataset.py            # single-sample PTB-XL latent dataset, not paired ECGTwin dataset
prompt_table.py       # super5 prompt table + nomic embedding cache
train.py              # diffusion training with AMP/EMA/logging
sample.py             # DDIM/CFG sampling and npz export
center_token.py       # optional per-block token after mainline works
train_center_token.py # optional target-center token training
```

Reuse from ECGTwin:

- `DiTBlock_ECGTwin`
- `CrossAttention`
- `TimestepEmbedder`
- `RoPEEmbedder`
- `PositionalEmbedder`
- `process_pat_info`
- `_pad_text_embed`
- VAE encoder/decoder checkpoint

For that historical no-IBE branch, delete or bypass:

- `IBExtractor`
- `ibe_path`
- paired ref/target training logic
- `base_vector` dependency

Preferred forward signature:

```python
def forward(
    self,
    x,
    t,
    y_super5,
    text_embed=None,
    text_embed_mask=None,
    pat_info=None,
    center_token=None,
):
    ...
```

## Logging Requirements

For every training/reproduction run, write artifacts useful for the thesis:

- `run_config.yaml`, `config.yaml`, or `run_config.json`
- `train.log`
- `metrics.jsonl` with step/epoch/loss/lr/grad_norm/val_loss if available
- `loss_curve.csv`
- `loss_curve.png`
- `checkpoints/best.pt` and `checkpoints/latest.pt`
- `samples/*.npz` for generated ECG
- `figures/*.png` for 12-lead examples
- downstream `eval_result.json` from PN2021 cross-center eval

Use stable output roots like:

```text
/root/autodl-tmp/ecgtwin_author_repro/<run_name>/
/root/autodl-tmp/triple_labels/<run_name>/
```

## 8-Hour Execution Policy

An 8-hour window is realistic only for a reduced proof-of-concept, not the full thesis package.

Prioritize:

1. keep ECGTwin author reproduction logs/checkpoints/curves complete;
2. keep EfficientNet1DV2 super5 baseline and PN2021 eval reproducible;
3. run target-anchor/TA-OMAT or synthetic-anchor ablations only with saved configs;
4. save logs, curves, metrics JSON/CSV, and checkpoint paths needed for the thesis.

Defer MIMIC pretraining, no-IBE revival, and center token unless the current ECGTwin reproduction + EfficientNet evaluation loop is stable.

## Classifier Augmentation Format

Simplest synthetic classifier input is `.npz`:

```text
signals: (N, 1000, 12) float32, PTB-XL lead order, 100 Hz, 10 s
labels:  (N, 5) float32, class order CD/HYP/MI/NORM/STTC
```

If an existing generator writes `(N, 12, 1000)`, transpose before feeding `train_ptbxl.py`.

`train_ptbxl.py --synth_npz` does not run filtering, z-score, or lead reorder on synthetic signals; it only normalizes shape and crops. Synthetic `.npz` must already be in PTB-XL lead order and on the classifier-preprocessed scale, or be explicitly documented as a raw-mV ablation.
