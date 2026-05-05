# ECG Adversarial Generation Project

## Overview
This worktree is the `graduate-project` branch. Its active thesis pipeline is:

1. PTB-XL super5 low-sample EfficientNet1DV2 baseline.
2. ECGTwin no-token hard-label synthetic pretrain, then real2000 fine-tune.
3. ECGTwin textual-inversion center-token hard-label synthetic pretrain, then
   real2000 fine-tune.
4. Fold10/PN2021 auxiliary evaluation, final medical-validity proxy ablation,
   and Streamlit demo/deploy artifacts.

Historical AugMix runners, old 256-d center-token hooks, style-translator
experiments, and old top-level scripts are archived under
`trash/cleanup_20260506_legacy/`.

## Environment
- Python: `/root/miniforge3/envs/ECGTwin/bin/python`
- Conda env: `ECGTwin` (activation not needed, use full python path)
- GPU: RTX 4090 24 GB
- Root disk: ~4 GB free → store large files in `/root/autodl-tmp/` (~28 GB free)

## Key Paths
- Current worktree: `/root/autodl-tmp/ECG_adv_Gen_graduate`
- PTB-XL data: `/root/ECG_adv_Gen/datasets/PTBXL/` (raw) + `/root/autodl-tmp/ptbxl/` (preprocessed cache)
- PTB-XL VAE-encoded: `/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt`
- PN2021: `/root/autodl-tmp/physionet2021/training/<center>/`
- MIMIC: `/root/autodl-tmp/MIMIC/` (cart reports + signal cache)
- EfficientNet 77-class JIT: `/root/ECG_adv_Gen/model/DeepECG/weights/efficientnetv2_77_classes/efficientnet_deepecg_unscaled.pt`
- ECGTwin: loaded via `util/ecgtwin_utils.py::ECGTwinWrapper`
- Trained checkpoints (large, on autodl-tmp):
  - v2 26-class baseline: `/root/autodl-tmp/crosscenter_v2/best_model.pt`
  - Tier-M 6-class victim: `/root/autodl-tmp/crosscenter_tierM/best_model.pt`
  - Triple-label heads: `/root/autodl-tmp/triple_labels/{super5,sub23,pn26}/best_model.pt`
- Outputs: keep large artifacts under `/root/autodl-tmp/`
- Docs: `/root/autodl-tmp/ECG_adv_Gen_graduate/docs/`
- Legacy archive: `/root/autodl-tmp/ECG_adv_Gen_graduate/trash/`

## Running Scripts
Always use the full Python path:
```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/train_ptbxl.py --scheme super5
```

## Architecture

### Core classifier pipelines
- `scripts/triple_labels/` — EfficientNet1DV2 super5 training/evaluation,
  including the custom train2000/val2000/test17799 split and PN2021 evaluation.
- `scripts/final_round/` — final reproduction and ablation entrypoints.
- `apps/streamlit_ecg_demo/` and `scripts/deploy/` — graduation demo and
  inference backend export/benchmark tooling.

### Adversarial / augmentation
- `adversarial/` — VAE-latent PGD (Mode A) attack pipeline
  - `pgd_advdiff.py` — PGD generator (replaces failed BoundaryAdvDiff)
  - `efficientnet_victim_tierM.py` — frozen Tier-M victim wrapper
  - `adv_validation.py` — ASR + medical-semantic gates
    (ASR overall ≥ 70%, per-class ≥ 30%; Einthoven residual p95 < 0.5)
  - `tierM_labels.py` — ECGTwin prompts/dirs; class list **re-exported**
    from `scripts/crosscenter_v2/label_alignment_v2.py::TIER_M`
    (single source of truth — never redefine the list here)
  - `efficientnet_victim.py` / `efficientnet_adapter.py` / `adv_generate.py`
    / `finetune.py` — older 77-class JIT-adapter path (kept; not on the
    Tier-M flow)
- `scripts/pgd_cross_center/` — PGD adv-buffer generator (driver for `adversarial/pgd_advdiff.py`)
- `scripts/{augmix_validation,augmix_adv_combo,online_vs_offline}/` — ablation runners

### Synthesis
- `methods/ecgtwin_gen/prompt_token/` — active textual-inversion center-class
  prompt-token implementation.
- `scripts/ecgtwin_gen/train_center_prompt_tokens.py` — active token training.
- `scripts/ecgtwin_gen/generate_center_prompt_token_synth.py` — active
  prompt-token generation.
- `scripts/ecgtwin_gen/gate_prompt_token_synth.py` — active quality-gated export.

### Other
- `sub_experiment/tsne_clustering_v1/` — t-SNE clustering ablation
- `util/` — shared utilities: `ecgtwin_utils.py::ECGTwinWrapper`,
  `ecg_viz.py` (12-lead plot, `engine={auto,ecgplot,matplotlib}`),
  `lead_mapping.py` / `lead_utils.py` (`ECGTWIN_TO_PTBXL_INDICES`)
- `model/ECGTwin/` — ECGTwin DiT + VAE (submodule, gitignored under `model/`)
- `model/DeepECG/` — DeepECG EfficientNet JIT (gitignored)
- `docs/` — technical docs + per-experiment reports
- `docs/papers/` — reference paper PDFs (gitignored as too large; only
  `INDEX.md` is tracked)
- `trash/` — archived legacy code (not on any active path)

## Important Notes

### Data integrity
- **PN2021 cross-center eval must hard-exclude the `ptb-xl` shard**
  (`/root/autodl-tmp/physionet2021/training/ptb-xl/` is exactly PTB-XL
  training data). Use `assert center.lower() not in {'ptb-xl', 'ptbxl'}`
  in any new eval loop. ningbo `ptb` is the older small dataset, allowed.
- **Single source of truth for SNOMED parsing**: `parse_header_snomed`
  lives in `scripts/triple_labels/eval_crosscenter.py:53`. The PN2021 .hea
  format uses `# Dx:` (with a space) — naive `startswith('#Dx:')` silently
  returns no codes → all-zero labels → all-NaN AUROC.
- **ningbo AF↔AFL quirk**: SNOMED 164890007 is officially AFL but is used
  for AF in ningbo. `_SNOMED_TO_IDX` patches it to map to both classes.
- **Super5 NORM exclusivity guard** (`label_schemes.py::snomed_list_to_super5`):
  PTB-XL trains NORM as "no abnormality" (4.65% co-occurrence with abnormal
  classes); PN2021 emits sinus rhythm (426783006) alongside pathology codes
  (98% co-occurrence on cpsc_2018_extra). Without the guard, NORM AUROC
  collapses on PN2021. Sub23 / PN26 schemes do not yet have an analogous
  guard — flag if NORM/NSR participates in macro.

### Project conventions
- **Tier-M class list canonical source**: `scripts/crosscenter_v2/label_alignment_v2.py::TIER_M`.
  `adversarial/tierM_labels.py` re-exports it. Don't redefine the 6-class
  list anywhere else.
- **Per-scheme PTBXL label cache** is keyed by class count
  (`*.C5.all.npy`, `*.C23.all.npy`, `*.C26.all.npy`) so different schemes
  can share an `--output_dir` without silently loading each other's array.
- **`util/test_wave.py`** is an autodl GPU-utilization keep-alive script
  (despite the "test_" prefix), already `.gitignore`d. Not a unittest.

### Numerical / framework gotchas
- EfficientNet 77-class is JIT-compiled (`torch.jit.load()`); supports autograd
  but weights are frozen. Input shape: `(B, 12, 2500)` @ 250 Hz, scaled by
  `MHI_FACTOR = 1 / 0.0048`.
- EfficientNet1DV2 (s_v2) used for Tier-M / Super5 / Sub23 / PN26 takes
  `(B, 12, 250)` @ 100 Hz (z-scored) directly; no MHI factor.
- VAE decoder has in-place `/= 0.18215` that breaks autograd → use the
  out-of-place division in `efficientnet_victim*.py::_decode_latent_differentiable`.
- ECGTwin lead order ≠ PTB-XL lead order. Reorder via
  `ECGTWIN_TO_PTBXL_INDICES` from `util/lead_utils.py` before feeding the
  victim or computing PTB-XL-shaped metrics.
- `unified_preprocess_to_1000(target_fs=100, target_len=1000, apply_filter=True, apply_zscore=True)`
  is the cross-dataset canonical preprocessing (per-record global z-score,
  not per-lead). Bypass with `apply_*=False` only for benchmark
  cross-comparison against external models that own their own scaler.

### ECGTwin generator (full spec — see `memory/ecgtwin_usage_guide.md`)

**Open-vocabulary**, not 12-class restricted. Trained on MIMIC-IV
`machine_measurements.csv` (800k × 18 GE/Marquette report cols, joined with
`|`). The 12 dirs in `model/ECGTwin/generation_result_by_disease/` are
demo-curated PNG galleries — **never confuse them with training classes**.
STTC content is abundant in training (~110k samples) so model has prior; the
`features.json` only exists for some 12 demo dirs.

**Prompt format (production `mix=False`, `config/DiT_ECGTwin.yaml`)**:
- pipe-separated short phrases, single Python str: `"phrase 1|phrase 2|..."`.
- `text.split('|')` is fed directly — `prompt_propcess()` is `mix=True` only,
  do NOT route through it for the production checkpoint.
- Each pipe-segment is encoded independently by nomic-embed-text-v1.5 → 768d
  → DiT 6-layer cross-attention treats each segment as a separate KV token.
- Tokenizer is `bert-base-uncased`; medical compounds get fragmented
  (`nstemi → ns/##tem/##i`, `lbbb → lb/##bb`).
- Do NOT add nomic instruction prefixes (`search_query:`, `clustering:`) —
  ECGTwin trained without them; symmetry matters.

**VAE I/O canonical** (`model/ECGTwin/module/vae_model.py`):
- Encoder input: `(B, L=1024, 12)` time-first, channels-last, raw mV (no
  z-score), MIMIC lead order, fs ≈ 102.4 Hz.
- Encoder output / Decoder input: `(B, 4, 128)` latent (already × 0.18215).
- Decoder output: `(B, 1024, 12)` raw mV, MIMIC order, time-first.
- L=1024 is **forced** by DiT positional encoding (encoder is fully-conv but
  the diffusion sampler hardcodes 128-token latent length).
- `0.18215` is SD-inherited, not ECG-recalibrated; resulting latent std ≈ 0.15.

**Lead order — MIMIC vs PTB-XL — only positions 4/5 swap (aVL↔aVF involution)**:
- ECGTwin MIMIC: `[I, II, III, aVR, aVF, aVL, V1..V6]`
- PTB-XL: `[I, II, III, aVR, aVL, aVF, V1..V6]`
- `ECGTWIN_TO_PTBXL_INDICES = [0,1,2,3,5,4,6,7,8,9,10,11]` — **its own inverse**.
- Forgetting the swap → silent ~5–10pp accuracy drop, no error.
- Forgetting fs (1024 @ 102.4 Hz vs classifier 1000 @ 100 Hz) → silent timing
  metric drift; resample 1024→1000 (linear interp, RMSE 0.02 mV) before
  feeding any 100Hz-trained classifier.

**Prebuilt PTB-XL latents already × 0.18215**: `PTBXL_vae_multi_nomic.pt[i]['data']`
is post-scaled z. Feed directly to `decode_latent`; do NOT re-scale.

**Channels-last vs channels-first**: ECGTwin VAE I/O is `(B, L, C)`; all our
classifiers are `(B, C, L)`. Always transpose at the boundary.

**`encode_ecg` is `@torch.no_grad`**: for gradient flow through the encoder,
call `wrapper.encoder(x)` directly. For gradient flow through the decoder
(PGD), use `_decode_latent_differentiable` from
`adversarial/efficientnet_victim_tierM.py` — the in-place `/= 0.18215` in
the production decoder breaks autograd.

### ECGTwin conditioning: prompt vs base_vector vs CenterToken (relative magnitude)

5-ref STTC ablation (seed=42, 50 DDPM steps) measured at signal level:

| Manipulation | rel ‖Δsignal‖ | Class-control? |
|---|---|---|
| Prompt swap MI ↔ NORM | **1.28** | yes — clean victim verdict flip |
| Prompt swap MI ↔ STTC | **0.62** | yes — clean victim verdict flip |
| Prompt swap STTC-clean ↔ STTC-v1 (same family) | 0.44 | no — same victim verdict (~0.98) |
| base_vector swap across 5 STTC refs | ~0.4 | no — within-class identity drift |
| CenterToken swap (extra ↔ nin) | **0.006** | no — 3-decimal identical victim probs |
| CenterToken swap (extra ↔ zero) | **0.012** | no — vanilla ≈ hooked |

- **PROMPT** drives clinical class via cross-attention KV (un-gated residual).
- **BASE_VECTOR** (`ib_projector(IBE(ref))`) drives patient/morphology
  identity *within class* via AdaLN-Zero modulation (per-block shift/scale/gate).
- **CENTERTOKEN v1** was additive to AdaLN driver via `forward_pre_hook`, but at
  trained norms 0.18–0.20 it sat in the noise floor (~1% of signal). Plan
  Rev 12 (2026-04-27) dropped v1 after finding 5 trainer bugs (detach kills
  L_inv gradient; best.pth selection saves epoch-1 near-zero; 1 batch/epoch ×
  30 epochs total; L_inv variance rewards uniform improvement not improvement;
  raw L2 reg pulls token to 0). Historical +0.83~1.38pp gains in
  `pgd_cross_center_ablation.md` and `center_token_data_efficiency.md` need
  re-attribution because they used `_best.pth` with norm 0.076 (epoch-1
  near-zero), not the 1.04 `_latest.pth`.
- **CENTERTOKEN v2** uses `CenterTokenPerBlock` with sphere projection. It has
  smoking-gun signal movement and small downstream gains in
  `docs/center_token_v2_results.md`, but it remains an ablation/extension, not
  the main thesis claim. The current mainline should not depend on CT to make
  the EfficientNetV2 result work.
- Critically: **base_vector uses the *reference's own* `text_embed`, not the
  target prompt**. Swapping target prompt does NOT change base_vector. See
  `util/ecgtwin_utils.py:266-276` and `model/ECGTwin/module/IBExtractor.py`.

### ECGTwin super5 generation scope: 3 classes only (NORM / MI / STTC)

Digital GT validation (36 raw tensors × 5 µV/ms-quantitative criteria,
2026-04-27) showed only 3 of 5 super5 classes pass medical thresholds at
3/3 best-cell seeds. **Production pipeline generates only NORM, MI, STTC.**

Pass per author prompt + `data/prepared_input/normal_1.pt` ref:

| super5 | prompt | digital pass (3 seeds best-cell) |
|---|---|---|
| **NORM** | `sinus rhythm\|normal ecg.` | **3/3** (also tachy/AF prompts) |
| **MI**   | `stemi\|st elevation myocardial infarction\|acute` | **3/3** (Q-path or 332 µV STE) |
| **STTC** | `nstemi\|non st elevation\|t wave inversion` | **3/3** (STD J+60 ≤ -50 µV in I/aVL) |
| HYP | `left ventricular hypertrophy\|high voltage` | **0/3** Sokolow 0.97-2.15 mV (need >3.5) |
| CD-LBBB | `left bundle branch block\|lbbb` | **0/3** QRS 132-176 ms passes L1; lateral R 0.08-0.22 mV fails L2 (need ≥0.5) |
| CD-RBBB | `right bundle branch block\|rbbb` | 0/3 (rsR' 1/3, QRS inconsistent) |
| CD-AVB | `atrioventricular block\|av block` | 0/3 (PR ≤ 88 ms, never crosses 200 ms) |

**Why HYP/CD fail (not an ECGTwin defect, a ref-amplitude inheritance fact)**:
`normal_1.pt` ref is a real normal female with Sokolow=1.51 mV. ECGTwin
**prompt controls morphology, base_vector controls absolute voltage**.
With normal-amplitude ref, prompt nudges voltage at most ~40% (1.51→2.15 mV)
— base_vector dominates. To reach LVH voltage threshold one would need
matched-class (LVH) ref + LVH prompt (untested; future work). CD has the
same lateral-R-amp issue plus inconsistent QRS-broadening for RBBB and
zero PR-prolongation capability for AVB.

**H4 per-class trust gate is therefore not transitional but structural**:
- `class_trust = {"NORM":1.0, "MI":1.0, "STTC":1.0, "HYP":0.0, "CD":0.0}`
- Pipeline never generates HYP/CD synth → never enters PGD adv buffer
- Super5 victim still evaluates HYP/CD at eval time (real PTB-XL records);
  AT just doesn't push HYP/CD adv signal

**Files**:
- Digital extractor: `util/ecg_digital_features.py`
- Quantitative thresholds spec: `docs/ecg_digital_thresholds.md`
- Validation report: `docs/ecgtwin_super5_digital_gt_validation.md`
- Memory: `memory/ecgtwin_super5_class_support.md`

**Author demo files in `model/ECGTwin/generation_result_by_disease/`** *do*
include HYP/CD demos with `features.json`, but those are 50-PNG curated
galleries (likely cherry-picked best-of-N). Sampling 3 random seeds with
the same prompt+ref shows the true distribution and fails strict digital
criteria. "Author demo exists" ≠ "any seed passes µV/ms thresholds".

### ECGTwin landmine: STTC reverse-supervision

Synth-victim STTC AUROC 0.115/0.205 in pilot is **NOT a vocabulary gap**
(model has 110k+ STTC training samples; ECGTwin is open-vocab). It's also
**not "prompt is architecturally a no-op"** — prompt swap MI↔STTC moves rel
0.62 of signal. The bit-exact AUROC across 3 STTC sub-variants happens
because (a) all 3 prompts share `t wave inversion` + `repolarization
abnormality` cores → nomic cos > 0.9 → same cross-attn neighborhood + (b)
the Super5 victim's STTC head fires at p≈0.98 on any "T-inverted-ST-depressed"
morphology regardless of sub-phrasing. The signals ARE different (rel 0.44
between cleanest and v1) but rank-equivalent for the victim.

**Real fix paths** (priority order, see `memory/ecgtwin_prompt_vs_basevector.md`):
1. **Prompt orthogonalization in nomic space**: `e_STTC_pure = e_STTC −
   α·proj(e_STTC → e_MI)`. Push out of MI manifold by Gram-Schmidt, not
   lexical synonyms.
2. **Per-class base_vector token** — train 5 learnable 256-d vectors added to
   `base_vector` at synth time, with class-discriminative loss + target norm
   ≥ 1.0 (vs current CT's 0.20). This is the **untapped lever**: AdaLN
   modulation channel for class control. CenterToken touches it but is
   class-agnostic and weakly trained.
3. **Replace IBE base_vector with class prototypes** for ambiguous refs:
   mean-pool over per-class STTC training records; bypasses learn-vs-discover
   tradeoff.
4. **H4 per-class trust gate** at AT training time (drop adv contributions
   on classes with synth-victim AUROC < 0.55). Cheapest fallback;
   sidesteps STTC entanglement at the cost of losing STTC adv signal.
5. **Don't tune STTC prompt micro-text further** — sub-variants with cos > 0.9
   produce victim-equivalent verdicts. Stop wasting compute on lexical fiddling.

See `memory/ecgtwin_prompt_vs_basevector.md` for the full ablation.
