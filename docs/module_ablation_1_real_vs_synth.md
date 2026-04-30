# Module Ablation Study #1: Real-Anchored vs Synth-Anchored PGD

**Date**: 2026-04-27
**Hypothesis**: ECGTwin diffusion synthesis is dispensable; pure VAE-latent PGD on real K=200 target-center records produces equal or better cross-center signal.

**Configuration** (single variable changed: anchor source):
- Both: K=200 ref pool per cell, K_anchor=300, K_pgd=10, ε=2.0, adv_w=0.5, mix `1.0:0.5:2.0` (PTBXL:roundtrip:adv)
- **Synth (Plan Rev 13.2 iter 4)**: ECGTwin DDPM 50-step → synth pool 300/cell → PGD on synth latents
- **Real (M1)**: K=200 real records → VAE encode → PGD on real latents (NO diffusion, NO CenterToken)
- 3 cells: cpsc_2018_extra (extra), ningbo (nin), georgia (geo)

## Headline result

| metric | synth iter 4 | M1 real-anchored | Δ (M1 − synth) |
|---|---|---|---|
| **PN2021 avg AUROC Δ** (3-cell mean) | **-0.14pp** | **-0.07pp** | **+0.07pp** ✓ |
| PN2021 avg AUPRC Δ (3-cell mean) | +0.14pp | +0.10pp | -0.04pp |
| PTBXL fold10 AUROC Δ | -0.02pp | -0.01pp | +0.01pp ✓ |
| MIMIC zero-shot AUROC Δ | +0.28pp | +0.30pp | +0.02pp ✓ |
| **cpsc_2018 cell drag** (target driving avg fail) | **-1.13pp** | **-0.86pp** | **+0.27pp** ✓ |
| Single-epoch wall clock | 150s | 25s | **6× faster** ✓ |

→ **M1 strictly improves average AUROC on all 3 cells** (extra: -0.08 vs -0.17, nin: -0.05 vs -0.12, geo: -0.08 vs -0.13). M1 reduces structural cpsc_2018 STTC drag by ~+0.27pp average, with extra cell showing the largest improvement (-0.65 vs -1.63 = **+0.98pp better**).

→ M1 trades slightly weaker peak signal on chap_shaoxing (avg +0.34 vs synth +0.39pp) for less cpsc_2018 reverse-supervision and 6× faster training.

→ Strict avg gate (Δ AUROC > +0.30pp AND Δ AUPRC > +0.50pp): both 0/3 pass — but M1 closer to threshold.

## Per-cell head-to-head — PN2021 avg

### AUROC delta vs baseline 0.8387

| cell | synth iter 4 | M1 real | Δ |
|---|---|---|---|
| extra | -0.17pp | **-0.08pp** | **+0.09pp** ✓ |
| nin | -0.12pp | **-0.05pp** | **+0.07pp** ✓ |
| geo | -0.13pp | **-0.08pp** | **+0.05pp** ✓ |

### AUPRC delta vs baseline 0.5888

| cell | synth iter 4 | M1 real | Δ |
|---|---|---|---|
| extra | +0.06pp | **+0.16pp** | +0.10pp ✓ |
| nin | +0.19pp | +0.11pp | -0.08pp |
| geo | +0.18pp | +0.03pp | -0.15pp |

→ **AUROC**: M1 strictly better on all 3 cells.
→ **AUPRC**: extra cell M1 wins; nin/geo synth slightly better.

## Per-target signal head-to-head

### chap_shaoxing AUROC (n=10247, baseline 0.8835)

| cell-run | synth iter 4 | M1 real |
|---|---|---|
| extra | +0.49pp | +0.33pp |
| nin | +0.28pp | +0.32pp |
| geo | +0.40pp | +0.38pp |
| **avg** | **+0.39pp** | **+0.34pp** | (synth +0.05pp better) |

### ningbo AUROC (n=34905, baseline 0.8820)

| cell-run | synth iter 4 | M1 real |
|---|---|---|
| extra | +0.49pp | +0.33pp |
| nin | +0.25pp | +0.32pp |
| geo | +0.38pp | +0.40pp |
| **avg** | **+0.37pp** | **+0.35pp** | (synth +0.02pp better) |

### MIMIC zero-shot AUROC (baseline 0.7857)

| cell-run | synth iter 4 | M1 real |
|---|---|---|
| extra | +0.33pp | +0.20pp |
| nin | +0.23pp | +0.36pp |
| geo | +0.27pp | +0.33pp |
| **avg** | **+0.28pp** | **+0.30pp** | **M1 +0.02pp better** ✓ |

### georgia AUROC (n=10344, baseline 0.8168)

| cell-run | synth iter 4 | M1 real |
|---|---|---|
| extra | +0.38pp | +0.26pp |
| nin | +0.22pp | +0.32pp |
| geo | +0.24pp | +0.29pp |
| **avg** | **+0.28pp** | **+0.29pp** | M1 +0.01pp better |

→ Per-target story preserved. Both pipelines produce consistent positive signal on 4 cross-center targets (chap +0.34-0.39, ningbo +0.35-0.37, MIMIC +0.28-0.30, georgia +0.28-0.29). Plan Rev 13.2 case (b) "cross-center per-target generalization" framing applies to both.

### cpsc_2018 AUROC (baseline 0.8052, n=6877, 3-class only — STTC vocab gap)

| cell-run | synth iter 4 | M1 real |
|---|---|---|
| extra | -1.63pp | -0.65pp |
| nin | -0.83pp | -0.88pp |
| geo | -0.95pp | -1.06pp |
| **avg** | **-1.13pp** | **-0.86pp** | **M1 +0.27pp less drag** ✓ |

→ M1 reduces structural drag, especially extra cell (-0.65 vs -1.63 = **+0.98pp better**). The reduction is consistent with the hypothesis that ECGTwin's STTC reverse-supervision (memory `ecgtwin_prompt_vs_basevector.md`: "STTC AUROC bit-exact across 3 prompts" — synth-victim STTC trust=0 in synth pipeline didn't actually prevent reverse signal because of class entanglement) is unique to the diffusion path. Real anchors carry true class identity so PGD attacks the right boundary.

## STTC class behavior (cpsc_2018 specifically)

| cell | iter 4 STTC drag | M1 STTC drag |
|---|---|---|
| extra | -4.82pp | **-2.06pp** | M1 **+2.76pp** less drag |
| nin | -2.49pp | -2.75pp |
| geo | -2.91pp | -3.27pp |

→ extra cell: M1 cuts STTC drag in half. nin/geo: similar. The ECGTwin "nstemi" prompt MI-shaped synth was specifically driving extra cell's STTC reverse-supervision; real anchors don't have this confusion.

### STTC peak on chap_shaoxing (extra cell)

| | iter 4 STTC chap | M1 STTC chap |
|---|---|---|
| extra | +1.03pp | +0.48pp |

→ M1 trades less peak STTC signal at chap for less cpsc_2018 STTC drag. Net effect on avg favors M1.

## Method-side diagnostics

| metric | synth iter 4 | M1 real |
|---|---|---|
| ASR_overall | 0.99-1.00 | **0.98-1.00** |
| Einthoven residual p95 | 0.094-0.108 | 0.222-0.267 |
| medical_pass | True every epoch | True every epoch |
| GPU memory peak | ~14GB | ~14GB |
| **Single epoch** | **150s** | **25s (6×)** |

→ M1 Einthoven p95 ~2× higher than synth (0.22 vs 0.09) — real anchors decoded after PGD have larger residual but still well under 0.5 medical gate. This is expected: synth latents start "near boundary" (already class-curated) while real anchors are deeper in the manifold and PGD pushes more.

## Per-cell early-stop / convergence

| cell | iter 4 ep_run | iter 4 best_ep | M1 ep_run | M1 best_ep |
|---|---|---|---|---|
| extra | 66 | 45 | 39 | 18 |
| nin | 42 | 21 | 51 | 30 |
| geo | 42 | 21 | 72 | 51 |

→ M1 best_ep occurs later in some cells (geo ep51 vs synth ep21) — real anchors require more training but reach final eval comparable or better.

## Decision

### **DROP DIFFUSION FROM THIS LINEAGE** ✓

Justification:
1. **Strictly better AUROC** on all 3 cells (avg +0.07pp delta in M1's favor)
2. **6× faster training** (no DDPM in loop, no Stage 1 generation)
3. **Simpler pipeline** (no ECGTwin, no CenterToken, no Stage 0.4 sanity gate, no class_trust map)
4. **Less reverse-supervision** on cpsc_2018 STTC structural drag (-0.86 vs -1.13pp avg)
5. **Same per-target story preserved** (chap/nin/MIMIC/georgia all +0.28-0.39pp consistent)
6. **Better in-domain protection** (PTBXL -0.01 vs -0.02pp)
7. **No memorization audit needed** (no synth → no Dar 2025 LDM concern; only standard PGD audit applies)

Trade-offs accepted:
- AUPRC mixed (extra +0.10pp better, nin/geo -0.08/-0.15pp worse) — net -0.04pp
- Slightly less peak STTC signal on chap_shaoxing (loses +0.55pp at peak)
- Memorization risk reappears (real K=200 patients used as anchors) — but anchors are PUBLIC PN2021 data, MIA risk acceptable; no diffusion-memorization concern

### Updated paper framing

Was: "synth-anchored on-manifold AT with ECGTwin diffusion"
Now: "**Target-Anchored On-Manifold Adversarial Training (TA-OMAT)** — VAE-latent PGD on K=200 real target-center records"

Backed by:
- Wong & Kolter ICLR 2021 (perturbation set learning) — closest method analog
- Stutz CVPR 2019 (on-manifold AT improves generalization)
- UCAT (TIP 2022/23) — anchored AT for cross-domain
- Madry 2018 (PGD core)

Not needed any more:
- TokenVerse SIGGRAPH'25 (CenterToken backing — drop, never worked anyway per Plan Rev 12)
- Dar 2025 Nat Biomed Eng (LDM memorization — drop, no LDM)
- Wilde MIDL 2024 (medical TI — drop, no TI)
- DataDream ECCV 2024 (K-shot synth — drop, no synth)

Still needed:
- Wang 2023 ICML DM-Improves-AT (frozen pool + per-epoch PGD — same scaffold)
- ADR ICLR 2024 (EWA anchor for AT)
- Han 2020 Nat Med (ECG adversarial baseline)
- Yang/La Cava 2025 (only ECG manifold AT — strict comparison)

### Next actions

- (a) **Confirmatory full 5-center pilot** — run real-anchored on all 5 PN2021 centers (chap_shaoxing, cpsc_2018, cpsc_2018_extra, georgia, ningbo) × K=200, ~5h compute (vs synth confirmatory ~7-9h). Lock decision gate at "≥3/5 cells improve PN2021 avg AUROC OR all per-target metrics positive".
- (b) **Module ablation #2** to test next: drop roundtrip_anchor (since real anchors carry their own clean reconstruction) — does w=0.5 roundtrip still help?
- (c) **K-sensitivity on real-anchored**: K∈{100, 200, 400} to check if larger real pool helps (cheap since no DDPM).
- (d) Memorization audit: cheap MIA-AUC test on K=200 anchors vs PN2021 holdout records (Song et al. Princeton DLS 2019 ATAT-MIA protocol).

## Files

- M1 pilot report: `docs/module_ablation_1_real_anchored_pilot.md`
- Synth iter 4 report: `docs/synth_anchored_super5_pilot_iter4.md`
- Synth final summary: `docs/synth_anchored_super5_pilot_summary_final.md`
- M1 prep script: `scripts/pgd_cross_center/prep_real_anchor_npz.py`
- M1 orchestrator: `scripts/pgd_cross_center/run_module_ablation_1_real_anchored.sh`
- M1 outputs: `/root/autodl-tmp/real_anchored_super5/{extra,nin,geo}_real_k200/`
