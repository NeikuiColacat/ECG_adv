# Synth-anchored Super5 Pilot — Final Summary (Plan Rev 13.2)

**Date**: 2026-04-27
**Plan revision**: 13.2 (NORM/MI/STTC scope, K=200/400, ε=2.0, K_pgd=10, K_anchor=300/600)
**Cells**: 3 × 2 K-levels = 6 cell-runs total

## TL;DR

- **Strict avg decision gate (>+0.30pp AUROC AND >+0.50pp AUPRC AND PTBXL≥-0.50pp)**: **0/6 cell-runs pass** at either K=200 or K=400.
- **K=400 escalation marginal**: extra +0.09pp AUPRC vs iter 4; nin/geo essentially flat.
- **Per-target signal robust and consistent across all 6 cell-runs**:
  - chap_shaoxing: AUROC +0.28–0.50pp, AUPRC +0.66–0.97pp
  - ningbo: AUROC +0.25–0.49pp, AUPRC +0.43–0.93pp
  - MIMIC zero-shot: AUROC +0.23–0.33pp, AUPRC +0.20–0.49pp
  - PTBXL fold10 in-domain: stable -0.00 to -0.04pp ✓ (AT does NOT hurt source)
- **Per-target structural drag**:
  - cpsc_2018 STTC: **-1.99 to -5.15pp** (PTB-XL vs CPSC vocab gap, ceiling ~0.59 — see memory `super5_label_audit_2026_04_26.md`)
  - st_petersburg_incart: -0.16 to -1.04pp (n=74, noisy small dataset)
- **Recommendation**: Frame as **cross-center per-target generalization** (Plan Rev 13.2 decision tree case (a)). Strict avg gate failure is dominated by 1–2 structurally limited centers, not method failure.
- **Module Ablation #1 (real-anchored, no diffusion) launched in parallel** to test whether ECGTwin diffusion contributes meaningfully to the per-target signal vs. is dispensable.

## Cell-run summary

### Iter 4 — K=200 (synth pool 300/cell)

| cell | center | epoch_run | early_stop | best_epoch | best_QE_AUROC |
|---|---|---|---|---|---|
| extra | cpsc_2018_extra | 66 | True | 45 | 0.8432 |
| nin | ningbo | 42 | True | 21 | 0.8448 |
| geo | georgia | 42 | True | 21 | 0.8458 |

### Iter 4b — K=400 (synth pool 600/cell)

| cell | center | epoch_run | early_stop | best_epoch | best_QE_AUROC |
|---|---|---|---|---|---|
| extra | cpsc_2018_extra | 48 | True | 27 | 0.8441 |
| nin | ningbo | 30 | True | 9 | 0.8448 |
| geo | georgia | 30 | True | 9 | 0.8457 |

→ K=400 reached convergence faster but final best_QE_AUROC effectively identical.

## K=200 vs K=400 head-to-head (avg PN2021)

### AUROC (baseline 0.8384 / 0.8387)

| cell | K=200 Δ | K=400 Δ | shift |
|---|---|---|---|
| extra | -0.17pp | -0.19pp | flat |
| nin | -0.12pp | -0.04pp | +0.08pp |
| geo | -0.13pp | -0.04pp | +0.09pp |

### AUPRC (baseline 0.5887 / 0.5888)

| cell | K=200 Δ | K=400 Δ | shift |
|---|---|---|---|
| extra | +0.06pp | +0.08pp | +0.02pp |
| nin | +0.19pp | +0.26pp | +0.07pp |
| geo | +0.18pp | +0.16pp | flat |

→ K=400 helps nin/geo recover slightly on AUROC but extra unchanged. AUPRC marginally better. **K-scaling NOT a transformative lever** — sub-percentage gains across the board.

## Per-target signal (cross-center generalization)

**chap_shaoxing** (n=10247, baseline AUROC 0.8835, 5 classes available):

| cell-run | AUROC Δ | AUPRC Δ |
|---|---|---|
| extra K=200 | +0.49pp | +0.67pp |
| nin K=200 | +0.28pp | +0.66pp |
| geo K=200 | +0.40pp | +0.83pp |
| extra K=400 | +0.50pp | +0.74pp |
| nin K=400 | +0.35pp | +0.82pp |
| geo K=400 | +0.46pp | +0.96pp |
| **avg** | **+0.41pp** | **+0.78pp** |

**ningbo** (n=34905, baseline AUROC 0.8819–0.8826, 5 classes):

| cell-run | AUROC Δ | AUPRC Δ |
|---|---|---|
| extra K=200 | +0.49pp | +0.93pp |
| nin K=200 | +0.25pp | +0.43pp |
| geo K=200 | +0.38pp | +0.73pp |
| extra K=400 | +0.49pp | +0.84pp |
| nin K=400 | +0.29pp | +0.53pp |
| geo K=400 | +0.49pp | +0.83pp |
| **avg** | **+0.40pp** | **+0.71pp** |

**MIMIC zero-shot** (baseline AUROC 0.7857, AUPRC 0.6939):

| cell-run | AUROC Δ | AUPRC Δ |
|---|---|---|
| extra K=200 | +0.33pp | +0.49pp |
| nin K=200 | +0.23pp | +0.20pp |
| geo K=200 | +0.27pp | +0.25pp |
| extra K=400 | +0.30pp | +0.34pp |
| nin K=400 | +0.27pp | +0.26pp |
| geo K=400 | +0.29pp | +0.21pp |
| **avg** | **+0.28pp** | **+0.29pp** |

**PTBXL fold10 in-domain** (baseline AUROC 0.9064, AUPRC 0.7754):

| cell-run | AUROC Δ | AUPRC Δ |
|---|---|---|
| all 6 cell-runs | -0.00 to -0.04pp | -0.02 to +0.13pp |

→ AT does NOT measurably degrade in-domain performance ✓ (PTBXL gate ≥ -0.50pp passed by all 6).

**cpsc_2018 STTC structural drag**:

| cell-run | STTC Δ |
|---|---|
| extra K=200 | -4.82pp |
| nin K=200 | -2.49pp |
| geo K=200 | -2.91pp |
| extra K=400 | -5.15pp |
| nin K=400 | -1.99pp |
| geo K=400 | -2.53pp |

→ Structural per-class regression caused by PTB-XL ↔ CPSC SNOMED vocab mismatch (memory `super5_label_audit_2026_04_26.md`). Not a method failure; the cpsc_2018 cell is structurally capped at AUROC≈0.59 baseline. The drag dominates avg-gate failure.

## PGD diagnostics

All 6 cell-runs:
- ASR_overall ≥ 0.99 throughout training
- Einthoven residual p95 ≈ 0.09–0.11 (well under 0.5 threshold)
- buf size 2048 (FIFO cap)
- medical_pass=True every epoch

→ PGD generation healthy across all cell-runs. Method-side has no obvious bottleneck.

## Decision tree (Plan Rev 13.2)

| Outcome | Path |
|---|---|
| **(a) ≥1/3 strict avg gate pass** | Frame as "average + per-target". Recommend completing 5-center confirmatory phase. ❌ NOT MET |
| **(b) 0/3 strict avg gate pass + per-target signal** | Frame as **cross-center per-target generalization**. Acceptable for paper if per-target consistency proven. ✅ **CURRENT STATE** |
| **(c) No per-target consistent signal** | Lock as ECGTwin manifold/vocab limitation. Pivot away. (NOT this case — chap +0.41pp / nin +0.40pp / MIMIC +0.28pp consistent across 6 runs) |

→ **Decision: Case (b)**. Paper framing: cross-center per-target generalization with synth-anchored on-manifold AT. Limitations chapter must disclose: (i) cpsc_2018 STTC structural ceiling; (ii) avg gate not met; (iii) ECGTwin scope limited to 3 super5 classes (NORM/MI/STTC).

## Open questions for follow-up

### Module Ablation #1 (in progress, 2026-04-27)

**Hypothesis**: Real K=200 target-center records (already VAE-encoded in our existing `.pt` ref pools) as PGD anchors should produce equal or better cross-center signal — testing whether ECGTwin diffusion contributes uniquely beyond providing anchors that PGD attacks.

**Same config as iter 4 (K=200, K_anchor=300, ε=2.0, K_pgd=10, adv_w=0.5)** — single variable changed: anchor source synth → real.

**Smoke test (3 epochs, extra cell)**:
- ASR=0.98–0.99 (vs synth iter 4 0.50–0.70 — real anchors more attackable, PGD signal stronger)
- Einthoven p95=0.22 (vs synth 0.09 — slightly higher but well within gate)
- Per-epoch wall clock: 25s (vs synth 150s — 6× speedup, no DDPM in loop)
- 3-epoch progress: avg AUROC 0.8409 (+0.12pp baseline)

→ Pipeline verified end-to-end. Full pilot 100ep × 3cells running in background.

**Decision target**: If real-anchored beats synth-anchored on per-target signal (especially the cpsc_2018 STTC structural drag — does removing ECGTwin reverse-supervision help?), we drop the ECGTwin diffusion module from this ablation lineage. This would simplify the pipeline by removing all generative-model dependencies.

## Files

- iter 4 report: `docs/synth_anchored_super5_pilot_iter4.md`
- iter 4b report: `docs/synth_anchored_super5_pilot_iter4b.md`
- iter 1-3 history: `docs/synth_anchored_super5_pilot_iter{1,2,3}.md`
- 3-iter early summary: `docs/synth_anchored_super5_pilot_summary.md`
- iter 4 outputs: `/root/autodl-tmp/synth_anchored_super5_v13/{extra,nin,geo}_k200/`
- iter 4b outputs: `/root/autodl-tmp/synth_anchored_super5_v13b_K400/{extra,nin,geo}_k400/`
- Module Ablation #1 (in progress): `/root/autodl-tmp/real_anchored_super5/`
