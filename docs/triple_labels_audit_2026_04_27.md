# Triple-label classifiers audit — 2026-04-27

Audit of three EfficientNetV2 heads (super5 / sub23 / pn26) trained on PTB-XL,
evaluated on PTB-XL fold 10 + PN2021 7 centers + MIMIC test.

## TL;DR

- **Super5** ✅ clean (post 2026-04-26 NORM-guard audit)
- **Sub23** ⚠️ **NORM exclusivity guard missing** → fixed this session; eval re-run
- **PN26** ⚠️ **3 PTB-XL-uncovered classes (Brady/RAD/PRWP) drag PN2021 macro**;
  NSR drift NOT present (different semantics from super5 NORM)
- **Preprocessing pipeline** ✅ uniform across schemes, no silent bugs

## 1. Pre-audit per-center AUROC / AUPRC

ckpt at `/root/autodl-tmp/triple_labels/{super5,sub23,pn26}/best_model.pt`.

### PTB-XL fold 10 (in-domain)

| Scheme | macro AUROC | macro AUPRC | n_classes used |
|---|---|---|---|
| super5 (post NORM-guard) | 0.9064 | 0.7754 | 5/5 |
| sub23 | 0.9158 | 0.4893 | 19/23 |
| pn26 | 0.9117 | 0.5293 | 23/26 |

### PN2021 macro AUROC / AUPRC per center

| center | n | super5 (guarded) | sub23 (no guard) | pn26 (no guard) |
|---|---|---|---|---|
| chapman_shaoxing | 10247 | **0.8835** / 0.547 | 0.8464 / 0.445 | 0.8402 / 0.400 |
| cpsc_2018 | 6877 | 0.8052 / 0.563 | **0.9283** / **0.735** | 0.9015 / 0.681 |
| cpsc_2018_extra | 3453 | 0.8052 / 0.622 | 0.8530 / 0.218 | 0.7939 / 0.288 |
| georgia | 10344 | 0.8173 / **0.646** | 0.8838 / 0.356 | 0.8321 / 0.371 |
| ningbo | 34905 | 0.8826 / 0.591 | 0.8849 / 0.317 | 0.8368 / 0.360 |
| ptb | 516 | 0.9009 / 0.595 | 0.9392 / 0.669 | 0.8883 / 0.423 |
| st_petersburg | 74 | 0.7785 / 0.558 | NaN | 0.5905 / 0.352 |

### MIMIC zero-shot

| Scheme | macro AUROC | macro AUPRC |
|---|---|---|
| super5 | missing (skip_mimic=true) | — |
| sub23 | 0.7828 | 0.2698 |
| pn26 | 0.7871 | 0.3520 |

## 2. Bug audit findings

### Bug 1 — sub23 NORM exclusivity guard missing **(HIGH severity, fixed)**

**Symptom**: sub23 NORM AUROC collapses on PN2021 centers heavy with sinus-rhythm-but-abnormal records:

| center | sub23 NORM AUROC (pre-guard) | super5 NORM AUROC (post-guard) | Δ |
|---|---|---|---|
| chap_shaoxing | 0.725 (n_pos=1826) | 0.906 (n_pos=4686) | **−18 pp** |
| ningbo | 0.707 (n_pos=6299) | 0.884 (n_pos=17327) | **−18 pp** |
| georgia | 0.847 | 0.859 | −1 pp |
| ptb | 0.939 | 0.932 | matches |

**Cause**: PTB-XL `diagnostic_subclass` NORM is exclusive to "no abnormality"
(NORM ↔ abnormal = ~5% co-occurrence). PN2021 `.hea` files emit SNOMED
426783006 (sinus rhythm) alongside pathology codes (~98% co-occurrence on
cpsc_2018_extra). Without exclusivity guard, NORM=1 in eval labels even when
the record has clear abnormality → labels diverge from training semantics →
NORM head AUROC collapses.

Identical drift previously documented for super5 (memory
`super5_label_audit_2026_04_26.md`); fix is the same pattern.

**Fix** (`scripts/triple_labels/label_schemes.py:323-352`): force NORM=0
whenever any SNOMED code mapping to a super5 abnormal class (CD/HYP/MI/STTC)
is present. Using super5's broader SNOMED-→-abnormal mapping (rather than
sub23's own mappable classes) catches the cpsc_2018_extra case where generic
MI 164865005 cannot be split into AMI/IMI/LMI/PMI subtypes (all -1
unmappable in sub23) but is still a real abnormality that must zero NORM.

```python
# label_schemes.py:323-326
_SUPER5_ABNORMAL_SNOMEDS = frozenset(
    c for c, cls in SNOMED_TO_SUPER5.items() if cls != 'NORM'
)
# In snomed_list_to_sub23, after assigning class labels:
if any(c in _SUPER5_ABNORMAL_SNOMEDS for c in snomed_codes):
    label[_SUB23_NORM_IDX] = 0.0
```

**Sanity tests** (all pass):
- pure sinus → NORM=1 ✓
- sinus + LBBB → NORM=0, CLBBB=1 ✓
- sinus + generic MI 164865005 → NORM=0, AMI/IMI/LMI/PMI=−1 ✓
- sinus + ST elevation → NORM=0, STTC=−1 ✓
- empty → NORM=0 ✓

**Expected impact on macro AUROC** (from analogue super5 audit, 5-ctr 0.755 → 0.839):
- chap_shaoxing macro 0.846 → ~0.86 (NORM +18pp / ~12 macro classes)
- ningbo macro 0.885 → ~0.90 (similar)
- cpsc_2018_extra: NORM remains <10 n_pos → no macro change
- avg PN2021 macro AUROC expected +1 to +2 pp

(actual numbers from re-run: see §3)

### Bug 2 — PN26 has 3 PTB-XL-uncovered classes that train as masked but test as 0/1 **(MEDIUM, drag only, structural limit)**

**Symptom**: pn26 per-center per-class on PN2021 shows below-random AUROC
on Brady (0.190 cpsc_2018_extra, n_pos=271), RAD (0.341 chap, 0.351 ningbo,
0.371 georgia), PRWP (0.667 ningbo).

**Cause** (`label_alignment_v2.py:131-137`): `PTBXL_SCP_TO_CLASSES` does not
emit Brady, RAD, or PRWP — comment line 137 explicitly states "Expected
uncovered: {Brady, PRWP, RAD}". PTB-XL training masks these classes (-1) so
the model's heads for these classes are unconstrained and produce noise.
At PN2021 eval, SNOMED labels DO populate Brady/RAD/PRWP as 0/1, so the
noisy model output is scored against real labels → arbitrary low AUROC.

**Drag estimate**:
- chap_shaoxing: RAD AUROC 0.341, n_pos=215 → ~−0.4 pp on macro
- cpsc_2018_extra: Brady 0.190, n_pos=271 → ~−2.2 pp on macro
- georgia: RAD 0.371, n_pos=83 → ~−0.8 pp
- ningbo: RAD 0.351 + PRWP 0.667 → ~−1.5 pp

**Recommended fix (out of scope of this audit)**: in
`eval_crosscenter.py::compute_macro_auroc_auprc`, accept a
`trained_indices` list and exclude classes outside it from macro
calculation when evaluating on PN2021/MIMIC. Pn26 macro on chap_shaoxing
would lift from 0.840 to ~0.85 with no model retraining.

NOT a label-mapping bug per se — `snomed_to_26` correctly emits Brady/RAD/PRWP from
PN2021 SNOMED. The issue is the eval treats trained and untrained classes
uniformly. Document as a known limitation.

### Non-bug — PN26 NSR exclusivity is intentionally absent

PN26 `NSR` class has DIFFERENT semantics from super5/sub23 `NORM`:

| dim | super5 NORM / sub23 NORM | pn26 NSR |
|---|---|---|
| training source | PTB-XL `diagnostic_class/subclass` "NORM" only | both 'NORM' SCP and 'SR' SCP code (`label_alignment_v2.py:115,125`) |
| co-occurrence with pathology in PTB-XL training | ~5% (exclusive) | ~25-40% ('SR' fires whenever rhythm is sinus) |
| behavior on PN2021 cpsc_2018_extra (sinus + pathology) | NORM head should output 0 | NSR head correctly outputs 1 (matches training) |

Empirically pn26 NSR per-center AUROC = 0.826–0.884 across 6 large centers
(n_pos ≥ 80). No collapse pattern → no guard needed.

### Non-bug — ningbo AF↔AFL dual mapping

Already correctly patched in `label_alignment_v2.py:88-91`: SNOMED 164890007
maps to BOTH AF and AFL class indices. Sanity test at line 326-336 of
that file confirms the patch with assertion. Sub23 has no AF class
(rhythm not in scheme); super5 has no AF (only super-classes).
✅ no action needed.

### Non-bug — PN2021 .hea parser

`scripts/triple_labels/eval_crosscenter.py:53::parse_header_snomed`
correctly handles "# Dx:" with space (line 57: `if line.startswith('#') and 'Dx' in line`).
Not vulnerable to naive `startswith('#Dx:')` regex. Sanity check (memory
`super5_label_audit_2026_04_26.md` line 36): 0/2500 mismatch.
✅ no action needed.

## 3. Pre-/post-guard sub23 PN2021 deltas

PTB-XL fold10: 0.9158 / 0.4893 — unchanged (guard only affects PN2021 path) ✅

| center | pre AUROC | post AUROC | ΔAUROC | pre AUPRC | post AUPRC | ΔAUPRC | NORM AUROC pre→post |
|---|---|---|---|---|---|---|---|
| chapman_shaoxing | 0.8464 | **0.8607** | **+1.43pp** | 0.4450 | 0.4482 | +0.32pp | 0.725 → **0.811** |
| ningbo | 0.8849 | **0.8924** | **+0.76pp** | 0.3166 | 0.3181 | +0.14pp | 0.707 → **0.790** |
| cpsc_2018 | 0.9283 | 0.9283 | 0 | 0.7353 | 0.7353 | 0 | 0.906 → 0.906 (already healthy — most cpsc records lack 426783006) |
| georgia | 0.8838 | 0.8838 | 0 | 0.3555 | 0.3555 | 0 | 0.847 → 0.847 |
| cpsc_2018_extra | 0.8530 | 0.8530 | 0 | 0.2175 | 0.2175 | 0 | NaN → NaN (n_pos<10) |
| ptb | 0.9392 | 0.9392 | 0 | 0.6688 | 0.6688 | 0 | 0.939 → 0.939 |
| st_petersburg | NaN | NaN | — | NaN | NaN | — | — |
| **avg 7-center** | **0.8892** | **0.8929** | **+0.36pp** ✅ | 0.4565 | 0.4572 | +0.08pp | |

Confirmation:
- chap NORM class AUROC +8.6pp (0.725 → 0.811) drives chap macro +1.43pp
- ningbo NORM class AUROC +8.3pp (0.707 → 0.790) drives ningbo macro +0.76pp
- chap and ningbo are the two centers where sinus-rhythm-with-pathology
  co-occurrence on PN2021 had been miscoded as NORM=1
- cpsc_2018 / georgia / ptb: NORM was already healthy; guard idempotent for
  records without abnormality SNOMEDs

**Note on NORM-AUROC ceiling**: post-guard sub23 NORM (0.811 chap / 0.790 ningbo)
remains below super5 NORM (0.906 / 0.884) by ~10pp. The gap is label-coverage,
not guard correctness: super5 maps 4 sinus-rhythm SNOMEDs to NORM (426783006
+ 426177001 brady + 427084000 tachy + 427393009 arrhythmia), while sub23
only maps 426783006. Records carrying sinus-bradycardia code without 426783006
get sub23 NORM=0 even when truly normal-by-no-abnormality. To match super5's
ceiling we would need to either (a) widen sub23 NORM SNOMED list (matches
super5 semantics) — but PTB-XL training has 'SBRAD' SCP → SB subclass (not
NORM), so adding sinus-brady SNOMED to sub23 NORM would conflict with
training labels — or (b) accept sub23 NORM as stricter than super5 NORM
(current state). Option (b) is correct per sub23's design intent.

## 4. Preprocessing pipeline audit (no bugs)

| Stage | parameter | super5 | sub23 | pn26 |
|---|---|---|---|---|
| input fs | resample target | 100 Hz | 100 Hz | 100 Hz |
| input length | crop / pad → samples | 1000 → 250 (center) | 1000 → 250 | 1000 → 250 |
| filter | bandpass + notch (fs>150) + median baseline | ✓ | ✓ | ✓ |
| z-score | per-record global (not per-lead) | ✓ | ✓ | ✓ |
| MIMIC lead reorder | aVF↔aVL swap via `ECGTWIN_TO_PTBXL_INDICES` analogue | ✓ | ✓ | ✓ |
| NaN / inf handling | `np.nan_to_num(nan=0.0)` | ✓ | ✓ | ✓ |
| PN2021 `ptb-xl` shard exclude | hard assert in eval_crosscenter.py:173-174 | ✓ | ✓ | ✓ |

All three schemes share `unified_preprocess_to_1000()` (single source of
truth at `scripts/crosscenter_v2/preprocess_utils.py:23-194`). Training
crop_signal_tc is `mode=random` for train, `center` for val/eval (consistent
across schemes).

No silent preprocessing drift between schemes.

## 5. Action items

- [x] Add NORM exclusivity guard to sub23 (label_schemes.py:323-352)
- [x] Backup pre-guard eval_result.json → eval_result_pre_normguard.json
- [x] Re-run sub23 PN2021 eval with new guard → eval_result_NORMguard.json
- [x] Compute pre/post deltas (§3 above): avg AUROC +0.36pp, chap +1.43pp / ningbo +0.76pp
- [x] Update memory `triple_labels_training.md` with sub23 post-guard numbers
- [ ] Document PN26 Brady/RAD/PRWP drag in `eval_crosscenter.py` limitation note (recommended but not blocking)
- [ ] (Optional, separate session) Implement `trained_indices` mask in compute_macro_auroc_auprc to drop pn26 untrained classes from PN2021 macro
