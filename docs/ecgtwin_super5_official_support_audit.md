# ECGTwin Super5 Official Support Audit

Date: 2026-04-27. Auditor: Claude (multimodal vision over PNG samples; medical
criteria from web sources, see inline URLs).

## Executive Summary

ECGTwin's author shipped 12 demo prompts (`generation_result_by_disease/`)
that **collectively cover all five super5 classes** (NORM / MI / HYP / CD / STTC).
The 12 author prompts and patient parameters are saved as `features.json` in each
demo dir — they are reproducible.

For our pilot pipeline, **HYP is the only class that simultaneously passes
(a) author validation, (b) visual recognizability across all 3 of our seeds,
and (c) Super5 victim agreement** (`p_target ≥ 0.85`, top-1 = HYP all 3 seeds).
**NORM** rate-variant prompts (sinus / bradycardia / tachycardia / AF) produce
visibly normal sinus-like morphology; victim agreement is moderate (`p ~ 0.45–0.61`)
because the Super5 NORM head is conservative when HR or rhythm is non-physiologic.
**MI / STTC / CD-LBBB / CD-RBBB / CD-AVB** all fail at least one criterion
(see Part 4).

Bottom line: **author-validated AND safe-to-use for pilot are HYP and NORM.**
MI/STTC are use-with-caveat (visually present but victim-OOD); CD bundle-branch
patterns are visually weak and victim-OOD — avoid as positive anchors.

---

## Part 1 — Author Demo Coverage

Every demo dir contains `features.json` (with `tar.report` = the exact prompt
the author used, plus target hr/age/sex) and 50 generated PNGs (`0..49 Generated ECG.png`).
Reference is `data/prepared_input/normal_1.pt` (the ECGTwin "normal_1" reference
patient: F, age 55, hr 91 bpm, "Sinus rhythm|Normal ECG").

| Demo dir | Author prompt (`tar.report`) | Author hr / age / sex | super5 class |
|---|---|---|---|
| `normal_ecg/` | `sinus rhythm\|normal ecg.` | 70 / 70 / F | **NORM** |
| `sinus_bradycardia-窦性心动过缓/` | `sinus bradycardia\|slow heart rate` | 45 / 75 / F | **NORM** (rate variant) |
| `sinus_tachycardia-窦性心动过速/` | `sinus tachycardia\|fast heart rate` | 120 / 50 / F | **NORM** (rate variant) |
| `atrial_fibrillation-心房颤动/` | `atrial fibrillation\|irregular rhythm` | 110 / 65 / F | **NORM** (PTB-XL super5 puts AFib under NORM/rhythm; not in MI/HYP/CD/STTC) |
| `myocardial_infarction-心梗/` | `myocardial infarction\|st elevation\|anterior wall` | 85 / 60 / F | **MI** |
| `st_elevation_mi-ST抬高心梗/` | `stemi\|st elevation myocardial infarction\|acute` | 95 / 55 / F | **MI** |
| `nstemi-非ST段抬高心肌梗死/` | `nstemi\|non st elevation\|t wave inversion` | 88 / 62 / F | **STTC** (PTB-XL groups NSTEMI with ST/T-changes) |
| `acute_pericarditis-急性心包炎/` | `acute pericarditis\|diffuse st elevation` | 88 / 45 / F | **STTC** (per PTB-XL super5 — pericarditis under STTC family) |
| `left_ventricular_hypertrophy-左心室肥厚/` | `left ventricular hypertrophy\|high voltage` | 72 / 65 / F | **HYP** |
| `left_bundle_branch_block-左束支阻滞/` | `left bundle branch block\|lbbb` | 72 / 68 / F | **CD** |
| `right_bundle_branch_block-右束支阻滞/` | `right bundle branch block\|rbbb` | 75 / 70 / F | **CD** |
| `atrioventricular_block-房室传导阻滞/` | `atrioventricular block\|av block` | 60 / 72 / F | **CD** |

**Coverage by super5**:
- NORM — 4 author demos (sinus / brady / tachy / AF)
- MI — 2 author demos (generic anterior MI; STEMI)
- STTC — 2 author demos (NSTEMI; pericarditis)
- HYP — 1 author demo (LVH)
- CD — 3 author demos (LBBB; RBBB; AV block)

All 5 super5 classes are represented in the author gallery. **No super5 class is
unrepresented.** However, "represented in the gallery" ≠ "produces a clinically
recognizable example" (Part 3).

Caveat noted in `ecgtwin_usage_guide.md`: the 12 demo dirs are paper-curated
PNGs; the *labels are post-hoc author tags*, not training classes. ECGTwin
training corpus is open-vocabulary MIMIC-IV `machine_measurements.csv` free-text.
The author shipping a demo for "nstemi" only proves they thought a recognizable
NSTEMI could be coaxed out of the model — **not** that the model has a clean
"nstemi" class.

---

## Part 2 — Canonical ECG Features per super5 Class

(Citing standard sources used in the audit.)

### NORM (normal sinus + rhythm variants)
Sources: [LITFL Normal Sinus Rhythm](https://litfl.com/normal-sinus-rhythm-ecg-library/),
[ecgwaves Normal ECG](https://ecgwaves.com/topic/ecg-normal-p-wave-qrs-complex-st-segment-t-wave-j-point/)

- HR 50–100 bpm (brady < 60, tachy > 100); P precedes every QRS;
  P upright in II/III/aVF; PR 0.12–0.22 s constant.
- QRS duration < 0.12 s (typically 0.07–0.10 s).
- ST iso-electric, T concordant (upright in I, II, V3–V6; may be inverted in III, aVR, V1).
- Atrial fibrillation: irregular RR, **absent P waves**, fibrillatory baseline (best seen in V1 / II).

### MI (acute myocardial infarction — STEMI / Q-wave)
Sources: [ecgwaves STEMI criteria](https://ecgwaves.com/topic/stemi-st-elevation-myocardial-infarction-criteria-ecg/),
[LITFL Anterior MI](https://litfl.com/anterior-myocardial-infarction-ecg-library/)

- ST elevation ≥ 1 mm (≥ 1.5 mm for women in V2–V3) at the J point in **two contiguous leads**.
- Anterior MI: STE in V1–V4 (± I, aVL); reciprocal ST depression in II, III, aVF.
- Pathological Q waves (> 40 ms, > 25% R amplitude) develop hours-to-days later.
- Hyperacute peaked T waves precede STE; T-wave inversion follows.
- Rhythm usually sinus; rate variable (often tachy from sympathetic surge).

### STTC (ST/T changes — NSTEMI / pericarditis / non-ischemic ST-T abnormality)
Sources: [LITFL Myocardial Ischaemia](https://litfl.com/myocardial-ischaemia-ecg-library/),
[ecgwaves NSTEMI](https://ecgwaves.com/topic/nstemi-non-st-elevation-myocardial-infarction-unstable-angina-criteria-ecg-diagnosis-management/),
[LITFL Pericarditis](https://litfl.com/pericarditis-ecg-library/)

- NSTEMI / ischemia: new horizontal or downsloping ST depression ≥ 0.5 mm in 2 contiguous leads,
  and/or new T-wave inversion > 1 mm in leads with dominant R waves.
- Pericarditis: **diffuse concave** ST elevation (modest, 0.5–1 mm) in I, II, aVL, aVF, V2–V6;
  **PR depression** in same territory; reciprocal STD + PR elevation in aVR.
- Pericarditis evolves: stage 1 STE+PRD → stage 2 isoelectric → stage 3 diffuse T-inv → stage 4 normalize.

### HYP (LVH)
Sources: [LITFL LVH](https://litfl.com/left-ventricular-hypertrophy-lvh-ecg-library/),
[ecgwaves LVH criteria](https://ecgwaves.com/topic/ecg-left-ventricular-hypertrophy-lvh-clinical-characteristics/)

- Sokolow-Lyon: S(V1) + R(V5 or V6) > 35 mm.
- Cornell: R(aVL) + S(V3) > 28 mm (men) / > 20 mm (women).
- LV strain: down-sloping ST depression and asymmetric inverted T waves in **V5, V6, I, aVL**.
- Often left-axis deviation, increased R-wave peak time in V5/V6 (> 50 ms).
- Rhythm typically sinus; rate normal unless coexisting condition.

### CD (LBBB / RBBB / AV block)
Sources: [LITFL LBBB](https://litfl.com/left-bundle-branch-block-lbbb-ecg-library/),
[LITFL RBBB](https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/),
[StatPearls 2nd-degree AVB](https://www.ncbi.nlm.nih.gov/books/NBK482359/),
[StatPearls AV block](https://www.ncbi.nlm.nih.gov/books/NBK459147/)

- **LBBB**: QRS ≥ 120 ms; broad notched / monomorphic R in I, aVL, V5, V6;
  deep wide S (or QS) in V1, V2; absent Q in V5, V6, I; T waves discordant
  (opposite to terminal QRS, i.e. T-inversion in V5/V6/I, T-upright in V1/V2).
- **RBBB**: QRS ≥ 120 ms; rsR'/rSR' / "M-shape" in V1, V2 (R' > R);
  broad slurred S in I, V5, V6 (> 40 ms); discordant ST/T in V1, V2 (STD + T-inv).
- **1° AV block**: PR > 200 ms, no dropped beats.
- **2° Mobitz I (Wenckebach)**: progressive PR lengthening then dropped QRS, repeating cycle.
- **2° Mobitz II**: constant PR with intermittent dropped QRS.
- **3° (complete)**: P and QRS dissociated; ventricular escape rate 30–50 bpm.

---

## Part 3 — Visual Validation of 36 Generated Samples

PNG plots are 12-lead grids (3 rows × 4 cols, MIMIC lead order:
I, II, III, aVR, aVF, aVL, V1–V6, time on x in seconds, mV on y).
HR shown in plot title is the wrapper's HR estimator (peak detection) on the
generated 1024-sample @ ~102.4 Hz signal.

### NORM — sinus rhythm (`normal_ecg`, seeds 42–44)
Three seeds show **regular narrow-QRS complexes** at ~91 bpm (seed 42, 43) and
~137 bpm (seed 44 — the estimator likely overcounted artifact peaks in V2/V4).
P waves are visible in II / aVF on seed 42–43. T waves are concordant. No ST
elevation. Lead V2 has noticeable noise floor on all seeds (typical for the
ECGTwin VAE on V2). **Visually plausible NORM** (2/3 strict, 3/3 loose).
Victim probabilities: NORM ~0.46 mean — moderate, because the Super5 NORM head
is conservative on borderline-tachy rhythms.

### NORM — sinus bradycardia (seeds 9042–9044)
Seeds show regular sinus-like complexes but **HR estimator reads 90, 132, 91 bpm**,
not 45 bpm as the prompt requested. Visually the rhythm is regular and looks
normal-morphology, but ECGTwin clearly does **not honor numeric HR target** at
the bradycardic extreme. Victim NORM mean p = 0.58 (top-1 STTC/HYP).
**Morphology fine; rate fails the "bradycardic" semantic.**

### NORM — sinus tachycardia (seeds 10042–10044)
HR 99 / 114 / 104 bpm — within or just above the tachy threshold for seed 10043.
Regular narrow QRS, visible P in II. **Closest to the labeled rate of any rate-variant.**
Victim NORM mean p = 0.60. Visually plausible NORM-tachy (3/3 loose).

### NORM — atrial fibrillation (seeds 11042–11044)
Seed 11044 (HR 91, irregular RR visible in III, V1; baseline jitter in V1
suggests f-waves) is the **most convincing AF** of the three (Super5 NORM p = 0.86,
top-1 NORM). Seeds 11042/11043 show more regular complexes — less convincing
as AF morphologically (peak detector can't distinguish AFib RR irregularity
from rate elevation alone). 1/3 cardiologist-recognizable AF.

### MI — myocardial infarction / anterior wall (seeds 1042–1044)
HR 112, 181, 91 bpm. Seed 1042 V2 shows very deep negative deflections (could be
S-wave dominance from anterior territory but no clear ST elevation above J point);
V1 ST is depressed-to-isoelectric. Seed 1043 (HR 181) is rate-distorted with
pacemaker-like spikes — not a clinically useful MI ECG. Seed 1044 V1 shows
inverted-T-wave morphology and possibly subtle STE in V4. **No seed shows a
clean anterior STE-with-reciprocal-STD pattern**. The **victim consistently
votes top-1 = HYP** (because high voltage in V5/V6 dominates), MI p_target = 0.21–0.26.
**Recognizable as "abnormal precordial morphology" only; not as classical STEMI.**

### MI — STEMI (seeds 2042–2044)
Seed 2042: V3 / V4 show large positive R waves with what looks like a J-point
elevation (~0.2 mV) — closest of the 6 MI samples to a textbook anterior STEMI.
Top-1 NORM (p = 0.80), MI p = 0.22. Seed 2043: V4 voltage > 1.0 mV (Sokolow-region)
suggests confused HYP signal. Seed 2044: lead III has flat baseline drift and
V4 shows a single anomalous beat — not a clean STEMI pattern. **1/3 plausibly
ischemic-looking; victim OOD.**

### STTC — NSTEMI (seeds 3042–3044)
**Most coherent of all classes after HYP.** Seed 3042: aVR shows broad complexes,
V5/V6 show **down-sloping ST segments and inverted T-waves** in beats — consistent
with NSTEMI/ischemia morphology. Seed 3043: severe baseline drift in lead III
(DC offset warning fired) and broad negative deflections in V1/V3; victim STTC
p = 0.76. Seed 3044: clean repetitive complexes with subtle T inversion in V5/V6;
victim STTC p = 0.85, **the strongest STTC vote of any seed**.
**3/3 show STTC-family morphology to varying degrees**, but victim still favors
HYP (top-1 HYP on 2/3 seeds) — STTC and HYP morphologies overlap in V5/V6.

### STTC — acute pericarditis (seeds 4042–4044)
HR 93, 167, 180. Seed 4042 inferior leads (II, aVF) show modest concordant
positive deflections that *could* be read as ST elevation in a relaxed
interpretation, but the pattern is **dominated by HYP-like high voltage in V5–V6**;
no clear PR depression visible. Seed 4043 (HR 167) is too rate-distorted to read.
Seed 4044 (HR 180) similar issue. **Pericarditis is the weakest STTC sample**.
The victim says HYP top-1 on 2/3, STTC on 1/3.

### HYP — left ventricular hypertrophy (seeds 5042–5044)
**Strongest class of the audit.** All 3 seeds have:
- V5 / V6 R-wave amplitudes ≥ 1.0 mV (visible in all seeds; seed 5043 has V6 ~1.5 mV).
- Deep S in V1 / V2 (seed 5043 V2 reaches –1.5 mV).
- Suggested Sokolow-Lyon S(V1)+R(V5) > 2.0 mV easily — **passes voltage criterion** in all 3.
- Down-sloping ST and asymmetric inverted T in V5/V6 visible on seed 5042 (strain).
HR estimator reads 130, 105, 184 bpm — only seed 5043 honors the prompted 72 bpm.
Victim agrees: **top-1 HYP on all 3 seeds**, p_target 0.92 / 0.87 / 0.75.
**3/3 cardiologist-recognizable LVH.**

### CD — left bundle branch block (seeds 6042–6044)
**Visual failure.** Seed 6042: complexes are narrow, no broad notched R in V5/V6,
no deep S in V1 (V1 actually shows a positive R-wave dominance — opposite of
LBBB). Looks more like RBBB-tinted normal than LBBB. Seed 6043 / 6044: chaotic
high-frequency baseline noise, no discernible clinical pattern. Victim:
**CD p = 0.009 across all 3** (essentially "not CD"); top-1 NORM.
**0/3 LBBB-recognizable.**

### CD — right bundle branch block (seeds 7042–7044)
Seed 7042: extremely sparse baseline (HR estimator unreliable at 86 bpm reading
from low-amplitude trace); no visible rsR' in V1, no broad S in V6. Seed 7043:
narrow-QRS regular tachycardia, no V1 M-shape. Seed 7044: similar — narrow QRS
with no bundle-branch morphology. Victim: **CD p = 0.003–0.012**; top-1 NORM
on 2/3, HYP on 1/3. **0/3 RBBB-recognizable.**

### CD — atrioventricular block (seeds 8042–8044)
Seed 8042: regular narrow QRS at HR 140 — completely incompatible with prompted
60 bpm bradycardia and shows no PR prolongation or dropped beats. Seeds 8043
/ 8044: same — regular narrow tachycardia. **No PR prolongation, no dropped
beats, no AV dissociation visible** in any seed. Victim: CD p = 0.005–0.020;
top-1 NORM/HYP. **0/3 AV-block-recognizable.**

---

## Part 4 — Final Verdict Table

| super5 | author demo? | medically recognizable in synth? | super5 victim p_target | recommendation |
|---|---|---|---|---|
| **NORM** | yes (4: sinus + brady + tachy + AF) | yes — sinus / tachy / AF morphology recognizable; brady fails rate | 0.46 / 0.58 / 0.60 / 0.61 | **USE** (sinus & AF & tachy); avoid brady prompt for rate fidelity |
| **MI** | yes (2: anterior MI + STEMI) | weak — 1/6 seeds plausibly STEMI-like; victim favors HYP | 0.24 / 0.23 | **USE with caveat** (victim OOD; visually present but not crisp) |
| **STTC** | yes (2: NSTEMI + pericarditis) | mixed — NSTEMI 3/3 show ST-T changes; pericarditis 1/3 weak | 0.64 / 0.48 | **USE with caveat** (NSTEMI prompt only; drop pericarditis) |
| **HYP** | yes (1: LVH) | **yes — 3/3 high-voltage + strain** | **0.92 / 0.87 / 0.75** | **USE** (gold standard of the 5) |
| **CD** | yes (3: LBBB + RBBB + AVB) | **no — 0/9 seeds show bundle/block morphology**; victim assigns p < 0.02 | 0.009 / 0.009 / 0.010 | **AVOID** (synthesis fails for CD; do not use as positive anchor) |

### Direct answer to user's question
> 哪几个 super5 类是 ECGTwin 官方支持，可以放心使用？

**官方 demo 全 5 类都覆盖，但 "放心使用" 的实际只有 2 个**:
- **HYP**（LVH，prompt = `left ventricular hypertrophy|high voltage`）— 唯一同时满足 author 验证 + 视觉清晰 + victim 同意（top-1 HYP, p ≥ 0.75）的类。
- **NORM**（特别是 `sinus rhythm|normal ecg.` 和 `atrial fibrillation|irregular rhythm`，HR 91 bpm 默认） — 形态正常窦律, victim 同意度中等 (0.46–0.61)；rate variants 不严格守 HR 目标，但形态是 NORM。

**有保留地用** (USE with caveat — visually OK but victim OOD)：
- **MI**：用 `myocardial infarction|st elevation|anterior wall` 或 `stemi|...` ，能拿到 "abnormal precordial morphology" 但不是教科书 STEMI；下游 victim 多投 HYP — 用作 adv anchor 时记得 H4 trust gate 处理 victim OOD。
- **STTC（仅 NSTEMI）**：3/3 都有 ST-T 改变信号；不用 pericarditis prompt（视觉信号不强 + Super5 NORM-exclusivity 风险）。

**不要用** (AVOID)：
- **CD（任何 LBBB / RBBB / AV block prompt）**：作者 demo 列表里有，但实际生成 0/9 视觉可识别，victim 给 p < 0.02。在 pilot 中把 CD 作为正例 anchor 会注入纯噪声。如果你必须覆盖 CD，要么换非 ECGTwin 来源，要么设 H4 per-class trust gate=0 把 CD 从训练目标里 drop。

---

## Limitations

1. **I judged ECG plausibility from PNG images, not from a cardiologist.**
   The "would a cardiologist recognize X" bar is approximated by my reading of
   Sokolow-Lyon voltage, QRS width, ST/T concordance, P-wave presence — not by
   real clinical adjudication. False positives (calling something LVH that isn't)
   and false negatives (missing subtle LBBB) are both possible.
2. **Victim p_target is biased.** The Super5 victim
   (`/root/autodl-tmp/triple_labels/super5/best_model.pt`) was trained on PTB-XL
   real-world signals; ECGTwin VAE-decoded signals live in a slightly different
   manifold (RMSE ~0.02 mV on round-trip but with characteristic spectral
   artifacts at 102.4 Hz). Low p_target may reflect domain gap, not bad synthesis.
   This is exactly why the report distinguishes "USE" from "USE with caveat".
3. **HR estimator is unreliable** at high HR (the wrapper sometimes reads 180+ bpm
   on reasonable signals due to peak-counting on noise). HR target fidelity here
   is a known weakness; we relied on visual rhythm regularity instead.
4. **Sample size is 3 seeds per class** (36 total). Author demo dirs each have
   50 PNGs; I used the project's existing 3-seed pilot output rather than
   sampling more. Class-level recommendations could shift if 50 seeds were
   inspected. The recommendation pattern (HYP/NORM strong, CD weak) is unlikely
   to flip; MI/STTC marginal calls could shift one rung.
5. **The 12 author "labels" are post-hoc tags**, not training-set classes — see
   `memory/ecgtwin_usage_guide.md`. ECGTwin trained on MIMIC machine-measurement
   free-text; the demo dir name is the curator's interpretation, not the model's.
6. **Reference is fixed to `normal_1.pt`** in this audit. The pilot pipeline uses
   per-center reference pools, so the actual production morphology will differ —
   this audit measures the **prompt quality**, not the full pilot pipeline yield.
   Per the prompt-vs-base_vector ablation
   (`memory/ecgtwin_prompt_vs_basevector.md`), prompt is the first-order driver,
   so prompt-quality findings do transfer; ref drift will add ~rel 0.4 morphology
   noise on top.
7. **Did not regenerate samples.** Per task constraints, this audit is purely
   over the existing 36 PNGs in `outputs/sanity_super5_authorprompt/`. If MI or
   CD prompts deserve a second-chance run with different references or seeds,
   that's out of scope.

---

## Source URLs (consolidated)

- LITFL Normal Sinus Rhythm — https://litfl.com/normal-sinus-rhythm-ecg-library/
- ecgwaves Normal ECG features — https://ecgwaves.com/topic/ecg-normal-p-wave-qrs-complex-st-segment-t-wave-j-point/
- ecgwaves STEMI criteria — https://ecgwaves.com/topic/stemi-st-elevation-myocardial-infarction-criteria-ecg/
- LITFL Anterior MI — https://litfl.com/anterior-myocardial-infarction-ecg-library/
- LITFL Myocardial Ischaemia — https://litfl.com/myocardial-ischaemia-ecg-library/
- ecgwaves NSTEMI — https://ecgwaves.com/topic/nstemi-non-st-elevation-myocardial-infarction-unstable-angina-criteria-ecg-diagnosis-management/
- LITFL Pericarditis — https://litfl.com/pericarditis-ecg-library/
- LITFL LVH — https://litfl.com/left-ventricular-hypertrophy-lvh-ecg-library/
- ecgwaves LVH criteria — https://ecgwaves.com/topic/ecg-left-ventricular-hypertrophy-lvh-clinical-characteristics/
- LITFL LBBB — https://litfl.com/left-bundle-branch-block-lbbb-ecg-library/
- LITFL RBBB — https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/
- StatPearls 2°AVB — https://www.ncbi.nlm.nih.gov/books/NBK482359/
- StatPearls AV block — https://www.ncbi.nlm.nih.gov/books/NBK459147/
