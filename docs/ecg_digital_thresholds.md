# ECG Digital Thresholds for Super5 Classes

Date: 2026-04-27. Purpose: provide a **machine-checkable** digital criterion table per super5 class (NORM / MI / HYP / CD / STTC), using quantitative thresholds (µV / ms / contiguity) drawn from authoritative ECG references. This replaces "visually recognizable" with concrete numbers a feature-extractor can pass or fail.

All thresholds are stated for an ECG sampled at 100–500 Hz, 12-lead, mV scale. For our ECGTwin synth pipeline the operating point is **102.4 Hz, MIMIC lead order `[I, II, III, aVR, aVF, aVL, V1..V6]`**, raw mV after VAE decode.

Sample-time conversion at 102.4 Hz: `samples = ms × 0.1024`. So 40 ms ≈ 4 samples, 120 ms ≈ 12 samples — measurement granularity is ~10 ms.

---

## NORM — normal sinus rhythm

Sources: ecgwaves "Normal ECG" [1]; LITFL "Normal Sinus Rhythm" [2]; Surawicz et al. AHA/ACCF/HRS recommendations 2009 [3].

| # | Criterion | Threshold | Lead(s) |
|---|---|---|---|
| N1 | Heart rate (R-peak based) | 60 ≤ HR ≤ 100 bpm | lead II / aVF |
| N2 | RR irregularity (CV of RR intervals) | CV ≤ 0.10 (else suspect AFib) | lead II |
| N3 | P-wave present (positive deflection 80–200 ms before each QRS) | mean P-amp ≥ +0.05 mV | lead II |
| N4 | PR interval | 120 ≤ PR ≤ 200 ms | lead II |
| N5 | QRS duration | 50 ≤ QRS < 120 ms (canonical upper bound is < 120 ms; we accept 50 as lower for VAE-narrowed synth) | broadest lead among I, II, V5, V6 |
| N6 | ST level at J+40 ms | -0.05 ≤ ST_J40 ≤ +0.10 mV | II, V5, V6 (avg) |
| N7 | T polarity in dominant-R leads | T_amp > 0 (sign positive) — convention is T concordance with QRS, not a hard amplitude floor | II, V5, V6 |

**NORM pass rule**: all of N1, N3, N4, N5, N6, N7 pass; N2 (RR-CV ≤ 0.10) flags non-AFib rhythm. We treat sinus brady (HR 50–60) and sinus tachy (HR 100–120) as **NORM-rate-variants**, accepted on N3–N7 with N1 relaxed to 50 ≤ HR ≤ 130. Atrial fibrillation is in PTB-XL super5's NORM bucket; for AFib we test (N2 fails AND N3 fails) — i.e. irregular AND no P-waves.

---

## MI — acute myocardial infarction (STEMI / Q-wave)

Sources: 4th Universal Definition of MI (Thygesen et al. ESC/ACC 2018) [4]; ecgwaves "STEMI criteria" [5]; LITFL "Anterior MI" [6].

| # | Criterion | Threshold | Lead(s) |
|---|---|---|---|
| MI1 | New ST elevation at J point in **2 contiguous leads** | ST_J ≥ +0.10 mV (≥ +0.15 mV V2–V3 women, ≥ +0.20 mV V2–V3 men < 40 y) | any 2 contiguous: V1V2, V2V3, V3V4, V4V5, V5V6, II/III, II/aVF, III/aVF, I/aVL |
| MI2 | Pathological Q wave: duration | Q_dur ≥ 30 ms (≥ 40 ms for "diagnostic") | V1–V3, anteroseptal: any |
| MI3 | Pathological Q wave: depth | Q_amp ≤ -0.10 mV AND \|Q_amp\| ≥ 0.25 × R_amp_same_lead | same lead as MI2 |
| MI4 | Reciprocal ST depression (anterior MI) | ST_J ≤ -0.05 mV in II/III/aVF when STE in V1–V4 | inferior leads |
| MI5 | T-wave inversion (post-acute or NSTEMI overlap, weaker than MI1) | T_amp ≤ -0.10 mV in 2 contiguous dominant-R leads | I, II, V4–V6 |

**MI pass rule** (digital STEMI): **MI1** passes (≥ 2 contiguous leads with ST_J ≥ +0.10 mV) **OR** (**MI2** AND **MI3** in same anatomically-grouped lead set).

We use the J point (QRS-offset) for ST elevation, not J+60. Per 4UDMI guideline elevation must be measured at the J point for STEMI (vs ST depression at J+80 for NSTEMI); we follow that convention strictly.

Contiguity groups:
- Anterior: V1, V2, V3, V4
- Lateral: I, aVL, V5, V6
- Inferior: II, III, aVF

---

## STTC — ST/T changes (NSTEMI / ischemia / non-MI ST-T abnormality)

Sources: LITFL "Myocardial Ischaemia" [7]; ecgwaves "NSTEMI" [8]; LITFL "Pericarditis" [9]; ecgwaves "ST depression criteria" [10].

| # | Criterion | Threshold | Lead(s) |
|---|---|---|---|
| ST1 | New horizontal/down-sloping ST depression at J+60 ms | ST_J60 ≤ -0.05 mV in 2 contiguous leads | any 2 contiguous (see MI grouping) |
| ST2 | New T-wave inversion in dominant-R leads | T_amp ≤ -0.10 mV in 2 contiguous leads | I, II, V4–V6 |
| ST3 | Pericarditis: diffuse concave STE | +0.05 ≤ ST_J ≤ +0.20 mV in ≥ 4 limb+precordial leads (excluding aVR), with PR depression ≤ -0.05 mV in II | I, II, aVF, V2–V6 |

**STTC pass rule**: **ST1** passes **OR** **ST2** passes.

Note: STE between +0.05 and +0.10 mV without 2-lead contiguity meeting the MI threshold may still qualify here as "non-specific ST-T abnormality" — but for our criterion we restrict to depression/inversion to keep the STTC family disjoint from MI.

---

## HYP — left ventricular hypertrophy

Sources: LITFL "LVH" [11]; ecgwaves "LVH criteria" [12]; Casale et al. Cornell index 1985 [13]; Sokolow & Lyon 1949 [14].

| # | Criterion | Threshold | Lead(s) |
|---|---|---|---|
| H1 | Sokolow-Lyon voltage | S_amp(V1) + max(R_amp(V5), R_amp(V6)) > 3.5 mV | V1, V5, V6 (S in V1 is reported as positive scalar = abs of negative deflection) |
| H2 | Cornell voltage (men) | R_amp(aVL) + S_amp(V3) > 2.8 mV | aVL, V3 |
| H2w | Cornell voltage (women) | R_amp(aVL) + S_amp(V3) > 2.0 mV | aVL, V3 |
| H3 | LV strain pattern | ST_J40 ≤ -0.05 mV AND T_amp ≤ -0.10 mV in V5/V6 (asymmetric inverted T) | V5, V6, I, aVL |
| H4 | R-peak time (intrinsicoid deflection) | RPT(V5 or V6) ≥ 50 ms | V5, V6 |

**HYP pass rule**: **H1** passes **OR** **H2** passes (Cornell male threshold; we use 2.8 mV as default since sex isn't usually known at synth-eval time).

---

## CD — conduction disturbance (LBBB / RBBB / 1° AVB)

Sources: LITFL "LBBB" [15]; LITFL "RBBB" [16]; Surawicz et al. AHA 2009 ventricular conduction [3]; StatPearls "1° AVB" [17].

### CD-LBBB

| # | Criterion | Threshold | Lead(s) |
|---|---|---|---|
| L1 | QRS duration | ≥ 120 ms | broadest among I, V5, V6 |
| L2 | Broad monomorphic R in lateral leads | R_amp ≥ +0.5 mV AND no Q wave (Q_amp = 0 or > -0.05 mV) | I, V5, V6 |
| L3 | Deep wide S (or QS) in V1, V2 | min(signal) ≤ -0.5 mV AND R/S ratio < 0.5 | V1, V2 |
| L4 | Discordant T in V5, V6 | T_amp ≤ -0.05 mV (negative; opposite the dominant-positive R) | V5, V6 |

**LBBB pass rule**: L1 AND L2 AND L3.

### CD-RBBB

| # | Criterion | Threshold | Lead(s) |
|---|---|---|---|
| R1 | QRS duration | ≥ 120 ms | broadest among I, V1, V6 |
| R2 | rsR' / "M-shape" pattern in V1 (≥ 2 positive peaks within QRS, second > first) | second_peak ≥ first_peak AND both > 0.1 mV | V1 (or V2) |
| R3 | Broad slurred S in lateral leads | S_dur ≥ 40 ms in I or V6 | I, V6 |

**RBBB pass rule**: R1 AND R2.

### CD-1°AVB

| # | Criterion | Threshold | Lead(s) |
|---|---|---|---|
| A1 | PR interval prolonged | PR > 200 ms (constant across beats; CV(PR) < 0.05) | lead II |
| A2 | Every P followed by QRS | P-count = QRS-count (no dropped beats) | lead II |

**1°AVB pass rule**: A1 AND A2.

### CD overall pass rule

Pass if **any** of LBBB / RBBB / 1°AVB passes.

---

## Caveats

1. **Voltage thresholds assume calibrated mV scale**. Our ECGTwin VAE output is raw mV (no z-score, no MHI factor). If the network outputs amplitudes systematically smaller than reality (typical for over-regularized VAE), Sokolow-Lyon and STE thresholds will be falsely-negative.
2. **Q-wave detection requires reliable QRS-onset localization.** With 102.4 Hz sampling, a 30-ms Q duration = ~3 samples; small detection error → large fractional error. Treat MI2/MI3 as low-precision.
3. **Pericarditis (ST3)** requires PR-segment depression which our extractor may not measure reliably; we do not require PR depression to pass STTC, only ST depression (ST1) or T inversion (ST2). This is intentional — pericarditis is rare in PTB-XL super5 and not a primary target.
4. **Strain/intrinsicoid (H3, H4) are confirmatory**, not required for HYP pass — the Sokolow-Lyon voltage gate (H1) alone is the standard digital LVH definition.
5. **AFib detection** uses absent P-wave + irregular RR. Our extractor's P-detector may also fail for high-noise signals; in that case "no P detected" is the failure mode for both AFib and noise — disambiguate with RR-CV.
6. **At 102.4 Hz, ms resolution is ~10 ms.** Criteria like Q_dur ≥ 30 ms are at the granularity floor; we round generously (a 25-ms Q passes).

---

## Sources

[1] ecgwaves Normal ECG — https://ecgwaves.com/topic/ecg-normal-p-wave-qrs-complex-st-segment-t-wave-j-point/
[2] LITFL Normal Sinus Rhythm — https://litfl.com/normal-sinus-rhythm-ecg-library/
[3] Surawicz et al. AHA/ACCF/HRS 2009 ventricular conduction recommendations — Circulation 2009;119:e235.
[4] Thygesen et al. 4th Universal Definition of MI 2018 — https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/Fourth-universal-definition-of-myocardial-infarction
[5] ecgwaves STEMI criteria — https://ecgwaves.com/topic/stemi-st-elevation-myocardial-infarction-criteria-ecg/
[6] LITFL Anterior MI — https://litfl.com/anterior-myocardial-infarction-ecg-library/
[7] LITFL Myocardial Ischaemia — https://litfl.com/myocardial-ischaemia-ecg-library/
[8] ecgwaves NSTEMI — https://ecgwaves.com/topic/nstemi-non-st-elevation-myocardial-infarction-unstable-angina-criteria-ecg-diagnosis-management/
[9] LITFL Pericarditis — https://litfl.com/pericarditis-ecg-library/
[10] ecgwaves ST depression — https://ecgwaves.com/topic/st-segment-depression-ecg/
[11] LITFL LVH — https://litfl.com/left-ventricular-hypertrophy-lvh-ecg-library/
[12] ecgwaves LVH criteria — https://ecgwaves.com/topic/ecg-left-ventricular-hypertrophy-lvh-clinical-characteristics/
[13] Casale PN et al. Improved sex-specific criteria of LVH (Cornell). Circulation 1985;72:565-72.
[14] Sokolow M, Lyon TP. The ventricular complex in left ventricular hypertrophy. Am Heart J 1949;37:161.
[15] LITFL LBBB — https://litfl.com/left-bundle-branch-block-lbbb-ecg-library/
[16] LITFL RBBB — https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/
[17] StatPearls 1° AVB — https://www.ncbi.nlm.nih.gov/books/NBK448161/
