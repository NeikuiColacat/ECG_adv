"""Three ECG label schemes with unified cross-dataset extractors.

Schemes:
  super5 — PTB-XL official `diagnostic_class` (5): CD / HYP / MI / NORM / STTC
  sub23  — PTB-XL official `diagnostic_subclass` (23): AMI / IMI / LMI / PMI
           + NORM / STTC / LVH / LAFB-LPFB / CLBBB / CRBBB / ILBBB / IRBBB
           + IVCD / _AVB / ISCA / ISCI / ISC_ / LAO-LAE / RVH / RAO-RAE / WPW
           + NST_ / SEHYP
  pn26   — PhysioNet/CinC 2021 official scored SNOMED, 4 equivalence pairs merged
           (re-exports from scripts.crosscenter_v2.label_alignment_v2)

Label convention: float32 array of shape (C,),
  1.0 = positive, 0.0 = known negative, -1.0 = unknown / uncovered (masked in loss)

PTB-XL labels are authoritative (from scp_statements.csv) — all super5 + sub23
classes are fully covered (0/1 only, no -1).

Current super5 mappings for PTB-XL, PN2021, and MIMIC output 0/1 labels only.
Finer schemes may produce -1 for classes that cannot be reliably extracted from
SNOMED codes or cart-report regex.
"""

import os
import re
import sys
import ast
import functools
import hashlib
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Re-export PN26 functions from the existing module (verified source of truth)
from scripts.crosscenter_v2.label_alignment_v2 import (
    NUM_CLASSES_26 as NUM_PN26,
    CLASS_NAMES_26 as CLASS_NAMES_PN26,
    ptbxl_scp_to_26 as ptbxl_scp_to_pn26,
    snomed_to_26 as snomed_list_to_pn26,
    mimic_report_to_26 as mimic_report_to_pn26,
    has_any_scored_class,
)


SCP_STATEMENTS_PATH = '/root/autodl-tmp/ptbxl/scp_statements.csv'


# ────────────────────────────────────────────────────────────────────────────
# PTB-XL SCP → diagnostic_class / diagnostic_subclass (authoritative)
# ────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_scp_maps():
    """Returns (scp_to_super5, scp_to_sub23) dicts.

    Only SCP codes with `diagnostic=1.0` contribute (form-only and rhythm-only
    codes have blank diagnostic_class / diagnostic_subclass and are skipped).
    """
    df = pd.read_csv(SCP_STATEMENTS_PATH, index_col=0)
    scp_to_super5 = {}
    scp_to_sub23 = {}
    for scp, row in df.iterrows():
        if row.get('diagnostic') != 1.0:
            continue
        super5 = row.get('diagnostic_class')
        sub23 = row.get('diagnostic_subclass')
        if isinstance(super5, str) and super5.strip():
            scp_to_super5[scp] = super5.strip()
        if isinstance(sub23, str) and sub23.strip():
            scp_to_sub23[scp] = sub23.strip()
    return scp_to_super5, scp_to_sub23


# ────────────────────────────────────────────────────────────────────────────
# SUPER5 — 5 classes from PTB-XL diagnostic_class
# ────────────────────────────────────────────────────────────────────────────

CLASS_NAMES_SUPER5 = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
NUM_SUPER5 = len(CLASS_NAMES_SUPER5)
SUPER5_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES_SUPER5)}
_SUPER5_NORM_IDX = SUPER5_TO_IDX['NORM']
_SUPER5_ABNORMAL_IDX = tuple(i for n, i in SUPER5_TO_IDX.items() if n != 'NORM')


def ptbxl_scp_to_super5(scp_codes_str_or_dict, confidence_threshold=0.0):
    """PTB-XL scp_codes → (5,) float32 multi-hot. 0/1 only, full coverage."""
    if isinstance(scp_codes_str_or_dict, str):
        scp_codes = ast.literal_eval(scp_codes_str_or_dict)
    else:
        scp_codes = scp_codes_str_or_dict

    scp_to_super5, _ = _load_scp_maps()
    label = np.zeros(NUM_SUPER5, dtype=np.float32)
    for code, conf in scp_codes.items():
        if conf >= confidence_threshold and code in scp_to_super5:
            label[SUPER5_TO_IDX[scp_to_super5[code]]] = 1.0
    return label


SUPER5_PN2021_MAPPING_VERSION = 'v3_super5_normsuppress_20260501'

# PN2021 SNOMED → PTB-XL Super5 semantic projection.
#
# Important: PhysioNet/CinC 2021 defines SNOMED-CT labels and Challenge scoring
# labels, not an official PN2021→PTB-XL-super5 crosswalk. This mapping is a
# project policy for external-center evaluation.
#
# v3 policy:
#   - direct positive mapping only for codes with a clear CD/HYP/MI/STTC target;
#   - strict NORM is only explicit sinus rhythm;
#   - rhythm/axis/ectopy/low-voltage/boundary codes suppress NORM without
#     becoming a super5 positive.
SNOMED_TO_SUPER5_POSITIVE = {
    # MI — infarction codes only (ischemia → STTC, hypertrophy → HYP)
    164865005: 'MI',     # myocardial infarction
    164867002: 'MI',     # old myocardial infarction
    57054005:  'MI',     # acute myocardial infarction
    54329005:  'MI',     # anterior MI
    22298006:  'MI',     # subacute MI
    401303003: 'MI',     # acute anterior MI alt
    233843008: 'MI',     # inferior MI alt
    # STTC — ST/T wave changes, ischemia (ischemia ≠ infarction)
    164934002: 'STTC',   # T wave abnormal
    111975006: 'STTC',   # prolonged QT
    164931005: 'STTC',   # ST elevation
    429622005: 'STTC',   # ST depression
    164930006: 'STTC',   # ST interval abnormal
    59931005:  'STTC',   # T wave inversion
    164861001: 'STTC',   # myocardial ischemia
    428750005: 'STTC',   # nonspecific ST-T abnormality
    55930002:  'STTC',   # ST changes
    425623009: 'STTC',   # lateral ischemia
    425419005: 'STTC',   # inferior ischemia
    426434006: 'STTC',   # anterior ischemia
    # CD — bundle branch blocks, AV blocks, conduction abnormalities, pacing
    270492004: 'CD',     # 1st degree AV block
    195042002: 'CD',     # 2nd degree AV block
    54016002:  'CD',     # 2nd degree Mobitz type I (Wenckebach)
    426183003: 'CD',     # Mobitz type II
    27885002:  'CD',     # 3rd degree AV block (complete heart block)
    233917008: 'CD',     # AV block generic
    164947007: 'CD',     # prolonged PR interval
    164909002: 'CD',     # LBBB
    733534002: 'CD',     # complete LBBB
    59118001:  'CD',     # RBBB
    713427006: 'CD',     # complete RBBB
    251120003: 'CD',     # incomplete LBBB
    713426002: 'CD',     # incomplete RBBB
    6374002:   'CD',     # bundle branch block (generic)
    445118002: 'CD',     # LAnFB (left anterior fascicular block)
    445211001: 'CD',     # left posterior fascicular block
    698252002: 'CD',     # nonspecific IV conduction block
    10370003:  'CD',     # pacing rhythm
    251268003: 'CD',     # atrial pacing pattern
    251266004: 'CD',     # ventricular pacing pattern
    74390002:  'CD',     # WPW (wolff-parkinson-white)
    26749005:  'CD',     # WPW alternate code
    195060002: 'CD',     # ventricular pre-excitation
    # HYP — hypertrophy and chamber enlargement
    164873001: 'HYP',    # left ventricular hypertrophy
    55827005:  'HYP',    # left ventricular high voltage
    89792004:  'HYP',    # right ventricular hypertrophy
    266249003: 'HYP',    # ventricular hypertrophy generic
    446358003: 'HYP',    # right atrial hypertrophy / RAE
    446813000: 'HYP',    # left atrial hypertrophy / LAE
    67741000119109: 'HYP',  # left atrial enlargement (alt)
    67751000119106: 'HYP',  # right atrial high voltage
    195126007: 'HYP',    # atrial hypertrophy
    164828000: 'HYP',    # atrial hypertrophy (alt)
}

NORM_POSITIVE_SNOMEDS = frozenset({
    426783006,  # sinus rhythm
})

NORM_SUPPRESS_SNOMEDS = frozenset({
    # Sinus rhythm variants are not equivalent to PTB-XL diagnostic NORM.
    426177001,  # sinus bradycardia
    427084000,  # sinus tachycardia
    427393009,  # sinus arrhythmia
    # Official scored rhythm/ectopy/axis/voltage labels without direct super5 target.
    164889003,  # atrial fibrillation
    164890007,  # atrial flutter
    284470004,  # premature atrial contraction
    63593006,   # supraventricular premature beats
    427172004,  # premature ventricular contractions
    17338001,   # ventricular premature beats
    39732003,   # left axis deviation
    47665007,   # right axis deviation
    251146004,  # low QRS voltages
    365413008,  # poor R wave progression
    426627000,  # bradycardia
    # Common unscored/non-super5 abnormalities and rhythm variants.
    164951009,  # abnormal QRS
    233892002,  # accelerated atrial escape rhythm
    251187003,  # atrial escape beat
    61277005,   # accelerated idioventricular rhythm
    426664006,  # accelerated junctional rhythm
    195080001,  # atrial fibrillation and flutter
    251173003,  # atrial bigeminy
    713422000,  # atrial tachycardia
    50799005,   # atrioventricular dissociation
    29320008,   # atrioventricular junctional rhythm
    251166008,  # AV nodal reentrant tachycardia
    233897008,  # AV reentrant tachycardia
    251170000,  # blocked premature atrial contraction
    74615001,   # brady tachy syndrome
    426749004,  # chronic atrial fibrillation
    698247007,  # cardiac dysrhythmia
    251198002,  # clockwise rotation
    251199005,  # counterclockwise rotation
    # Boundary codes: suppress NORM but do not directly create STTC in v3.
    164917005,  # Q wave abnormal
    428417006,  # early repolarization
})

# Backward-compatible public view used by legacy scripts that need to inspect
# direct positive labels. Suppress-only codes are intentionally absent.
SNOMED_TO_SUPER5 = {
    **SNOMED_TO_SUPER5_POSITIVE,
    **{code: 'NORM' for code in NORM_POSITIVE_SNOMEDS},
}


def _stable_hash_mapping(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]


SUPER5_PN2021_MAPPING_HASH = _stable_hash_mapping({
    'version': SUPER5_PN2021_MAPPING_VERSION,
    'positive': SNOMED_TO_SUPER5_POSITIVE,
    'norm_positive': sorted(NORM_POSITIVE_SNOMEDS),
    'norm_suppress': sorted(NORM_SUPPRESS_SNOMEDS),
})


def get_super5_pn2021_mapping_metadata():
    return {
        'mapping_version': SUPER5_PN2021_MAPPING_VERSION,
        'mapping_hash': SUPER5_PN2021_MAPPING_HASH,
        'class_names': CLASS_NAMES_SUPER5,
    }


def snomed_list_to_super5(snomed_codes):
    """PN2021 SNOMED list → (5,) float32 multi-hot.

    v3 NORM policy: explicit sinus rhythm may become NORM only when no direct
    super5 abnormal class and no suppress-only abnormality are present.
    """
    label = np.zeros(NUM_SUPER5, dtype=np.float32)
    has_norm_candidate = False
    has_norm_suppress = False
    for code in snomed_codes:
        cls = SNOMED_TO_SUPER5_POSITIVE.get(code)
        if cls is not None:
            label[SUPER5_TO_IDX[cls]] = 1.0
        if code in NORM_POSITIVE_SNOMEDS:
            has_norm_candidate = True
        if code in NORM_SUPPRESS_SNOMEDS:
            has_norm_suppress = True
    if has_norm_candidate and not any(label[i] for i in _SUPER5_ABNORMAL_IDX) and not has_norm_suppress:
        label[_SUPER5_NORM_IDX] = 1.0
    else:
        label[_SUPER5_NORM_IDX] = 0.0
    return label


MIMIC_SUPER5_PATTERNS = {
    'NORM': [r'sinus rhythm', r'\bnsr\b', r'normal ecg', r'normal sinus',
             r'within normal limits', r'otherwise normal', r'sinus arrhythmia',
             r'sinus bradycardia', r'sinus tachycardia'],
    'MI':   [r'myocardial infarct', r'\bmi\b(?! \w)', r'\binfarct',
             r'(anterior|inferior|lateral|posterior|septal|anteroseptal|anterolateral|inferolateral|inferoposterior) (mi|infarct)',
             r'\bstemi\b', r'\bnstemi\b', r'st elevation myocardial',
             r'old infarct', r'recent infarct', r'acute infarct'],
    'STTC': [r'st elevation', r'st depression', r'st[- ]?t chang',
             r't[- ]?wave (invert|abnorm|chang)', r'\bischemi',
             r'prolonged qt', r'\blong qt', r'qt prolong',
             r'q[- ]?wave abnormal', r'pathologic(al)? q[- ]?wave',
             r'nonspecific (st|t)[- ]?wave', r'\bstc', r'repolarization abnormal'],
    'CD':   [r'bundle branch block', r'\blbbb\b', r'\brbbb\b',
             r'\bav block\b', r'heart block', r'(1st|2nd|3rd|first|second|third)[- ]degree',
             r'fascicular block', r'\blafb\b', r'\blpfb\b',
             r'intraventricular.*(block|conduction (defect|delay|disturb))',
             r'\bivcd\b', r'wolff[- ]?parkinson', r'\bwpw\b', r'preexcitation',
             r'pace[dr]?', r'pacemaker', r'paced rhythm'],
    'HYP':  [r'(ventricular|atrial|left|right) hypertrophy',
             r'\blvh\b', r'\brvh\b', r'atrial enlargement',
             r'chamber enlargement', r'(left|right) atrial enlargement',
             r'\blae\b', r'\brae\b', r'biventricular hypertrophy',
             r'septal hypertrophy'],
}
_MIMIC_SUPER5_COMPILED = {
    cls: [re.compile(p) for p in pats]
    for cls, pats in MIMIC_SUPER5_PATTERNS.items()
}


def mimic_report_to_super5(report_text):
    """MIMIC cart report text → (5,) float32 multi-hot. 0/1 only.

    Applies the same NORM exclusivity guard used for PN2021 so report phrases
    like "sinus rhythm" do not coexist with abnormal super5 labels.
    """
    label = np.zeros(NUM_SUPER5, dtype=np.float32)
    if not report_text:
        return label
    text = report_text.lower()
    for cls, patterns in _MIMIC_SUPER5_COMPILED.items():
        for pat in patterns:
            if pat.search(text):
                label[SUPER5_TO_IDX[cls]] = 1.0
                break
    if any(label[i] for i in _SUPER5_ABNORMAL_IDX):
        label[_SUPER5_NORM_IDX] = 0.0
    return label


# ────────────────────────────────────────────────────────────────────────────
# SUB23 — 23 classes from PTB-XL diagnostic_subclass
# ────────────────────────────────────────────────────────────────────────────

# Frozen in first-appearance order from scp_statements.csv
# (verified via pd.read_csv(...).dropna(subset='diagnostic').diagnostic_subclass.dropna().unique())
CLASS_NAMES_SUB23 = [
    'STTC',       # 0  non-diagnostic T, LNGQT, DIG, ANEUR, EL  (generic ST-T)
    'NST_',       # 1  non-specific ST changes
    'NORM',       # 2  normal ECG
    'IMI',        # 3  inferior MI (incl. ILMI, IPLMI, IPMI, INJIN, INJIL)
    'AMI',        # 4  anterior MI (incl. ASMI, ALMI, INJAS, INJAL, INJLA)
    'LVH',        # 5  left ventricular hypertrophy
    'LAFB/LPFB',  # 6  left anterior / posterior fascicular block
    'ISC_',       # 7  non-specific ischemic (generic)
    'IRBBB',      # 8  incomplete right bundle branch block
    '_AVB',       # 9  1st/2nd/3rd degree AV block
    'IVCD',       # 10 non-specific intraventricular conduction disturbance
    'ISCA',       # 11 anterior ischemia (ISCAL/ISCAS/ISCAN/ISCLA)
    'CRBBB',      # 12 complete right bundle branch block
    'CLBBB',      # 13 complete left bundle branch block
    'LAO/LAE',    # 14 left atrial overload/enlargement
    'ISCI',       # 15 inferior ischemia (ISCIN/ISCIL)
    'LMI',        # 16 lateral MI
    'RVH',        # 17 right ventricular hypertrophy
    'RAO/RAE',    # 18 right atrial overload/enlargement
    'WPW',        # 19 Wolff-Parkinson-White
    'ILBBB',      # 20 incomplete left bundle branch block
    'SEHYP',      # 21 septal hypertrophy
    'PMI',        # 22 posterior MI
]
NUM_SUB23 = len(CLASS_NAMES_SUB23)
assert NUM_SUB23 == 23
SUB23_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES_SUB23)}


def ptbxl_scp_to_sub23(scp_codes_str_or_dict, confidence_threshold=0.0):
    """PTB-XL scp_codes → (23,) float32 multi-hot. 0/1 only, full coverage."""
    if isinstance(scp_codes_str_or_dict, str):
        scp_codes = ast.literal_eval(scp_codes_str_or_dict)
    else:
        scp_codes = scp_codes_str_or_dict

    _, scp_to_sub23 = _load_scp_maps()
    label = np.zeros(NUM_SUB23, dtype=np.float32)
    for code, conf in scp_codes.items():
        if conf >= confidence_threshold and code in scp_to_sub23:
            sub = scp_to_sub23[code]
            if sub in SUB23_TO_IDX:
                label[SUB23_TO_IDX[sub]] = 1.0
    return label


# PN2021 SNOMED → Sub23.
# Only classes that can be reliably derived from SNOMED codes get 0/1.
# Classes that cannot be discriminated at subclass level (MI subtypes,
# ischemia subtypes, septal hypertrophy, generic STTC) get -1.
SNOMED_TO_SUB23 = {
    426783006: 'NORM',      # sinus rhythm
    733534002: 'CLBBB',
    164909002: 'CLBBB',     # LBBB (treat as CLBBB since complete is default)
    713427006: 'CRBBB',
    59118001:  'CRBBB',     # RBBB → CRBBB (complete default)
    251120003: 'ILBBB',
    713426002: 'IRBBB',
    445118002: 'LAFB/LPFB', # LAnFB
    445211001: 'LAFB/LPFB', # LPFB
    698252002: 'IVCD',
    270492004: '_AVB',      # 1st degree
    195042002: '_AVB',      # 2nd degree
    54016002:  '_AVB',      # 2nd degree Mobitz II
    27885002:  '_AVB',      # 3rd degree
    55827005:  'LVH',
    89792004:  'RVH',
    446813000: 'LAO/LAE',
    67741000119109: 'LAO/LAE',
    446358003: 'RAO/RAE',
    74390002:  'WPW',
    26749005:  'WPW',
}

# Classes that CANNOT be derived from PN2021 SNOMED (stay -1 forever)
SUB23_UNMAPPABLE_PN2021 = {
    'IMI', 'AMI', 'LMI', 'PMI',   # MI subtypes (SNOMED 164865005 = generic MI)
    'ISC_', 'ISCA', 'ISCI',       # ischemia subtypes
    'STTC', 'NST_',               # non-specific ST-T
    'SEHYP',                      # septal hypertrophy (no standard SNOMED)
}
SUB23_UNMAPPABLE_PN2021_IDX = [SUB23_TO_IDX[c] for c in SUB23_UNMAPPABLE_PN2021]

_SUB23_NORM_IDX = SUB23_TO_IDX['NORM']
# Reuse super5's SNOMED → abnormality mapping as a broad "any abnormality"
# detector. Catches abnormalities that sub23 itself cannot map to a subclass
# (e.g., generic MI 164865005 — sub23 needs an MI subtype, but the
# abnormality is real, so NORM must still be 0).
_SUPER5_ABNORMAL_SNOMEDS = frozenset(
    set(SNOMED_TO_SUPER5_POSITIVE) | set(NORM_SUPPRESS_SNOMEDS)
)


def snomed_list_to_sub23(snomed_codes):
    """PN2021 SNOMED list → (23,) float32 with -1 for unmappable classes.

    NORM exclusivity guard mirrors super5: PTB-XL trains diagnostic_subclass
    NORM as "no abnormality" (~5% co-occurrence with abnormal subclasses);
    PN2021 emits sinus rhythm 426783006 alongside pathology codes (~98% on
    cpsc_2018_extra), and many PN2021 abnormalities (generic MI, ischemia)
    cannot be split into sub23 subtypes. We therefore key the guard on
    super5-level abnormality presence, not just sub23 mappable hits.
    """
    label = np.zeros(NUM_SUB23, dtype=np.float32)
    for idx in SUB23_UNMAPPABLE_PN2021_IDX:
        label[idx] = -1.0
    for code in snomed_codes:
        cls = SNOMED_TO_SUB23.get(code)
        if cls is not None:
            label[SUB23_TO_IDX[cls]] = 1.0
    if any(c in _SUPER5_ABNORMAL_SNOMEDS for c in snomed_codes):
        label[_SUB23_NORM_IDX] = 0.0
    return label


MIMIC_SUB23_PATTERNS = {
    'NORM':      [r'sinus rhythm', r'\bnsr\b', r'normal ecg', r'normal sinus',
                  r'within normal limits', r'otherwise normal'],
    'AMI':       [r'anterior (mi|infarct)', r'anteroseptal (mi|infarct)',
                  r'anterolateral (mi|infarct)', r'antero[\s-]?septal',
                  r'antero[\s-]?lateral', r'anterior wall'],
    'IMI':       [r'inferior (mi|infarct)', r'inferolateral (mi|infarct)',
                  r'inferoposterior (mi|infarct)', r'infero[\s-]?lateral',
                  r'infero[\s-]?posterior', r'inferior wall'],
    'LMI':       [r'^lateral (mi|infarct)', r'(?<!antero|infero)lateral (mi|infarct)',
                  r'lateral wall (mi|infarct)'],
    'PMI':       [r'posterior (mi|infarct)', r'posterior wall (mi|infarct)',
                  r'postero[\s-]?lateral'],
    'LVH':       [r'left ventricular hypertrophy', r'\blvh\b'],
    'LAFB/LPFB': [r'left anterior fascicular', r'\blafb\b',
                  r'left anterior hemiblock', r'left posterior fascicular',
                  r'\blpfb\b', r'left posterior hemiblock'],
    'ISC_':      [r'\bischemi[ac]', r'ischemia unspecified'],
    'IRBBB':     [r'incomplete right bundle', r'\birbbb\b', r'incomplete rbbb'],
    '_AVB':      [r'(1st|first)[- ]degree (av block|heart block)',
                  r'(2nd|second)[- ]degree (av block|heart block)',
                  r'(3rd|third)[- ]degree (av block|heart block)',
                  r'mobitz', r'complete heart block', r'av block'],
    'IVCD':      [r'intraventricular conduction (defect|delay|disturb|abnormal)',
                  r'\bivcd\b', r'nonspecific.*conduction'],
    'ISCA':      [r'anterior ischemia', r'anterolateral ischemia',
                  r'anteroseptal ischemia'],
    'CRBBB':     [r'complete right bundle branch', r'\bcrbbb\b',
                  r'(?<!incomplete )right bundle branch block', r'(?<!incomplete )\brbbb\b'],
    'CLBBB':     [r'complete left bundle branch', r'\bclbbb\b',
                  r'(?<!incomplete )left bundle branch block', r'(?<!incomplete )\blbbb\b'],
    'LAO/LAE':   [r'left atrial enlargement', r'\blae\b',
                  r'left atrial overload', r'left atrial abnormality'],
    'ISCI':      [r'inferior ischemia', r'inferolateral ischemia'],
    'RVH':       [r'right ventricular hypertrophy', r'\brvh\b'],
    'RAO/RAE':   [r'right atrial enlargement', r'\brae\b',
                  r'right atrial overload', r'right atrial abnormality'],
    'WPW':       [r'wolff[- ]?parkinson', r'\bwpw\b', r'preexcitation',
                  r'pre[- ]?excitation'],
    'ILBBB':     [r'incomplete left bundle', r'\bilbbb\b'],
    'NST_':      [r'nonspecific st', r'non[- ]?specific st',
                  r'nonspecific.*st (change|depression|elevation)'],
    'STTC':      [r'st[- ]?t wave change', r'st[- ]?t abnormality',
                  r'abnormal repolarization', r'prolonged qt', r'long qt',
                  r'q[- ]?wave abnormal'],
    'SEHYP':     [r'septal hypertrophy', r'interventricular septal hypertroph'],
}
_MIMIC_SUB23_COMPILED = {
    cls: [re.compile(p) for p in pats]
    for cls, pats in MIMIC_SUB23_PATTERNS.items()
}


def mimic_report_to_sub23(report_text):
    """MIMIC report text → (23,) float32 multi-hot. 0/1 for all 23 classes
    (every class has a regex; missing match = 0). No -1 from MIMIC."""
    label = np.zeros(NUM_SUB23, dtype=np.float32)
    if not report_text:
        return label
    text = report_text.lower()
    for cls, patterns in _MIMIC_SUB23_COMPILED.items():
        idx = SUB23_TO_IDX[cls]
        for pat in patterns:
            if pat.search(text):
                label[idx] = 1.0
                break
    return label


# ────────────────────────────────────────────────────────────────────────────
# Unified SCHEME_REGISTRY
# ────────────────────────────────────────────────────────────────────────────

SCHEME_REGISTRY = {
    'super5': {
        'num_classes': NUM_SUPER5,
        'class_names': CLASS_NAMES_SUPER5,
        'ptbxl_fn':    ptbxl_scp_to_super5,
        'pn2021_fn':   snomed_list_to_super5,
        'mimic_fn':    mimic_report_to_super5,
    },
    'sub23': {
        'num_classes': NUM_SUB23,
        'class_names': CLASS_NAMES_SUB23,
        'ptbxl_fn':    ptbxl_scp_to_sub23,
        'pn2021_fn':   snomed_list_to_sub23,
        'mimic_fn':    mimic_report_to_sub23,
    },
    'pn26': {
        'num_classes': NUM_PN26,
        'class_names': CLASS_NAMES_PN26,
        'ptbxl_fn':    ptbxl_scp_to_pn26,
        'pn2021_fn':   snomed_list_to_pn26,
        'mimic_fn':    mimic_report_to_pn26,
    },
}


def get_scheme(name):
    if name not in SCHEME_REGISTRY:
        raise ValueError(f"Unknown scheme: {name}. Choices: {list(SCHEME_REGISTRY)}")
    return SCHEME_REGISTRY[name]


# ────────────────────────────────────────────────────────────────────────────
# Sanity check (`python label_schemes.py --sanity`)
# ────────────────────────────────────────────────────────────────────────────

def _print_label_distribution(labels, class_names):
    n_total = len(labels)
    for i, name in enumerate(class_names):
        col = labels[:, i]
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        n_mask = int((col == -1).sum())
        print(f"  {i:2d} {name:<12} pos={n_pos:>6}  neg={n_neg:>6}  mask={n_mask:>6}  "
              f"(pos_rate={100 * n_pos / max(n_total, 1):.2f}%)")


def _sanity_ptbxl(scheme_name, n_samples=None):
    """Summarize per-class pos/neg/mask counts on PTB-XL fold 1-8."""
    scheme = get_scheme(scheme_name)
    csv = '/root/autodl-tmp/ptbxl/ptbxl_database.csv'
    df = pd.read_csv(csv)
    df = df[df.strat_fold.isin(list(range(1, 9)))]  # fold 1-8
    if n_samples:
        df = df.iloc[:n_samples]
    labels = np.stack([scheme['ptbxl_fn'](r) for r in df.scp_codes])
    print(f"\n[{scheme_name}] PTB-XL fold 1-8  (n={len(labels)})")
    _print_label_distribution(labels, scheme['class_names'])
    return labels


def _sanity_pn2021(scheme_name, per_center_limit=500):
    """Sample some records per center and summarize label distribution."""
    # Single source of truth for PN2021 .hea SNOMED parsing — eval uses the
    # same function, so sanity counts match what the trained model will see.
    from scripts.triple_labels.eval_crosscenter import parse_header_snomed
    scheme = get_scheme(scheme_name)
    centers = ['chapman_shaoxing', 'cpsc_2018', 'cpsc_2018_extra',
               'georgia', 'ningbo', 'ptb', 'st_petersburg_incart']
    print(f"\n[{scheme_name}] PN2021 (per-center sample n<={per_center_limit})")

    all_labels = []
    for center in centers:
        base = f'/root/autodl-tmp/physionet2021/training/{center}'
        if not os.path.isdir(base):
            continue
        heas = []
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith('.hea'):
                    heas.append(os.path.join(root, f))
                    if len(heas) >= per_center_limit:
                        break
            if len(heas) >= per_center_limit:
                break
        labels = []
        for hea in heas:
            try:
                codes = parse_header_snomed(hea)
            except (OSError, ValueError):
                continue
            labels.append(scheme['pn2021_fn'](codes))
        if not labels:
            continue
        labels = np.stack(labels)
        all_labels.append(labels)
        pos_per_class = (labels == 1).sum(axis=0).tolist()
        mask_frac = float((labels == -1).any(axis=1).mean())
        print(f"  {center:<22} n={len(labels):>4}  "
              f"mask_frac={mask_frac:.2f}  pos_per_class={pos_per_class}")
    if all_labels:
        combined = np.vstack(all_labels)
        print(f"  ---- combined ({len(combined)}) ----")
        _print_label_distribution(combined, scheme['class_names'])


def _sanity_mimic(scheme_name, n_samples=1000):
    """Sample machine_measurements rows and apply regex."""
    scheme = get_scheme(scheme_name)
    mm_path = '/root/autodl-tmp/MIMIC/machine_measurements.csv'
    report_cols = [f'report_{i}' for i in range(18)]
    df = pd.read_csv(mm_path, usecols=report_cols, nrows=n_samples, low_memory=False)

    def _join(row):
        parts = [str(v).strip() for v in row.values if isinstance(v, str) and v.strip()]
        return ' | '.join(parts)
    texts = df.apply(_join, axis=1).values

    labels = np.stack([scheme['mimic_fn'](t) for t in texts])
    print(f"\n[{scheme_name}] MIMIC (n={len(texts)})")
    _print_label_distribution(labels, scheme['class_names'])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sanity', action='store_true')
    p.add_argument('--scheme', default=None, help='run sanity for one scheme only')
    p.add_argument('--pn_limit', type=int, default=300,
                   help='PN2021 per-center sample cap (small = fast)')
    p.add_argument('--mimic_n', type=int, default=2000,
                   help='MIMIC sample count for sanity')
    args = p.parse_args()

    if args.sanity:
        schemes = [args.scheme] if args.scheme else ['super5', 'sub23', 'pn26']
        for s in schemes:
            print(f"\n{'=' * 70}\nSanity check — scheme: {s}\n{'=' * 70}")
            _sanity_ptbxl(s)
            _sanity_pn2021(s, per_center_limit=args.pn_limit)
            _sanity_mimic(s, n_samples=args.mimic_n)
    else:
        print(f"Available schemes: {list(SCHEME_REGISTRY)}")
        for name, s in SCHEME_REGISTRY.items():
            print(f"  {name}: {s['num_classes']} classes — {s['class_names']}")
