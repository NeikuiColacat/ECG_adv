"""ECG label helpers for the thesis PTB-XL super5 route.

Schemes:
  super5 — PTB-XL official `diagnostic_class` (5): CD / HYP / MI / NORM / STTC

Label convention: float32 array of shape (C,),
  1.0 = positive, 0.0 = known negative, -1.0 = unknown / uncovered (masked in loss)

PTB-XL super5 labels are authoritative (from scp_statements.csv) and fully
covered (0/1 only, no -1). PN2021 super5 helpers remain for prompt-token cache
construction; PN2021/MIMIC evaluation scripts are archived under `legacy/`.
"""

import os
import sys
import ast
import functools
import hashlib
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_ROOT = os.path.expanduser(os.environ.get('ECG_ADV_DATA_ROOT', '~/autodl-tmp'))
PTBXL_ROOT = os.path.expanduser(os.environ.get('ECG_ADV_PTBXL_ROOT', os.path.join(DATA_ROOT, 'ptbxl')))
SCP_STATEMENTS_PATH = os.environ.get(
    'ECG_ADV_SCP_STATEMENTS_PATH',
    os.path.join(PTBXL_ROOT, 'scp_statements.csv'),
)


# ────────────────────────────────────────────────────────────────────────────
# PTB-XL SCP → diagnostic_class / diagnostic_subclass (authoritative)
# ────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_scp_maps():
    """Return SCP-code to PTB-XL diagnostic class mapping.

    Only SCP codes with `diagnostic=1.0` contribute (form-only and rhythm-only
    codes have blank diagnostic_class and are skipped).
    """
    df = pd.read_csv(SCP_STATEMENTS_PATH, index_col=0)
    scp_to_super5 = {}
    for scp, row in df.iterrows():
        if row.get('diagnostic') != 1.0:
            continue
        super5 = row.get('diagnostic_class')
        if isinstance(super5, str) and super5.strip():
            scp_to_super5[scp] = super5.strip()
    return scp_to_super5


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

    scp_to_super5 = _load_scp_maps()
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


# ────────────────────────────────────────────────────────────────────────────
# Unified SCHEME_REGISTRY
# ────────────────────────────────────────────────────────────────────────────

SCHEME_REGISTRY = {
    'super5': {
        'num_classes': NUM_SUPER5,
        'class_names': CLASS_NAMES_SUPER5,
        'ptbxl_fn':    ptbxl_scp_to_super5,
        'pn2021_fn':   snomed_list_to_super5,
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
    csv = os.path.join(PTBXL_ROOT, 'ptbxl_database.csv')
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
    from util.pn2021_headers import parse_header_snomed
    scheme = get_scheme(scheme_name)
    centers = ['chapman_shaoxing', 'cpsc_2018', 'cpsc_2018_extra',
               'georgia', 'ningbo', 'ptb', 'st_petersburg_incart']
    print(f"\n[{scheme_name}] PN2021 (per-center sample n<={per_center_limit})")

    all_labels = []
    for center in centers:
        base = os.path.join(DATA_ROOT, 'physionet2021', 'training', center)
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


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sanity', action='store_true')
    p.add_argument('--scheme', default=None, help='run sanity for one scheme only')
    p.add_argument('--pn_limit', type=int, default=300,
                   help='PN2021 per-center sample cap (small = fast)')
    args = p.parse_args()

    if args.sanity:
        schemes = [args.scheme] if args.scheme else ['super5']
        for s in schemes:
            print(f"\n{'=' * 70}\nSanity check — scheme: {s}\n{'=' * 70}")
            _sanity_ptbxl(s)
            _sanity_pn2021(s, per_center_limit=args.pn_limit)
    else:
        print(f"Available schemes: {list(SCHEME_REGISTRY)}")
        for name, s in SCHEME_REGISTRY.items():
            print(f"  {name}: {s['num_classes']} classes — {s['class_names']}")
