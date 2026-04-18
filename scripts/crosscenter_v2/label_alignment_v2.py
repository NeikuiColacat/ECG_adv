"""
26-class unified label head based on PhysioNet 2021 Challenge scored diagnoses.

Uses SNOMED-CT as the canonical vocabulary. Provides mapping utilities for:
  - PTBXL (SCP codes -> 26-dim vector, with -1 for unrepresented classes)
  - PhysioNet 2021 (SNOMED -> 26-dim vector, full coverage)
  - MIMIC-IV ECG (machine-report keywords -> 26-dim vector, partial coverage)

Label convention: for each class dimension,
  1.0 = positive, 0.0 = known negative, -1.0 = unknown / not represented by dataset
  (MaskedFocalLoss will ignore -1.0)
"""

import ast
import re
import numpy as np
import pandas as pd


# ── 26 class definitions (PN2021 official scored, equivalents merged) ─────────
# (abbr, [SNOMED codes including officially equated codes])
SCORED_26 = [
    ("IAVB",   [270492004]),                        # 1st degree AV block
    ("AF",     [164889003]),                        # atrial fibrillation
    ("AFL",    [164890007]),                        # atrial flutter
    ("BBB",    [6374002]),                          # bundle branch block (generic)
    ("Brady",  [426627000]),                        # bradycardia
    ("LBBB",   [164909002, 733534002]),             # LBBB ≡ CLBBB
    ("RBBB",   [59118001, 713427006]),              # RBBB ≡ CRBBB
    ("ILBBB",  [251120003]),                        # incomplete LBBB
    ("IRBBB",  [713426002]),                        # incomplete RBBB
    ("LAD",    [39732003]),                         # left axis deviation
    ("LAnFB",  [445118002]),                        # left anterior fascicular block
    ("LQRSV",  [251146004]),                        # low QRS voltages
    ("NSIVCB", [698252002]),                        # nonspecific IV conduction block
    ("PR",     [10370003]),                         # pacing rhythm
    ("PRWP",   [365413008]),                        # poor R wave progression
    ("PAC",    [284470004, 63593006]),              # PAC ≡ SVPB
    ("PVC",    [427172004, 17338001]),              # PVC ≡ VPB
    ("LPR",    [164947007]),                        # prolonged PR interval
    ("LQT",    [111975006]),                        # prolonged QT interval
    ("QAb",    [164917005]),                        # Q wave abnormal
    ("RAD",    [47665007]),                         # right axis deviation
    ("SA",     [427393009]),                        # sinus arrhythmia
    ("SB",     [426177001]),                        # sinus bradycardia
    ("NSR",    [426783006]),                        # sinus rhythm (normal)
    ("STach",  [427084000]),                        # sinus tachycardia
    ("TAb",    [164934002]),                        # T wave abnormal
]

NUM_CLASSES_26 = len(SCORED_26)
assert NUM_CLASSES_26 == 26

CLASS_NAMES_26 = [abbr for abbr, _ in SCORED_26]
CLASS_TO_IDX = {abbr: i for i, (abbr, _) in enumerate(SCORED_26)}


# ── Tier definitions ──────────────────────────────────────────────────────────
# Tier-1: classes present in ALL 5 large PN2021 centers with >=5 positives each
TIER1 = ['AF', 'LBBB', 'RBBB', 'IAVB', 'NSR']

# Tier-2: mirrors legacy 15-class baseline for direct comparison with old results
TIER2 = ['AF', 'AFL', 'BBB', 'LBBB', 'RBBB', 'IAVB', 'IRBBB', 'LAD', 'LAnFB',
         'LQT', 'NSIVCB', 'NSR', 'PVC', 'PR', 'TAb']

TIER1_IDX = [CLASS_TO_IDX[c] for c in TIER1]
TIER2_IDX = [CLASS_TO_IDX[c] for c in TIER2]
ALL_IDX = list(range(NUM_CLASSES_26))

# SNOMED reverse lookup (code -> list of class indices)
_SNOMED_TO_IDX = {}
for i, (_, codes) in enumerate(SCORED_26):
    for c in codes:
        _SNOMED_TO_IDX.setdefault(c, []).append(i)

ALL_SCORED_SNOMED = set(_SNOMED_TO_IDX.keys())


# ── PTBXL SCP -> 26 mapping ───────────────────────────────────────────────────
# Each SCP code maps to a list of 26-class indices (may include multiple, e.g.
# CLBBB -> [LBBB, BBB]). Classes NOT produced by ANY PTBXL SCP are considered
# "unrepresented" and get label -1 (ignore in loss).
PTBXL_SCP_TO_CLASSES = {
    '1AVB':  ['IAVB'],
    'AFIB':  ['AF'],
    'AFLT':  ['AFL'],
    'CLBBB': ['LBBB', 'BBB'],
    'CRBBB': ['RBBB', 'BBB'],
    'ILBBB': ['ILBBB', 'BBB'],
    'IRBBB': ['IRBBB'],
    'IVCD':  ['NSIVCB'],
    'LAFB':  ['LAnFB', 'LAD'],
    'LNGQT': ['LQT'],
    'LOWT':  ['TAb'],         # low T amplitude is a T-wave abnormality
    'LPR':   ['LPR'],
    'LVOLT': ['LQRSV'],
    'NDT':   ['TAb'],
    'NORM':  ['NSR'],
    'NST_':  ['TAb'],
    'NT_':   ['TAb'],
    'INVT':  ['TAb'],         # inverted T is a T-wave abnormality
    'PAC':   ['PAC'],
    'PACE':  ['PR'],
    'PVC':   ['PVC'],
    'QWAVE': ['QAb'],
    'SARRH': ['SA'],
    'SBRAD': ['SB'],
    'SR':    ['NSR'],
    'STACH': ['STach'],
    'TAB_':  ['TAb'],
}

# Set of class indices that PTBXL can represent (observable classes).
PTBXL_COVERED_INDICES = set()
for scp, cls_list in PTBXL_SCP_TO_CLASSES.items():
    for abbr in cls_list:
        PTBXL_COVERED_INDICES.add(CLASS_TO_IDX[abbr])

PTBXL_UNCOVERED_INDICES = set(range(NUM_CLASSES_26)) - PTBXL_COVERED_INDICES
# Expected uncovered: {Brady, PRWP, RAD}


def ptbxl_scp_to_26(scp_codes_str_or_dict, confidence_threshold=0.0):
    """Convert PTBXL scp_codes -> 26-dim label vector.

    Covered classes default to 0 (known negative).
    Uncovered classes get -1 (masked in loss).

    Args:
        scp_codes_str_or_dict: raw string like "{'NORM': 100.0}" or dict
        confidence_threshold: min confidence to count as positive. Default 0 =
            presence-based (any SCP code listed counts as positive). PTBXL uses
            confidence 0 for automated diagnoses that are still valid positives;
            a strict threshold of 50 drops ~97% of AFIB cases. Presence-based
            labeling also matches PN2021 (SNOMED binary) and MIMIC (keyword
            binary) so cross-dataset label semantics are consistent.
    Returns: np.ndarray (26,) float32
    """
    if isinstance(scp_codes_str_or_dict, str):
        scp_codes = ast.literal_eval(scp_codes_str_or_dict)
    else:
        scp_codes = scp_codes_str_or_dict

    label = np.full(NUM_CLASSES_26, -1.0, dtype=np.float32)
    for i in PTBXL_COVERED_INDICES:
        label[i] = 0.0

    for code, confidence in scp_codes.items():
        if confidence >= confidence_threshold and code in PTBXL_SCP_TO_CLASSES:
            for abbr in PTBXL_SCP_TO_CLASSES[code]:
                label[CLASS_TO_IDX[abbr]] = 1.0
    return label


def get_ptbxl_26_labels(csv_path, folds=None, confidence_threshold=0.0):
    """Load PTBXL metadata and produce 26-dim label matrix.

    Returns:
        indices: list of row indices into raw100.npy (DataFrame row order)
        labels: np.ndarray (N, 26) float32, with -1 for uncovered classes
        df: filtered DataFrame
    """
    df_full = pd.read_csv(csv_path)
    if folds is not None:
        mask = df_full.strat_fold.isin(folds)
        indices = np.where(mask)[0].tolist()
        df = df_full[mask].reset_index(drop=True)
    else:
        indices = list(range(len(df_full)))
        df = df_full.copy()

    labels = np.stack([
        ptbxl_scp_to_26(row, confidence_threshold) for row in df.scp_codes
    ])
    return indices, labels, df


# ── PhysioNet 2021 SNOMED -> 26 mapping ───────────────────────────────────────
def snomed_to_26(snomed_codes):
    """Convert list of SNOMED codes -> 26-dim binary label vector.

    PN2021 fully covers all 26 classes so no -1 is produced here.
    """
    label = np.zeros(NUM_CLASSES_26, dtype=np.float32)
    for code in snomed_codes:
        if code in _SNOMED_TO_IDX:
            for idx in _SNOMED_TO_IDX[code]:
                label[idx] = 1.0
    return label


def has_any_scored_class(snomed_codes):
    return any(c in ALL_SCORED_SNOMED for c in snomed_codes)


# ── MIMIC report keyword -> 26 mapping ────────────────────────────────────────
# Keys are class abbreviations; values are lists of regex patterns matched
# case-insensitively against the concatenated report_* text.
MIMIC_KEYWORD_PATTERNS = {
    'IAVB':   [r'1st[-\s]degree a[\s-]?v block', r'first[-\s]degree a[\s-]?v block',
               r'1st[-\s]degree av', r'1 av block', r'1st degree heart block'],
    'AF':     [r'atrial fibrillation', r'\bafib\b', r'a[\s-]?fib'],
    'AFL':    [r'atrial flutter', r'\baflutter\b'],
    'BBB':    [r'\bbundle branch block\b'],
    'Brady':  [r'\bbradycardia\b', r'\bbrady\b'],
    'LBBB':   [r'left bundle branch block', r'\blbbb\b'],
    'RBBB':   [r'right bundle branch block', r'\brbbb\b'],
    'ILBBB':  [r'incomplete left bundle'],
    'IRBBB':  [r'incomplete right bundle', r'\birbbb\b'],
    'LAD':    [r'left axis deviation', r'leftward axis'],
    'LAnFB':  [r'left anterior fascicular', r'\blafb\b', r'left anterior hemiblock'],
    'LQRSV':  [r'low qrs voltage', r'low voltage'],
    'NSIVCB': [r'nonspecific.*conduction', r'intraventricular conduction (defect|delay|disturbance)',
               r'\bivcd\b'],
    'PR':     [r'paced rhythm', r'pacemaker rhythm', r'ventricular[\s-]paced',
               r'atrial[\s-]paced', r'a[\s-]?v paced', r'\bpacing\b'],
    'PRWP':   [r'poor r[\s-]wave progression', r'poor r progression'],
    'PAC':    [r'premature atrial', r'\bpac\b', r'atrial premature'],
    'PVC':    [r'premature ventricular', r'\bpvc\b', r'ventricular premature'],
    'LPR':    [r'prolonged pr interval', r'long pr interval'],
    'LQT':    [r'prolonged qt', r'long qt', r'qt prolongation'],
    'QAb':    [r'q[-\s]?wave abnormal', r'abnormal q[\s-]?wave', r'pathologic(al)? q'],
    'RAD':    [r'right axis deviation', r'rightward axis'],
    'SA':     [r'sinus arrhythmia'],
    'SB':     [r'sinus bradycardia'],
    'NSR':    [r'sinus rhythm', r'\bnsr\b', r'normal ecg', r'normal sinus'],
    'STach':  [r'sinus tachycardia'],
    'TAb':    [r't wave abnormal', r't[-\s]?wave abnorm', r'abnormal t[\s-]?wave',
               r't wave (inversion|inverted|change)', r'nonspecific t wave',
               r'nonspecific st.*t', r'non-specific t'],
}

# Precompile regex
_MIMIC_COMPILED = {
    abbr: [re.compile(p, re.IGNORECASE) for p in plist]
    for abbr, plist in MIMIC_KEYWORD_PATTERNS.items()
}

MIMIC_COVERED_INDICES = set(CLASS_TO_IDX[a] for a in MIMIC_KEYWORD_PATTERNS.keys())
MIMIC_UNCOVERED_INDICES = set(range(NUM_CLASSES_26)) - MIMIC_COVERED_INDICES


def mimic_report_to_26(report_text):
    """Convert concatenated MIMIC report text -> 26-dim label vector.

    Covered classes default to 0 (assumed negative if no matching keyword).
    Uncovered classes get -1 (none currently; all 26 are covered by default).
    """
    label = np.full(NUM_CLASSES_26, -1.0, dtype=np.float32)
    for i in MIMIC_COVERED_INDICES:
        label[i] = 0.0

    if not report_text:
        return label

    text = report_text.lower()
    for abbr, patterns in _MIMIC_COMPILED.items():
        idx = CLASS_TO_IDX[abbr]
        for pat in patterns:
            if pat.search(text):
                label[idx] = 1.0
                break
    return label


def mimic_row_to_report_text(row, report_prefix='report_', max_fields=18):
    """Join non-NaN report_0..report_17 into single text blob."""
    fields = []
    for i in range(max_fields):
        col = f'{report_prefix}{i}'
        if col in row.index:
            val = row[col]
            if isinstance(val, str) and val.strip():
                fields.append(val.strip())
    return ' | '.join(fields)


# ── Quick integrity check ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"NUM_CLASSES_26 = {NUM_CLASSES_26}")
    print(f"Classes: {CLASS_NAMES_26}")
    print(f"TIER1 = {TIER1}")
    print(f"TIER2 = {TIER2} ({len(TIER2)} classes)")
    print(f"TIER1 indices = {TIER1_IDX}")
    print(f"TIER2 indices = {TIER2_IDX}")
    print(f"PTBXL covers {len(PTBXL_COVERED_INDICES)}/26 classes")
    print(f"PTBXL uncovered (forced -1): "
          f"{[CLASS_NAMES_26[i] for i in sorted(PTBXL_UNCOVERED_INDICES)]}")
    print(f"MIMIC covers {len(MIMIC_COVERED_INDICES)}/26 classes")

    # Verify Tier-1 is covered by PTBXL
    for cls in TIER1:
        assert CLASS_TO_IDX[cls] in PTBXL_COVERED_INDICES, \
            f"PTBXL doesn't cover Tier-1 class {cls}"
    print(f"All Tier-1 classes covered by PTBXL: OK")

    # Sanity on SNOMED map
    assert 164889003 in _SNOMED_TO_IDX, "AF SNOMED missing"
    assert _SNOMED_TO_IDX[164889003] == [CLASS_TO_IDX['AF']]
    print(f"SNOMED map: OK")
