"""Package-owned PTB-XL Super5 label conversion policy.

This module owns the Super5 class order, PN2021 SNOMED projection, MIMIC report
regex mapping, and PTB-XL diagnostic-class lookup used by legacy wrappers. Keep
changes here tightly controlled: any mapping drift changes cited metrics and
requires a mapping version/hash bump plus cache rebuild.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES_SUPER5 = ("CD", "HYP", "MI", "NORM", "STTC")
NUM_SUPER5 = len(CLASS_NAMES_SUPER5)
SUPER5_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES_SUPER5)}
_SUPER5_NORM_IDX = SUPER5_TO_IDX["NORM"]
_SUPER5_ABNORMAL_IDX = tuple(i for name, i in SUPER5_TO_IDX.items() if name != "NORM")


def _default_scp_statements_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "datasets" / "PTBXL" / "scp_statements.csv"


SCP_STATEMENTS_PATH = Path(
    os.environ.get("PTBXL_SCP_STATEMENTS", str(_default_scp_statements_path()))
)


@functools.lru_cache(maxsize=1)
def _load_scp_super5_map() -> dict[str, str]:
    """Return PTB-XL SCP code to diagnostic_class mapping."""
    import pandas as pd

    df = pd.read_csv(SCP_STATEMENTS_PATH, index_col=0)
    scp_to_super5: dict[str, str] = {}
    for scp, row in df.iterrows():
        if row.get("diagnostic") != 1.0:
            continue
        super5 = row.get("diagnostic_class")
        if isinstance(super5, str) and super5.strip():
            scp_to_super5[str(scp)] = super5.strip()
    return scp_to_super5


def ptbxl_scp_to_super5(scp_codes_str_or_dict: str | dict[str, Any], confidence_threshold: float = 0.0) -> np.ndarray:
    """PTB-XL scp_codes -> (5,) float32 multi-hot. 0/1 only."""
    if isinstance(scp_codes_str_or_dict, str):
        scp_codes = ast.literal_eval(scp_codes_str_or_dict)
    else:
        scp_codes = scp_codes_str_or_dict

    scp_to_super5 = _load_scp_super5_map()
    label = np.zeros(NUM_SUPER5, dtype=np.float32)
    for code, conf in scp_codes.items():
        if conf >= confidence_threshold and code in scp_to_super5:
            label[SUPER5_TO_IDX[scp_to_super5[code]]] = 1.0
    return label


SUPER5_PN2021_MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"

# PN2021 SNOMED -> PTB-XL Super5 semantic projection.
#
# PhysioNet/CinC 2021 defines SNOMED-CT labels and Challenge scoring labels,
# not an official PN2021 to PTB-XL-super5 crosswalk. This mapping is a project
# policy for external-center evaluation.
SNOMED_TO_SUPER5_POSITIVE = {
    # MI - infarction codes only.
    164865005: "MI",
    164867002: "MI",
    57054005: "MI",
    54329005: "MI",
    22298006: "MI",
    401303003: "MI",
    233843008: "MI",
    # STTC - ST/T wave changes and ischemia.
    164934002: "STTC",
    111975006: "STTC",
    164931005: "STTC",
    429622005: "STTC",
    164930006: "STTC",
    59931005: "STTC",
    164861001: "STTC",
    413444003: "STTC",
    413844008: "STTC",
    428750005: "STTC",
    55930002: "STTC",
    704997005: "STTC",
    425623009: "STTC",
    425419005: "STTC",
    426434006: "STTC",
    370365005: "STTC",
    77867006: "STTC",
    164937009: "STTC",
    # CD - bundle branch blocks, AV blocks, and conduction abnormalities.
    270492004: "CD",
    195042002: "CD",
    54016002: "CD",
    426183003: "CD",
    204384007: "CD",
    27885002: "CD",
    233917008: "CD",
    65778007: "CD",
    164947007: "CD",
    164909002: "CD",
    733534002: "CD",
    59118001: "CD",
    713427006: "CD",
    251120003: "CD",
    713426002: "CD",
    6374002: "CD",
    445118002: "CD",
    445211001: "CD",
    698252002: "CD",
    82226007: "CD",
    74390002: "CD",
    26749005: "CD",
    195060002: "CD",
    418818005: "CD",
    49578007: "CD",
    # HYP - hypertrophy and chamber enlargement.
    164873001: "HYP",
    89792004: "HYP",
    266249003: "HYP",
    446358003: "HYP",
    446813000: "HYP",
    67741000119109: "HYP",
    195126007: "HYP",
    253352002: "HYP",
    253339007: "HYP",
    55827005: "HYP",
    67751000119106: "HYP",
    164912004: "HYP",
    251223006: "HYP",
    # SJR/RGQ-reviewed repolarization morphology.
    251259000: "STTC",
}

NORM_POSITIVE_SNOMEDS = frozenset({
    426783006,  # sinus rhythm
})

NORM_SUPPRESS_SNOMEDS = frozenset({
    426177001,
    427084000,
    427393009,
    164889003,
    164890007,
    284470004,
    63593006,
    427172004,
    17338001,
    39732003,
    47665007,
    251146004,
    365413008,
    426627000,
    10370003,
    251268003,
    251266004,
    164951009,
    164942001,
    233892002,
    251187003,
    61277005,
    426664006,
    106068003,
    195080001,
    251173003,
    713422000,
    50799005,
    29320008,
    251166008,
    233897008,
    251170000,
    74615001,
    426749004,
    698247007,
    13640000,
    49260003,
    251200008,
    426995002,
    251164006,
    426648003,
    251182009,
    282825002,
    67198005,
    425856008,
    251205003,
    164921003,
    314208002,
    5609005,
    60423000,
    17366009,
    251168009,
    426761007,
    11157007,
    164884008,
    75532003,
    81898007,
    164896001,
    111288001,
    164895002,
    251180001,
    195101003,
    164917005,
})

SUPER5_PN2021_IGNORED_SNOMEDS = frozenset({
    53741008,
    84114007,
    368009,
    266257000,
    251198002,
    251199005,
    428417006,
    61721007,
    251139008,
})

SNOMED_TO_SUPER5 = {
    **SNOMED_TO_SUPER5_POSITIVE,
    **{code: "NORM" for code in NORM_POSITIVE_SNOMEDS},
}


def _stable_hash_mapping(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


SUPER5_PN2021_MAPPING_HASH = _stable_hash_mapping({
    "version": SUPER5_PN2021_MAPPING_VERSION,
    "positive": SNOMED_TO_SUPER5_POSITIVE,
    "norm_positive": sorted(NORM_POSITIVE_SNOMEDS),
    "norm_suppress": sorted(NORM_SUPPRESS_SNOMEDS),
    "ignored": sorted(SUPER5_PN2021_IGNORED_SNOMEDS),
})


def get_super5_pn2021_mapping_metadata() -> dict[str, Any]:
    return {
        "mapping_version": SUPER5_PN2021_MAPPING_VERSION,
        "mapping_hash": SUPER5_PN2021_MAPPING_HASH,
        "class_names": list(CLASS_NAMES_SUPER5),
        "ignored_snomeds": sorted(SUPER5_PN2021_IGNORED_SNOMEDS),
    }


def snomed_list_to_super5(snomed_codes: list[int] | tuple[int, ...]) -> np.ndarray:
    """PN2021 SNOMED list -> (5,) float32 multi-hot."""
    label = np.zeros(NUM_SUPER5, dtype=np.float32)
    has_norm_candidate = False
    has_norm_suppress = False
    for code in snomed_codes:
        cls = SNOMED_TO_SUPER5_POSITIVE.get(int(code))
        if cls is not None:
            label[SUPER5_TO_IDX[cls]] = 1.0
        if int(code) in NORM_POSITIVE_SNOMEDS:
            has_norm_candidate = True
        if int(code) in NORM_SUPPRESS_SNOMEDS:
            has_norm_suppress = True
    if has_norm_candidate and not any(label[i] for i in _SUPER5_ABNORMAL_IDX) and not has_norm_suppress:
        label[_SUPER5_NORM_IDX] = 1.0
    else:
        label[_SUPER5_NORM_IDX] = 0.0
    return label


MIMIC_SUPER5_PATTERNS = {
    "NORM": [
        r"sinus rhythm",
        r"\bnsr\b",
        r"normal ecg",
        r"normal sinus",
        r"within normal limits",
        r"otherwise normal",
        r"sinus arrhythmia",
        r"sinus bradycardia",
        r"sinus tachycardia",
    ],
    "MI": [
        r"myocardial infarct",
        r"\bmi\b(?! \w)",
        r"\binfarct",
        r"(anterior|inferior|lateral|posterior|septal|anteroseptal|anterolateral|inferolateral|inferoposterior) (mi|infarct)",
        r"\bstemi\b",
        r"\bnstemi\b",
        r"st elevation myocardial",
        r"old infarct",
        r"recent infarct",
        r"acute infarct",
    ],
    "STTC": [
        r"st elevation",
        r"st depression",
        r"st[- ]?t chang",
        r"t[- ]?wave (invert|abnorm|chang)",
        r"\bischemi",
        r"prolonged qt",
        r"\blong qt",
        r"qt prolong",
        r"q[- ]?wave abnormal",
        r"pathologic(al)? q[- ]?wave",
        r"nonspecific (st|t)[- ]?wave",
        r"\bstc",
        r"repolarization abnormal",
    ],
    "CD": [
        r"bundle branch block",
        r"\blbbb\b",
        r"\brbbb\b",
        r"\bav block\b",
        r"heart block",
        r"(1st|2nd|3rd|first|second|third)[- ]degree",
        r"fascicular block",
        r"\blafb\b",
        r"\blpfb\b",
        r"intraventricular.*(block|conduction (defect|delay|disturb))",
        r"\bivcd\b",
        r"wolff[- ]?parkinson",
        r"\bwpw\b",
        r"preexcitation",
        r"pace[dr]?",
        r"pacemaker",
        r"paced rhythm",
    ],
    "HYP": [
        r"(ventricular|atrial|left|right) hypertrophy",
        r"\blvh\b",
        r"\brvh\b",
        r"atrial enlargement",
        r"chamber enlargement",
        r"(left|right) atrial enlargement",
        r"\blae\b",
        r"\brae\b",
        r"biventricular hypertrophy",
        r"septal hypertrophy",
    ],
}
_MIMIC_SUPER5_COMPILED = {
    cls: [re.compile(pattern) for pattern in patterns]
    for cls, patterns in MIMIC_SUPER5_PATTERNS.items()
}


def mimic_report_to_super5(report_text: str | None) -> np.ndarray:
    """MIMIC cart report text -> (5,) float32 multi-hot."""
    label = np.zeros(NUM_SUPER5, dtype=np.float32)
    if not report_text:
        return label
    text = report_text.lower()
    for cls, patterns in _MIMIC_SUPER5_COMPILED.items():
        for pattern in patterns:
            if pattern.search(text):
                label[SUPER5_TO_IDX[cls]] = 1.0
                break
    if any(label[i] for i in _SUPER5_ABNORMAL_IDX):
        label[_SUPER5_NORM_IDX] = 0.0
    return label
