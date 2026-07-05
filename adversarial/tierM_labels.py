"""Tier-M ECGTwin generation prompts + dir names.

Class list itself is the canonical package-owned TIER_M, re-exported here so
the list never drifts between training and adversarial generation.
"""

from typing import Dict, List

from ecg_adv_gen.labels.tier26 import TIER_M as TIER_M_CLASSES

TIER_M_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(TIER_M_CLASSES)}

# /root/ECG_adv_Gen/model/ECGTwin/generation_result_by_disease/<dir>/
TIER_M_TO_ECGTWIN_DIR: Dict[str, str] = {
    "NSR":   "normal_ecg",
    "STach": "sinus_tachycardia-窦性心动过速",
    "AF":    "atrial_fibrillation-心房颤动",
    "IAVB":  "atrioventricular_block-房室传导阻滞",
    "LBBB":  "left_bundle_branch_block-左束支阻滞",
    "RBBB":  "right_bundle_branch_block-右束支阻滞",
}

# ECGTwin DDPM text prompts (pipe-delimited; get_text_embedding splits on |).
TIER_M_TEXT_PROMPT: Dict[str, str] = {
    "NSR":   "sinus rhythm|normal ecg",
    "STach": "sinus tachycardia|abnormal ecg",
    "AF":    "atrial fibrillation|abnormal ecg",
    "IAVB":  "first degree atrioventricular block|abnormal ecg",
    "LBBB":  "left bundle branch block|abnormal ecg",
    "RBBB":  "right bundle branch block|abnormal ecg",
}


def get_target_indices_tierM(class_name: str) -> List[int]:
    """Return the Tier-M head index for a given class name (wrapped in a list)."""
    if class_name not in TIER_M_CLASS_TO_IDX:
        raise KeyError(
            f"Unknown Tier-M class: {class_name!r}. "
            f"Valid: {TIER_M_CLASSES}"
        )
    return [TIER_M_CLASS_TO_IDX[class_name]]


if __name__ == "__main__":
    assert TIER_M_CLASS_TO_IDX == {
        "NSR": 0, "STach": 1, "AF": 2, "IAVB": 3, "LBBB": 4, "RBBB": 5
    }
    assert get_target_indices_tierM("NSR") == [0]
    assert get_target_indices_tierM("RBBB") == [5]
    try:
        get_target_indices_tierM("FooBar")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    assert set(TIER_M_TO_ECGTWIN_DIR.keys()) == set(TIER_M_CLASSES)
    assert set(TIER_M_TEXT_PROMPT.keys()) == set(TIER_M_CLASSES)
    print("tierM_labels.py smoke OK")
