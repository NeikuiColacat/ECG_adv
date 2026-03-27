"""
PTBXL SCP 编码 ↔ EfficientNet 77 类 pattern 索引映射

ECG_PATTERNS 列表来自 model/DeepECG/utils/constants.py
索引 = pattern 在 ECG_PATTERNS 列表中的位置

PTBXL superdiagnostic classes:
  0: CD  (Conduction Disorder)
  1: HYP (Hypertrophy)
  2: MI  (Myocardial Infarction)
  3: NORM (Normal ECG)
  4: STTC (ST/T-Change)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ECG_PATTERNS 索引速查（来自 constants.py 的列表顺序）
# 0: Sinusal, 1: Regular, 2: Monomorph, 3: QS complex V1-V2-V3, 4: R complex V5-V6
# 5: T wave inv (inferior), 6: LBBB, 7: RaVL>11mm, 8: SV1+RV5/RV6>35mm
# 9: T wave inv (lateral), 10: T wave inv (anterior), 11: Left axis deviation
# 12: LVH, 13: Bradycardia, 14: Q wave (inferior), 15: Afib
# 16: Irregularly irregular, 17: Atrial tachy, 18: NICD, 19: PVC
# 20: Polymorph, 21: T wave inv (septal), 22: RBBB, 23: Ventricular paced
# 24: ST elev (anterior), 25: ST elev (septal), 26: 1st AVB, 27: PAC
# 28: Atrial flutter, 29: rSR' V1-V2, 30: qRS V5-V6-I-aVL
# 31: Left anterior fascicular block, 32: Right axis deviation
# 33: 2nd AVB mobitz1, 34: ST depression (inferior), 35: Acute pericarditis
# 36: ST elev (inferior), 37: Low voltage, 38: Regularly irregular
# 39: Junctional rhythm, 40: Left atrial enlargement
# 41: ST elev (lateral), 42: Atrial paced, 43: RVH, 44: Delta wave
# 45: WPW, 46: Prolonged QT, 47: ST depression (anterior), 48: QRS neg in III
# 49: Q wave (lateral), 50: SVT, 51: ST downsloping, 52: ST depression (lateral)
# 53: 2nd AVB mobitz2, 54: U wave, 55: R/S ratio V1-V2>1, 56: RV1+SV6>11mm
# 57: Left posterior fascicular block, 58: Right atrial enlargement
# 59: ST depression (septal), 60: Q wave (septal), 61: Q wave (anterior)
# 62: ST upsloping, 63: Right superior axis, 64: Ventricular tachy
# 65: ST elev (posterior), 66: Ectopic atrial rhythm, 67: Lead misplacement
# 68: Third Degree AVB, 69: Acute MI, 70: Early repolarization
# 71: Q wave (posterior), 72: Bi-atrial enlargement, 73: LV pacing
# 74: Brugada, 75: Ventricular Rhythm, 76: no_qrs

# ——— PTBXL SCP code → EfficientNet 77 类 pattern 索引 ———
# 每个 SCP code 对应的 pattern 索引列表
# 来源：roadmap.md Step 1 + ECG_PATTERNS 顺序核对

SCP_TO_EFFICIENTNET: Dict[str, List[int]] = {
    # ---- Myocardial Infarction (MI) ----
    "IMI":  [14, 36],        # Q wave inferior + ST elev inferior
    "ASMI": [60, 25],        # Q wave septal + ST elev septal
    "LMI":  [49, 41],        # Q wave lateral + ST elev lateral
    "AMI":  [61, 24],        # Q wave anterior + ST elev anterior
    "ALMI": [49, 41],        # Q wave lateral + ST elev lateral
    "INJAS":[60, 25],        # Q wave septal + ST elev septal
    "IPLMI":[49, 41],        # Q wave lateral + ST elev lateral
    "IPMI": [14, 36],        # Q wave inferior + ST elev inferior
    "ILMI": [49, 41],        # Q wave lateral + ST elev lateral
    "PMI":  [71, 65],        # Q wave posterior + ST elev posterior
    "INJAL":[49, 41],        # Q wave lateral + ST elev lateral
    "INJIN":[14, 36],        # Q wave inferior + ST elev inferior
    "INJLA":[49, 41],        # Q wave lateral + ST elev lateral
    "INJIL":[14, 36, 49],    # Q wave inferior/lateral + ST elev inferior
    "INJP": [71, 65],        # Q wave posterior + ST elev posterior
    "INJAP":[61, 24],        # Q wave anterior + ST elev anterior
    "INJAS2":[60, 25],
    "ANEUR":[61, 24],        # Q wave anterior (aneurysm)

    # ---- ST/T Changes (STTC) ----
    "NDT":  [5, 9, 10, 21],  # T wave inversions (non-diagnostic)
    "NST_": [34, 47, 52, 59],# ST depressions (non-specific)
    "DIG":  [34, 47],        # ST depression (digitalis)
    "LNGQT":[46],            # Prolonged QT
    "ISCA": [47, 52],        # ST depression anterior/lateral (ischemia)
    "ISCI": [34, 47],        # ST depression inferior (ischemia)
    "ISC_": [34, 47, 52],    # Non-specific ischemia
    "ISCAL":[47, 52],        # Ischemia anterior/lateral
    "ISCIN":[34],            # Ischemia inferior
    "ISCLA":[52],            # Ischemia lateral
    "ISCAS":[47],            # Ischemia anterior/septal
    "ISCIL":[52, 34],        # Ischemia inferior/lateral
    "ADRH": [36, 41],        # ST elevation (general)
    "EL":   [62],            # ST upsloping
    "DE":   [51],            # ST downsloping

    # ---- Normal ----
    "NORM": [0, 1],          # Sinusal + Regular
    "SR":   [0, 1],          # Sinus rhythm

    # ---- Conduction Disorders (CD) ----
    "CRBBB":[22],            # Right bundle branch block
    "CLBBB":[6],             # Left bundle branch block
    "IRBBB":[22],            # Incomplete RBBB
    "1AVB": [26],            # 1st degree AV block
    "2AVB": [33],            # 2nd degree AV block mobitz 1
    "2AVB2":[53],            # 2nd degree AV block mobitz 2
    "3AVB": [68],            # 3rd degree AV block
    "LAFB": [31],            # Left anterior fascicular block
    "LPFB": [57],            # Left posterior fascicular block
    "WPW":  [45, 44],        # WPW + delta wave
    "IVCD": [18],            # NICD

    # ---- Hypertrophy (HYP) ----
    "LVH":  [12],            # Left ventricular hypertrophy
    "RVH":  [43],            # Right ventricular hypertrophy
    "LAO/LAE": [40],         # Left atrial enlargement
    "RAO/RAE": [58],         # Right atrial enlargement
    "BAO/BAE": [72],         # Bi-atrial enlargement
    "SEHYP":[8, 7],          # ST elevation hypertrophy (SV1+RV5>35 + RaVL>11)

    # ---- Rhythm Disorders ----
    "AFIB": [15, 16],        # Afib + irregularly irregular
    "AFLT": [28],            # Atrial flutter
    "SARRH":[0],             # Sinus arrhythmia (Sinusal)
    "SBRAD":[13],            # Sinus bradycardia
    "STACH":[0, 1],          # Sinus tachycardia (sinusal+regular)
    "SVTAC":[50],            # SVT
    "BIGU": [19],            # Bigeminy (PVC)
    "TRIGU":[19],            # Trigeminy (PVC)
    "PACE": [23, 42],        # Paced (ventricular + atrial)
    "PSVT": [50],            # Paroxysmal SVT
    "JUNCTIONAL":[39],       # Junctional rhythm
    "ECTAC":[66],            # Ectopic atrial rhythm
    "VTACH":[64, 20],        # Ventricular tachycardia + polymorph
    "SVARR":[50],            # SVA

    # ---- Other ----
    "LOWV": [37],            # Low voltage
    "LNGQT2":[46],
    "ABQRS":[76],            # no_qrs
    "RAD":  [32],            # Right axis deviation
    "LAD":  [11],            # Left axis deviation
    "QWAVE":[14],            # Q wave (generic → inferior)
    "PEAC": [35],            # Acute pericarditis
    "BRUGADA":[74],          # Brugada
    "ERP":  [70],            # Early repolarization
    "UWAVE":[54],            # U wave
}

# ——— Prompt Key → 目标 EfficientNet pattern 索引 ———
# 用于 adv_generate.py 中计算 boundary loss
PROMPT_TO_PATTERN_INDICES: Dict[str, List[int]] = {
    "MI_inferior":  [14, 36],       # Q wave inferior + ST elev inferior
    "MI_anterior":  [61, 24],       # Q wave anterior + ST elev anterior
    "MI_lateral":   [49, 41],       # Q wave lateral + ST elev lateral
    "MI_acute":     [69],           # Acute MI
    "ISCHEMIA_ant": [47],           # ST depression anterior
    "ISCHEMIA_inf": [34],           # ST depression inferior
    "NORM":         [0, 1],         # Sinusal + Regular
    "LBBB":         [6],            # LBBB
    "RBBB":         [22],           # RBBB
    "LVH":          [12],           # LVH
    # ── MIMIC-IV 风格 MI 困难样本（仅使用 EfficientNet 实测 prob > 0.3 的 pattern）──
    # MIMIC ECG 报告用小写 | 管道分隔，prompt_process 会转换为自然语言
    "MIMIC_MI_inferior":       [14, 36],        # Q wave inferior + ST elev inferior (0.549/0.476)
    "MIMIC_MI_acute":          [69],            # Acute MI (0.622) - EfficientNet 最敏感
    "MIMIC_MI_acute_inferior": [69, 14, 36],    # Acute MI + 下壁特征联合引导
    "MIMIC_MI_inferolateral":  [14, 36],          # Q wave inferior + ST elev inferior (侧壁 49,41 置信度太低)
}

# PTBXL superdiagnostic class → prompt key（用于选择参考 ECG）
SUPERDIAG_TO_PROMPTS: Dict[str, List[str]] = {
    "MI":   ["MI_inferior", "MI_anterior", "MI_lateral", "MI_acute"],
    "STTC": ["ISCHEMIA_ant", "ISCHEMIA_inf"],
    "NORM": ["NORM"],
    "CD":   ["LBBB", "RBBB"],
    "HYP":  ["LVH"],
}


def scp_codes_to_77_labels(scp_dict: Dict[str, float], threshold: float = 0.0) -> torch.Tensor:
    """
    将 PTBXL scp_codes 字典转换为 EfficientNet 77 维二值标签向量

    Args:
        scp_dict: PTBXL scp_codes 字典，如 {"IMI": 100.0, "NORM": 0.0}
        threshold: SCP code 置信度阈值（> threshold 才算阳性）

    Returns:
        labels: shape (77,), float32, 二值多标签向量
    """
    labels = torch.zeros(77, dtype=torch.float32)

    for scp_code, confidence in scp_dict.items():
        if confidence <= threshold:
            continue
        if scp_code in SCP_TO_EFFICIENTNET:
            for idx in SCP_TO_EFFICIENTNET[scp_code]:
                if 0 <= idx < 77:
                    labels[idx] = 1.0

    return labels


def get_target_indices(prompt_key: str) -> List[int]:
    """
    获取给定 prompt key 对应的 EfficientNet pattern 索引列表

    Args:
        prompt_key: 如 "MI_inferior", "NORM", "LBBB"

    Returns:
        indices: pattern 索引列表
    """
    return PROMPT_TO_PATTERN_INDICES.get(prompt_key, [])


def batch_scp_to_77_labels(
    scp_dicts: List[Dict[str, float]],
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    批量转换 SCP codes → 77-dim 标签矩阵

    Args:
        scp_dicts: List of scp_codes 字典
        threshold: 置信度阈值

    Returns:
        labels: shape (N, 77), float32
    """
    return torch.stack([scp_codes_to_77_labels(d, threshold) for d in scp_dicts])
