"""
AUPRC / AUROC 评估

在 PTBXL test set (fold 10) 上比较：
  - Baseline: 原始 frozen EfficientNet
  - Finetuned: EfficientNetAdapter（frozen backbone + trained adapter）

输出格式：
    Category                   | Baseline AUPRC | Finetuned AUPRC | Δ
    Infarction or ischemia     | 0.089          | ???             | +???
      ST elevation (anterior)  | ...            | ...             |
    ...
"""

import sys
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.metrics import average_precision_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adversarial.label_mapping import batch_scp_to_77_labels
from adversarial.efficientnet_victim import load_efficientnet_jit, MHI_FACTOR, EFFICIENTNET_INPUT_LENGTH
from adversarial.efficientnet_adapter import EfficientNetAdapter

# ECG_PATTERNS 索引顺序（来自 model/DeepECG/utils/constants.py）
# 直接复制避免 import DeepECG（会触发 TensorFlow 导致 segfault）
ECG_PATTERNS = [
    "Sinusal", "Regular", "Monomorph", "QS complex in V1-V2-V3", "R complex in V5-V6",
    "T wave inversion (inferior - II, III, aVF)", "Left bundle branch block", "RaVL > 11 mm",
    "SV1 + RV5 or RV6 > 35 mm", "T wave inversion (lateral -I, aVL, V5-V6)",
    "T wave inversion (anterior - V3-V4)", "Left axis deviation", "Left ventricular hypertrophy",
    "Bradycardia", "Q wave (inferior - II, III, aVF)", "Afib", "Irregularly irregular",
    "Atrial tachycardia (>= 100 BPM)", "Nonspecific intraventricular conduction delay",
    "Premature ventricular complex", "Polymorph", "T wave inversion (septal- V1-V2)",
    "Right bundle branch block", "Ventricular paced", "ST elevation (anterior - V3-V4)",
    "ST elevation (septal - V1-V2)", "1st degree AV block", "Premature atrial complex",
    "Atrial flutter", "rSR' in V1-V2", "qRS in V5-V6-I, aVL", "Left anterior fascicular block",
    "Right axis deviation", "2nd degree AV block - mobitz 1",
    "ST depression (inferior - II, III, aVF)", "Acute pericarditis",
    "ST elevation (inferior - II, III, aVF)", "Low voltage", "Regularly irregular",
    "Junctional rhythm", "Left atrial enlargement", "ST elevation (lateral - I, aVL, V5-V6)",
    "Atrial paced", "Right ventricular hypertrophy", "Delta wave",
    "Wolff-Parkinson-White (Pre-excitation syndrome)", "Prolonged QT",
    "ST depression (anterior - V3-V4)", "QRS complex negative in III",
    "Q wave (lateral- I, aVL, V5-V6)", "Supraventricular tachycardia", "ST downslopping",
    "ST depression (lateral - I, avL, V5-V6)", "2nd degree AV block - mobitz 2", "U wave",
    "R/S ratio in V1-V2 >1", "RV1 + SV6 > 11 mm", "Left posterior fascicular block",
    "Right atrial enlargement", "ST depression (septal- V1-V2)", "Q wave (septal- V1-V2)",
    "Q wave (anterior - V3-V4)", "ST upslopping", "Right superior axis",
    "Ventricular tachycardia", "ST elevation (posterior - V7-V8-V9)",
    "Ectopic atrial rhythm (< 100 BPM)", "Lead misplacement", "Third Degree AV Block",
    "Acute MI", "Early repolarization", "Q wave (posterior - V7-V9)",
    "Bi-atrial enlargement", "LV pacing", "Brugada", "Ventricular Rhythm", "no_qrs",
]

ECG_CATEGORIES = {
    "Rhythm Disorders": [
        "Ventricular tachycardia", "Bradycardia", "Brugada",
        "Wolff-Parkinson-White (Pre-excitation syndrome)", "Atrial flutter",
        "Ectopic atrial rhythm (< 100 BPM)", "Atrial tachycardia (>= 100 BPM)",
        "Sinusal", "Ventricular Rhythm", "Supraventricular tachycardia",
        "Junctional rhythm", "Regular", "Regularly irregular", "Irregularly irregular",
        "Afib", "Premature ventricular complex", "Premature atrial complex",
    ],
    "Conduction Disorder": [
        "Left anterior fascicular block", "Delta wave", "2nd degree AV block - mobitz 2",
        "Left bundle branch block", "Right bundle branch block", "Left axis deviation",
        "Atrial paced", "Right axis deviation", "Left posterior fascicular block",
        "1st degree AV block", "Right superior axis",
        "Nonspecific intraventricular conduction delay", "Third Degree AV Block",
        "2nd degree AV block - mobitz 1", "Prolonged QT", "U wave", "LV pacing",
        "Ventricular paced",
    ],
    "Enlargement of the heart chambers": [
        "Bi-atrial enlargement", "Left atrial enlargement", "Right atrial enlargement",
        "Left ventricular hypertrophy", "Right ventricular hypertrophy",
    ],
    "Pericarditis": ["Acute pericarditis"],
    "Infarction or ischemia": [
        "Q wave (septal- V1-V2)", "ST elevation (anterior - V3-V4)",
        "Q wave (posterior - V7-V9)", "Q wave (inferior - II, III, aVF)",
        "Q wave (anterior - V3-V4)", "ST elevation (lateral - I, aVL, V5-V6)",
        "Q wave (lateral- I, aVL, V5-V6)", "ST depression (lateral - I, avL, V5-V6)",
        "Acute MI", "ST elevation (septal - V1-V2)",
        "ST elevation (inferior - II, III, aVF)", "ST elevation (posterior - V7-V8-V9)",
        "ST depression (inferior - II, III, aVF)", "ST depression (anterior - V3-V4)",
    ],
    "Other diagnoses": [
        "ST downslopping", "ST depression (septal- V1-V2)", "R/S ratio in V1-V2 >1",
        "RV1 + SV6 > 11 mm", "Polymorph", "rSR' in V1-V2", "QRS complex negative in III",
        "qRS in V5-V6-I, aVL", "QS complex in V1-V2-V3", "R complex in V5-V6",
        "RaVL > 11 mm", "T wave inversion (septal- V1-V2)", "SV1 + RV5 or RV6 > 35 mm",
        "T wave inversion (inferior - II, III, aVF)", "Monomorph",
        "T wave inversion (anterior - V3-V4)", "T wave inversion (lateral -I, aVL, V5-V6)",
        "Low voltage", "Lead misplacement", "ST depression (anterior - V3-V4)",
        "Early repolarization", "ST upslopping", "no_qrs",
    ],
}


def load_ptbxl_split(
    ptbxl_data_root: str,
    folds,
    n_per_superdiag: Optional[int] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载 PTBXL 指定 fold 的数据，返回 (ecg, labels_77)

    Args:
        ptbxl_data_root: PTBXL 根目录
        folds: fold 编号（int 或 list of int）
        n_per_superdiag: 每个 superdiagnostic class 最多取多少样本
                         None = 全部数据；设置小值（如 100）模拟少样本场景
        seed: 随机种子

    Returns:
        X: (N, 12, 2500) float32
        Y: (N, 77) float32 二值标签
    """
    if isinstance(folds, int):
        folds = [folds]

    data_root = Path(ptbxl_data_root)
    X_all = np.load(str(data_root / "raw100.npy"), allow_pickle=True)  # (21799, 1000, 12)
    df = pd.read_csv(str(data_root / "ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # 用 reset_index 获取位置索引（0-based），避免 ecg_id 间隙问题
    df_reset = df.reset_index()

    mask = df_reset["strat_fold"].isin(folds)
    df_split = df_reset[mask].copy()

    # 少样本采样（按 diagnostic_class 分层）
    if n_per_superdiag is not None and "diagnostic_class" in df_split.columns:
        rng = np.random.RandomState(seed)
        sampled_parts = []
        for cls, grp in df_split.groupby("diagnostic_class"):
            n = min(len(grp), n_per_superdiag)
            sampled_parts.append(grp.sample(n=n, random_state=rng))
        df_split = pd.concat(sampled_parts).sort_index()
        print(f"  Few-shot sampling: {n_per_superdiag}/class → {len(df_split)} total")

    pos_idx = df_split.index.values   # 0-based positional index
    scp_list = df_split["scp_codes"].tolist()

    X = X_all[pos_idx]                                         # (N, 1000, 12)
    X_resampled = resample(X, EFFICIENTNET_INPUT_LENGTH, axis=1)  # (N, 2500, 12)
    X_tensor = torch.from_numpy(X_resampled).float().transpose(1, 2)  # (N, 12, 2500)
    Y = batch_scp_to_77_labels(scp_list, threshold=0.0)        # (N, 77)

    return X_tensor, Y


def load_ptbxl_test(
    ptbxl_data_root: str,
    test_fold: int = 10,
    n_per_superdiag: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载 PTBXL test set，返回 (ecg, labels_77)

    Args:
        ptbxl_data_root: PTBXL 根目录
        test_fold: test fold 编号（默认 10）
        n_per_superdiag: 每类最多样本数（None = 全部；设小值模拟少样本场景）

    Returns:
        X: (N, 12, 2500) float32
        Y: (N, 77) float32 二值标签
    """
    X, Y = load_ptbxl_split(ptbxl_data_root, folds=test_fold, n_per_superdiag=n_per_superdiag)
    print(f"Test set: {X.shape[0]} samples (fold={test_fold})")
    return X, Y


@torch.no_grad()
def run_inference(
    model,
    X: torch.Tensor,
    batch_size: int = 64,
    device: str = "cuda",
    use_sigmoid: bool = True,
    is_adapter: bool = False,
) -> np.ndarray:
    """
    批量推理，返回概率矩阵 (N, 77)

    Args:
        model: EfficientNet JIT 模型 或 EfficientNetAdapter
        X: (N, 12, 2500)
        batch_size: 推理 batch size
        device: 设备
        use_sigmoid: 是否对输出做 sigmoid（JIT 模型输出 logits 时需要）
        is_adapter: 是否是 EfficientNetAdapter（其 forward 返回 logits）

    Returns:
        probs: (N, 77) numpy array
    """
    all_probs = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size].to(device)
        if is_adapter:
            logits = model(batch)
            probs = torch.sigmoid(logits)
        else:
            # JIT 模型：需要乘 mhi_factor，输出 logits
            logits = model(batch * MHI_FACTOR)
            probs = torch.sigmoid(logits) if use_sigmoid else logits
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pattern_names: List[str],
    categories: Dict[str, List[str]],
    min_positive: int = 5,
) -> Dict:
    """
    计算 per-pattern 和 per-category 的 AUPRC / AUROC

    Args:
        y_true: (N, 77) 真实标签
        y_pred: (N, 77) 预测概率
        pattern_names: 77 个 pattern 名称
        categories: 类别 → pattern 名称列表的字典
        min_positive: 最少正样本数（少于此数跳过计算）

    Returns:
        metrics: dict with keys 'per_pattern', 'per_category'
    """
    pattern_to_idx = {name: i for i, name in enumerate(pattern_names)}

    per_pattern = {}
    for i, name in enumerate(pattern_names):
        n_pos = int(y_true[:, i].sum())
        if n_pos < min_positive:
            per_pattern[name] = {"auprc": float("nan"), "auroc": float("nan"), "n_pos": n_pos}
            continue
        try:
            auprc = average_precision_score(y_true[:, i], y_pred[:, i])
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except Exception:
            auprc = float("nan")
            auroc = float("nan")
        per_pattern[name] = {"auprc": auprc, "auroc": auroc, "n_pos": n_pos}

    per_category = {}
    for cat_name, pattern_list in categories.items():
        valid_patterns = [p for p in pattern_list if p in pattern_to_idx]
        indices = [pattern_to_idx[p] for p in valid_patterns]
        if not indices:
            continue

        cat_true = y_true[:, indices]
        cat_pred = y_pred[:, indices]

        # Micro-average over valid patterns (有足够正样本)
        valid_auprcs = []
        valid_aurocs = []
        for p, idx in zip(valid_patterns, indices):
            m = per_pattern.get(p, {})
            if not np.isnan(m.get("auprc", float("nan"))):
                valid_auprcs.append(m["auprc"])
            if not np.isnan(m.get("auroc", float("nan"))):
                valid_aurocs.append(m["auroc"])

        per_category[cat_name] = {
            "macro_auprc": np.mean(valid_auprcs) if valid_auprcs else float("nan"),
            "macro_auroc": np.mean(valid_aurocs) if valid_aurocs else float("nan"),
            "n_patterns": len(valid_auprcs),
        }

    return {"per_pattern": per_pattern, "per_category": per_category}


def print_comparison_table(
    baseline_metrics: Dict,
    finetuned_metrics: Optional[Dict],
    categories: Dict[str, List[str]],
):
    """打印基线 vs 微调后的对比表格"""
    print("\n" + "=" * 85)
    header = f"{'Category / Pattern':<45} | {'Baseline AUPRC':>14}"
    if finetuned_metrics:
        header += f" | {'Finetuned AUPRC':>15} | {'Δ':>6}"
    print(header)
    print("-" * 85)

    for cat_name, pattern_list in categories.items():
        b_cat = baseline_metrics["per_category"].get(cat_name, {})
        b_cat_auprc = b_cat.get("macro_auprc", float("nan"))

        if finetuned_metrics:
            f_cat = finetuned_metrics["per_category"].get(cat_name, {})
            f_cat_auprc = f_cat.get("macro_auprc", float("nan"))
            delta = f_cat_auprc - b_cat_auprc if not (np.isnan(f_cat_auprc) or np.isnan(b_cat_auprc)) else float("nan")
            row = f"{cat_name:<45} | {b_cat_auprc:>14.3f} | {f_cat_auprc:>15.3f} | {delta:>+6.3f}"
        else:
            row = f"{cat_name:<45} | {b_cat_auprc:>14.3f}"
        print(row)

        # Per-pattern detail
        for pattern in pattern_list:
            b_p = baseline_metrics["per_pattern"].get(pattern, {})
            b_auprc = b_p.get("auprc", float("nan"))
            if np.isnan(b_auprc):
                continue

            name_short = f"  {pattern}"[:44]
            if finetuned_metrics:
                f_p = finetuned_metrics["per_pattern"].get(pattern, {})
                f_auprc = f_p.get("auprc", float("nan"))
                delta_p = f_auprc - b_auprc if not np.isnan(f_auprc) else float("nan")
                row = f"{name_short:<45} | {b_auprc:>14.3f} | {f_auprc:>15.3f} | {delta_p:>+6.3f}"
            else:
                row = f"{name_short:<45} | {b_auprc:>14.3f}"
            print(row)

    print("=" * 85)


def evaluate(
    ptbxl_data_root: str,
    adapter_path: Optional[str] = None,
    efficientnet_weight_path: Optional[str] = None,
    device: str = "cuda",
    output_path: Optional[str] = None,
    batch_size: int = 64,
) -> Dict:
    """
    主评估函数

    Args:
        ptbxl_data_root: PTBXL 根目录
        adapter_path: adapter 权重路径（None 则只评估 baseline）
        efficientnet_weight_path: EfficientNet JIT 权重路径
        device: 设备
        output_path: 保存结果的 JSON 路径
        batch_size: 推理 batch size

    Returns:
        results dict
    """
    # 加载测试集
    X_test, Y_test = load_ptbxl_test(ptbxl_data_root)

    # ——— Baseline 评估 ———
    print("\nEvaluating baseline EfficientNet...")
    baseline_model = load_efficientnet_jit(efficientnet_weight_path, device)
    baseline_model.eval()

    baseline_probs = run_inference(
        baseline_model, X_test, batch_size=batch_size, device=device, use_sigmoid=True
    )
    baseline_metrics = compute_metrics(
        Y_test.numpy(), baseline_probs, ECG_PATTERNS, ECG_CATEGORIES
    )

    # ——— Finetuned 评估（可选）———
    finetuned_metrics = None
    if adapter_path is not None and Path(adapter_path).exists():
        print(f"\nEvaluating finetuned adapter from {adapter_path}...")
        adapter = EfficientNetAdapter(
            weight_path=efficientnet_weight_path, device=device
        )
        adapter.load_adapter(adapter_path)
        adapter.eval()
        adapter = adapter.to(device)

        finetuned_probs = run_inference(
            adapter, X_test, batch_size=batch_size, device=device, is_adapter=True
        )
        finetuned_metrics = compute_metrics(
            Y_test.numpy(), finetuned_probs, ECG_PATTERNS, ECG_CATEGORIES
        )

    # 打印对比表格
    print_comparison_table(baseline_metrics, finetuned_metrics, ECG_CATEGORIES)

    results = {
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
    }

    # 保存结果
    if output_path is not None:
        import json
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # 转换 nan 为 None（JSON serializable）
        def convert(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            return obj

        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            else:
                return convert(d)

        with open(output_path, "w") as f:
            json.dump(recursive_convert(results), f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results
