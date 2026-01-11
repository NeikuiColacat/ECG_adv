"""
对抗样本保存工具

使用 ECGTwin 的 ecg_plot 库将生成的对抗 ECG 样本保存为图像
命名格式: GT标签-攻击目标标签-模型预测结果标签.png
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import torch
import numpy as np
from matplotlib import pyplot as plt

# 添加 ECGTwin 路径以使用 ecg_plot
_PROJECT_ROOT = Path(__file__).parent.parent
_ECGTWIN_ROOT = _PROJECT_ROOT / "model" / "ECGTwin"
sys.path.insert(0, str(_ECGTWIN_ROOT))

import ecg_plot

# PTB-XL Superdiagnostic 类别名称
CLASS_NAMES = {
    0: "CD",     # Conduction Disturbance
    1: "HYP",    # Hypertrophy
    2: "MI",     # Myocardial Infarction
    3: "NORM",   # Normal
    4: "STTC",   # ST/T Change
}

# ECGTwin 的标准导联顺序
LEAD_INDEX = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def save_ecg_image(
    ecg: Union[torch.Tensor, np.ndarray],
    save_path: str,
    title: Optional[str] = None,
    sample_rate: float = 102.4,
    lead_index: List[str] = None,
    columns: int = 1,
    row_height: float = 4,
) -> str:
    """
    将单个 ECG 信号保存为图像
    
    Args:
        ecg: ECG 信号，shape (L, 12) 或 (12, L) 或 (1024, 12)
        save_path: 保存路径
        title: 图像标题
        sample_rate: 采样率 (ECGTwin 默认 102.4 Hz)
        lead_index: 导联名称列表
        columns: 列数
        row_height: 行高
        
    Returns:
        保存的文件路径
    """
    if lead_index is None:
        lead_index = LEAD_INDEX
    
    # 转换为 numpy
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.detach().cpu().numpy()
    
    # 确保是 2D
    if ecg.ndim == 3:
        ecg = ecg[0]
    
    # ecg_plot 需要 (12, L) 格式
    # ECGTwin decoder 输出是 (B, 1024, 12)，所以单个样本是 (1024, 12)
    if ecg.shape[0] == 12:
        # 已经是 (12, L)
        ecg_for_plot = ecg
    elif ecg.shape[-1] == 12:
        # (L, 12) -> (12, L)
        ecg_for_plot = ecg.transpose(1, 0)
    else:
        raise ValueError(f"Invalid ECG shape: {ecg.shape}, expected (L, 12) or (12, L)")
    
    # 绘制并保存
    ecg_plot.plot(ecg_for_plot, sample_rate, lead_index=lead_index, title=title, columns=columns, row_height=row_height)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def save_adversarial_sample(
    ecg: Union[torch.Tensor, np.ndarray],
    gt_label: int,
    target_label: int,
    pred_label: int,
    save_dir: str = "./result",
    sample_idx: int = None,
    add_title: bool = True,
) -> str:
    """
    保存单个对抗样本，使用指定的命名格式
    
    命名格式: GT标签-攻击目标标签-模型预测结果标签.png
    例如: NORM-CD-CD.png (原始为NORM，目标攻击为CD，模型预测为CD)
    
    Args:
        ecg: ECG 信号，shape (L, 12) 或 (12, L)
        gt_label: Ground Truth 标签 (0-4)
        target_label: 攻击目标标签 (0-4)
        pred_label: 模型预测结果标签 (0-4)
        save_dir: 保存目录
        sample_idx: 样本索引（用于区分同类型的多个样本）
        add_title: 是否在图像中添加标题
        
    Returns:
        保存的文件路径
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取类别名称
    gt_name = CLASS_NAMES.get(gt_label, str(gt_label))
    target_name = CLASS_NAMES.get(target_label, str(target_label))
    pred_name = CLASS_NAMES.get(pred_label, str(pred_label))
    
    # 构建文件名
    if sample_idx is not None:
        filename = f"{gt_name}-{target_name}-{pred_name}_{sample_idx}.png"
    else:
        filename = f"{gt_name}-{target_name}-{pred_name}.png"
    
    save_path = os.path.join(save_dir, filename)
    
    # 构建标题
    title = None
    if add_title:
        attack_status = "✓ Success" if target_label == pred_label else "✗ Failed"
        title = f"GT: {gt_name} | Target: {target_name} | Pred: {pred_name} | {attack_status}"
    
    # 保存图像
    save_ecg_image(ecg, save_path, title=title)
    
    return save_path


def save_adversarial_batch(
    ecg_batch: Union[torch.Tensor, np.ndarray],
    gt_label: int,
    target_label: int,
    pred_labels: Union[torch.Tensor, np.ndarray, List[int]],
    save_dir: str = "./result",
    add_title: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    批量保存对抗样本
    
    Args:
        ecg_batch: ECG 批次，shape (B, L, 12) 或 (B, 12, L)
        gt_label: Ground Truth 标签 (所有样本相同)
        target_label: 攻击目标标签
        pred_labels: 每个样本的预测标签，shape (B,)
        save_dir: 保存目录
        add_title: 是否添加标题
        verbose: 是否打印保存信息
        
    Returns:
        保存的文件路径列表
    """
    # 转换预测标签
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().tolist()
    elif isinstance(pred_labels, np.ndarray):
        pred_labels = pred_labels.tolist()
    
    # 转换 ECG 批次
    if isinstance(ecg_batch, torch.Tensor):
        ecg_batch = ecg_batch.detach().cpu().numpy()
    
    batch_size = ecg_batch.shape[0]
    saved_paths = []
    
    # 统计每种组合的数量（用于命名）
    combo_counts = {}
    
    for i in range(batch_size):
        pred_label = pred_labels[i]
        
        # 创建组合键
        gt_name = CLASS_NAMES.get(gt_label, str(gt_label))
        target_name = CLASS_NAMES.get(target_label, str(target_label))
        pred_name = CLASS_NAMES.get(pred_label, str(pred_label))
        combo_key = f"{gt_name}-{target_name}-{pred_name}"
        
        # 获取当前组合的索引
        if combo_key not in combo_counts:
            combo_counts[combo_key] = 0
        sample_idx = combo_counts[combo_key]
        combo_counts[combo_key] += 1
        
        # 保存单个样本
        save_path = save_adversarial_sample(
            ecg=ecg_batch[i],
            gt_label=gt_label,
            target_label=target_label,
            pred_label=pred_label,
            save_dir=save_dir,
            sample_idx=sample_idx,
            add_title=add_title,
        )
        saved_paths.append(save_path)
        
        if verbose:
            status = "✓" if target_label == pred_label else "✗"
            print(f"  {status} Saved: {save_path}")
    
    return saved_paths


def save_attack_results(
    results: Dict[str, Any],
    gt_label: int,
    save_dir: str = "./result",
    add_title: bool = True,
    verbose: bool = True,
    save_all: bool = True,
) -> Dict[str, Any]:
    """
    保存 AdvDiffAttacker.attack() 返回的攻击结果
    
    Args:
        results: attack() 返回的结果字典，包含:
                 - final_ecg: (B, 1024, 12) 所有 ECG 样本
                 - adv_samples: 成功的对抗样本（可能只包含部分）
                 - final_predictions: (B,) 最终预测
                 - success_mask: (B,) 成功掩码
                 - target_label: 攻击目标标签
        gt_label: Ground Truth 标签
        save_dir: 保存目录
        add_title: 是否添加标题
        verbose: 是否打印信息
        save_all: 是否保存所有样本（True）或只保存成功的样本（False）
        
    Returns:
        包含保存路径和统计信息的字典
    """
    target_label = results.get("target_label", 0)
    final_ecg = results.get("final_ecg")  # 所有样本
    adv_samples = results.get("adv_samples")  # 只有成功的样本
    final_predictions = results.get("final_predictions")
    success_mask = results.get("success_mask")
    
    # 选择要保存的样本
    if save_all:
        samples_to_save = final_ecg
    else:
        samples_to_save = adv_samples
        # 只保存成功样本时，需要过滤预测结果
        if success_mask is not None and isinstance(final_predictions, torch.Tensor):
            final_predictions = final_predictions[success_mask]
    
    if samples_to_save is None:
        print("Warning: No samples to save in results")
        return {"saved_paths": [], "success_count": 0, "total_count": 0}
    
    gt_name = CLASS_NAMES.get(gt_label, str(gt_label))
    target_name = CLASS_NAMES.get(target_label, str(target_label))
    
    if verbose:
        mode_str = "所有样本" if save_all else "仅成功样本"
        print(f"\n{'='*60}")
        print(f"保存对抗样本: GT={gt_name} -> Target={target_name} ({mode_str})")
        print(f"{'='*60}")
    
    # 批量保存
    saved_paths = save_adversarial_batch(
        ecg_batch=samples_to_save,
        gt_label=gt_label,
        target_label=target_label,
        pred_labels=final_predictions,
        save_dir=save_dir,
        add_title=add_title,
        verbose=verbose,
    )
    
    # 统计
    success_count = int(success_mask.sum().item()) if isinstance(success_mask, torch.Tensor) else sum(success_mask)
    total_count = len(saved_paths)
    
    if verbose:
        print(f"\n保存完成: {total_count} 个样本")
        print(f"攻击成功: {success_count}/{total_count} ({success_count/total_count:.0%})")
        print(f"保存目录: {save_dir}")
    
    return {
        "saved_paths": saved_paths,
        "success_count": success_count,
        "total_count": total_count,
        "success_rate": success_count / total_count if total_count > 0 else 0,
    }


if __name__ == "__main__":
    # 测试代码
    print("测试 save_tool.py")
    print("=" * 60)
    
    # 创建模拟数据
    fake_ecg = np.random.randn(1024, 12) * 0.5  # 模拟 ECG 信号
    
    # 测试单个样本保存
    save_path = save_adversarial_sample(
        ecg=fake_ecg,
        gt_label=3,      # NORM
        target_label=0,  # CD
        pred_label=0,    # CD (攻击成功)
        save_dir="./result/test",
        sample_idx=0,
    )
    print(f"单个样本已保存: {save_path}")
    
    # 测试批量保存
    fake_batch = np.random.randn(4, 1024, 12) * 0.5
    pred_labels = [0, 0, 3, 0]  # 3个成功，1个失败
    
    saved_paths = save_adversarial_batch(
        ecg_batch=fake_batch,
        gt_label=3,
        target_label=0,
        pred_labels=pred_labels,
        save_dir="./result/test",
    )
    print(f"\n批量保存完成: {len(saved_paths)} 个文件")
