"""
导联对齐工具

ECGTwin 使用 MIMIC-IV-ECG 数据集，导联顺序为:
    ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
PTB-XL 数据集的导联顺序为:
    ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

差异在于位置 4 和 5 对调了 (aVF <-> aVL)
"""

import torch

# 标准导联顺序
ECGTWIN_LEADS = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
PTBXL_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# ECGTwin -> PTB-XL 的通道重排索引
# PTB-XL 的顺序：[I, II, III, AVR, AVL, AVF, V1-V6]
# 对应 ECGTwin 的索引：[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
ECGTWIN_TO_PTBXL_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]

# PTB-XL -> ECGTwin 的通道重排索引（反向）
PTBXL_TO_ECGTWIN_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]  # 对称的，所以一样


def ecgtwin_to_ptbxl(ecg: torch.Tensor) -> torch.Tensor:
    """
    将 ECGTwin 输出的 ECG 转换为 PTB-XL 分类器的输入格式
    
    Args:
        ecg: ECGTwin 输出，shape 可以是:
             - (B, L, 12): ECGTwin VAE_Decoder 的直接输出
             - (B, 12, L): 已经转置的格式
             
    Returns:
        ptbxl_ecg: PTB-XL 分类器需要的格式 (B, 12, L)
                   导联顺序已对齐为 PTB-XL 标准
    """
    if ecg.dim() == 2:
        ecg = ecg.unsqueeze(0)
    
    # 检测输入格式
    if ecg.shape[-1] == 12:
        # (B, L, 12) -> (B, 12, L)
        ecg = ecg.transpose(-1, -2)
    
    # 重排通道顺序: ECGTwin -> PTB-XL
    # ecg: (B, 12, L)
    ecg = ecg[:, ECGTWIN_TO_PTBXL_INDICES, :]
    
    return ecg


def ptbxl_to_ecgtwin(ecg: torch.Tensor) -> torch.Tensor:
    """
    将 PTB-XL 格式的 ECG 转换为 ECGTwin 的格式
    
    Args:
        ecg: PTB-XL 格式，shape (B, 12, L)
        
    Returns:
        ecgtwin_ecg: ECGTwin 格式 (B, L, 12)，导联顺序已对齐
    """
    if ecg.dim() == 2:
        ecg = ecg.unsqueeze(0)
    
    # 确保输入是 (B, 12, L)
    if ecg.shape[-1] == 12:
        ecg = ecg.transpose(-1, -2)
    
    # 重排通道顺序: PTB-XL -> ECGTwin
    ecg = ecg[:, PTBXL_TO_ECGTWIN_INDICES, :]
    
    # 转换为 ECGTwin 格式 (B, L, 12)
    ecg = ecg.transpose(-1, -2)
    
    return ecg


def resample_ecg(ecg: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    重采样 ECG 信号到目标长度
    
    ECGTwin 生成的 ECG 长度为 1024（采样率 102.4 Hz, 10秒）
    PTB-XL 使用 100 Hz 采样率，10秒 = 1000 个采样点
    
    Args:
        ecg: shape (B, C, L) 或 (B, L, C)
        target_length: 目标长度
        
    Returns:
        resampled_ecg: 重采样后的 ECG
    """
    import torch.nn.functional as F
    
    # 确保是 (B, C, L) 格式用于插值
    is_channel_last = ecg.shape[-1] in [12, 1, 2, 3, 4, 8]
    if is_channel_last and ecg.shape[-1] != ecg.shape[-2]:
        ecg = ecg.transpose(-1, -2)
    
    # 使用线性插值重采样
    resampled = F.interpolate(ecg, size=target_length, mode='linear', align_corners=True)
    
    if is_channel_last:
        resampled = resampled.transpose(-1, -2)
    
    return resampled


def prepare_ecg_for_classifier(
    ecg: torch.Tensor,
    source: str = "ecgtwin",
    target_length: int = 1000,
    normalize: bool = True,
    mean: torch.Tensor = None,
    std: torch.Tensor = None,
) -> torch.Tensor:
    """
    完整的 ECG 预处理流程：从生成模型输出到分类器输入
    
    Args:
        ecg: 生成的 ECG
        source: 来源模型 ("ecgtwin" 或 "ptbxl")
        target_length: 目标序列长度（PTB-XL 默认 1000）
        normalize: 是否进行标准化
        mean: 标准化均值 (12,)
        std: 标准化标准差 (12,)
        
    Returns:
        prepared_ecg: 准备好的 ECG (B, 12, target_length)
    """
    if ecg.dim() == 2:
        ecg = ecg.unsqueeze(0)
    
    # 1. 导联对齐
    if source == "ecgtwin":
        ecg = ecgtwin_to_ptbxl(ecg)  # -> (B, 12, L)
    else:
        # 确保是 (B, 12, L)
        if ecg.shape[-1] == 12:
            ecg = ecg.transpose(-1, -2)
    
    # 2. 重采样
    if ecg.shape[-1] != target_length:
        ecg = resample_ecg(ecg, target_length)
    
    # 3. 标准化
    if normalize and mean is not None and std is not None:
        # mean, std: (12,) -> (1, 12, 1)
        mean = mean.view(1, -1, 1).to(ecg.device)
        std = std.view(1, -1, 1).to(ecg.device)
        ecg = (ecg - mean) / (std + 1e-8)
    
    return ecg
