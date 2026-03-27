"""
可微分 EfficientNet 封装

将 ECGTwin VAE latent 转换为 EfficientNet 77 类 sigmoid 概率的完整可微分链路：

  latent (B,4,128)
    → VAE decode (out-of-place，避免 in-place /= 0.18215)
    → ECG (B,1024,12)
    → transpose → (B,12,1024)
    → lead reorder (ECGTWIN_TO_PTBXL_INDICES)
    → F.interpolate(size=2500, mode='linear')
    → × mhi_factor
    → JIT model → logits (B,77)
    → sigmoid → probs (B,77)
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).parent.parent
_ECGTWIN_ROOT = _PROJECT_ROOT / "model" / "ECGTwin"

for p in [str(_PROJECT_ROOT), str(_ECGTWIN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES

# EfficientNet 输入为 250Hz × 10s = 2500 samples
EFFICIENTNET_INPUT_LENGTH = 2500

# EfficientNet 缩放因子（MHI 数据集归一化参数）
MHI_FACTOR = 1.0 / 0.0048  # ≈ 208.33

# JIT 模型权重路径
DEFAULT_WEIGHT_PATH = (
    _PROJECT_ROOT
    / "model"
    / "DeepECG"
    / "weights"
    / "efficientnetv2_77_classes"
    / "efficientnet_deepecg_unscaled.pt"
)


def load_efficientnet_jit(
    weight_path: str = None,
    device: str = "cuda",
) -> torch.jit.ScriptModule:
    """加载 JIT-compiled EfficientNet 模型"""
    if weight_path is None:
        weight_path = str(DEFAULT_WEIGHT_PATH)
    model = torch.jit.load(weight_path, map_location=device)
    model.eval()
    return model


class EfficientNetVictim(nn.Module):
    """
    可微分 EfficientNet 推理路径

    支持两种使用模式：
    1. 从 VAE latent 出发（用于 AdvDiff 对抗采样，梯度可回传到 latent）
    2. 从 ECG 信号出发（用于微调后评估）

    JIT 模型本身支持 autograd，梯度可以流过（JIT 只是编译优化，非冻结 no_grad）。
    """

    def __init__(
        self,
        weight_path: str = None,
        device: str = "cuda",
        ecgtwin_wrapper=None,
    ):
        """
        Args:
            weight_path: JIT 模型权重路径
            device: 设备
            ecgtwin_wrapper: ECGTwinWrapper 实例（用于 VAE decode）
                             如果为 None，仅支持从 ECG 信号推理
        """
        super().__init__()
        self.device = torch.device(device)
        self.mhi_factor = MHI_FACTOR
        self.jit_model = load_efficientnet_jit(weight_path, device)
        self.ecgtwin = ecgtwin_wrapper

    def forward_from_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        从 PTB-XL 格式 ECG 推理

        Args:
            ecg: (B, 12, L)，任意采样长度（会自动 resample 到 2500）

        Returns:
            probs: (B, 77)，sigmoid 概率
        """
        if ecg.shape[-1] != EFFICIENTNET_INPUT_LENGTH:
            ecg = F.interpolate(
                ecg, size=EFFICIENTNET_INPUT_LENGTH, mode="linear", align_corners=True
            )
        ecg_scaled = ecg * self.mhi_factor
        logits = self.jit_model(ecg_scaled)
        return torch.sigmoid(logits)

    def _latent_to_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """
        从 VAE latent 到 raw logits 的可微分链路（供内部复用）

        Args:
            latent: (B, 4, 128)，调用前已设置 requires_grad

        Returns:
            logits: (B, 77)，raw logits（sigmoid 之前）
        """
        assert self.ecgtwin is not None, "需要传入 ecgtwin_wrapper 才能从 latent 推理"

        ecg = self._decode_latent_differentiable(latent)

        # ECGTwin 输出 (B, 1024, 12) → (B, 12, 1024)
        if ecg.shape[-1] == 12:
            ecg = ecg.transpose(-1, -2)

        # 导联重排 ECGTwin → PTBXL
        ecg = ecg[:, ECGTWIN_TO_PTBXL_INDICES, :]  # (B, 12, 1024)

        # ECG 幅度 clamp（对应 AdvDiff 原版的 torch.clamp(img, 0, 1)）
        # 正常 ECG 幅度 ±2 mV，clamp 到 ±3 mV 防止 OOD latent 产生极端幅度
        # clamp 处梯度为 0，自动阻止 latent 继续向极端方向移动
        ecg = torch.clamp(ecg, min=-3.0, max=3.0)

        # 重采样到 2500
        ecg = F.interpolate(
            ecg, size=EFFICIENTNET_INPUT_LENGTH, mode="linear", align_corners=True
        )

        ecg_scaled = ecg * self.mhi_factor
        return self.jit_model(ecg_scaled)

    def forward_from_latent_to_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """
        从 VAE latent 到 raw logits（用于 boundary_loss 在 logit 空间计算，避免梯度消失）

        Args:
            latent: (B, 4, 128)，需要梯度

        Returns:
            logits: (B, 77)，raw logits
        """
        return self._latent_to_logits(latent)

    def forward_from_latent(
        self,
        latent: torch.Tensor,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """
        从 VAE latent 出发的全可微分推理链路

        关键：VAE decoder 内部有 `x /= 0.18215` 的 in-place 操作。
        通过手动迭代 decoder modules 并用 out-of-place 除法避免梯度截断。

        Args:
            latent: (B, 4, 128)
            enable_grad: 是否保留梯度（用于对抗生成时应为 True）

        Returns:
            probs: (B, 77)，sigmoid 概率
        """
        assert self.ecgtwin is not None, "需要传入 ecgtwin_wrapper 才能从 latent 推理"

        if enable_grad:
            logits = self._latent_to_logits(latent)
        else:
            with torch.no_grad():
                ecg = self.ecgtwin.decoder(latent.clone())  # (B, 1024, 12)
                if ecg.shape[-1] == 12:
                    ecg = ecg.transpose(-1, -2)
                ecg = ecg[:, ECGTWIN_TO_PTBXL_INDICES, :]
                ecg = F.interpolate(
                    ecg, size=EFFICIENTNET_INPUT_LENGTH, mode="linear", align_corners=True
                )
                ecg_scaled = ecg * self.mhi_factor
                logits = self.jit_model(ecg_scaled)

        return torch.sigmoid(logits)

    def _decode_latent_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        """
        out-of-place VAE decode，使梯度可以回传到 latent

        复用 advdiff_attack.py L188-195 的模式：
        先做 out-of-place 除法（x = latent / 0.18215），然后手动迭代
        decoder 的子模块，跳过 VAE_Decoder.forward 中的 in-place `/= 0.18215`。

        VAE_Decoder.forward 结构（伪代码）：
            def forward(self, x):
                x /= 0.18215      # ← in-place，会截断梯度
                x = self.layer1(x)
                ...
                return x

        本方法：先 out-of-place 做缩放，再手动跑各层。
        """
        # VAE_Decoder.forward 结构：
        #   x /= 0.18215          ← in-place（截断梯度）
        #   for module in self:   ← nn.Sequential 迭代
        #       x = module(x)
        #   x = x.transpose(1,2) ← (B,12,L) → (B,L,12)
        # 复现步骤但用 out-of-place 除法：
        x = latent / 0.18215      # out-of-place

        decoder = self.ecgtwin.decoder
        for module in decoder:    # nn.Sequential 直接迭代
            x = module(x)

        x = x.transpose(1, 2)    # (B,12,1024) → (B,1024,12)
        return x

    def compute_logits_from_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        从 ECG 计算 raw logits（未 sigmoid），用于微调训练时的 BCEWithLogitsLoss

        Args:
            ecg: (B, 12, L)

        Returns:
            logits: (B, 77)
        """
        if ecg.shape[-1] != EFFICIENTNET_INPUT_LENGTH:
            ecg = F.interpolate(
                ecg, size=EFFICIENTNET_INPUT_LENGTH, mode="linear", align_corners=True
            )
        ecg_scaled = ecg * self.mhi_factor
        return self.jit_model(ecg_scaled)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """默认 forward：从 ECG 信号推理概率"""
        return self.forward_from_ecg(ecg)
