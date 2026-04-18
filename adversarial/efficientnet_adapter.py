"""
EfficientNet Residual Adapter

由于 EfficientNet 是 JIT-frozen，不能直接微调权重。
使用 residual adapter 方案：在 frozen backbone 的 logits 上叠加一个可训练的小网络。

架构：
    frozen_logits (B,77)
      → LayerNorm
      → Linear(77, 128)
      → GELU
      → Dropout(0.3)
      → Linear(128, 77)
      → + frozen_logits   (residual 连接)
      = adapted_logits (B,77)

零初始化 fc2 保证初始时 adapter 输出 = 原模型输出（安全初始化）。

参数量：77×128 + 128 + 128×77 + 77 + 77 ≈ 19,893 ≈ 20K
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adversarial.efficientnet_victim import EfficientNetVictim, load_efficientnet_jit, MHI_FACTOR, EFFICIENTNET_INPUT_LENGTH


class EfficientNetAdapter(nn.Module):
    """
    Frozen EfficientNet backbone + 可训练 Residual Adapter

    推理流程：
        ECG (B,12,L)
          → [no_grad] JIT EfficientNet → frozen_logits (B,77)
          → adapter → delta_logits (B,77)
          → frozen_logits + delta_logits = adapted_logits (B,77)

    训练时只更新 adapter 参数，backbone 始终冻结。
    """

    def __init__(
        self,
        num_classes: int = 77,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        weight_path: str = None,
        device: str = "cuda",
        backbone=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mhi_factor = MHI_FACTOR
        self.device_str = device

        # Frozen JIT backbone (shared or self-loaded)
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = load_efficientnet_jit(weight_path, device)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable residual adapter
        self.norm = nn.LayerNorm(num_classes)
        self.fc1 = nn.Linear(num_classes, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # 零初始化 fc2 → 初始行为 = 原模型（residual 为 0）
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ecg: (B, 12, L)，任意长度（自动 resample 到 2500）

        Returns:
            adapted_logits: (B, 77) raw logits（未 sigmoid）
        """
        # 确保长度正确
        if ecg.shape[-1] != EFFICIENTNET_INPUT_LENGTH:
            ecg = F.interpolate(
                ecg, size=EFFICIENTNET_INPUT_LENGTH, mode="linear", align_corners=True
            )

        # Frozen backbone forward
        with torch.no_grad():
            frozen_logits = self.backbone(ecg * self.mhi_factor)  # (B, 77)

        # Adapter forward
        x = self.norm(frozen_logits)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Residual 连接
        adapted_logits = frozen_logits + x
        return adapted_logits

    def predict_proba(self, ecg: torch.Tensor) -> torch.Tensor:
        """返回 sigmoid 概率"""
        return torch.sigmoid(self.forward(ecg))

    def forward_frozen_only(self, ecg: torch.Tensor) -> torch.Tensor:
        """只用 frozen backbone 推理（用于 baseline 对比）"""
        if ecg.shape[-1] != EFFICIENTNET_INPUT_LENGTH:
            ecg = F.interpolate(
                ecg, size=EFFICIENTNET_INPUT_LENGTH, mode="linear", align_corners=True
            )
        with torch.no_grad():
            return self.backbone(ecg * self.mhi_factor)

    def save(self, path: str):
        """只保存 adapter 参数（不含 backbone）"""
        state = {
            "adapter_state_dict": {
                "norm": self.norm.state_dict(),
                "fc1": self.fc1.state_dict(),
                "fc2": self.fc2.state_dict(),
            },
            "num_classes": self.num_classes,
            "hidden_dim": self.fc1.out_features,
        }
        torch.save(state, path)

    def load_adapter(self, path: str):
        """加载 adapter 参数"""
        state = torch.load(path, map_location=self.device_str)
        self.norm.load_state_dict(state["adapter_state_dict"]["norm"])
        self.fc1.load_state_dict(state["adapter_state_dict"]["fc1"])
        self.fc2.load_state_dict(state["adapter_state_dict"]["fc2"])

    def adapter_parameters(self):
        """返回只包含 adapter 参数的迭代器（排除 backbone）"""
        return list(self.norm.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters())

    def count_adapter_params(self) -> int:
        return sum(p.numel() for p in self.adapter_parameters())
