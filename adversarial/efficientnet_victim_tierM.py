"""
可微分 Tier-M 6 类 EfficientNet1DV2 victim（对抗 / 在线微调路径）

与 77-class JIT 版本 efficientnet_victim.py 的区别：
  - 权重来自 state_dict（默认位于 `ECG_ADV_DATA_ROOT/crosscenter_tierM/best_model.pt`）
  - 输出维度 6（NSR / STach / AF / IAVB / LBBB / RBBB）
  - 输入预处理与 `unified_preprocess_to_1000` 一致：
      1024 @ 102.4Hz (ECGTwin VAE decode 输出) → 1000 @ 100Hz → global per-sample zscore → center crop 250
  - 模型本身是 nn.Module（非 JIT），所以可以训练（不需要单独 adapter 层）

可微分链路：
  latent (B,4,128)
    → out-of-place `/ 0.18215`
    → VAE decoder（手动迭代 Sequential，避开 in-place /=）
    → (B, 1024, 12)  [ECGTwin lead 顺序]
    → transpose → (B, 12, 1024)
    → lead reorder → (B, 12, 1024)  [PTBXL 顺序]
    → clamp(-3, 3)（幅度稳定性）
    → F.interpolate(1024 → 1000)
    → per-sample global zscore（全 12×1000 上算 mean/std）
    → center crop 1000 → 250
    → EfficientNet1DV2 → logits (B, 6)

API 与 77-class victim 对齐，`BoundaryAdvDiffGenerator` 无需改动即可 swap-in。
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).parent.parent
_ECGTWIN_ROOT = _PROJECT_ROOT / "model" / "ECGTwin"
_DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()
_DEEPECG_NB_CANDIDATES = [
    _PROJECT_ROOT / "model" / "DeepECG" / "notebooks",
    _DATA_ROOT / "models" / "DeepECG" / "notebooks",
]


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


for p in [str(_PROJECT_ROOT), str(_ECGTWIN_ROOT)] + [
    str(p) for p in _DEEPECG_NB_CANDIDATES if _path_exists(p)
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402


DEFAULT_TIERM_CKPT = str(_DATA_ROOT / "crosscenter_tierM" / "best_model.pt")

# Tier-M victim training input length (match scripts/crosscenter_tierM/train_ptbxl_tierM.py crop_len=250)
TIERM_INPUT_LENGTH = 250

# ECGTwin VAE decoder output is 1024 samples @ 102.4Hz; resample to 1000 for PTBXL preprocessing alignment
TIERM_PREPROC_LENGTH = 1000

# Amplitude clamp for stability (unified_preprocess_to_1000 produces zscored signals ~ N(0,1); ±3 is ~3σ)
TIERM_AMP_CLAMP = 3.0


def _build_efficientnet_tierM(num_classes: int = 6) -> EfficientNet1DV2:
    """Construct a Tier-M EfficientNet1DV2 with the exact config used during training."""
    return EfficientNet1DV2(
        variant='s_v2',
        input_channels=12,
        num_classes=num_classes,
        activation='leaky_relu',
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type='batch',
    )


def load_efficientnet_tierM(
    weight_path: str = DEFAULT_TIERM_CKPT,
    device: str = "cuda",
    num_classes: int = 6,
) -> EfficientNet1DV2:
    """Load Tier-M EfficientNet1DV2 from a plain state_dict checkpoint."""
    model = _build_efficientnet_tierM(num_classes=num_classes)
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    return model


class EfficientNetVictimTierM(nn.Module):
    """
    Tier-M 6-class EfficientNet1DV2 可微分 victim。

    同时支持 4 种前向入口：
      1. forward_from_latent_to_logits(latent) — 对抗生成时用（raw logits，支持梯度）
      2. forward_from_latent(latent, enable_grad=True) — 返回 sigmoid 概率
      3. forward_from_ecg(ecg_ct) — 输入 (B, 12, L)，自动 center crop 到 250 后推理
      4. forward(ecg_ct) — 默认入口，等价 forward_from_ecg
    """

    def __init__(
        self,
        weight_path: str = DEFAULT_TIERM_CKPT,
        device: str = "cuda",
        ecgtwin_wrapper=None,
        num_classes: int = 6,
        crop_len: int = TIERM_INPUT_LENGTH,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.ecgtwin = ecgtwin_wrapper
        self.num_classes = num_classes
        self.crop_len = crop_len
        self.model = load_efficientnet_tierM(
            weight_path=weight_path,
            device=device,
            num_classes=num_classes,
        )
        # Default to eval — BatchNorm collapses to bias-only output at batch=1
        # in train mode (yields identical "fake" probs across distinct inputs).
        # Training loops explicitly call .train()/.eval() per phase, so zero
        # regression risk; ad-hoc inference probes get correct behavior by default.
        self.eval()

    # ---- static utilities ----
    @staticmethod
    def _center_crop(ecg_ct: torch.Tensor, crop_len: int) -> torch.Tensor:
        L = ecg_ct.shape[-1]
        if L == crop_len:
            return ecg_ct
        if L < crop_len:
            pad = crop_len - L
            left = pad // 2
            right = pad - left
            return F.pad(ecg_ct, (left, right))
        start = (L - crop_len) // 2
        return ecg_ct[..., start:start + crop_len]

    @staticmethod
    def _global_zscore(ecg_bct: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Global per-sample zscore over all (12 × time) flattened (matches
        per_sample_zscore in preprocess_utils.py)."""
        B = ecg_bct.shape[0]
        flat = ecg_bct.reshape(B, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp(min=eps)
        return (ecg_bct - mean.unsqueeze(-1)) / std.unsqueeze(-1)

    # ---- gradient-preserving VAE decode ----
    def _decode_latent_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        """Out-of-place VAE decode — same pattern as 77-class victim."""
        assert self.ecgtwin is not None, "need ecgtwin_wrapper for latent decode"
        x = latent / 0.18215                   # out-of-place (avoids VAE_Decoder's in-place /=)
        decoder = self.ecgtwin.decoder
        for module in decoder:                 # nn.Sequential: iterate submodules
            x = module(x)
        x = x.transpose(1, 2)                  # (B, 12, 1024) → (B, 1024, 12)
        return x

    # ---- shared latent->logits internal path ----
    def _latent_to_logits(self, latent: torch.Tensor) -> torch.Tensor:
        ecg_tc = self._decode_latent_differentiable(latent)     # (B, 1024, 12)
        ecg_ct = ecg_tc.transpose(-1, -2)                       # (B, 12, 1024) [ECGTwin order]
        ecg_ct = ecg_ct[:, ECGTWIN_TO_PTBXL_INDICES, :]         # PTBXL canonical order
        ecg_ct = torch.clamp(ecg_ct, min=-TIERM_AMP_CLAMP, max=TIERM_AMP_CLAMP)
        ecg_ct = F.interpolate(
            ecg_ct, size=TIERM_PREPROC_LENGTH, mode="linear", align_corners=True
        )                                                       # (B, 12, 1000) @ 100Hz
        ecg_ct = self._global_zscore(ecg_ct)                    # global per-sample zscore
        ecg_ct = self._center_crop(ecg_ct, self.crop_len)       # (B, 12, 250)
        return self.model(ecg_ct)                               # (B, 6)

    # ---- public API ----
    def forward_from_latent_to_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self._latent_to_logits(latent)

    def forward_from_latent(
        self, latent: torch.Tensor, enable_grad: bool = True
    ) -> torch.Tensor:
        if enable_grad:
            logits = self._latent_to_logits(latent)
        else:
            with torch.no_grad():
                logits = self._latent_to_logits(latent.clone())
        return torch.sigmoid(logits)

    def forward_from_ecg(self, ecg_ct: torch.Tensor) -> torch.Tensor:
        """(B, 12, L) → sigmoid probs (B, 6). Center-crops to `self.crop_len` if needed."""
        if ecg_ct.shape[-1] != self.crop_len:
            ecg_ct = self._center_crop(ecg_ct, self.crop_len)
        return torch.sigmoid(self.model(ecg_ct))

    def compute_logits_from_ecg(self, ecg_ct: torch.Tensor) -> torch.Tensor:
        """(B, 12, L) → raw logits (B, 6)."""
        if ecg_ct.shape[-1] != self.crop_len:
            ecg_ct = self._center_crop(ecg_ct, self.crop_len)
        return self.model(ecg_ct)

    def forward(self, ecg_ct: torch.Tensor) -> torch.Tensor:
        return self.forward_from_ecg(ecg_ct)


if __name__ == "__main__":
    # Minimal smoke: construct victim from the Tier-M checkpoint and forward random (B, 12, 250)
    import time
    t0 = time.time()
    print("[smoke] Loading Tier-M victim state_dict...")
    victim = EfficientNetVictimTierM(
        weight_path=DEFAULT_TIERM_CKPT,
        device="cuda",
        ecgtwin_wrapper=None,   # not needed for ecg-input path
    )
    victim.eval()
    x = torch.randn(2, 12, TIERM_INPUT_LENGTH, device="cuda")
    with torch.no_grad():
        probs = victim(x)
        logits = victim.compute_logits_from_ecg(x)
    assert probs.shape == (2, 6), probs.shape
    assert logits.shape == (2, 6), logits.shape
    print(f"[smoke] OK in {time.time()-t0:.2f}s. Example probs[0]={probs[0].cpu().tolist()}")
