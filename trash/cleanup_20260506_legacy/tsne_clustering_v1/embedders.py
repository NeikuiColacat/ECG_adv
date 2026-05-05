"""ECGFounder (Net1D) feature extractor used as the independent embedder.

We set `return_features=True` on Net1D and take the pooled 1024-d vector
produced by `out.mean(-1)` before the classifier head. This gives a
semantically rich, center-agnostic embedding trained on 10M ECGs — crucial
for clustering analysis that must not be biased by ECGTwin or by the
downstream victim EfficientNet classifier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ECGFounder ships a top-level `util.py` that would shadow this project's
# `util/` package. Insert its directory, import only the module we need,
# then remove the entry so subsequent imports see the project util package.
_ECGFOUNDER_ROOT = "/root/autodl-tmp/ecgfounder"
sys.path.insert(0, _ECGFOUNDER_ROOT)
try:
    from net1d import Net1D  # noqa: E402
finally:
    try:
        sys.path.remove(_ECGFOUNDER_ROOT)
    except ValueError:
        pass
    # Also drop any top-level 'util' module that may have been picked up
    # from ecgfounder's util.py before we can correctly reload ours.
    if "util" in sys.modules and getattr(sys.modules["util"], "__file__", "").startswith(
        _ECGFOUNDER_ROOT
    ):
        del sys.modules["util"]


class ECGFounderEmbedder:
    """Load 12-lead ECGFounder and extract penultimate features.

    Input to `extract_features`: (B, 12, 1000) float tensor @ 100Hz.
    Output: (B, 1024) float CPU numpy array (penultimate global-pooled features).
    """

    def __init__(self, weights_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        # Match the architecture used in finetune_model.ft_12lead_ECGFounder
        self.model = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=150,   # placeholder; we drop the dense head
            return_features=True,
        )

        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        # Drop mismatched classifier head — we only need the backbone
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dense.")}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # 'dense' is expected to be missing (head dropped on purpose)
        critical_missing = [k for k in missing if not k.startswith("dense.")]
        if critical_missing:
            raise RuntimeError(
                f"Unexpected missing ECGFounder keys: {critical_missing[:5]} ..."
            )

        # Re-attach a dummy dense so forward() doesn't crash (we ignore its output)
        self.model.dense = nn.Linear(self.model.dense.in_features, 1)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """x: (N, 12, 1000) on any device; returns (N, 1024) numpy on CPU."""
        if x.dim() != 3 or x.shape[1] != 12:
            raise ValueError(f"expected (N, 12, L), got {tuple(x.shape)}")
        feats = []
        for i in range(0, x.shape[0], batch_size):
            chunk = x[i : i + batch_size].to(self.device).float()
            _, f = self.model(chunk)  # (B, 1024)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)
