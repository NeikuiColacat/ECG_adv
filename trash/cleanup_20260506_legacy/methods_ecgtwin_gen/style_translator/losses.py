"""Losses for Style Translator training.

- SupConLoss (Khosla 2020): supervised contrastive loss on center labels.
- style_translator_loss: composite (L_supcon + λ_id·L_identity + λ_style·L_style_transfer).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    Pull features with same label together, push different labels apart on unit sphere.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (B, D) — will be L2-normalized internally
        labels:   (B,) integer class labels
        """
        if features.dim() != 2:
            raise ValueError(f"features must be (B, D); got {features.shape}")
        if labels.dim() != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(f"labels shape {labels.shape} mismatch features {features.shape}")

        device = features.device
        B = features.shape[0]

        features = F.normalize(features, p=2, dim=-1)
        # similarity: (B, B)
        sim = features @ features.t() / self.temperature

        # Numerical stability: subtract row max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # mask: same-label (including self) vs others
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B) bool
        # Exclude self-pairs from both positives and normalization
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = labels_eq & ~self_mask                        # (B, B)

        # exp(sim) with self-pairs zeroed for denominator
        exp_sim = torch.exp(sim) * (~self_mask).float()

        # denominator: sum over all non-self pairs
        denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12)   # (B, 1)
        log_prob = sim - torch.log(denom)                            # (B, B)

        # positives per-row count
        n_pos = pos_mask.float().sum(dim=1)                          # (B,)

        # Average log-prob over positives for rows with ≥1 positive
        row_loss = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos.clamp(min=1.0)
        valid = n_pos > 0
        if valid.any():
            return row_loss[valid].mean()
        # Degenerate batch (every row singleton class): return 0 (not a grad signal)
        return torch.zeros((), device=device, requires_grad=True)


def style_translator_loss(
    *,
    feat_translated: torch.Tensor,          # (B, D) SIBE(x, proto_c)
    feat_base: torch.Tensor,                # (B, D) base_IBE(x)       [detached]
    center_labels: torch.Tensor,            # (B,) int — which proto each row used
    # For L_identity: rows where proto came from SAME center as x's true center
    identity_mask: Optional[torch.Tensor] = None,  # (B,) bool
    # For L_style_transfer: pairs (i, j) where x_i from center A, style=proto_B, and
    # x_j from center B same disease as x_i; feat_translated[i] should ≈ feat_translated[j]
    pair_i: Optional[torch.Tensor] = None,  # (P,) long
    pair_j: Optional[torch.Tensor] = None,  # (P,) long
    lambda_id: float = 1.0,
    lambda_style: float = 0.5,
    lambda_sup: float = 1.0,
    supcon_temp: float = 0.1,
) -> dict:
    """Composite Style Translator loss.

    Returns a dict with total + component scalars (tensor values).
    """
    sup = SupConLoss(temperature=supcon_temp)
    l_sup = sup(feat_translated, center_labels)

    # L_identity: rows where proto center == x's true center -> translated should ≈ base
    if identity_mask is not None and identity_mask.any():
        sel = feat_translated[identity_mask]
        ref = feat_base[identity_mask].detach()
        l_id = F.mse_loss(sel, ref)
    else:
        l_id = torch.zeros((), device=feat_translated.device)

    # L_style_transfer: translated[i] (x_A, proto_B) should match translated[j] (x_B, proto_B)
    if pair_i is not None and pair_j is not None and len(pair_i) > 0:
        a = feat_translated[pair_i]
        b = feat_translated[pair_j]
        l_style = F.mse_loss(a, b)
    else:
        l_style = torch.zeros((), device=feat_translated.device)

    total = lambda_sup * l_sup + lambda_id * l_id + lambda_style * l_style
    return {
        "total": total,
        "supcon": l_sup.detach(),
        "identity": l_id.detach(),
        "style": l_style.detach(),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 12, 256
    feats = torch.randn(B, D, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])

    # SupCon basic smoke
    sup = SupConLoss()
    loss = sup(feats, labels)
    print(f"SupCon loss: {loss.item():.4f}  (should be ~positive)")
    loss.backward()
    assert feats.grad is not None

    # Full composite smoke
    feat_translated = torch.randn(B, D, requires_grad=True)
    feat_base = torch.randn(B, D)
    center_labels = labels

    # half identity mask, 2 pairs
    identity_mask = torch.tensor([True, False, True, False, False, True, False, True, False, True, True, False])
    pair_i = torch.tensor([0, 3])
    pair_j = torch.tensor([1, 4])

    result = style_translator_loss(
        feat_translated=feat_translated,
        feat_base=feat_base,
        center_labels=center_labels,
        identity_mask=identity_mask,
        pair_i=pair_i,
        pair_j=pair_j,
    )
    print(f"Composite: total={result['total'].item():.4f}  supcon={result['supcon'].item():.4f}  id={result['identity'].item():.4f}  style={result['style'].item():.4f}")
    result["total"].backward()
    assert feat_translated.grad is not None

    # Empty pair case (no same-disease cross-center pairs in batch)
    result2 = style_translator_loss(
        feat_translated=torch.randn(B, D, requires_grad=True),
        feat_base=torch.randn(B, D),
        center_labels=labels,
        identity_mask=None,
        pair_i=torch.tensor([], dtype=torch.long),
        pair_j=torch.tensor([], dtype=torch.long),
    )
    print(f"Empty-pair case: style={result2['style'].item():.4f}  (should be 0)")
    assert result2["style"].item() == 0.0

    # Degenerate SupCon (all unique labels)
    feats2 = torch.randn(5, D, requires_grad=True)
    labels2 = torch.arange(5)
    loss2 = SupConLoss()(feats2, labels2)
    print(f"All-singleton SupCon: {loss2.item():.4f}  (should be 0)")

    print("✓ losses smoke test passed")
