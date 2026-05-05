"""StyleTranslatorIBE: wraps a frozen IBExtractor with style-conditioned residual.

IBE(x, style=None)  == base_IBE(x)                               # backward compat
IBE(x, style=s)     == base_IBE(x) + style_fusion([base, s])     # style-translated
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ECGTWIN_ROOT = PROJECT_ROOT / "model" / "ECGTwin"
if str(ECGTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ECGTWIN_ROOT))

from module.IBExtractor import IBExtractor  # noqa: E402


class StyleTranslatorIBE(nn.Module):
    """Counterfactual style translator wrapping a frozen base IBExtractor.

    Args:
        base_ibe: pretrained IBExtractor instance (will be frozen)
        prototype_encoder: PrototypeEncoder module
        embed_dim: feature dim (default 256, must match base IBE output)
    """

    def __init__(
        self,
        base_ibe: IBExtractor,
        prototype_encoder: nn.Module,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.base_ibe = base_ibe
        for p in self.base_ibe.parameters():
            p.requires_grad = False
        self.base_ibe.eval()

        self.prototype_encoder = prototype_encoder

        # Style fusion: [base_feat; style_vec] -> delta
        self.style_fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # zero-init last linear so initial behavior ≡ base IBE (delta = 0)
        nn.init.zeros_(self.style_fusion[-1].weight)
        nn.init.zeros_(self.style_fusion[-1].bias)

    def train(self, mode: bool = True):
        """Keep base_ibe in eval mode regardless of parent mode."""
        super().train(mode)
        self.base_ibe.eval()
        return self

    def extract_features(
        self,
        x: torch.Tensor,
        text_embed: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        p: torch.Tensor,
        style_vec: Optional[torch.Tensor] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Extract features with optional style conditioning.

        Args:
            x: (B, L, in_ch=4) VAE latent (IBE convention: channels last)
            text_embed: (B, L_text, 768) or None for classfree
            mask: (B, L_text) or None
            p: (B, 3) patient info
            style_vec: (embed_dim,) or (B, embed_dim) or None
            reduce: if True, mean-pool to (B, embed_dim); else (B, L, embed_dim)

        Returns:
            features: same shape as base IBE output (reduce controls)
        """
        with torch.no_grad():
            base_feat = self.base_ibe.extract_features(
                x, text_embed, mask, p, reduce=reduce
            )

        if style_vec is None:
            return base_feat

        if reduce:
            B = base_feat.shape[0]
            if style_vec.dim() == 1:
                style = style_vec.unsqueeze(0).expand(B, -1)  # (B, D)
            elif style_vec.dim() == 2:
                if style_vec.shape[0] == 1:
                    style = style_vec.expand(B, -1)
                elif style_vec.shape[0] == B:
                    style = style_vec
                else:
                    raise ValueError(
                        f"style_vec batch dim {style_vec.shape[0]} != base_feat batch {B}"
                    )
            else:
                raise ValueError(f"style_vec must be 1D or 2D; got {style_vec.shape}")
            delta = self.style_fusion(torch.cat([base_feat, style], dim=-1))
            return base_feat + delta
        else:
            # Non-reduced path: apply style per-token (broadcast)
            B, L, D = base_feat.shape
            if style_vec.dim() == 1:
                style = style_vec.view(1, 1, D).expand(B, L, D)
            elif style_vec.dim() == 2:
                style = style_vec.view(style_vec.shape[0], 1, D).expand(-1, L, D)
            else:
                raise ValueError(f"style_vec must be 1D or 2D; got {style_vec.shape}")
            flat = base_feat.reshape(B * L, D)
            flat_style = style.reshape(B * L, D)
            delta = self.style_fusion(torch.cat([flat, flat_style], dim=-1))
            return base_feat + delta.reshape(B, L, D)

    def encode_support(self, support_latents_BCL: torch.Tensor) -> torch.Tensor:
        """Convenience: run prototype encoder on support set (IBE-convention channel-first).

        support_latents_BCL: (K, 4, 128)
        returns: (embed_dim,)
        """
        return self.prototype_encoder(support_latents_BCL)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    from prototype_encoder import PrototypeEncoder

    print("Loading IBExtractor (random init for smoke test)...")
    base_ibe = IBExtractor(
        embed_dim=256,
        num_heads=8,
        ff_hidden_size=1024,
        num_layers=3,
        text_embed_dim=768,
        patient_info_size=3,
    )
    proto = PrototypeEncoder()
    model = StyleTranslatorIBE(base_ibe, proto)
    print(f"Trainable params: {model.trainable_params():,}")
    print(f"Frozen params:    {model.frozen_params():,}")

    B = 4
    L_sig = 128
    L_text = 10
    x = torch.randn(B, L_sig, 4)
    text = torch.randn(B, L_text, 768)
    mask = torch.ones(B, L_text)
    p = torch.randn(B, 3)

    # Without style: behavior == base IBE
    base_feat = model.base_ibe.extract_features(x, text, mask, p, reduce=True)
    out_nostyle = model.extract_features(x, text, mask, p, style_vec=None)
    diff = (out_nostyle - base_feat).abs().max().item()
    print(f"No-style forward matches base IBE: max abs diff = {diff:.2e}  (should be 0)")
    assert diff < 1e-6

    # With style (zero-init last layer => delta should be ~0 initially)
    K = 50
    support = torch.randn(K, 4, L_sig)
    style = proto(support)
    out_style = model.extract_features(x, text, mask, p, style_vec=style)
    diff_init = (out_style - base_feat).abs().max().item()
    print(f"Initial style delta (zero-init): max abs diff = {diff_init:.2e}  (should be 0)")
    assert diff_init < 1e-6

    # Backward
    loss = out_style.sum()
    loss.backward()
    print("Backward pass ok")
    # Check that only trainable params have grads
    assert model.base_ibe.signal_embedding.weight.grad is None, "base IBE should not have grads"
    assert model.style_fusion[0].weight.grad is not None, "style_fusion should have grads"
    assert model.prototype_encoder.per_sample[0].weight.grad is not None, "prototype_encoder should have grads"

    # Nudge style_fusion last layer to have nonzero init, verify delta nonzero
    with torch.no_grad():
        model.style_fusion[-1].weight.normal_(std=0.01)
        model.style_fusion[-1].bias.zero_()
    out_style2 = model.extract_features(x, text, mask, p, style_vec=style)
    diff2 = (out_style2 - base_feat).abs().max().item()
    print(f"After perturbing fusion: delta max = {diff2:.4f}  (should be > 0)")
    assert diff2 > 1e-4

    print("✓ StyleTranslatorIBE smoke test passed")
