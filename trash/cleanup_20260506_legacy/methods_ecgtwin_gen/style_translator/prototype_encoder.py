"""PrototypeEncoder: K support ECG latents -> 256-d center style vector.

Permutation invariant via mean pool. Supports any K (train K=50, deploy K=50-200).
"""
from typing import Optional

import torch
import torch.nn as nn


class PrototypeEncoder(nn.Module):
    """Map a support set of VAE latents (K, 4, 128) to a 256-d style vector.

    Design:
      - per-sample Conv1d encoder captures local waveform patterns (P/QRS/T positioning)
      - mean-pool over the K support samples => permutation-invariant
      - final LayerNorm stabilizes style vec scale across enrollments
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden: int = 128,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.out_dim = out_dim

        self.per_sample = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # -> (K, hidden, 1)
        )
        self.aggregate = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, support_latents: torch.Tensor) -> torch.Tensor:
        """support_latents: (K, 4, 128) -> style_vec: (out_dim,)."""
        if support_latents.dim() != 3:
            raise ValueError(
                f"expected (K, in_ch, L); got {tuple(support_latents.shape)}"
            )
        h = self.per_sample(support_latents).squeeze(-1)   # (K, hidden)
        pooled = h.mean(dim=0)                              # (hidden,) perm-invariant
        return self.aggregate(pooled)                       # (out_dim,)

    def forward_batch(self, support_batch: torch.Tensor) -> torch.Tensor:
        """Process a batch of support sets in parallel.

        support_batch: (n_sets, K, 4, 128)
        returns: (n_sets, out_dim)
        """
        if support_batch.dim() != 4:
            raise ValueError(
                f"expected (n_sets, K, in_ch, L); got {tuple(support_batch.shape)}"
            )
        n_sets, K = support_batch.shape[:2]
        flat = support_batch.reshape(n_sets * K, *support_batch.shape[2:])
        h = self.per_sample(flat).squeeze(-1)           # (n_sets*K, hidden)
        h = h.reshape(n_sets, K, self.hidden)
        pooled = h.mean(dim=1)                           # (n_sets, hidden)
        return self.aggregate(pooled)                    # (n_sets, out_dim)


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    enc = PrototypeEncoder()
    total_params = sum(p.numel() for p in enc.parameters())
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"PrototypeEncoder params: total={total_params}, trainable={trainable}")

    # Single support set
    K = 50
    support = torch.randn(K, 4, 128)
    style = enc(support)
    print(f"single forward: support {tuple(support.shape)} -> style {tuple(style.shape)}")
    assert style.shape == (256,)

    # Permutation invariance check
    idx = torch.randperm(K)
    style_perm = enc(support[idx])
    diff = (style - style_perm).abs().max().item()
    print(f"permutation invariance check: max abs diff = {diff:.2e}  (should be ~0)")
    assert diff < 1e-5

    # Batch forward
    n_sets = 3
    support_batch = torch.randn(n_sets, K, 4, 128)
    styles = enc.forward_batch(support_batch)
    print(f"batch forward: {tuple(support_batch.shape)} -> {tuple(styles.shape)}")
    assert styles.shape == (n_sets, 256)

    # Consistency: forward_batch should match per-set forward
    styles_manual = torch.stack([enc(support_batch[i]) for i in range(n_sets)])
    diff = (styles - styles_manual).abs().max().item()
    print(f"batch vs manual max abs diff = {diff:.2e}  (should be ~0)")
    assert diff < 1e-5

    # Backward
    loss = styles.sum()
    loss.backward()
    print("backward pass ok")

    print("✓ PrototypeEncoder smoke test passed")
