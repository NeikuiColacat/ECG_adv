import torch
import torch.nn as nn
from typing import List


class CenterToken(nn.Module):
    """
    单个可学习的 center token，表示某个医院中心的域特征（设备噪声、基线漂移等）。
    注入到 DiT_ECGTwin 的 AdaX 调制路径中：c = t_emb + ib_proj(base_vector) + center_embedding

    维度 256，与 DiT hidden_size 一致。
    初始化为零向量，使训练起始时模型行为与原始 ECGTwin 一致。

    v1 设计 (Plan Rev 1-11): 单 256-d 向量, 广播到所有 DiT block.
    Plan Rev 12 (2026-04-27) audit: 实测训出来 norm ~0.20, rel-diff vs zero
    < 0.012 — 在 noise floor.  v2 用 CenterTokenPerBlock (per-block, 6 个独立向量).
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        """返回 (B, dim) 的 center embedding"""
        return self.embedding.unsqueeze(0).expand(batch_size, -1)

    @property
    def norm(self) -> float:
        return self.embedding.data.norm(p=2).item()


class CenterTokenPerBlock(nn.Module):
    """
    Per-DiT-block learnable center tokens (TokenVerse SIGGRAPH'25 paradigm).

    6 个独立的 256-d ParameterList entries, 每个 DiT block 一个 token.
    Hook 必须传 block_idx 才能拿到对应 block 的 token.

    维度 (num_blocks=6, dim=256), 全部 init zero.

    与 v1 CenterToken 的差异:
      - v1: 单 256-d, 广播到 6 个 block (受限于 broadcast, 没有 per-block bandwidth)
      - v2: 6 × 256-d, 每个 block 一个 (TokenVerse stage-2 / P+ 范式)

    `forward(block_idx, batch_size)` 而非 `forward(batch_size)`,
    需要新的 hook closure 传递 block_idx (见 generate_center_synth.py
    的 per_block 路径 / trainer_v2.py 的 _make_hook(block_idx)).
    """

    def __init__(self, dim: int = 256, num_blocks: int = 6):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.embeddings = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(num_blocks)]
        )

    def forward(self, block_idx: int, batch_size: int) -> torch.Tensor:
        return self.embeddings[block_idx].unsqueeze(0).expand(batch_size, -1)

    @property
    def norms(self) -> List[float]:
        """Per-block L2 norms — diagnostic during training to detect collapse."""
        return [emb.data.norm(p=2).item() for emb in self.embeddings]

    @property
    def total_norm(self) -> float:
        """Concatenated flat-vector norm (for global magnitude tracking)."""
        return torch.cat([emb.data for emb in self.embeddings]).norm(p=2).item()

    def flat(self) -> torch.Tensor:
        """Concatenated (num_blocks*dim,) flat tensor for cosine matrix
        comparisons across centers (smoking gun gate G3)."""
        return torch.cat([emb.data for emb in self.embeddings])

    def project_to_sphere(self, target_norms: List[float]):
        """In-place per-block sphere projection.

        Plan CT v2 fix for Bug H (L_reg pulls token to 0): instead of L2
        regularization, project each block's token to the magnitude of the
        AdaLN driver scale (typically ~1.0 since ib_projector outputs are
        Linear(256, 256) of unit-stable inputs).

        Args:
          target_norms: list of length num_blocks, target L2 per block.
        """
        assert len(target_norms) == self.num_blocks
        with torch.no_grad():
            for i, emb in enumerate(self.embeddings):
                cur = emb.data.norm(p=2).item()
                if cur > 1e-8:
                    emb.data.mul_(target_norms[i] / cur)
