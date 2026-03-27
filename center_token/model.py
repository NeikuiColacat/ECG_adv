import torch
import torch.nn as nn


class CenterToken(nn.Module):
    """
    单个可学习的 center token，表示某个医院中心的域特征（设备噪声、基线漂移等）。
    注入到 DiT_ECGTwin 的 AdaX 调制路径中：c = t_emb + ib_proj(base_vector) + center_embedding

    维度 256，与 DiT hidden_size 一致。
    初始化为零向量，使训练起始时模型行为与原始 ECGTwin 一致。
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
