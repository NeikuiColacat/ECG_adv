from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class CenterClassPromptTokenBank(nn.Module):
    """Learnable 768-d prompt tokens indexed by target center and super5 class.

    This is intentionally a text-embedding bank, not a tokenizer modification and
    not the older 256-d AdaLN/base-vector hook. At generation/training time the
    selected row is appended to ECGTwin's `text_embed` sequence and its mask bit
    is set to 1.
    """

    def __init__(
        self,
        centers: Iterable[str],
        class_names: Iterable[str],
        dim: int = 768,
        init_by_class: Optional[Dict[str, torch.Tensor]] = None,
        init_noise_std: float = 0.0,
        n_token_vectors: int = 1,
        token_mode: str = "direct",
        center_vectors: int = 1,
        class_vectors: int = 1,
        residual_vectors: int = 1,
        residual_init_std: float = 0.0,
    ):
        super().__init__()
        self.centers: List[str] = list(centers)
        self.class_names: List[str] = list(class_names)
        self.dim = dim
        self.n_token_vectors = int(n_token_vectors)
        self.token_mode = str(token_mode)
        self.center_vectors = int(center_vectors)
        self.class_vectors = int(class_vectors)
        self.residual_vectors = int(residual_vectors)
        self.residual_init_std = float(residual_init_std)
        if not self.centers:
            raise ValueError("centers must be non-empty")
        if not self.class_names:
            raise ValueError("class_names must be non-empty")
        if self.token_mode not in {"direct", "factorized"}:
            raise ValueError(f"token_mode must be direct|factorized, got {token_mode!r}")
        if self.n_token_vectors < 1:
            raise ValueError("n_token_vectors must be >= 1")

        class_init = torch.zeros(len(self.class_names), dim)
        if init_by_class:
            for class_idx, cls in enumerate(self.class_names):
                if cls not in init_by_class:
                    continue
                emb = init_by_class[cls].detach().float()
                if emb.dim() == 2:
                    emb = emb.mean(dim=0)
                if emb.shape != (dim,):
                    raise ValueError(f"init embedding for {cls} has shape {tuple(emb.shape)}")
                class_init[class_idx] = emb
        global_init = class_init.mean(dim=0)

        if self.token_mode == "direct":
            init = class_init.view(1, len(self.class_names), 1, dim).repeat(
                len(self.centers), 1, self.n_token_vectors, 1
            )
            if init_noise_std > 0:
                init.add_(torch.randn_like(init) * init_noise_std)
            self.embeddings = nn.Parameter(init)
            self.register_buffer("init_embeddings", init.clone(), persistent=True)
        else:
            if self.center_vectors < 1 or self.class_vectors < 1 or self.residual_vectors < 0:
                raise ValueError("factorized vectors must satisfy center>=1, class>=1, residual>=0")
            center_init = global_init.view(1, 1, dim).repeat(
                len(self.centers), self.center_vectors, 1
            )
            class_token_init = class_init.view(len(self.class_names), 1, dim).repeat(
                1, self.class_vectors, 1
            )
            residual_init = torch.zeros(
                len(self.centers), len(self.class_names), self.residual_vectors, dim
            )
            if init_noise_std > 0:
                center_init.add_(torch.randn_like(center_init) * init_noise_std)
                class_token_init.add_(torch.randn_like(class_token_init) * init_noise_std)
            if residual_init_std > 0 and self.residual_vectors > 0:
                residual_init.add_(torch.randn_like(residual_init) * residual_init_std)

            self.center_embeddings = nn.Parameter(center_init)
            self.class_embeddings = nn.Parameter(class_token_init)
            self.residual_embeddings = nn.Parameter(residual_init)
            self.register_buffer("init_center_embeddings", center_init.clone(), persistent=True)
            self.register_buffer("init_class_embeddings", class_token_init.clone(), persistent=True)
            self.register_buffer("init_residual_embeddings", residual_init.clone(), persistent=True)

    def forward(self, center_idx: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        seq = self.token_sequence(center_idx, class_idx)
        if seq.shape[1] == 1:
            return seq[:, 0, :]
        return seq

    def token_sequence(self, center_idx: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        center_idx = center_idx.long()
        class_idx = class_idx.long()
        if self.token_mode == "direct":
            return self.embeddings[center_idx, class_idx]
        parts = [
            self.center_embeddings[center_idx],
            self.class_embeddings[class_idx],
        ]
        if self.residual_vectors > 0:
            parts.append(self.residual_embeddings[center_idx, class_idx])
        return torch.cat(parts, dim=1)

    def init_for(self, center_idx: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        seq = self.init_sequence_for(center_idx, class_idx)
        if seq.shape[1] == 1:
            return seq[:, 0, :]
        return seq

    def init_sequence_for(self, center_idx: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        center_idx = center_idx.long()
        class_idx = class_idx.long()
        if self.token_mode == "direct":
            return self.init_embeddings[center_idx, class_idx]
        parts = [
            self.init_center_embeddings[center_idx],
            self.init_class_embeddings[class_idx],
        ]
        if self.residual_vectors > 0:
            parts.append(self.init_residual_embeddings[center_idx, class_idx])
        return torch.cat(parts, dim=1)

    @property
    def center_to_idx(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.centers)}

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.class_names)}

    @torch.no_grad()
    def token_norm_table(self) -> Dict[str, Dict[str, float]]:
        rows = []
        device = next(self.parameters()).device
        for center_idx in range(len(self.centers)):
            center = torch.full((len(self.class_names),), center_idx, dtype=torch.long, device=device)
            cls = torch.arange(len(self.class_names), dtype=torch.long, device=device)
            rows.append(self.token_sequence(center, cls).detach().float().norm(dim=-1).mean(dim=-1))
        norms = torch.stack(rows, dim=0).cpu()
        return {
            center: {
                cls: float(norms[center_idx, class_idx])
                for class_idx, cls in enumerate(self.class_names)
            }
            for center_idx, center in enumerate(self.centers)
        }

    @torch.no_grad()
    def delta_norm_table(self) -> Dict[str, Dict[str, float]]:
        rows = []
        device = next(self.parameters()).device
        for center_idx in range(len(self.centers)):
            center = torch.full((len(self.class_names),), center_idx, dtype=torch.long, device=device)
            cls = torch.arange(len(self.class_names), dtype=torch.long, device=device)
            seq = self.token_sequence(center, cls).detach().float()
            init = self.init_sequence_for(center, cls).detach().float()
            rows.append((seq - init).norm(dim=-1).mean(dim=-1))
        delta = torch.stack(rows, dim=0).cpu()
        return {
            center: {
                cls: float(delta[center_idx, class_idx])
                for class_idx, cls in enumerate(self.class_names)
            }
            for center_idx, center in enumerate(self.centers)
        }

    def export_state(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "version": "ecgtwin_center_class_prompt_token_v2",
            "centers": list(self.centers),
            "class_names": list(self.class_names),
            "dim": self.dim,
            "token_mode": self.token_mode,
            "n_token_vectors": self.n_token_vectors,
            "center_vectors": self.center_vectors,
            "class_vectors": self.class_vectors,
            "residual_vectors": self.residual_vectors,
            "token_norms": self.token_norm_table(),
            "delta_norms": self.delta_norm_table(),
        }
        if self.token_mode == "direct":
            out.update({
                "embeddings": self.embeddings.detach().cpu(),
                "init_embeddings": self.init_embeddings.detach().cpu(),
            })
        else:
            out.update({
                "center_embeddings": self.center_embeddings.detach().cpu(),
                "class_embeddings": self.class_embeddings.detach().cpu(),
                "residual_embeddings": self.residual_embeddings.detach().cpu(),
                "init_center_embeddings": self.init_center_embeddings.detach().cpu(),
                "init_class_embeddings": self.init_class_embeddings.detach().cpu(),
                "init_residual_embeddings": self.init_residual_embeddings.detach().cpu(),
            })
        return out
