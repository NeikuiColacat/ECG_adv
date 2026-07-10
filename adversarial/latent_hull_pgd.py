"""Same-label ECGTwin VAE latent-hull adversarial generator.

This keeps the public output format of ``PGDAdvDiffGenerator`` but replaces the
free latent perturbation ``z0 + delta`` with a constrained convex-hull move:

    z_mix = sum_i softmax(a_i) * z_i
    z_adv = (1 - lambda) * z0 + lambda * z_mix

Only the mixture logits ``a`` are optimized. Candidate latents are expected to
come from the same primary class or exact same multi-hot label as ``z0``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adversarial.efficientnet_victim_tierM import EfficientNetVictimTierM  # noqa: E402
from adversarial.pgd_advdiff import PGDAdvDiffGenerator  # noqa: E402
from ecg_adv_gen.adaptation.latent_hull_torch import (  # noqa: E402
    initial_hull_weights,
    latent_hull_geometry,
)


class LatentHullPGDGenerator(PGDAdvDiffGenerator):
    """Constrained same-label convex-hull attack in ECGTwin VAE latent space."""

    def __init__(
        self,
        ecgtwin_wrapper,
        victim: EfficientNetVictimTierM,
        epsilon: Optional[float] = None,
        hull_lambda: float = 0.25,
        hull_steps: int = 5,
        hull_lr: float = 0.3,
        weight_mode: str = "optimized",
        dirichlet_alpha: float = 1.0,
        init_logit_gap: float = 4.0,
        device: str = "cuda",
    ):
        super().__init__(
            ecgtwin_wrapper=ecgtwin_wrapper,
            victim=victim,
            epsilon=float(epsilon) if epsilon is not None else 1.0,
            K_pgd=0,
            alpha=None,
            delta_init_scale=0.0,
            device=device,
        )
        self.hull_epsilon = float(epsilon) if epsilon is not None else None
        self.hull_lambda = float(hull_lambda)
        self.hull_steps = int(hull_steps)
        self.hull_lr = float(hull_lr)
        if weight_mode not in {"optimized", "one_hot", "uniform", "dirichlet"}:
            raise ValueError(
                "weight_mode must be optimized|one_hot|uniform|dirichlet, "
                f"got {weight_mode!r}"
            )
        self.weight_mode = weight_mode
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.init_logit_gap = float(init_logit_gap)
        self.attack_pos_weight: Optional[torch.Tensor] = None
        self.last_info: Dict[str, Any] = {}
        self.last_weights: Optional[torch.Tensor] = None
        self.last_initial_signals: Optional[torch.Tensor] = None

    @staticmethod
    def _enable_rnn_backward_only(module: torch.nn.Module) -> list[tuple[torch.nn.Module, bool]]:
        """Temporarily put only cuDNN RNN modules in train mode for input-gradient attacks.

        cuDNN RNN backward is unavailable for eval-mode RNNs, but switching the
        whole victim to train would also change BatchNorm/Dropout behavior.  The
        latent-hull inner loop only needs gradients through the recurrent layer,
        so we toggle RNNBase children and restore their original states later.
        """
        states: list[tuple[torch.nn.Module, bool]] = []
        for child in module.modules():
            if isinstance(child, torch.nn.modules.rnn.RNNBase):
                states.append((child, child.training))
                child.train(True)
        return states

    @staticmethod
    def _restore_module_training_states(states: list[tuple[torch.nn.Module, bool]]) -> None:
        for child, was_training in states:
            child.train(was_training)

    @staticmethod
    def _require_finite_diagnostics(diagnostics: Dict[str, Any]) -> None:
        for field, value in diagnostics.items():
            if value is None:
                continue
            values = np.asarray(value)
            if values.dtype.kind not in "fc":
                continue
            bad = np.argwhere(~np.isfinite(values))
            if bad.size:
                location = f"sample {int(bad[0, 0])}" if values.ndim else "aggregate"
                raise RuntimeError(
                    f"latent-hull generator diagnostic field {field!r} is non-finite "
                    f"at {location}"
                )

    def _fixed_weights(self, bsz: int, m: int) -> Optional[torch.Tensor]:
        if self.weight_mode == "optimized":
            return None
        weights = initial_hull_weights(
            bsz, m, weight_mode=self.weight_mode, init_logit_gap=self.init_logit_gap,
            device=self.device, dtype=torch.float32,
        )
        if weights is not None:
            return weights
        concentration = torch.full((m,), self.dirichlet_alpha, device=self.device)
        return torch.distributions.Dirichlet(concentration).sample((bsz,))

    def attack_from_latent(
        self,
        z0: torch.Tensor,
        y0: torch.Tensor,
        candidate_latents: torch.Tensor,
        init_logits: Optional[torch.Tensor] = None,
        *,
        anchor_pool_indices: Optional[torch.Tensor] = None,
        candidate_pool_indices: Optional[torch.Tensor] = None,
        candidate_is_anchor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize same-label mixture weights and return decoded adv signals.

        Args:
          z0: (B,4,128) anchor latent.
          y0: (B,C) multi-hot label.
          candidate_latents: (B,M,4,128), same-label candidates.
          init_logits: optional (B,M). If omitted, candidate 0 starts dominant.

        Returns:
          x_adv_1000: (B,12,1000) PTBXL-order, z-scored.
          delta: (B,4,128) = z_adv - z0.
        """
        z0 = z0.to(self.device).detach().float()
        y0 = y0.to(self.device).detach().float()
        cand = candidate_latents.to(self.device).detach().float()
        if cand.dim() != 4 or cand.shape[0] != z0.shape[0] or cand.shape[2:] != (4, 128):
            raise ValueError(f"bad candidate_latents shape: {tuple(cand.shape)}")
        if y0.dim() != 2 or y0.shape[-1] != self.victim.num_classes:
            raise ValueError(f"bad y0 shape: {tuple(y0.shape)}")

        bsz, m = cand.shape[:2]
        anchor_ids = None if anchor_pool_indices is None else torch.as_tensor(anchor_pool_indices).long().cpu()
        candidate_ids = (
            None if candidate_pool_indices is None else torch.as_tensor(candidate_pool_indices).long().cpu()
        )
        if ((anchor_ids is not None and anchor_ids.shape != (bsz,)) or
                (candidate_ids is not None and candidate_ids.shape != (bsz, m))):
            raise ValueError("latent-hull pool identities do not match the attack batch")
        derived_mask = (
            candidate_ids == anchor_ids[:, None]
            if anchor_ids is not None and candidate_ids is not None else None
        )
        identity_mask = (
            torch.as_tensor(candidate_is_anchor).bool().cpu()
            if candidate_is_anchor is not None else derived_mask
        )
        if identity_mask is None:
            identity_mask = torch.zeros((bsz, m), dtype=torch.bool)
        if identity_mask.shape != (bsz, m) or (derived_mask is not None and not torch.equal(identity_mask, derived_mask)):
            raise ValueError("candidate anchor identities do not match the supplied pool identities")
        identity_mask_device = identity_mask.to(self.device)

        last_loss = None
        fixed_w = self._fixed_weights(bsz, m)
        if fixed_w is None:
            if init_logits is None:
                initial_w = initial_hull_weights(
                    bsz,
                    m,
                    weight_mode="optimized",
                    init_logit_gap=self.init_logit_gap,
                    device=self.device,
                    dtype=z0.dtype,
                )
                assert initial_w is not None
                logits_a = initial_w.clamp_min(1e-12).log()
            else:
                logits_a = init_logits.to(self.device).detach().float().clone()
                if logits_a.shape != (bsz, m):
                    raise ValueError(f"bad init_logits shape: {tuple(logits_a.shape)}")
                initial_w = torch.softmax(logits_a.detach(), dim=-1)
            logits_a.requires_grad_(True)

            opt = torch.optim.Adam([logits_a], lr=self.hull_lr)
            rnn_states = self._enable_rnn_backward_only(self.victim.model)
            try:
                for _ in range(self.hull_steps):
                    w = torch.softmax(logits_a, dim=-1)
                    z_mix = (w.view(bsz, m, 1, 1) * cand).sum(dim=1)
                    z_adv = (1.0 - self.hull_lambda) * z0 + self.hull_lambda * z_mix
                    logits = self.victim.forward_from_latent_to_logits(z_adv)
                    pos_weight = None
                    if self.attack_pos_weight is not None:
                        pos_weight = self.attack_pos_weight.to(device=logits.device, dtype=logits.dtype)
                    loss = F.binary_cross_entropy_with_logits(
                        logits,
                        y0,
                        pos_weight=pos_weight,
                        reduction="mean",
                    )
                    opt.zero_grad(set_to_none=True)
                    (-loss).backward()
                    opt.step()
                    last_loss = loss.detach()
            finally:
                self._restore_module_training_states(rnn_states)
        else:
            logits_a = None
            initial_w = fixed_w.detach().clone()

        with torch.no_grad():
            initial_geometry = latent_hull_geometry(
                z0,
                cand,
                initial_w,
                hull_lambda=self.hull_lambda,
                epsilon=self.hull_epsilon,
                candidate_is_anchor=identity_mask_device,
            )

        with torch.no_grad():
            if fixed_w is None:
                w = torch.softmax(logits_a.detach(), dim=-1)
            else:
                w = fixed_w
            final_geometry = latent_hull_geometry(
                z0,
                cand,
                w,
                hull_lambda=self.hull_lambda,
                epsilon=self.hull_epsilon,
                candidate_is_anchor=identity_mask_device,
            )
            z_adv = final_geometry["latent"]
            delta = final_geometry["delta_post"]
            x_adv_1000 = self._decode_to_ptbxl_1000(z_adv)
            self.last_initial_signals = self._decode_to_ptbxl_1000(
                initial_geometry["latent_pre"]
            ).detach().cpu()
            self.last_weights = w.detach().cpu()
            entropy = -(w * w.clamp(min=1e-12).log()).sum(dim=-1)
            initial_cpu = initial_w.detach().cpu()
            final_cpu = w.detach().cpu()
            initial_top1_weights, initial_top1_positions = initial_cpu.max(dim=-1)
            final_top1_weights, final_top1_positions = final_cpu.max(dim=-1)

            def top1_pool_ids(positions: torch.Tensor) -> list[int]:
                if candidate_ids is None:
                    return positions.tolist()
                return candidate_ids.gather(1, positions[:, None]).squeeze(1).tolist()

            def geometry_lists(prefix: str, geometry: Dict[str, torch.Tensor]) -> Dict[str, list[float]]:
                return {
                    f"{prefix}_{key}": geometry[key].detach().float().cpu().tolist()
                    for key in (
                        "pre_projection_norm",
                        "post_projection_norm",
                        "projection_scale",
                        "effective_lambda",
                        "candidate_anchor_weight",
                        "effective_original_share",
                    )
                }

            last_info = {
                "schema_version": 1,
                "anchor_pool_indices": None if anchor_ids is None else anchor_ids.tolist(),
                "candidate_pool_indices": None if candidate_ids is None else candidate_ids.tolist(),
                "candidate_is_anchor": identity_mask.tolist(),
                "initial_weights": initial_cpu.tolist(),
                "final_weights": final_cpu.tolist(),
                "initial_top1_candidate_pool_indices": top1_pool_ids(initial_top1_positions),
                "final_top1_candidate_pool_indices": top1_pool_ids(final_top1_positions),
                "initial_top1_weights": initial_top1_weights.tolist(),
                "final_top1_weights": final_top1_weights.tolist(),
                **geometry_lists("initial", initial_geometry),
                **geometry_lists("final", final_geometry),
                "hull_weight_entropy_mean": float(entropy.mean().cpu()),
                "hull_weight_entropy_max": float(entropy.max().cpu()),
                "hull_weight_top1_mean": float(w.max(dim=-1).values.mean().cpu()),
                "hull_delta_norm_mean": float(delta.flatten(1).norm(dim=1).mean().cpu()),
                "hull_delta_norm_max": float(delta.flatten(1).norm(dim=1).max().cpu()),
                "hull_inner_loss": float(last_loss.cpu()) if last_loss is not None else None,
                "hull_weight_mode": self.weight_mode,
            }
            self._require_finite_diagnostics(last_info)
            self.last_info = last_info
        return x_adv_1000, delta.detach()
