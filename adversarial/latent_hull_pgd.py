"""Same-label ECGTwin VAE latent-hull adversarial generator.

This keeps the public output format of ``PGDAdvDiffGenerator`` but replaces the
free latent perturbation ``z0 + delta`` with a constrained convex-hull move:

    z_mix = sum_i softmax(a_i) * z_i
    z_adv = (1 - lambda) * z0 + lambda * z_mix

Only the mixture logits ``a`` are optimized. Candidate latents are expected to
come from the same primary class or exact same multi-hot label as ``z0``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adversarial.efficientnet_victim_tierM import EfficientNetVictimTierM  # noqa: E402
from adversarial.pgd_advdiff import PGDAdvDiffGenerator  # noqa: E402


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
        self.last_info: Dict[str, float] = {}
        self.last_weights: Optional[torch.Tensor] = None

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

    def _fixed_weights(self, bsz: int, m: int) -> Optional[torch.Tensor]:
        if self.weight_mode == "optimized":
            return None
        if self.weight_mode == "one_hot":
            w = torch.zeros((bsz, m), device=self.device)
            w[:, 0] = 1.0
            return w
        if self.weight_mode == "uniform":
            return torch.full((bsz, m), 1.0 / float(m), device=self.device)
        concentration = torch.full((m,), self.dirichlet_alpha, device=self.device)
        return torch.distributions.Dirichlet(concentration).sample((bsz,))

    def attack_from_latent(
        self,
        z0: torch.Tensor,
        y0: torch.Tensor,
        candidate_latents: torch.Tensor,
        init_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize same-label mixture weights and return decoded adv signals.

        Args:
          z0: (B,C,L) anchor latent.
          y0: (B,C) multi-hot label.
          candidate_latents: (B,M,C,L), same-label candidates.
          init_logits: optional (B,M). If omitted, candidate 0 starts dominant.

        Returns:
          x_adv_1000: (B,12,1000) PTBXL-order, z-scored.
          delta: (B,4,128) = z_adv - z0.
        """
        z0 = z0.to(self.device).detach().float()
        y0 = y0.to(self.device).detach().float()
        cand = candidate_latents.to(self.device).detach().float()
        if cand.dim() != 4 or cand.shape[0] != z0.shape[0] or cand.shape[2:] != z0.shape[1:]:
            raise ValueError(f"bad candidate_latents shape: {tuple(cand.shape)}")
        if y0.dim() != 2 or y0.shape[-1] != self.victim.num_classes:
            raise ValueError(f"bad y0 shape: {tuple(y0.shape)}")

        bsz, m = cand.shape[:2]
        last_loss = None
        fixed_w = self._fixed_weights(bsz, m)
        if fixed_w is None:
            if init_logits is None:
                logits_a = torch.full((bsz, m), -self.init_logit_gap, device=self.device)
                logits_a[:, 0] = self.init_logit_gap
            else:
                logits_a = init_logits.to(self.device).detach().float().clone()
                if logits_a.shape != (bsz, m):
                    raise ValueError(f"bad init_logits shape: {tuple(logits_a.shape)}")
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

        with torch.no_grad():
            if fixed_w is None:
                w = torch.softmax(logits_a.detach(), dim=-1)
            else:
                w = fixed_w
            self.last_weights = w.detach().cpu()
            z_mix = (w.view(bsz, m, 1, 1) * cand).sum(dim=1)
            z_adv = (1.0 - self.hull_lambda) * z0 + self.hull_lambda * z_mix
            delta = z_adv - z0
            if self.hull_epsilon is not None and self.hull_epsilon > 0:
                delta = self._project_l2_ball(delta)
                z_adv = z0 + delta
            x_adv_1000 = self._decode_to_ptbxl_1000(z_adv)
            entropy = -(w * w.clamp(min=1e-12).log()).sum(dim=-1)
            self.last_info = {
                "hull_weight_entropy_mean": float(entropy.mean().cpu()),
                "hull_weight_entropy_max": float(entropy.max().cpu()),
                "hull_weight_top1_mean": float(w.max(dim=-1).values.mean().cpu()),
                "hull_delta_norm_mean": float(delta.flatten(1).norm(dim=1).mean().cpu()),
                "hull_delta_norm_max": float(delta.flatten(1).norm(dim=1).max().cpu()),
                "hull_inner_loss": float(last_loss.cpu()) if last_loss is not None else float("nan"),
                "hull_weight_mode": self.weight_mode,
            }
        return x_adv_1000, delta.detach()


def _smoke(args: argparse.Namespace) -> None:
    from util.ecgtwin_utils import ECGTwinWrapper

    print("[smoke] loading ECGTwin wrapper ...")
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)
    print(f"[smoke] loading victim from {args.ckpt} ...")
    victim = EfficientNetVictimTierM(
        weight_path=args.ckpt,
        device=args.device,
        ecgtwin_wrapper=wrapper,
        num_classes=args.num_classes,
    )
    gen = LatentHullPGDGenerator(
        ecgtwin_wrapper=wrapper,
        victim=victim,
        epsilon=args.epsilon,
        hull_lambda=args.hull_lambda,
        hull_steps=args.hull_steps,
        hull_lr=args.hull_lr,
        weight_mode=args.weight_mode,
        dirichlet_alpha=args.dirichlet_alpha,
        device=args.device,
    )
    z0 = torch.randn(args.batch_size, 4, 128, device=args.device) * 0.5
    cand = z0[:, None] + torch.randn(args.batch_size, args.M, 4, 128, device=args.device) * 0.1
    y = torch.zeros(args.batch_size, args.num_classes, device=args.device)
    y[:, min(2, args.num_classes - 1)] = 1.0
    t0 = time.time()
    x, delta = gen.attack_from_latent(z0, y, cand)
    print(f"[smoke] {args.batch_size} samples in {time.time() - t0:.2f}s")
    print(f"  x: {tuple(x.shape)} finite={torch.isfinite(x).all().item()}")
    print(f"  delta: {tuple(delta.shape)} mean_norm={delta.flatten(1).norm(dim=1).mean().item():.4f}")
    print(f"  info: {gen.last_info}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/triple_labels/super5/best_model.pt")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--hull_lambda", type=float, default=0.25)
    ap.add_argument("--hull_steps", type=int, default=2)
    ap.add_argument("--hull_lr", type=float, default=0.3)
    ap.add_argument(
        "--weight_mode",
        choices=["optimized", "one_hot", "uniform", "dirichlet"],
        default="optimized",
    )
    ap.add_argument("--dirichlet_alpha", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=2.0)
    args = ap.parse_args()
    if args.smoke:
        _smoke(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
