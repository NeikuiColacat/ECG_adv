"""ECGTwin-PGD adversarial sample generator (Mode A: VAE-latent PGD).

Replaces the BoundaryAdvDiffGenerator's logit-squared boundary loss with
standard Madry-style PGD adversarial training philosophy adapted to
ECGTwin's VAE manifold:

  Anchor  : real PN2021 ECG x0 with ground-truth Tier-M label y0
  Encode  : z0 = VAE.encode(x0)   (pre-encoded via prep_center_dataset.py)
  Attack  : for K_pgd steps, gradient ASCENT on BCE(victim(decode(z0+δ)), y0)
  Project : L2 ε-ball  ||δ||_2 <= epsilon_latent
  Output  : x_adv = decode(z0+δ) in PTBXL-order, 1000-sample, z-scored
  Label   : y0 (anchor's truth)  — NO synthesis, NO ref+soft_target

The VAE decoder acts as a manifold projection, so adv samples remain
physiologically plausible (Einthoven's Law, QRS morphology) while δ is
constrained in latent space.

Mode B (DiT-denoising PGD) left as future work — see plan §Mode B.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adversarial.efficientnet_victim_tierM import EfficientNetVictimTierM  # noqa: E402


ATTACK_LOSS_MODES = {"bce", "positive_hide", "negative_add", "pos_hide_neg_add"}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / denom


def _masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    pieces = []
    for row_values, row_mask in zip(values, mask, strict=True):
        selected = row_values[row_mask]
        if selected.numel() == 0:
            continue
        kk = min(int(k), selected.numel())
        pieces.append(selected.topk(kk).values.mean())
    if not pieces:
        return values.sum() * 0.0
    return torch.stack(pieces).mean()


def multilabel_attack_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str = "bce",
    pos_weight: torch.Tensor | None = None,
    pos_hide_weight: float = 1.0,
    neg_add_weight: float = 0.25,
    negative_exclude_indices: Sequence[int] | None = None,
    neg_topk: int = 0,
) -> torch.Tensor:
    """Attack objective for multi-label ECG classification.

    ``bce`` preserves the historical behavior: maximize full BCE against the
    original multi-hot label. The explicit modes split that loss into positive
    label hiding and negative label addition so experiments can control which
    failure mode the latent attack should search for.
    """
    if mode not in ATTACK_LOSS_MODES:
        raise ValueError(f"unknown attack loss mode {mode!r}; valid={sorted(ATTACK_LOSS_MODES)}")
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="mean",
        )

    pos_mask = targets > 0.5
    neg_mask = targets < 0.5
    if negative_exclude_indices:
        exclude = [int(i) for i in negative_exclude_indices]
        valid = [i for i in exclude if 0 <= i < logits.shape[1]]
        if valid:
            neg_mask = neg_mask.clone()
            neg_mask[:, valid] = False

    pos_terms = F.softplus(-logits)
    neg_terms = F.softplus(logits)
    pos_loss = _masked_mean(pos_terms, pos_mask) if pos_mask.any() else logits.sum() * 0.0
    if neg_mask.any():
        if neg_topk and neg_topk > 0:
            neg_loss = _masked_topk_mean(neg_terms, neg_mask, int(neg_topk))
        else:
            neg_loss = _masked_mean(neg_terms, neg_mask)
    else:
        neg_loss = logits.sum() * 0.0

    if mode == "positive_hide":
        return float(pos_hide_weight) * pos_loss
    if mode == "negative_add":
        return float(neg_add_weight) * neg_loss
    return float(pos_hide_weight) * pos_loss + float(neg_add_weight) * neg_loss


class PGDAdvDiffGenerator:
    """Mode A: PGD in VAE latent space, victim-guided BCE maximization.

    Usage:
      gen = PGDAdvDiffGenerator(ecgtwin_wrapper, victim, epsilon=1.0, K_pgd=10)
      x_adv, delta = gen.attack_from_latent(z0, y0)  # z0 (1,4,128), y0 (1,6)
      # x_adv: (1, 12, 1000) PTBXL-order, z-scored, clamp ±3
    """

    def __init__(
        self,
        ecgtwin_wrapper,
        victim: EfficientNetVictimTierM,
        epsilon: float = 1.0,
        K_pgd: int = 10,
        alpha: Optional[float] = None,
        delta_init_scale: float = 0.1,
        device: str = "cuda",
        attack_loss_mode: str = "bce",
        attack_pos_hide_weight: float = 1.0,
        attack_neg_add_weight: float = 0.25,
        attack_negative_exclude_indices: Sequence[int] | None = None,
        attack_neg_topk: int = 0,
    ):
        self.ecgtwin = ecgtwin_wrapper
        self.victim = victim
        self.epsilon = float(epsilon)
        self.K_pgd = int(K_pgd)
        # Default step size: 2ε/K (classical PGD step scaling)
        self.alpha = float(alpha) if alpha is not None else 2.0 * self.epsilon / max(self.K_pgd, 1)
        self.delta_init_scale = float(delta_init_scale)
        self.device = torch.device(device)
        if attack_loss_mode not in ATTACK_LOSS_MODES:
            raise ValueError(
                f"attack_loss_mode must be one of {sorted(ATTACK_LOSS_MODES)}, "
                f"got {attack_loss_mode!r}"
            )
        self.attack_loss_mode = attack_loss_mode
        self.attack_pos_hide_weight = float(attack_pos_hide_weight)
        self.attack_neg_add_weight = float(attack_neg_add_weight)
        self.attack_negative_exclude_indices = tuple(int(i) for i in (attack_negative_exclude_indices or ()))
        self.attack_neg_topk = int(attack_neg_topk)
        # Freeze victim
        self.victim.eval()
        for p in self.victim.parameters():
            p.requires_grad_(False)

    # ----- public attack API -----

    def attack_from_latent(
        self,
        z0: torch.Tensor,       # (B, 4, 128) anchor's VAE latent
        y0: torch.Tensor,       # (B, 6) float multi-hot label
        delta_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run K_pgd steps of PGD on the latent, return (x_adv_ecg_1000, final_delta).

        x_adv_ecg_1000: (B, 12, 1000) PTBXL-order, clamp(±3), z-scored.
                        This matches SynthCenterDataset's expected format.
        final_delta:    (B, 4, 128) final perturbation, useful for logging/analysis.
        """
        z0 = z0.to(self.device).detach()
        y0 = y0.to(self.device).detach()
        assert z0.dim() == 3, f"bad z0 shape: {z0.shape}"
        # num_classes is read off the victim so the same code works for Tier-M (6),
        # Super5 (5), or any future scheme — this is the only line that was
        # hard-coded to 6 in the Mode A pipeline (Plan Rev 8 Issue #4).
        nc = self.victim.num_classes
        assert y0.dim() == 2 and y0.shape[-1] == nc, \
            f"bad y0 shape: {y0.shape}, expected (B, {nc})"

        if delta_init is None:
            delta = torch.randn_like(z0) * self.delta_init_scale
        else:
            delta = delta_init.to(self.device).detach().clone()

        # Ensure initial δ already inside ε-ball
        delta = self._project_l2_ball(delta)
        delta.requires_grad_(True)

        for step in range(self.K_pgd):
            # Re-attach gradient at each iteration
            if not delta.requires_grad:
                delta = delta.detach().requires_grad_(True)

            z_pert = z0 + delta
            # Use victim's grad-preserving latent→logits path
            logits = self.victim.forward_from_latent_to_logits(z_pert)  # (B, 6)
            # Attack direction is gradient ASCENT on the selected multi-label objective.
            loss = self._attack_loss(logits, y0)

            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            # Normalize gradient (per-sample L2) for stable step size
            grad_norm = grad.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1)
            normed_grad = grad / grad_norm

            # Gradient ascent (maximize BCE)
            delta = delta.detach() + self.alpha * normed_grad
            # L2 ball projection per sample
            delta = self._project_l2_ball(delta)

        # Final decode to PTBXL 1000 z-scored (no crop; leave crop to training-time dataset)
        with torch.no_grad():
            x_adv_1000 = self._decode_to_ptbxl_1000(z0 + delta.detach())

        return x_adv_1000, delta.detach()

    def generate_for_anchor(
        self,
        z0: torch.Tensor,         # (4, 128) single anchor latent
        y0_multi_hot: np.ndarray,  # (6,) float multi-hot
        n_per_anchor: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate `n_per_anchor` adv samples from a single anchor.

        Returns (signals, labels, info):
          signals: (n_per_anchor, 12, 1000) float32 PTBXL-order, z-scored
          labels:  (n_per_anchor, 6) float32 = y0_multi_hot replicated
          info:    {'final_delta_norms': list[float], ...}
        """
        rng = rng or np.random.default_rng()
        if z0.dim() == 2:
            z0 = z0.unsqueeze(0)  # (1, 4, 128)
        z0 = z0.to(self.device).float()

        # Batch n_per_anchor trials in a single PGD call — 10× faster than the
        # old per-trial loop. Each trial gets its own independent δ init; the
        # PGD loop runs on the batched tensors and per-sample L2 projection
        # keeps trials independent. Gradient normalization is per-sample
        # (see attack_from_latent), so mean-loss scaling does not bias any
        # trial's step size vs the unbatched version.
        z0_batch = z0.expand(n_per_anchor, -1, -1).contiguous()         # (n, 4, 128)
        y_batch = (
            torch.from_numpy(y0_multi_hot.astype(np.float32))
            .to(self.device)
            .unsqueeze(0)
            .expand(n_per_anchor, -1)
            .contiguous()
        )                                                                # (n, 6)
        delta_init = torch.randn_like(z0_batch) * self.delta_init_scale

        x_adv_batch, final_delta_batch = self.attack_from_latent(
            z0_batch, y_batch, delta_init=delta_init
        )                                                                # (n, 12, 1000), (n, 4, 128)

        signals_arr = x_adv_batch.detach().cpu().numpy().astype(np.float32)   # (n, 12, 1000)
        labels_arr = np.tile(
            y0_multi_hot.astype(np.float32)[None, :], (n_per_anchor, 1)
        )                                                                 # (n, 6)
        deltas_norm = final_delta_batch.detach().flatten(1).norm(dim=1).cpu().tolist()

        info = {
            "final_delta_norms": deltas_norm,
            "mean_delta_norm": float(np.mean(deltas_norm)),
            "max_delta_norm": float(np.max(deltas_norm)),
        }
        return signals_arr, labels_arr, info

    # ----- internal helpers -----

    def _attack_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return multilabel_attack_loss(
            logits,
            targets,
            mode=self.attack_loss_mode,
            pos_weight=pos_weight,
            pos_hide_weight=self.attack_pos_hide_weight,
            neg_add_weight=self.attack_neg_add_weight,
            negative_exclude_indices=self.attack_negative_exclude_indices,
            neg_topk=self.attack_neg_topk,
        )

    def _project_l2_ball(self, delta: torch.Tensor) -> torch.Tensor:
        """Per-sample L2 ball projection. delta shape (B, 4, 128) or (1, 4, 128)."""
        flat = delta.flatten(1)                        # (B, D)
        norm = flat.norm(dim=1).clamp(min=1e-12)       # (B,)
        factor = torch.clamp(self.epsilon / norm, max=1.0)  # (B,)
        return delta * factor.view(-1, 1, 1)

    def _decode_to_ptbxl_1000(self, z: torch.Tensor) -> torch.Tensor:
        """Replicate victim's latent preprocessing chain up to z-score (no crop).

        Output: (B, 12, L) PTBXL-order, optional clamp, z-scored.
        This matches SynthCenterDataset.__init__ expected signals format.
        """
        return self.victim.decode_latent_to_ecg_ct(z)


if __name__ == "__main__":
    # Minimal smoke test: random latent + random label, verify shapes + grad flow
    import time
    from util.ecgtwin_utils import ECGTwinWrapper

    print("[smoke] loading ECGTwin wrapper (encoder not needed — we pass pre-encoded z0)...")
    wrapper = ECGTwinWrapper(device="cuda:0", load_encoder=False, load_text_model=False)

    print("[smoke] loading Tier-M victim...")
    victim = EfficientNetVictimTierM(
        weight_path="/root/autodl-tmp/crosscenter_tierM/best_model.pt",
        device="cuda:0",
        ecgtwin_wrapper=wrapper,
    )

    gen = PGDAdvDiffGenerator(
        ecgtwin_wrapper=wrapper, victim=victim,
        epsilon=1.0, K_pgd=10, device="cuda:0",
    )

    # Fake anchor
    z0 = torch.randn(1, 4, 128, device="cuda:0") * 0.5
    y0 = np.zeros(6, dtype=np.float32); y0[4] = 1.0  # LBBB

    t0 = time.time()
    signals, labels, info = gen.generate_for_anchor(z0, y0, n_per_anchor=3)
    print(f"[smoke] 3 adv samples in {time.time()-t0:.2f}s")
    print(f"  signals: {signals.shape} dtype={signals.dtype} "
          f"finite={np.isfinite(signals).all()}")
    print(f"  labels:  {labels.shape} dtype={labels.dtype}")
    print(f"  delta norms (ε=1.0): {info['final_delta_norms']}")
    # Verify attack actually moves the victim prediction
    with torch.no_grad():
        x_adv_t = torch.from_numpy(signals).float().to("cuda:0")
        probs_adv = victim.forward_from_ecg(x_adv_t)        # (B, 6) sigmoid
        print(f"  victim sigmoid(LBBB) on adv: {probs_adv[:, 4].cpu().tolist()}")
