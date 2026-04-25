"""AugMix executed in ECGTwin's VAE latent space.

The stock AugMix (methods/augmix/augmix.py) composes W augmentation chains in
time domain and Dirichlet-mixes them there. Here we push the mixing into the
VAE latent:

  1. Encode the original ECG -> z0 (1, 4, 128).
  2. For each of W chains:
       a. Apply a random composition of d time-domain ops (severity-scaled).
       b. Encode the perturbed signal -> z_i (1, 4, 128).
  3. Dirichlet-mix: z_mix = sum_i ws_i * z_i   where ws ~ Dir(alpha).
  4. Beta-blend:   z_final = (1-m) * z0 + m * z_mix   where m ~ Beta(a,a).
  5. Decode z_final via the VAE decoder -> final ECG.

Mixing latents (step 3) exploits the VAE's continuous manifold: a convex
combination stays near the data manifold, whereas mixing raw signals can land
in physiologically implausible territory (e.g. two misaligned QRS peaks
super-imposed). This module exists to verify that claim empirically via
ecg_viz.sanity_check on the decoded output.

The time-domain ops live in `methods/augmix/ecg_ops.py` and are instantiated
through `methods/augmix/severity.build_op`. We reuse AugMix's random op
sampling + Dirichlet + Beta logic from `methods/augmix/augmix.py` verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from methods.augmix.augmix import DEFAULT_OPS, _apply_op
from methods.augmix.severity import AVAILABLE_OPS
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES


# canonical (PTBXL-style) <-> ECGTwin lead permutation; involutive
_LEAD_SWAP = ECGTWIN_TO_PTBXL_INDICES


def _canonical_to_ecgtwin_ct(sig_ct: np.ndarray) -> np.ndarray:
    return sig_ct[_LEAD_SWAP, :]


def _ecgtwin_to_canonical_ct(sig_ct: np.ndarray) -> np.ndarray:
    return sig_ct[_LEAD_SWAP, :]


def _per_sample_zscore(sig_ct: np.ndarray) -> np.ndarray:
    m = float(sig_ct.mean())
    s = float(sig_ct.std())
    return (sig_ct - m) if s < 1e-8 else (sig_ct - m) / s


def _encode_canonical(wrapper, sig_ct_100hz: np.ndarray, device) -> Tensor:
    """canonical (12, 1000) @100Hz -> VAE latent (1, 4, 128)."""
    sig_et = _canonical_to_ecgtwin_ct(sig_ct_100hz)
    t = torch.from_numpy(sig_et).float().unsqueeze(0)                  # (1, 12, 1000)
    t = F.interpolate(t, size=1024, mode="linear", align_corners=True)  # (1, 12, 1024)
    t = t.transpose(1, 2).contiguous()                                  # (1, 1024, 12)
    return wrapper.encode_ecg(t.to(device))


def _decode_to_canonical(wrapper, latent: Tensor) -> np.ndarray:
    """VAE latent (1, 4, 128) -> canonical (12, 1000) @100Hz, per-sample zscored."""
    ecg = wrapper.decode_latent(latent)                 # (1, 1024, 12)
    ecg_ct = ecg.transpose(1, 2).squeeze(0).detach().cpu().numpy()  # (12, 1024) ECGTwin order
    ecg_ct = _ecgtwin_to_canonical_ct(ecg_ct)
    t = torch.from_numpy(ecg_ct).float().unsqueeze(0)   # (1, 12, 1024)
    t = F.interpolate(t, size=1000, mode="linear", align_corners=True).squeeze(0).numpy()
    return _per_sample_zscore(t).astype(np.float32)


@dataclass
class LatentAugMixResult:
    """All intermediate artifacts for one sample, enough to drive plots + reports."""
    original_ct: np.ndarray                     # (12, 1000)
    vae_roundtrip_ct: np.ndarray                # (12, 1000) — decode(encode(orig)), no aug
    chain_signals_ct: List[np.ndarray]          # W entries of (12, 1000) — time-domain chain outputs (empty slot for injected chains)
    chain_op_sequences: List[List[str]]         # the op list chosen per chain (empty for injected chains)
    augmix_latent_ct: np.ndarray                # (12, 1000) — decode(z_final)
    dirichlet_weights: np.ndarray               # (W,)
    beta_m: float
    severity: int = 0
    width: int = 0
    depth: int = 0
    injected_chain_indices: tuple = ()          # indices of chains sourced from inject_latents


def latent_augmix_on_signal(
    wrapper,
    signal_ct_100hz: np.ndarray,
    *,
    severity: int = 5,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    ops: Optional[List[str]] = None,
    device: str = "cuda:0",
    rng_seed: Optional[int] = None,
    inject_latents: Optional[List[Tensor]] = None,
) -> LatentAugMixResult:
    """Run one AugMix chain in VAE latent space for a single canonical ECG.

    Args:
        wrapper: ECGTwinWrapper loaded with load_encoder=True.
        signal_ct_100hz: canonical (12, 1000) float32 @100Hz, per-sample zscored.
        severity: in [1,10]; passed to each time-domain op via build_op.
        width: number of AugMix chains.
        depth: depth per chain; -1 -> random in {1,2,3}.
        alpha: Dirichlet/Beta parameter.
        ops: subset of AVAILABLE_OPS; defaults to all five.
        device: torch device string.
        rng_seed: if set, make sampling deterministic for this call.
        inject_latents: optional list of pre-computed VAE latents, each shape
            (1, 4, 128) or (4, 128). The first N=len(inject_latents) chains
            (up to ``width``) use these latents verbatim instead of being
            generated via time-domain ops + encode. Remaining chains still
            go through the time-domain pipeline.
    """
    if ops is None:
        ops = list(DEFAULT_OPS)
    for name in ops:
        if name not in AVAILABLE_OPS:
            raise ValueError(f"unknown op: {name}")
    if width < 1:
        raise ValueError(f"width must be >=1, got {width}")

    rng = np.random.default_rng(rng_seed)
    dev = torch.device(device)

    z0 = _encode_canonical(wrapper, signal_ct_100hz, dev)            # (1, 4, 128)

    ws = rng.dirichlet([alpha] * width).astype(np.float32)           # (W,)
    m = float(rng.beta(alpha, alpha))

    n_inject = 0
    inject_prepared: List[Tensor] = []
    if inject_latents is not None:
        if len(inject_latents) > width:
            raise ValueError(
                f"inject_latents length ({len(inject_latents)}) exceeds width ({width})"
            )
        for z in inject_latents:
            t = z if torch.is_tensor(z) else torch.as_tensor(z)
            if t.dim() == 2:
                t = t.unsqueeze(0)                                    # (4, 128) -> (1, 4, 128)
            if t.shape != z0.shape:
                raise ValueError(
                    f"injected latent shape {tuple(t.shape)} != z0 shape {tuple(z0.shape)}"
                )
            inject_prepared.append(t.to(dev).to(z0.dtype))
        n_inject = len(inject_prepared)

    chain_signals: List[np.ndarray] = []
    chain_ops: List[List[str]] = []
    z_mix = torch.zeros_like(z0)
    injected_idx: List[int] = []
    for i in range(width):
        if i < n_inject:
            # Injected chain: use provided latent verbatim; placeholder entries for bookkeeping
            z_i = inject_prepared[i]
            chain_signals.append(np.zeros((12, 1000), dtype=np.float32))
            chain_ops.append([])
            injected_idx.append(i)
        else:
            d = depth if depth > 0 else int(rng.integers(1, 4))  # {1,2,3}
            op_seq: List[str] = []
            sig = torch.from_numpy(signal_ct_100hz).clone().float()      # (12, 1000)
            for _ in range(d):
                op_name = str(rng.choice(ops))
                op_seq.append(op_name)
                sig = _apply_op(sig, op_name, severity)
            chain_signal_np = sig.detach().cpu().numpy().astype(np.float32)
            chain_signals.append(chain_signal_np)
            chain_ops.append(op_seq)
            z_i = _encode_canonical(wrapper, chain_signal_np, dev)       # (1, 4, 128)

        z_mix = z_mix + float(ws[i]) * z_i

    z_final = (1.0 - m) * z0 + m * z_mix
    augmix_ct = _decode_to_canonical(wrapper, z_final)
    vae_roundtrip_ct = _decode_to_canonical(wrapper, z0)

    return LatentAugMixResult(
        original_ct=signal_ct_100hz.astype(np.float32),
        vae_roundtrip_ct=vae_roundtrip_ct,
        chain_signals_ct=chain_signals,
        chain_op_sequences=chain_ops,
        augmix_latent_ct=augmix_ct,
        dirichlet_weights=ws,
        beta_m=m,
        severity=int(severity),
        width=int(width),
        depth=int(depth),
        injected_chain_indices=tuple(injected_idx),
    )


def latent_augmix_batch(
    wrapper,
    signals_bct_100hz: Tensor,
    *,
    severity: int = 5,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    ops: Optional[List[str]] = None,
) -> Tensor:
    """Batched latent-space AugMix for training-time augmentation.

    Input: (B, 12, 1000) canonical PTBXL-order, zscored, on device.
    Output: (B, 12, 1000) same device/dtype, per-sample zscored.

    Pipeline per batch:
      1. encode originals  (single batched encode)
      2. for each of W chains: per-sample random op composition in numpy,
         move to device, single batched encode
      3. per-sample Dirichlet weights + Beta blend in latent space
      4. single batched decode -> (B, 12, 1000), lead-reorder + zscore

    VAE is assumed frozen; runs under torch.no_grad.
    """
    if ops is None:
        ops = list(DEFAULT_OPS)
    for name in ops:
        if name not in AVAILABLE_OPS:
            raise ValueError(f"unknown op: {name}")
    if width < 1:
        raise ValueError(f"width must be >=1, got {width}")

    B = signals_bct_100hz.shape[0]
    if B == 0:
        return signals_bct_100hz
    dev = signals_bct_100hz.device

    def _to_et_1024_ct(x_bct: Tensor) -> Tensor:
        """(B,12,L) canonical PTBXL -> (B,12,1024) ECGTwin order."""
        x_et = x_bct[:, _LEAD_SWAP, :]
        return F.interpolate(x_et, size=1024, mode="linear", align_corners=True)

    with torch.no_grad():
        # 1. batched encode originals
        x0_et = _to_et_1024_ct(signals_bct_100hz)
        z0 = wrapper.encode_ecg(x0_et)                                       # (B, 4, 128)

        # 2. W chains — time-domain ops in numpy, then batched encode
        sig_np = signals_bct_100hz.detach().cpu().numpy()                    # (B, 12, 1000)
        chain_zs: List[Tensor] = []
        for _w in range(width):
            chain_np = np.empty_like(sig_np)
            for b in range(B):
                d = depth if depth > 0 else int(np.random.randint(1, 4))
                sig_t = torch.from_numpy(sig_np[b].copy()).float()
                for _ in range(d):
                    op_name = str(np.random.choice(ops))
                    sig_t = _apply_op(sig_t, op_name, severity)
                chain_np[b] = sig_t.cpu().numpy()
            chain_torch = torch.from_numpy(chain_np).float().to(dev)
            chain_et = _to_et_1024_ct(chain_torch)
            z_w = wrapper.encode_ecg(chain_et)                               # (B, 4, 128)
            chain_zs.append(z_w)

        # 3. per-sample Dirichlet + Beta
        ws_np = np.random.dirichlet([alpha] * width, size=B).astype(np.float32)  # (B, W)
        ms_np = np.random.beta(alpha, alpha, size=B).astype(np.float32)          # (B,)
        ws_t = torch.from_numpy(ws_np).to(dev)
        ms_t = torch.from_numpy(ms_np).to(dev)

        # 4. mix
        z_mix = torch.zeros_like(z0)
        for i, z_w in enumerate(chain_zs):
            z_mix = z_mix + ws_t[:, i:i + 1, None] * z_w                     # (B, 4, 128)
        m_bcast = ms_t[:, None, None]                                        # (B, 1, 1)
        z_final = (1.0 - m_bcast) * z0 + m_bcast * z_mix

        # 5. batched decode -> (B, 1024, 12) ECGTwin order
        ecg_et_tc = wrapper.decode_latent(z_final)
        ecg_et_ct_1024 = ecg_et_tc.transpose(1, 2)                           # (B, 12, 1024)

        # 6. resample 1024 -> 1000, lead reorder back to PTBXL, zscore
        ecg_ct_1000 = F.interpolate(
            ecg_et_ct_1024, size=1000, mode="linear", align_corners=True
        )                                                                    # (B, 12, 1000) ECGTwin order
        ecg_ptbxl = ecg_ct_1000[:, _LEAD_SWAP, :]                            # involutive swap
        flat = ecg_ptbxl.reshape(B, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp(min=1e-8)
        ecg_norm = (ecg_ptbxl - mean.unsqueeze(-1)) / std.unsqueeze(-1)

    return ecg_norm.to(torch.float32)
