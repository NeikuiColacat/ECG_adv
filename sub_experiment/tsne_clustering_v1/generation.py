"""Drive ECGTwin to produce synthetic ECG from per-center reference samples.

For each center C we pick M real samples, encode them via ECGTwin VAE encoder
to get ref_latents, then call `generate_ecg(ref_latent, ref_label)` with no
target override — the reference supplies both the latent conditioning and the
diagnostic / patient-info conditioning. The output is post-processed back to
canonical lead order and unified 100Hz / 1000-sample / per-sample-zscore so
it is directly comparable to the real samples loaded by `data_loader`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from util.ecgtwin_utils import load_ecgtwin
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # [0,1,2,3,5,4,6..11]
from sub_experiment.tsne_clustering_v1.data_loader import RealSample


# The canonical lead order used by unified_preprocess_to_1000 (PTBXL-style:
# aVL at 4, aVF at 5) differs from ECGTwin's native order (aVF at 4, aVL at 5)
# only in positions 4<->5. The permutation in lead_utils is involutive so we
# use the same index array in both directions.
_PTBXL_TO_ECGTWIN = ECGTWIN_TO_PTBXL_INDICES  # also valid for inverse


def _canonical_to_ecgtwin(sig_ct: np.ndarray) -> np.ndarray:
    """(12, L) canonical -> (12, L) ECGTwin-order (swap aVL/aVF)."""
    return sig_ct[_PTBXL_TO_ECGTWIN, :]


def _ecgtwin_to_canonical(sig_ct: np.ndarray) -> np.ndarray:
    return sig_ct[_PTBXL_TO_ECGTWIN, :]  # involutive


def _per_sample_zscore_ct(sig_ct: np.ndarray) -> np.ndarray:
    m = float(sig_ct.mean())
    s = float(sig_ct.std())
    if s < 1e-8:
        return sig_ct - m
    return (sig_ct - m) / s


def _build_ref_label(sample: RealSample) -> dict:
    """Build the label dict ECGTwin expects.

    The text field drives nomic text embedding; we use a generic placeholder
    because the goal of this experiment is to isolate center-style transfer,
    not diagnostic content. HR defaults to 60 when un-estimable.
    """
    hr = sample.hr if (sample.hr is not None and 30 < sample.hr < 200) else 60.0
    return {
        "hr": float(hr),
        "age": int(sample.age),
        "sex": sample.sex if sample.sex in ("Male", "Female") else "Male",
        "text": "Normal sinus rhythm",
    }


class ECGTwinGenerator:
    """Wraps ECGTwinWrapper and handles resampling + lead-order bookkeeping."""

    def __init__(
        self,
        device: str = "cuda:0",
        load_encoder: bool = True,
        load_text_model: bool = True,
        num_inference_steps: int = 50,
    ):
        self.wrapper = load_ecgtwin(
            device=device,
            load_encoder=load_encoder,
            load_text_model=load_text_model,
        )
        self.device = torch.device(device)
        self.num_inference_steps = num_inference_steps

    def _encode_real(self, sig_ct_100hz: np.ndarray) -> torch.Tensor:
        """canonical (12, 1000) @ 100Hz -> ECGTwin VAE latent (1, 4, 128)."""
        sig_ecgtwin_ct = _canonical_to_ecgtwin(sig_ct_100hz)
        sig_t = torch.from_numpy(sig_ecgtwin_ct).float().unsqueeze(0)   # (1, 12, 1000)
        sig_t = F.interpolate(sig_t, size=1024, mode="linear", align_corners=True)  # (1, 12, 1024)
        sig_tc = sig_t.transpose(1, 2).contiguous()                     # (1, 1024, 12)
        return self.wrapper.encode_ecg(sig_tc.to(self.device))

    def generate_for_sample(self, sample: RealSample) -> np.ndarray:
        """Run ECGTwin once, return (12, 1000) canonical float32 signal."""
        ref_latent = self._encode_real(sample.signal)
        ref_label = _build_ref_label(sample)
        ecg, _ = self.wrapper.generate_ecg(
            ref_latent=ref_latent,
            ref_label=ref_label,
            batch_size=1,
            num_inference_steps=self.num_inference_steps,
        )
        # ecg: (1, 1024, 12) in ECGTwin lead order
        ecg_ct = ecg.transpose(1, 2).squeeze(0).detach().cpu().numpy()   # (12, 1024)
        ecg_ct = _ecgtwin_to_canonical(ecg_ct)
        ecg_ct = torch.from_numpy(ecg_ct).float().unsqueeze(0)           # (1, 12, 1024)
        ecg_ct = F.interpolate(ecg_ct, size=1000, mode="linear", align_corners=True).squeeze(0).numpy()
        ecg_ct = _per_sample_zscore_ct(ecg_ct)
        return ecg_ct.astype(np.float32)

    def generate_for_samples(
        self,
        samples: list[RealSample],
        progress: bool = True,
    ) -> list[np.ndarray]:
        outs = []
        for i, s in enumerate(samples):
            outs.append(self.generate_for_sample(s))
            if progress and ((i + 1) % 10 == 0 or i + 1 == len(samples)):
                print(f"  [gen] {i + 1}/{len(samples)}")
        return outs


def pack_generated(
    gen_signals: list[np.ndarray],
    ref_samples: list[RealSample],
    centers: list[str],
) -> dict:
    """Stack generated signals and attach metadata mirroring data_loader output."""
    assert len(gen_signals) == len(ref_samples)
    center_to_id = {c: i for i, c in enumerate(centers)}
    signals = np.stack(gen_signals, axis=0).astype(np.float32)
    target_center_ids = np.array(
        [center_to_id[s.center] for s in ref_samples], dtype=np.int64
    )
    return {
        "signals": signals,                          # (N, 12, 1000)
        "target_center_ids": target_center_ids,
        "center_names": list(centers),
        "ref_record_ids": [s.record_id for s in ref_samples],
    }
