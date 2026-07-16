"""NumPy and device-native Torch ECG waveform augmentation operators.

All public operators accept one ECG shaped ``(time, 12)`` and return a new
contiguous ``float32`` object with the same shape. The unprefixed functions
use NumPy; the ``torch_`` functions preserve the input tensor's device.
"""

from .operators import (
    UPSTREAM_COMMIT,
    UPSTREAM_SOURCE_URL,
    baseline_shift,
    baseline_wander,
    emg_noise,
    powerline_noise,
    random_leads_masking,
)
from .torch_operators import (
    baseline_shift as torch_baseline_shift,
    baseline_wander as torch_baseline_wander,
    emg_noise as torch_emg_noise,
    powerline_noise as torch_powerline_noise,
    random_leads_masking as torch_random_leads_masking,
)

__all__ = [
    "UPSTREAM_COMMIT",
    "UPSTREAM_SOURCE_URL",
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
    "torch_powerline_noise",
    "torch_emg_noise",
    "torch_baseline_wander",
    "torch_baseline_shift",
    "torch_random_leads_masking",
]
