from __future__ import annotations

import numpy as np
import pytest

from data_preprocess import PN2021_preprocess, PTBXL_preprocess


@pytest.mark.parametrize(
    "module",
    [PTBXL_preprocess, PN2021_preprocess],
)
def test_sparse_nonfinite_values_are_repaired_with_nearest_edges(module) -> None:
    signal = np.tile(np.arange(10, dtype=np.float64)[:, None], (1, 12))
    signal[0, 0] = np.nan
    signal[5, 0] = np.inf
    signal[-1, 0] = -np.inf

    repaired, details = module._repair_nonfinite_per_lead(
        signal,
        max_record_fraction=0.1,
        max_lead_fraction=0.5,
    )

    assert np.isfinite(repaired).all()
    assert repaired[0, 0] == repaired[1, 0]
    assert repaired[5, 0] == pytest.approx(5.0)
    assert repaired[-1, 0] == repaired[-2, 0]
    assert details["nonfinite_count"] == 3
    assert details["repaired_nonfinite_count"] == 3
    assert details["repair_method"] == "linear_per_lead_nearest_edge"


@pytest.mark.parametrize(
    "module",
    [PTBXL_preprocess, PN2021_preprocess],
)
def test_all_nonfinite_lead_is_rejected(module) -> None:
    signal = np.ones((100, 12), dtype=np.float64)
    signal[:, 4] = np.nan

    with pytest.raises(module.WaveformQualityError) as exc_info:
        module._repair_nonfinite_per_lead(
            signal,
            max_record_fraction=1.0,
            max_lead_fraction=1.0,
        )

    assert exc_info.value.reason == "all_nonfinite_lead"


@pytest.mark.parametrize(
    "module",
    [PTBXL_preprocess, PN2021_preprocess],
)
def test_excessive_nonfinite_fraction_is_rejected(module) -> None:
    signal = np.ones((100, 12), dtype=np.float64)
    signal[:20, 0] = np.nan

    with pytest.raises(module.WaveformQualityError) as exc_info:
        module._repair_nonfinite_per_lead(
            signal,
            max_record_fraction=1.0,
            max_lead_fraction=0.05,
        )

    assert exc_info.value.reason == "lead_nonfinite_fraction_exceeded"
