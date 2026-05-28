"""Paper-protocol data and preprocessing contracts.

These constants describe the current PTB-XL -> PN2021 Super5 evaluation
contract used by the YAML-managed runs. They are not full data loaders; legacy
dataset reading and waveform conversion remain in existing scripts until the
data pipeline is extracted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SOURCE_DATASET = "ptbxl"
TARGET_DATASET = "pn2021"

PN2021_TARGET_CENTERS_4 = (
    "ningbo",
    "chapman_shaoxing",
    "cpsc_2018",
    "georgia",
)
PN2021_EVAL_CENTERS_7 = (
    "chapman_shaoxing",
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "ningbo",
    "ptb",
    "st_petersburg_incart",
)
PN2021_LEAK_EXCLUDED_CENTERS = ("ptb-xl", "ptbxl")

CLASSIFIER_FS = 100
CLASSIFIER_LEN = 1000
CROP_LEN = 1000
PREPROCESS_MODE = "minimal_resample"
NORM_MODE = "per_sample_global"
PTBXL_LEAD_ORDER = "ptbxl"

ECGTWIN_DECODE_INPUT_LEN = 1024
ECGTWIN_DECODE_OUTPUT_LEN = 1000
ECGTWIN_TO_PTBXL_INDICES = (0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11)


class DataContractError(ValueError):
    """Raised when a config drifts from the current data/preprocess contract."""


@dataclass(frozen=True)
class DataPreprocessContract:
    source_dataset: str
    target_dataset: str
    target_centers: tuple[str, ...]
    eval_centers: tuple[str, ...]
    leak_excluded_centers: tuple[str, ...]
    classifier_fs: int
    classifier_len: int
    crop_len: int
    preprocess_mode: str
    norm_mode: str
    lead_order: str
    ecgtwin_decode_input_len: int
    ecgtwin_decode_output_len: int
    ecgtwin_to_ptbxl_indices: tuple[int, ...]


def get_data_preprocess_contract() -> DataPreprocessContract:
    return DataPreprocessContract(
        source_dataset=SOURCE_DATASET,
        target_dataset=TARGET_DATASET,
        target_centers=PN2021_TARGET_CENTERS_4,
        eval_centers=PN2021_EVAL_CENTERS_7,
        leak_excluded_centers=PN2021_LEAK_EXCLUDED_CENTERS,
        classifier_fs=CLASSIFIER_FS,
        classifier_len=CLASSIFIER_LEN,
        crop_len=CROP_LEN,
        preprocess_mode=PREPROCESS_MODE,
        norm_mode=NORM_MODE,
        lead_order=PTBXL_LEAD_ORDER,
        ecgtwin_decode_input_len=ECGTWIN_DECODE_INPUT_LEN,
        ecgtwin_decode_output_len=ECGTWIN_DECODE_OUTPUT_LEN,
        ecgtwin_to_ptbxl_indices=ECGTWIN_TO_PTBXL_INDICES,
    )


def _require(config: dict[str, Any], key: str) -> Any:
    cur: Any = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise DataContractError(f"Missing required data/preprocess key: {key}")
        cur = cur[part]
    if cur in (None, "", []):
        raise DataContractError(f"Missing required data/preprocess key: {key}")
    return cur


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise DataContractError(f"{name} mismatch: config={actual!r} expected={expected!r}")


def validate_data_preprocess_config(config: dict[str, Any]) -> None:
    """Validate YAML data/preprocess fields against the current paper protocol."""
    expected = get_data_preprocess_contract()

    _assert_equal("data.source_dataset", _require(config, "data.source_dataset"), expected.source_dataset)
    _assert_equal("data.target_dataset", _require(config, "data.target_dataset"), expected.target_dataset)
    _assert_equal(
        "paper_protocol.centers.target_4",
        _as_tuple(_require(config, "paper_protocol.centers.target_4")),
        expected.target_centers,
    )
    _assert_equal(
        "paper_protocol.centers.eval_7",
        _as_tuple(_require(config, "paper_protocol.centers.eval_7")),
        expected.eval_centers,
    )
    _assert_equal(
        "data.pn2021_centers",
        _as_tuple(_require(config, "data.pn2021_centers")),
        expected.target_centers,
    )
    excluded = set(str(v) for v in _as_tuple(_require(config, "data.exclude_centers")))
    missing_excluded = [c for c in expected.leak_excluded_centers if c not in excluded]
    if missing_excluded:
        raise DataContractError(
            f"data.exclude_centers must include leak centers: {missing_excluded}"
        )

    _assert_equal("preprocess.classifier_fs", int(_require(config, "preprocess.classifier_fs")), expected.classifier_fs)
    _assert_equal(
        "preprocess.classifier_len",
        int(_require(config, "preprocess.classifier_len")),
        expected.classifier_len,
    )
    _assert_equal("preprocess.crop_len", int(_require(config, "preprocess.crop_len")), expected.crop_len)
    _assert_equal("preprocess.mode", _require(config, "preprocess.mode"), expected.preprocess_mode)
    _assert_equal("preprocess.norm_mode", _require(config, "preprocess.norm_mode"), expected.norm_mode)
    _assert_equal("preprocess.lead_order", _require(config, "preprocess.lead_order"), expected.lead_order)
    _assert_equal(
        "preprocess.ecgtwin_decode.input_len",
        int(_require(config, "preprocess.ecgtwin_decode.input_len")),
        expected.ecgtwin_decode_input_len,
    )
    _assert_equal(
        "preprocess.ecgtwin_decode.output_len",
        int(_require(config, "preprocess.ecgtwin_decode.output_len")),
        expected.ecgtwin_decode_output_len,
    )
    _assert_equal(
        "preprocess.ecgtwin_decode.reorder_indices",
        tuple(int(v) for v in _as_tuple(_require(config, "preprocess.ecgtwin_decode.reorder_indices"))),
        expected.ecgtwin_to_ptbxl_indices,
    )
