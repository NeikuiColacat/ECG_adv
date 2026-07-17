from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from util.augmentations.profile import (
    CANONICAL_OPERATOR_ORDER,
    load_augmentation_profile,
)


REPO = Path(__file__).resolve().parents[2]
OPERATORS = REPO / "configs" / "augmentation" / "operators.yaml"
SEED = REPO / "configs" / "random_seed.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "configs"
    operators = root / "augmentation" / "operators.yaml"
    seed = root / "random_seed.yaml"
    operators.parent.mkdir(parents=True)
    operators.write_bytes(OPERATORS.read_bytes())
    seed.write_bytes(SEED.read_bytes())
    return operators, seed


def test_profile_loader_records_validated_config_and_seed_identity() -> None:
    profile = load_augmentation_profile(
        OPERATORS,
        profile_name="pn2021c_paper_anchored_s5_v1",
        severity=5,
        expected_seed_config_path=SEED,
    )

    assert profile.schema_version == 1
    assert profile.profile_name == "pn2021c_paper_anchored_s5_v1"
    assert profile.severity == 5
    assert profile.canonical_order == CANONICAL_OPERATOR_ORDER
    assert profile.config_path == OPERATORS.resolve()
    assert profile.config_sha256 == _sha256(OPERATORS)
    assert profile.random_seed_config.path == SEED.resolve()
    assert profile.random_seed_config.sha256 == _sha256(SEED)
    assert tuple(profile.operator_parameters) == CANONICAL_OPERATOR_ORDER
    assert profile.parameters_for("baseline_shift")["amplitude_mode"] == (
        "signed_shared_uniform"
    )

    description = profile.describe()
    assert description["config_sha256"] == _sha256(OPERATORS)
    assert description["random_seed_sha256"] == _sha256(SEED)
    assert description["canonical_order"] == list(CANONICAL_OPERATOR_ORDER)


def test_profile_loader_resolves_seed_within_copied_config_bundle(
    tmp_path: Path,
) -> None:
    operators, seed = _write_bundle(tmp_path)

    profile = load_augmentation_profile(operators)

    assert profile.config_path == operators.resolve()
    assert profile.random_seed_config.path == seed.resolve()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda payload: payload["metadata"].__setitem__(
                "profile_name", "wrong_profile"
            ),
            "profile",
        ),
        (
            lambda payload: payload["metadata"].__setitem__(
                "public_severity", 4
            ),
            "severity",
        ),
        (
            lambda payload: payload["metadata"]["composite"].__setitem__(
                "canonical_order",
                list(reversed(CANONICAL_OPERATOR_ORDER)),
            ),
            "canonical order",
        ),
        (
            lambda payload: payload["metadata"].__setitem__(
                "random_seed_file", "missing_seed.yaml"
            ),
            "random seed",
        ),
    ],
)
def test_profile_loader_rejects_contract_drift(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    operators, _ = _write_bundle(tmp_path)
    payload = yaml.safe_load(operators.read_text(encoding="utf-8"))
    mutation(payload)
    operators.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_augmentation_profile(
            operators,
            profile_name="pn2021c_paper_anchored_s5_v1",
            severity=5,
        )


def test_profile_parameters_are_returned_as_an_independent_copy() -> None:
    profile = load_augmentation_profile(OPERATORS)

    first = profile.parameters_for("powerline_noise")
    first["max_amplitude"] = 999.0

    assert profile.parameters_for("powerline_noise")["max_amplitude"] == 0.5
