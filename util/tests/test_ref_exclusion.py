from pathlib import Path

import pytest

from ecg_adv_gen.data import apply_ref_exclusion
from ecg_adv_gen.evaluation.ref_exclusion import (
    append_target_ref_exclusion_args,
    target_ref_meta_paths,
    target_ref_meta_paths_from_anchor_base,
)


def test_target_ref_meta_paths_cover_all_paper_target_centers():
    paths = target_ref_meta_paths("/refs", k=500, seed=20260531)

    assert [path.name for path in paths] == [
        "ningbo_real_k500_seed20260531.ref_meta.json",
        "chapman_shaoxing_real_k500_seed20260531.ref_meta.json",
        "cpsc_2018_real_k500_seed20260531.ref_meta.json",
        "georgia_real_k500_seed20260531.ref_meta.json",
    ]
    assert paths[0] == Path("/refs/ningbo/k500_seed20260531/ningbo_real_k500_seed20260531.ref_meta.json")


def test_target_ref_meta_paths_from_anchor_base_parses_standard_kshot_layout():
    paths = target_ref_meta_paths_from_anchor_base(
        "/refs/georgia/k500_seed20260531/georgia_real_k500_seed20260531"
    )

    assert len(paths) == 4
    assert paths[-1] == Path("/refs/georgia/k500_seed20260531/georgia_real_k500_seed20260531.ref_meta.json")


def test_target_ref_meta_paths_from_anchor_base_rejects_nonstandard_layout():
    with pytest.raises(ValueError, match="anchor_base parent"):
        target_ref_meta_paths_from_anchor_base("/refs/georgia/georgia_real_k500_seed20260531")


def test_append_target_ref_exclusion_args_uses_single_nargs_block():
    argv = ["python", "eval.py"]

    append_target_ref_exclusion_args(argv, "/refs", k=500, seed=20260531)

    assert argv[2] == "--exclude_ref_ids"
    assert len(argv[3:]) == 4


def test_apply_ref_exclusion_preserves_complete_center_order():
    records = ["a", "b", "c"]

    assert apply_ref_exclusion(records, excluded_record_ids={"b"}) == ["a", "c"]
