from ecg_adv_gen.evaluation.pn2021c_protocol import (
    LOCKED_ECGFOUNDER_CORRUPTION_INPUT,
    OFFICIAL_S5_COMPOSITE_OPS,
    ecgfounder_pn2021c_corruption_order,
    locked_protocol_metadata,
    official_s5_depth23_composites,
)
import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_official_s5_depth23_composites_are_stable_pairs_and_triples():
    names = official_s5_depth23_composites()

    pairs = [name for name in names if len(name.split("+")) == 2]
    triples = [name for name in names if len(name.split("+")) == 3]

    assert len(names) == 20
    assert len(pairs) == 10
    assert len(triples) == 10
    assert len(set(names)) == 20
    for name in names:
        ops = name.split("+")
        assert len(ops) == len(set(ops))
        assert set(ops).issubset(set(OFFICIAL_S5_COMPOSITE_OPS))
    assert names[0] == "baseline_shift+baseline_wander"
    assert names[-1] == "emg_noise+powerline_noise+random_leads_masking"


def test_ecgfounder_pn2021c_corruption_order_is_package_owned():
    locked = ecgfounder_pn2021c_corruption_order(LOCKED_ECGFOUNDER_CORRUPTION_INPUT)
    assert locked == locked_protocol_metadata("ecgfounder")["waveform_order"]

    native = ecgfounder_pn2021c_corruption_order("native_raw_first")
    assert native == [
        "wfdb_read_native_fs_native_length",
        "lead_reorder_nan_guard_no_resample_no_zscore",
        "corruption_with_native_sample_rate",
        "resample_pad_or_truncate_to_100hz_1000",
        "per_sample_global_zscore",
        "center_crop",
        "ecgfounder_resample_1000_to_5000_no_second_zscore",
        "model",
    ]
    assert len(native) == len(set(native))

    assert ecgfounder_pn2021c_corruption_order("raw_first") == [
        "wfdb_read",
        "lead_reorder_nan_guard_resample_pad_no_zscore",
        "corruption",
        "per_sample_global_zscore",
        "center_crop",
        "ecgfounder_resample_1000_to_5000_no_second_zscore",
        "model",
    ]
    assert ecgfounder_pn2021c_corruption_order("preprocessed_cache") == [
        "load_100hz1000_per_sample_global_zscore_cache",
        "center_crop_or_full_signal",
        "corruption",
        "ecgfounder_resample_1000_to_5000_and_global_zscore",
        "model",
    ]


def test_ecgfounder_pn2021c_evaluator_uses_package_corruption_order():
    tree = ast.parse((REPO / "ecg_adv_gen" / "runner" / "ecgfounder_pn2021c_eval.py").read_text())

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ecg_adv_gen.evaluation.pn2021c_protocol"
        and any(alias.name == "ecgfounder_pn2021c_corruption_order" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "ecgfounder_pn2021c_corruption_order"
        for node in ast.walk(tree)
    )


def test_pn2021c_eval_scripts_use_package_cache_defaults():
    from ecg_adv_gen.evaluation.pn2021c import (
        PN2021C_DEFAULT_CENTERS,
        PN2021C_DEFAULT_CORRUPTIONS,
    )

    assert PN2021C_DEFAULT_CENTERS == (
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    )
    assert PN2021C_DEFAULT_CORRUPTIONS == (
        "powerline_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    )

    for relative in (
        "ecg_adv_gen/runner/pn2021c_eval.py",
        "ecg_adv_gen/runner/ecgfounder_pn2021c_eval.py",
    ):
        tree = ast.parse((REPO / relative).read_text())
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert "scripts.triple_labels.build_pn2021_corruptions" not in modules
        assert "ecg_adv_gen.evaluation.pn2021c" in modules


def test_effnet_pn2021c_evaluator_uses_package_super5_scheme():
    tree = ast.parse((REPO / "ecg_adv_gen" / "runner" / "pn2021c_eval.py").read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "scripts.triple_labels.label_schemes" not in modules
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ecg_adv_gen.labels"
        and {alias.name for alias in node.names}
        >= {"get_super5_scheme", "get_super5_pn2021_mapping_metadata"}
        for node in ast.walk(tree)
    )
