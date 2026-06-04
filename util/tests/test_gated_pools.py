"""Tests for prompt-token gated pool artifact contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ecg_adv_gen.data.gated_pools import (
    GatedPoolArtifactPaths,
    GatedPoolError,
    build_gated_class_trust,
    class_counts_from_labels,
    load_gated_pool_npzs,
    merge_gated_pool_artifacts,
    write_gated_class_trust_json,
    write_gated_pool_npzs,
    write_gated_ref_meta_json,
    write_selected_gated_pool_artifacts,
)
from ecg_adv_gen.labels.super5 import CLASS_NAMES_SUPER5


REPO = Path(__file__).resolve().parents[2]


def _labels(indices: list[int]) -> np.ndarray:
    labels = np.zeros((len(indices), len(CLASS_NAMES_SUPER5)), dtype=np.float32)
    for row, idx in enumerate(indices):
        labels[row, idx] = 1.0
    return labels


def test_gated_pool_artifact_paths_use_canonical_filenames(tmp_path: Path):
    paths = GatedPoolArtifactPaths.from_dir(tmp_path / "pool" / "gated")

    assert paths.samples.name == "gated_samples.npz"
    assert paths.latents.name == "gated_samples.latent.npz"
    assert paths.class_trust.name == "gated_samples.class_trust.json"
    assert paths.ref_meta.name == "gated_samples.ref_meta.json"
    assert paths.gate_report.name == "gate_report.json"
    assert paths.manifest_paths() == {
        "gated_samples_npz": paths.samples,
        "gated_latent_npz": paths.latents,
        "class_trust_json": paths.class_trust,
        "ref_meta_json": paths.ref_meta,
        "gate_report_json": paths.gate_report,
    }


def test_write_gated_pool_npzs_writes_matching_sample_and_latent_artifacts(tmp_path: Path):
    paths = GatedPoolArtifactPaths.from_dir(tmp_path)
    signals = np.ones((2, 12, 1000), dtype=np.float64)
    raw_signal_ct = np.ones((2, 12, 1024), dtype=np.float32) * 2
    latents = np.ones((2, 4, 128), dtype=np.float64) * 3
    labels = _labels([2, 4])
    source_indices = np.asarray([3, 8], dtype=np.int64)

    write_gated_pool_npzs(
        paths,
        signals=signals,
        raw_signal_ct=raw_signal_ct,
        latents=latents,
        labels=labels,
        center_name="ningbo",
        class_names=CLASS_NAMES_SUPER5,
        source_indices=source_indices,
    )

    loaded = load_gated_pool_npzs(paths)
    assert loaded["center_name"] == "ningbo"
    assert loaded["signals"].dtype == np.float32
    assert loaded["latents"].shape == (2, 4, 128)
    np.testing.assert_array_equal(loaded["source_indices"], source_indices)
    np.testing.assert_allclose(loaded["sample_latents"], loaded["latents"])
    np.testing.assert_allclose(loaded["sample_labels"], loaded["labels"])


def test_load_gated_pool_npzs_rejects_sample_latent_metadata_drift(tmp_path: Path):
    paths = GatedPoolArtifactPaths.from_dir(tmp_path)
    signals = np.ones((1, 12, 1000), dtype=np.float32)
    raw_signal_ct = np.ones((1, 12, 1024), dtype=np.float32)
    latents = np.ones((1, 4, 128), dtype=np.float32)
    labels = _labels([2])
    write_gated_pool_npzs(
        paths,
        signals=signals,
        raw_signal_ct=raw_signal_ct,
        latents=latents,
        labels=labels,
        center_name="ningbo",
    )
    np.savez_compressed(
        paths.latents,
        latents=latents,
        labels=labels,
        center_name="georgia",
        class_names=np.asarray(list(CLASS_NAMES_SUPER5[::-1])),
    )

    with np.testing.assert_raises(GatedPoolError):
        load_gated_pool_npzs(paths)


def test_build_gated_class_trust_defaults_hyp_cd_to_zero():
    labels = _labels([0, 0, 1, 2, 4, 4])
    counts = class_counts_from_labels(labels)

    result = build_gated_class_trust(counts, min_pass_per_class=2)

    assert result.class_trust == {
        "CD": 0.0,
        "HYP": 0.0,
        "MI": 0.0,
        "NORM": 0.0,
        "STTC": 1.0,
    }
    assert result.hardcoded_zero == ("HYP", "CD")

    allow_all = build_gated_class_trust(counts, min_pass_per_class=1, allow_hyp_cd_trust=True)
    assert allow_all.class_trust["HYP"] == 1.0
    assert allow_all.hardcoded_zero == ()


def test_write_gated_metadata_jsons_are_stable_and_ref_ids_are_deduplicated(tmp_path: Path):
    paths = GatedPoolArtifactPaths.from_dir(tmp_path)
    trust = build_gated_class_trust({"NORM": 3, "MI": 1, "STTC": 0}, min_pass_per_class=1)

    trust_path = write_gated_class_trust_json(
        paths,
        tag="ningbo",
        center="ningbo",
        source_samples=Path("samples.npz"),
        class_trust=trust.class_trust,
        counts_total={"NORM": 5, "MI": 2},
        counts_passed={"NORM": 3, "MI": 1},
        policy={"hardcoded_zero": list(trust.hardcoded_zero)},
    )
    ref_path = write_gated_ref_meta_json(
        paths,
        center="ningbo",
        ref_record_ids=["b", "a", "b"],
        k=500,
        selection_seed=42,
    )

    trust_blob = json.loads(trust_path.read_text(encoding="utf-8"))
    ref_blob = json.loads(ref_path.read_text(encoding="utf-8"))
    assert trust_blob["gated_latent_pool"].endswith("gated_samples.latent.npz")
    assert trust_blob["class_trust"]["MI"] == 1.0
    assert ref_blob["ref_record_ids"] == ["a", "b"]
    assert ref_blob["K"] == 500
    assert ref_blob["selection_seed"] == 42


def test_merge_gated_pool_artifacts_unions_ref_ids_and_source_trust(tmp_path: Path):
    src_a = GatedPoolArtifactPaths.from_dir(tmp_path / "src_a")
    src_b = GatedPoolArtifactPaths.from_dir(tmp_path / "src_b")
    labels_a = _labels([2, 3])
    labels_b = _labels([2, 4])

    write_gated_pool_npzs(
        src_a,
        signals=np.ones((2, 12, 1000), dtype=np.float32),
        raw_signal_ct=np.ones((2, 12, 1024), dtype=np.float32),
        latents=np.ones((2, 4, 128), dtype=np.float32),
        labels=labels_a,
        center_name="ningbo",
    )
    write_gated_class_trust_json(
        src_a,
        tag="a",
        center="ningbo",
        source_samples="a.npz",
        class_trust={"MI": 1.0, "NORM": 1.0, "STTC": 0.0, "CD": 0.0, "HYP": 0.0},
        counts_total={"MI": 1, "NORM": 1},
        counts_passed={"MI": 1, "NORM": 1},
        policy="test",
    )
    write_gated_ref_meta_json(src_a, center="ningbo", ref_record_ids=["b", "a"])

    write_gated_pool_npzs(
        src_b,
        signals=np.ones((2, 12, 1000), dtype=np.float32) * 2,
        raw_signal_ct=np.ones((2, 12, 1024), dtype=np.float32) * 2,
        latents=np.ones((2, 4, 128), dtype=np.float32) * 2,
        labels=labels_b,
        center_name="ningbo",
    )
    write_gated_class_trust_json(
        src_b,
        tag="b",
        center="ningbo",
        source_samples="b.npz",
        class_trust={"MI": 1.0, "NORM": 0.0, "STTC": 0.0, "CD": 0.0, "HYP": 0.0},
        counts_total={"MI": 1, "STTC": 1},
        counts_passed={"MI": 1, "STTC": 1},
        policy="test",
    )
    write_gated_ref_meta_json(src_b, center="ningbo", ref_record_ids=["c", "a"])

    result = merge_gated_pool_artifacts([src_a.out_dir, src_b.out_dir], tmp_path / "merged", tag="merged_v1")

    loaded = load_gated_pool_npzs(result.paths)
    trust_blob = json.loads(result.paths.class_trust.read_text(encoding="utf-8"))
    ref_blob = json.loads(result.paths.ref_meta.read_text(encoding="utf-8"))
    report_blob = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert result.n_samples == 4
    assert loaded["center_name"] == "ningbo"
    assert loaded["labels"].shape == (4, len(CLASS_NAMES_SUPER5))
    assert trust_blob["class_trust"]["MI"] == 1.0
    assert trust_blob["class_trust"]["STTC"] == 0.0
    assert trust_blob["class_trust"]["HYP"] == 0.0
    assert ref_blob["ref_record_ids"] == ["a", "b", "c"]
    assert report_blob["tag"] == "merged_v1"
    assert report_blob["n_samples"] == 4


def test_merge_gated_pool_artifacts_rejects_mixed_centers(tmp_path: Path):
    src_a = GatedPoolArtifactPaths.from_dir(tmp_path / "src_a")
    src_b = GatedPoolArtifactPaths.from_dir(tmp_path / "src_b")
    for paths, center in [(src_a, "ningbo"), (src_b, "georgia")]:
        write_gated_pool_npzs(
            paths,
            signals=np.ones((1, 12, 1000), dtype=np.float32),
            raw_signal_ct=np.ones((1, 12, 1024), dtype=np.float32),
            latents=np.ones((1, 4, 128), dtype=np.float32),
            labels=_labels([2]),
            center_name=center,
        )

    with np.testing.assert_raises(GatedPoolError):
        merge_gated_pool_artifacts([src_a.out_dir, src_b.out_dir], tmp_path / "merged")


def test_merge_gated_pool_artifacts_rejects_class_name_drift(tmp_path: Path):
    src_a = GatedPoolArtifactPaths.from_dir(tmp_path / "src_a")
    src_b = GatedPoolArtifactPaths.from_dir(tmp_path / "src_b")
    for paths, class_names in [(src_a, CLASS_NAMES_SUPER5), (src_b, CLASS_NAMES_SUPER5[::-1])]:
        write_gated_pool_npzs(
            paths,
            signals=np.ones((1, 12, 1000), dtype=np.float32),
            raw_signal_ct=np.ones((1, 12, 1024), dtype=np.float32),
            latents=np.ones((1, 4, 128), dtype=np.float32),
            labels=_labels([2]),
            center_name="ningbo",
            class_names=class_names,
        )

    with np.testing.assert_raises(GatedPoolError):
        merge_gated_pool_artifacts([src_a.out_dir, src_b.out_dir], tmp_path / "merged")


def test_write_selected_gated_pool_artifacts_writes_selector_style_report(tmp_path: Path):
    labels = _labels([2, 4, 4])

    result = write_selected_gated_pool_artifacts(
        tmp_path / "selected",
        tag="quality_v1",
        center="ningbo",
        signals=np.ones((3, 12, 1000), dtype=np.float32),
        raw_signal_ct=np.ones((3, 12, 1024), dtype=np.float32),
        latents=np.ones((3, 4, 128), dtype=np.float32),
        labels=labels,
        ref_record_ids=["r2", "r1", "r2"],
        report_name="quality_report.json",
        report_payload={
            "gated_dirs": ["a", "b"],
            "quota": 2,
            "selected": [{"record_id": "x"}],
        },
        trust_policy="quality-aware top quota; HYP/CD hardcoded 0",
        ref_policy="Union of source K-ref exclusions.",
    )

    loaded = load_gated_pool_npzs(result.paths)
    trust_blob = json.loads(result.paths.class_trust.read_text(encoding="utf-8"))
    ref_blob = json.loads(result.paths.ref_meta.read_text(encoding="utf-8"))
    report_blob = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert result.n_samples == 3
    assert loaded["labels"].shape == (3, len(CLASS_NAMES_SUPER5))
    assert trust_blob["counts"] == {"CD": 0, "HYP": 0, "MI": 1, "NORM": 0, "STTC": 2}
    assert trust_blob["class_trust"] == {"CD": 0.0, "HYP": 0.0, "MI": 1.0, "NORM": 0.0, "STTC": 1.0}
    assert ref_blob["ref_record_ids"] == ["r1", "r2"]
    assert report_blob["tag"] == "quality_v1"
    assert report_blob["quota"] == 2
    assert report_blob["n_samples"] == 3
    assert report_blob["samples"].endswith("gated_samples.npz")
    assert report_blob["class_trust"].endswith("gated_samples.class_trust.json")


def test_quality_selector_rejects_mixed_center_gated_dirs(tmp_path: Path):
    gated_dirs = []
    for center in ["ningbo", "georgia"]:
        paths = GatedPoolArtifactPaths.from_dir(tmp_path / center / "gated")
        write_gated_pool_npzs(
            paths,
            signals=np.ones((1, 12, 1000), dtype=np.float32),
            raw_signal_ct=np.ones((1, 12, 1024), dtype=np.float32),
            latents=np.ones((1, 4, 128), dtype=np.float32),
            labels=_labels([2]),
            center_name=center,
            source_indices=[0],
        )
        paths.gate_report.write_text(
            json.dumps(
                {
                    "per_sample": [
                        {
                            "index": 0,
                            "class": "MI",
                            "keep": True,
                            "p_target": 0.9,
                            "top1": "MI",
                            "digital_metrics": {},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        write_gated_ref_meta_json(paths, center=center, ref_record_ids=[f"{center}_ref"])
        gated_dirs.append(paths.out_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "ecgtwin_gen" / "select_quality_prompt_token_pool.py"),
            "--gated_dirs",
            *(str(path) for path in gated_dirs),
            "--out_dir",
            str(tmp_path / "selected"),
            "--quota",
            "1",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "input centers differ" in result.stderr
