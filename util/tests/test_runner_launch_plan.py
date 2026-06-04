from __future__ import annotations

import json
from pathlib import Path

from ecg_adv_gen.runner.launch_plan import render_launch_command, write_launch_plan_files


def test_render_launch_command_quotes_env_and_argv():
    rendered = render_launch_command(
        {
            "env": {"TMPDIR": "/tmp/with spaces", "XDG_CACHE_HOME": "/tmp/cache"},
            "argv": ["python", "script.py", "--name", "value with spaces"],
        }
    )

    assert rendered.startswith("TMPDIR='/tmp/with spaces' XDG_CACHE_HOME=/tmp/cache python")
    assert "--name 'value with spaces'" in rendered


def test_write_launch_plan_files_materializes_agent_handoff_artifacts(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "ptbxl").mkdir(parents=True)
    (data_root / "pn2021" / "training" / "ningbo").mkdir(parents=True)
    (data_root / "cache" / "pn2021").mkdir(parents=True)
    (data_root / "cache" / "pn2021_mmap").mkdir(parents=True)
    refs = tmp_path / "refs"
    refs.mkdir()
    ref_meta = refs / "ningbo_ref_meta.json"
    ref_meta.write_text(
        json.dumps(
            {
                "K": 2,
                "selection_seed": 7,
                "ref_record_ids": ["ningbo/a", "ningbo/b"],
                "policy": "kshot_internal_validation",
            }
        ),
        encoding="utf-8",
    )
    signals_npz = refs / "signals.npz"
    signals_npz.write_bytes(b"npz-placeholder")

    config = {
        "experiment": {
            "name": "pytest_launch_plan",
            "description": "unit launch plan",
        },
        "data": {
            "source_dataset": "ptbxl",
            "target_dataset": "pn2021",
            "roots": {
                "ptbxl": str(data_root / "ptbxl"),
                "pn2021": str(data_root / "pn2021"),
            },
            "cache": {
                "pn2021_cache_dir": str(data_root / "cache" / "pn2021"),
                "pn2021_mmap_cache_dir": str(data_root / "cache" / "pn2021_mmap"),
            },
            "pn2021_centers": ["ningbo"],
            "exclude_centers": [],
        },
        "preprocess": {
            "classifier_fs": 100,
            "classifier_len": 1000,
            "crop_len": 1000,
            "mode": "resample_crop",
            "norm_mode": "zscore_per_record",
            "lead_order": "ptbxl",
            "ecgtwin_decode": {
                "input_len": 1024,
                "output_len": 1000,
                "reorder_indices": [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11],
            },
        },
        "paper_protocol": {
            "mapping_version": "v7_super5_sjr_rgq_review_20260528",
            "mapping_hash": "555ec85d5b51",
            "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            "centers": {"target_4": ["ningbo"], "eval_7": ["ningbo"]},
            "kshot": {"k": 2, "seed": 7, "subset_seed": 7, "exclude_refs_from_eval": True},
            "selection": {
                "policy": "k500_internal_val_plus_source_floor",
                "allowed_data": [
                    "target_k500_train_split",
                    "target_k500_internal_val",
                    "ptbxl_source_floor",
                ],
                "forbid_heldout_target_labels": True,
                "forbid_full_target_distribution_tuning": True,
            },
        },
    }
    commands = [
        {
            "name": "legacy_child",
            "env": {"TMPDIR": tmp_path / "tmp space"},
            "argv": ["python", "legacy.py", "--out_dir", tmp_path / "run space"],
        }
    ]
    postprocess_commands = [
        {
            "name": "post",
            "env": {},
            "argv": ["python", "post.py", "--input", tmp_path / "run space"],
        }
    ]
    manifest = {
        "status": "dry_run",
        "run_id": "pytest_plan",
        "config_hash_sha256": "abc123",
        "experiment": config["experiment"],
        "paper_protocol": config["paper_protocol"],
        "local_paths": {"write_boundary": str(tmp_path)},
        "artifact_trace": {
            "metrics": {
                "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                "mapping_hash": "555ec85d5b51",
                "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            },
            "selection_policy": config["paper_protocol"]["selection"],
            "protocol_audit": {
                "command_audit": {"passed": True},
                "postprocess_audit": {"passed": True},
            },
            "inputs": {
                "k500_refs": [
                    {
                        "center": "ningbo",
                        "anchor_base": str(refs),
                        "ref_meta_json": {"path": str(ref_meta)},
                        "signals_npz": {"path": str(signals_npz)},
                    }
                ]
            },
            "expected_outputs": {
                "launch_artifacts_declared": [
                    "run_config.resolved.yaml",
                    "run_config.resolved.json",
                    "run_manifest.json",
                    "command.sh",
                    "data_manifest.json",
                    "k500_ref_ids.json",
                    "selection.json",
                    "run_card.json",
                    "run_file_index.json",
                    "summary.md",
                ]
            },
        },
    }

    written = write_launch_plan_files(
        tmp_path / "run",
        config,
        manifest,
        commands,
        postprocess_commands,
    )

    run_dir = tmp_path / "run"
    for name in [
        "run_config.resolved.yaml",
        "run_config.resolved.json",
        "run_manifest.json",
        "command.sh",
        "data_manifest.json",
        "k500_ref_ids.json",
        "selection.json",
        "run_card.json",
        "run_file_index.json",
        "summary.md",
    ]:
        assert (run_dir / name).exists()
    assert "legacy_child" not in (run_dir / "command.sh").read_text(encoding="utf-8")
    assert "--out_dir" in (run_dir / "command.sh").read_text(encoding="utf-8")
    assert "# Managed postprocess commands" in (run_dir / "command.sh").read_text(encoding="utf-8")
    ref_artifact = json.loads((run_dir / "k500_ref_ids.json").read_text(encoding="utf-8"))
    assert ref_artifact["centers"]["ningbo"]["ref_record_ids_ordered"] == ["ningbo/a", "ningbo/b"]
    assert written["run_record"]["outcome"] == "dry_run"
