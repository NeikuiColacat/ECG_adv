"""CPU-only contracts for the prospective PULSE AugMix SFT experiment."""
from __future__ import annotations

import json
from pathlib import Path

from util.config_bundle import load_yaml_mapping, resolve_config_reference
from util.pn2021_artifact_contract import sha256_file

CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
PULSE_REVISION = "3b497b425ade2717f76ba0b14202823b83668011"


def validate_config(config: dict) -> None:
    expected = {"schema_version", "mode", "references", "center", "width", "model",
                "training", "paths", "resume_from"}
    visual = config.get("schema_version") == 3
    if config.get("schema_version") in (2, 3):
        expected.add("mixing_domain")
        if config.get("mixing_domain") != "rendered_rgb":
            raise ValueError("schema 2 is the matched rendered-RGB mixing ablation")
    if set(config) != expected or config["schema_version"] not in (1, 2, 3):
        raise ValueError("invalid PULSE training config schema")
    if config["mode"] not in {"prepare", "smoke", "train"}:
        raise ValueError("unsupported PULSE training mode")
    if config["center"] not in CENTERS or type(config["width"]) is not int or config["width"] not in ((0, 1, 3) if visual else (1, 2)):
        raise ValueError("invalid center or AugMix width")
    if set(config["references"]) != {"split_config", "data_load_config", "seed_config", "operators_config"}:
        raise ValueError("incomplete data/config closure")
    model = config["model"]
    if set(model) != {"directory", "revision", "precision", "lora_rank", "lora_alpha", "lora_dropout"}:
        raise ValueError("invalid model configuration")
    if model["revision"] != PULSE_REVISION or model["precision"] != "float16":
        raise ValueError("only the locked unquantized PULSE checkpoint is admitted")
    if (model["lora_rank"], model["lora_alpha"], model["lora_dropout"]) != ((8, 16, 0.0) if visual else (16, 32, 0.05)):
        raise ValueError("LoRA settings must be frozen across the pair")
    training = config["training"]
    if set(training) != {"seed", "optimizer_steps", "effective_batch", "learning_rate",
                         "projector_learning_rate", "warmup_steps", "max_length",
                         "save_every", "smoke_records", "cpu_threads", "loader_workers"}:
        raise ValueError("invalid training configuration")
    for key in ("seed", "optimizer_steps", "effective_batch", "max_length", "save_every",
                "smoke_records", "cpu_threads", "loader_workers"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"invalid training {key}")
    if training["max_length"] != 4096 or training["effective_batch"] > 32:
        raise ValueError("unsafe batch or changed PULSE context")
    if not 0 <= training["warmup_steps"] < training["optimizer_steps"]:
        raise ValueError("invalid warmup")
    if not 0 < training["learning_rate"] <= 0.0002 or not 0 < training["projector_learning_rate"] <= 0.00002:
        raise ValueError("unsafe learning rate")
    if training["cpu_threads"] > 4 or training["loader_workers"] > 4 or training["smoke_records"] > 32:
        raise ValueError("unsafe preprocessing concurrency")
    if config["mode"] == "smoke" and training["optimizer_steps"] > 4:
        raise ValueError("smoke budget must be bounded")
    paths = config["paths"]
    if set(paths) != {"raw_root", "toolkit_dir", "temporary_root"}:
        raise ValueError("invalid training paths")
    for raw in [model["directory"], *paths.values(), *([config["resume_from"]] if config["resume_from"] else [])]:
        Path(raw).resolve().relative_to(Path("/home/linbinhao"))


def load_config(path: Path, root: Path) -> tuple[dict, dict[str, Path]]:
    config = load_yaml_mapping(path, description="PULSE training config")
    validate_config(config)
    refs = {key: resolve_config_reference(value, owner_config_path=path, config_root=root,
             description=key, must_exist=True) for key, value in config["references"].items()}
    return config, refs


def label_text(label: list[int]) -> str:
    if len(label) != 5 or any(type(x) is not int or x not in (0, 1) for x in label) or not any(label):
        raise ValueError("expected nonzero binary Super5 label")
    return ";".join(name for name, present in zip(CLASS_ORDER, label, strict=True) if present)


def validate_result(payload: dict, path: Path) -> None:
    if payload.get("artifact_type") != "pulse_train_result" or payload.get("schema_version") != 1:
        raise ValueError("invalid PULSE result identity")
    if payload.get("status") != "complete" or payload.get("mode") not in {"prepare", "smoke", "train"}:
        raise ValueError("incomplete PULSE training result")
    protocol = payload["protocol"]
    visual = protocol.get("training_scope") == "clip_last4_qv_lora8_projector_frozen_llm"
    if protocol["center"] not in CENTERS or protocol["width"] not in ((0, 1, 3) if visual else (1, 2)):
        raise ValueError("invalid PULSE arm identity")
    if protocol["mapping_hash"] != "555ec85d5b51" or tuple(protocol["class_order"]) != CLASS_ORDER:
        raise ValueError("invalid PULSE label identity")
    if protocol["adaptation_records"] != 500 or protocol["partition"] != "k500":
        raise ValueError("invalid PULSE K500 identity")
    if protocol["simclr"] or protocol["vae_lhat"] or protocol["jsd"] != (visual and protocol["width"] > 0):
        raise ValueError("extra objective is outside the first pair")
    if "mixing_domain" in protocol and protocol["mixing_domain"] != "rendered_rgb":
        raise ValueError("invalid pixel mixing protocol")
    if visual and payload["mode"] != "prepare":
        audit = json.loads((path.parent / "visual_training_admission.json").read_text())
        if ("visual_training_admission.json" not in payload["files"]
                or audit.get("frozen_parameter_versions_unchanged") is not True
                or audit.get("trainable_vision_outputs_cached") != 0
                or audit.get("vision_lora_updated") is not True
                or audit.get("projector_updated") is not True):
            raise ValueError("visual-only training admission failed")
        if (protocol["dirichlet_alpha"] != 1.0 or protocol["clean_mix"] != "beta_1_1"
                or protocol["jsd_weight"] != (12.0 if protocol["width"] else 0.0)):
            raise ValueError("visual JSD recipe changed")
    if not visual and protocol.get("mixing_domain") == "rendered_rgb" and payload["mode"] != "prepare":
        if "pixel_mixing_admission.json" not in payload["files"]:
            raise ValueError("pixel admission must be fingerprinted")
        audit = json.loads((path.parent / "pixel_mixing_admission.json").read_text())
        if (audit.get("single_chain_pixels_exact") is not True
                or audit.get("single_chain_processor_exact") is not True
                or audit.get("global_torch_rng_unchanged") is not True):
            raise ValueError("pixel mixing equivalence admission missing")
    if payload["mode"] != "prepare" and payload["optimizer_steps"] != protocol["training"]["optimizer_steps"]:
        raise ValueError("incomplete optimizer budget")
    for relative, digest in payload["files"].items():
        member = (path.parent / relative).resolve()
        member.relative_to(path.parent.resolve())
        if sha256_file(member) != digest:
            raise ValueError(f"PULSE result file changed: {relative}")
    rows = json.loads((path.parent / "k500_records.json").read_text())
    if len(rows) != 500 or len({r["hash_id"] for r in rows}) != 500:
        raise ValueError("PULSE K500 manifest is incomplete or duplicated")
    if any(r["logical_center"] != protocol["center"] for r in rows):
        raise ValueError("PULSE adaptation center mismatch")
    if payload["mode"] != "prepare" and "checkpoint.pt" not in payload["files"]:
        raise ValueError("PULSE final adapter checkpoint missing")
