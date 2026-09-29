"""CPU-only contracts for the prospective PULSE AugMix SFT experiment."""
from __future__ import annotations

import json
from pathlib import Path

from util.config_bundle import load_yaml_mapping, resolve_config_reference
from util.pn2021_artifact_contract import sha256_file

CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
PULSE_REVISION = "3b497b425ade2717f76ba0b14202823b83668011"
FULL_LORA_SCOPE = "llm_all_linear_lora32_clip_last4_qv_lora8_projector"


def validate_config(config: dict) -> None:
    expected = {"schema_version", "mode", "references", "center", "width", "model",
                "training", "paths", "resume_from"}
    visual = config.get("schema_version") in (3, 4, 5)
    full_lora = config.get("schema_version") == 5
    if full_lora:
        expected.add("training_scope")
        if config["training_scope"] != FULL_LORA_SCOPE:
            raise ValueError("schema 5 requires the explicit full LLM LoRA scope")
    if config.get("schema_version") in (4, 5):
        from core.image_corruption import validate_image_config
        expected.add("image_augmentation")
        validate_image_config(config["image_augmentation"])
    if config.get("schema_version") in (2, 3, 4, 5):
        expected.add("mixing_domain")
        if config.get("mixing_domain") != "rendered_rgb":
            raise ValueError("schema 2 is the matched rendered-RGB mixing ablation")
    if set(config) != expected or config["schema_version"] not in (1, 2, 3, 4, 5):
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
    if (model["lora_rank"], model["lora_alpha"], model["lora_dropout"]) != ((32, 64, 0.0) if full_lora else (8, 16, 0.0) if visual else (16, 32, 0.05)):
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
    for raw in [model["directory"], paths["raw_root"], paths["toolkit_dir"]]:
        Path(raw).resolve().relative_to(Path("/home/linbinhao"))
    if config["resume_from"]:
        resume = Path(config["resume_from"]).resolve()
        if not resume.is_relative_to("/home/linbinhao") and not resume.is_relative_to("/dev/shm/linbinhao-pulse-hybrid"):
            raise ValueError("resume checkpoint requires home-owned data or scoped RAM")
    temporary = Path(paths["temporary_root"]).resolve()
    if not temporary.is_relative_to("/home/linbinhao") and not temporary.is_relative_to("/dev/shm/linbinhao-pulse-hybrid"):
        raise ValueError("temporary files require user-owned data root or scoped RAM directory")


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
    full_lora = protocol.get("training_scope") == FULL_LORA_SCOPE
    visual = full_lora or protocol.get("training_scope") == "clip_last4_qv_lora8_projector_frozen_llm"
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
        if "visual_training_admission.json" not in payload["files"]:
            raise ValueError("visual training admission is missing")
        audit = json.loads((path.parent / "visual_training_admission.json").read_text())
        if full_lora and (audit.get("language_lora_updated") is not True
                or (protocol["model"]["lora_rank"], protocol["model"]["lora_alpha"]) != (32, 64)):
            raise ValueError("full language LoRA training admission failed")
        if (audit.get("frozen_parameter_versions_unchanged") is not True
                or audit.get("trainable_vision_outputs_cached") != 0
                or audit.get("vision_lora_updated") is not True
                or audit.get("projector_updated") is not True):
            raise ValueError("visual-only training admission failed")
        weight = 12.0
        if "image_augmentation" in protocol:
            from core.image_corruption import validate_image_config
            validate_image_config(protocol["image_augmentation"])
            weight = protocol["image_augmentation"]["jsd_weight"]
            expected_topology = ("image_only_gpu_branches_v1"
                if protocol["image_augmentation"].get("implementation") == "augmix_torch_gpu_v2"
                else "shared_waveform_image_branches_v1")
            if protocol.get("augmentation_topology") != expected_topology:
                raise ValueError("hybrid augmentation topology missing")
            if "implementation" in protocol["image_augmentation"]:
                implementation = protocol["image_augmentation"]["implementation"]
                if implementation == "augmix_pil_reference_v1":
                    from core.image_augmix_c import AUGMIX_IMPLEMENTATION, AUGMIX_UPSTREAM_COMMIT
                    reference = protocol.get("image_reference", {})
                    if (reference.get("implementation") != AUGMIX_IMPLEMENTATION
                            or reference.get("upstream_commit") != AUGMIX_UPSTREAM_COMMIT
                            or not reference.get("pillow_version")
                            or protocol.get("mix_residual") != "corrupted_waveform_render"):
                        raise ValueError("missing reference AugMix implementation identity")
                elif implementation == "augmix_torch_gpu_v2":
                    gpu = protocol.get("image_gpu", {})
                    if (gpu.get("implementation") != "augmix_torch_gpu_v2"
                        or gpu.get("device_policy") != "same_device_as_rendered_rgb"
                        or gpu.get("parity") != "visual_approximation_not_reference_pixel_equivalence"
                        or gpu.get("host_tensor_transfer") != "none_inside_operator"
                        or gpu.get("waveform_corruption") != "disabled"
                        or protocol.get("mix_residual") != "clean_render"):
                        raise ValueError("missing GPU AugMix implementation identity")
                else:
                    raise ValueError("unknown AugMix image implementation identity")
        if (protocol["dirichlet_alpha"] != 1.0 or protocol["clean_mix"] != "beta_1_1"
                or protocol["jsd_weight"] != (weight if protocol["width"] else 0.0)):
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
