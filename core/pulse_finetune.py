"""One-stage native500 PULSE LoRA SFT, with matched single/two-chain views.

Only fixed target-center K500 identities enter training. Images are rendered
online; durable files contain manifests, adapter/projector state and recovery
state, never an expanded ECG image dataset.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from core.augmix import native500_augmix_one
from data_preprocess.data_runtime import PN2021K500LoaderPlan
from util.augmentations.profile import load_augmentation_profile
from util.ecg_image_renderer import PulseECGTensorRenderer
from util.evaluation.ecg_image_data import QUESTION, native500_waveform
from util.evaluation.ecg_image_queue import atomic_json, digest_json
from util.pn2021_artifact_contract import sha256_file
from util.pulse_training_contract import CLASS_ORDER, label_text, load_config, validate_result

REPO = Path(__file__).resolve().parents[1]
TRAINABLE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODEL_HASHES = {
    "model-00001-of-00003.safetensors": "2690665841131164f9ef0b8d67ae47ee89e18e288e9b9c721124be3bbe25537b",
    "model-00002-of-00003.safetensors": "09717b1627cabad1693a4efd921baf73178f1fb28868ee73bd34895a80c2691a",
    "model-00003-of-00003.safetensors": "7039b3c407a504bf9fb13a34c47934e30f026ea32e9efaf258725f27a0322af0",
}


def k500_records(center: str, refs: dict) -> tuple[list[dict], dict]:
    plan = PN2021K500LoaderPlan(center=center, batch_size=1, num_workers=0,
        pin_memory=False, persistent_workers=False, prefetch_factor=2,
        cache_mode="mmap", validate_values="none", selection_resident=False,
        selection_resident_pin_memory=False, drop_last=False,
        seed_namespace="pulse_k500_identity_v1", split_config_path=refs["split_config"],
        data_load_config_path=refs["data_load_config"], seed_config_path=refs["seed_config"])
    loader = plan.open_ordered()
    try:
        dataset = loader.dataset
        selection, cache = dataset.selection, dataset._get_cache()
        rows = []
        for position, index in enumerate(selection.indices):
            metadata = cache.records.iloc[int(index)]
            hash_id = str(selection.hash_ids[position])
            if hash_id != str(metadata["hash_id"]) or float(metadata["original_fs"]) != 500.0:
                raise ValueError("K500 hash or native500 source mismatch")
            label = np.asarray(cache.labels[int(index)], dtype=np.uint8).astype(int).tolist()
            answer = label_text(label)
            rows.append({"sample_key": f"{center}:{hash_id}", "hash_id": hash_id,
                "logical_center": center, "source_center": str(metadata["center"]),
                "record_id": str(metadata["record_id"]), "source_record": str(metadata["source_record"]),
                "label": label, "label_names": answer.split(";"),
                "original_num_samples": int(metadata["original_num_samples"]),
                "repaired_nonfinite_count": int(metadata["repaired_nonfinite_count"])})
        rows.sort(key=lambda row: row["hash_id"])
        if len(rows) != 500 or len({r["hash_id"] for r in rows}) != 500:
            raise ValueError("K500 identity is not exactly 500 unique records")
        return rows, {"selection_hash_id_set_sha256": selection.hash_id_set_sha256,
                      "cache_manifest_sha256": cache.identity.manifest_sha256}
    finally:
        loader.close()


def sample_schedule(count: int, total: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    output = []
    while len(output) < total:
        output.extend(rng.permutation(count).tolist())
    return output[:total]


def tokenize_supervision(tokenizer, label: list[int]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    answer = label_text(label)
    conversation = conv_templates["llava_v1"].copy()
    conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + QUESTION)
    conversation.append_message(conversation.roles[1], None)
    prefix = tokenizer_image_token(conversation.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    conversation.messages[-1][1] = answer
    tokens = tokenizer_image_token(conversation.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    if not torch.equal(tokens[:len(prefix)], prefix):
        raise ValueError("answer tokenization changed the inference prompt prefix")
    labels = tokens.clone()
    labels[:len(prefix)] = -100
    decoded = tokenizer.decode(tokens[len(prefix):], skip_special_tokens=True).strip()
    if decoded != answer or tokens[-1].item() != tokenizer.eos_token_id:
        raise ValueError("assistant supervision mask or EOS is incorrect")
    if int((tokens == IMAGE_TOKEN_INDEX).sum()) != 1:
        raise ValueError("expected exactly one ECG image placeholder")
    return tokens.unsqueeze(0), labels.unsqueeze(0), {"answer": answer,
        "prompt_tokens": len(prefix), "supervised_tokens": len(tokens) - len(prefix),
        "expanded_sequence_tokens": len(tokens) - 1 + 5 * 576}


def load_pulse_for_training(config: dict):
    from llava.model.builder import load_pretrained_model
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    directory = Path(config["model"]["directory"])
    warnings.filterwarnings("ignore", message=r"for vision_model\..*copying from a non-meta parameter", category=UserWarning)
    tokenizer, model, _, _ = load_pretrained_model(str(directory), None, "pulse-7b",
        device="cpu", device_map="cpu", attn_implementation="sdpa", local_files_only=True)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("PULSE load left meta parameters")
    # This specific trained vision parameter catches accidental bare-CLIP reload.
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.vision_tower.vision_tower.vision_model.embeddings.class_embedding"
    with safe_open(directory / index[name], framework="pt", device="cpu") as handle:
        expected = handle.get_tensor(name).to(torch.float16)
    actual = dict(model.named_parameters())[name].detach().cpu()
    if not torch.equal(actual, expected):
        raise ValueError("trained PULSE vision tower does not match checkpoint")
    model.requires_grad_(False)
    visual = (config.get("schema_version") == 3 or config.get("training_scope")
              == "clip_last4_qv_lora8_projector_frozen_llm")
    targets = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)
               and name.startswith("model.layers.") and name.rsplit(".", 1)[-1] in TRAINABLE_TARGETS]
    if len(targets) != 32 * len(TRAINABLE_TARGETS):
        raise ValueError("unexpected PULSE language LoRA target count")
    if visual:
        from core.pulse_visual import visual_lora_targets
        targets = visual_lora_targets(model)
    options = config["model"]
    model = get_peft_model(model, LoraConfig(r=options["lora_rank"], lora_alpha=options["lora_alpha"],
        lora_dropout=options["lora_dropout"], bias="none", task_type="CAUSAL_LM", target_modules=targets))
    base = model.get_base_model()
    base.config.use_cache = False
    base.config.tokenizer_model_max_length = config["training"]["max_length"]
    base.get_model().mm_projector.requires_grad_(True)
    # AdamW master parameters and gradients stay FP32; frozen base remains FP16.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name and ".mm_projector." not in name:
                raise ValueError(f"unexpected trainable parameter: {name}")
            parameter.data = parameter.data.float()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    if not visual:
        base.get_vision_tower().requires_grad_(False)
    base.get_vision_tower().eval()
    return tokenizer, model, {"trained_vision_probe_exact": True, "lora_target_modules": targets,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "base_precision": "float16", "trainable_precision": "float32", "attention": "sdpa",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def trainable_state(model) -> dict:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def pack_flat_visual_features(base, input_ids, labels, features):
    """Exact one-image flat merge; admitted against the author's GPU output."""
    from llava.constants import IMAGE_TOKEN_INDEX
    if base.config.mm_patch_merge_type != "flat" or input_ids.shape[0] != 1 or tuple(features.shape) != (5, 576, 1024):
        raise ValueError("unsupported precomputed-vision packing contract")
    positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
    if len(positions) != 1:
        raise ValueError("expected one image placeholder")
    index = int(positions.item())
    ids = torch.cat((input_ids[0, :index], input_ids[0, index + 1:]))
    embedded = base.get_model().embed_tokens(ids)
    projected = base.get_model().mm_projector(features).flatten(0, 1)
    packed = torch.cat((embedded[:index], projected, embedded[index:])).unsqueeze(0)
    masked = torch.cat((labels[0, :index], labels.new_full((len(projected),), -100), labels[0, index + 1:])).unsqueeze(0)
    if packed.shape[1] > base.config.tokenizer_model_max_length:
        raise ValueError("packed ECG exceeds context window")
    return packed, masked


def answer_only_loss(hidden, labels, lm_head):
    """Exact causal CE, omitting unused image/prompt vocabulary logits."""
    shifted = labels[:, 1:]
    selected = shifted != -100
    if not bool(selected.any()):
        raise ValueError("no supervised answer tokens")
    logits = lm_head(hidden[:, :-1][selected]).float()
    return F.cross_entropy(logits, shifted[selected])


class FrozenVisionCache:
    """Bounded per-center RAM cache; augmented views are never cached."""
    def __init__(self, renderer, base, limit: int, *, train_vision=False):
        self.renderer, self.base, self.limit = renderer, base, limit
        self.clean = {}
        self.packing_verified = False
        self.train_vision = train_vision

    def features(self, waveform, *, key=None, input_is_image=False):
        if input_is_image and key is not None:
            raise ValueError("augmented RGB must not enter the clean feature cache")
        if key is not None and key in self.clean:
            return self.clean[key].to("cuda", non_blocking=True), None
        image = waveform if input_is_image else self.renderer.render(waveform)
        pixels = self.renderer.preprocess_for_pulse(image).clone().to(torch.float16)
        if self.train_vision:
            from core.pulse_visual import vision_features_with_grad
            with torch.autocast("cuda", dtype=torch.float16):
                features = vision_features_with_grad(self.base, pixels[0])
            return features, pixels
        # Match the author's multimodal forward under the training AMP scope.
        # FP16 weights alone do not reproduce autocast LayerNorm/softmax rules.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            features = self.base.get_vision_tower()(pixels[0])
        if key is not None:
            if len(self.clean) >= self.limit:
                raise ValueError("clean feature cache exceeds K500 bound")
            self.clean[key] = features.detach().to("cpu").pin_memory()
        return features, pixels

    def verify_packing(self, ids, labels, features, pixels):
        if self.packing_verified:
            return
        if pixels is None:
            raise ValueError("packing admission needs an uncached image")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            reference = self.base.prepare_inputs_labels_for_multimodal(ids, None, None, None, labels,
                pixels, image_sizes=[(2200, 1700)])
            packed, masked = pack_flat_visual_features(self.base, ids, labels, features)
        if not torch.equal(reference[4], packed) or not torch.equal(reference[5], masked):
            delta = float((reference[4].float() - packed.float()).abs().max())
            raise ValueError(f"cached feature packing differs from the author implementation: max_abs={delta}, "
                             f"labels_equal={torch.equal(reference[5], masked)}")
        self.packing_verified = True


def restore_trainables(model, state: dict) -> None:
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state):
        raise ValueError("adapter/projector parameter identity changed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape or not bool(torch.isfinite(state[name]).all()):
                raise ValueError("invalid trainable checkpoint tensor")
            parameter.copy_(state[name])


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, *, step: int,
                    protocol_identity: str, history: list[dict]) -> None:
    state = {"schema_version": 1, "protocol_identity": protocol_identity, "step": step,
        "trainables": trainable_state(model), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
        "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
        "rng_numpy": np.random.get_state(), "rng_python": random.getstate()}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(config_path: Path, config_root: Path, output: Path) -> None:
    config, refs = load_config(config_path, config_root)
    output.mkdir(parents=True, exist_ok=False)
    Path(config["paths"]["temporary_root"]).mkdir(parents=True, exist_ok=True)
    settings = config["training"]
    visual = config["schema_version"] == 3
    torch.set_num_threads(settings["cpu_threads"])
    rows, data_audit = k500_records(config["center"], refs)
    atomic_json(output / "k500_records.json", rows)
    profile = load_augmentation_profile(refs["operators_config"], config_root=config_root)
    protocol = {"center": config["center"], "width": config["width"], "partition": "k500",
        "adaptation_records": 500, "mapping_hash": "555ec85d5b51",
        "mapping_version": "v7_super5_sjr_rgq_review_20260528", "class_order": list(CLASS_ORDER),
        "simclr": False, "vae_lhat": False, "jsd": False, "stage_count": 1,
        "model": config["model"], "training": settings, "data_audit": data_audit,
        "k500_records_sha256": sha256_file(output / "k500_records.json"),
        "operator_profile": {k: v for k, v in profile.describe().items() if not k.endswith("path")},
        "input": "native500_physical_mV_5000x12_no_zscore", "clean_mix": "none",
        "clean_augmented_loss_weights": [0.5, 0.5], "images_per_record_exposure": 2,
        "dirichlet_alpha": 0.5, "selection": "last", "paper_claim_allowed": False}
    if config.get("mixing_domain") == "rendered_rgb":
        protocol["mixing_domain"] = "rendered_rgb"
    if visual:
        protocol.update(training_scope="clip_last4_qv_lora8_projector_frozen_llm",
            vision_blocks=[19, 20, 21, 22], jsd=config["width"] > 0,
            jsd_weight=12.0 if config["width"] else 0.0,
            jsd_distribution="aligned_teacher_forced_answer_token_vocabulary",
            jsd_reduction="mean_over_answer_tokens_and_three_views",
            dirichlet_alpha=1.0, clean_mix="beta_1_1", depths=[1, 2, 3],
            operator_sampling="uniform_with_replacement",
            clean_augmented_loss_weights=[1.0, 0.0],
            images_per_record_exposure=3 if config["width"] else 1)
    identity = digest_json(protocol)
    atomic_json(output / "protocol.json", protocol)
    sources = ("core/pulse_finetune.py", "core/augmix.py", "util/pulse_training_contract.py",
        "util/ecg_image_renderer.py", "util/evaluation/ecg_image_data.py",
        "util/augmentations/torch_operators.py", "util/augmentations/profile.py",
        "data_preprocess/preprocess_primitives.py", "data_preprocess/PN2021_preprocess.py",
        "util/evaluation/ecg_image_elastic.py")
    if visual:
        sources += ("core/pulse_visual.py", "core/consistency.py")
    source_hashes = {}
    for relative in sources:
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
        source_hashes[relative] = sha256_file(destination)
    atomic_json(output / "source_identity.json", source_hashes)
    protocol["implementation_sha256"] = source_hashes
    identity = digest_json(protocol)
    atomic_json(output / "protocol.json", protocol)
    if config["mode"] == "prepare":
        _finalize(output, config, protocol, 0)
        return
    if torch.cuda.device_count() != 1:
        raise ValueError("each PULSE training task must bind exactly one free GPU UUID")
    gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
        raise ValueError("PULSE training requires an explicit single GPU UUID")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True)
    from util.evaluation.ecg_image_elastic import authorized_idle_contexts
    allowed = authorized_idle_contexts(gpu_uuid)
    atomic_json(output / "gpu_admission.json", {"gpu_uuid": gpu_uuid, "allowed_idle_context_pids": sorted(allowed),
        "permit_sha256": os.environ.get("PULSE_IDLE_CONTEXT_PERMIT_SHA256")})
    for line in processes.splitlines():
        identity_gpu, pid = [item.strip() for item in line.split(",")]
        if identity_gpu == gpu_uuid and int(pid) not in {os.getpid(), *allowed}:
            raise RuntimeError(f"GPU acquired by another process {pid}; refusing to load")
    free, _ = torch.cuda.mem_get_info()
    if free < 22 * 1024**3:
        raise RuntimeError("PULSE admission requires at least 22 GiB free")
    admission_reservation = torch.empty(20 * 1024**3, dtype=torch.uint8, device="cuda")
    started = time.time()
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Torch 2.1 memory-efficient SDPA otherwise has nondeterministic backward
    # reductions; RNG restoration alone is insufficient for exact resume.
    torch.use_deterministic_algorithms(True)
    directory = Path(config["model"]["directory"])
    for name, digest in MODEL_HASHES.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"PULSE model shard mismatch: {name}")
    tokenizer, model, model_audit = load_pulse_for_training(config)
    # Keep the CUDA claim throughout CPU hashing/loading, then transfer once.
    del admission_reservation
    torch.cuda.empty_cache()
    model.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    atomic_json(output / "model_audit.json", model_audit)
    print(json.dumps({"event": "model_ready", "trainable_parameters": model_audit["trainable_parameters"],
        "trained_vision_probe_exact": model_audit["trained_vision_probe_exact"]}), flush=True)
    selected = rows[:settings["smoke_records"]] if config["mode"] == "smoke" else rows
    with ThreadPoolExecutor(max_workers=settings["loader_workers"]) as executor:
        waveforms = list(executor.map(lambda row: native500_waveform(row, Path(config["paths"]["raw_root"]))[0], selected))
    # <=120 MB per full center; augmented images are never persisted.
    waves = torch.from_numpy(np.stack(waveforms)).to("cuda")
    del waveforms
    tokens = [tokenize_supervision(tokenizer, row["label"]) for row in selected]
    if any(audit["expanded_sequence_tokens"] > settings["max_length"] for _, _, audit in tokens):
        raise ValueError("visual tokens would truncate answer supervision")
    atomic_json(output / "token_audit.json", [audit for _, _, audit in tokens])
    renderer, rendering = PulseECGTensorRenderer.from_ecg_image_kit(config["paths"]["toolkit_dir"],
        device=torch.device("cuda:0"), tmp_root=config["paths"]["temporary_root"])
    atomic_json(output / "renderer.json", rendering)
    if config.get("mixing_domain") == "rendered_rgb" and not visual:
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state().clone()
        for index in range(min(4, len(selected))):
            kwargs = {"width": 1, "profile": profile, "seed": seed,
                      "identity": f"pixel-admission|{selected[index]['hash_id']}"}
            wave, _ = native500_augmix_one(waves[index:index+1], **kwargs)
            expected = renderer.render(wave)
            actual, _ = native500_augmix_one(waves[index:index+1], renderer=renderer, **kwargs)
            if not torch.equal(expected, actual) or not torch.equal(
                    renderer.preprocess_for_pulse(expected), renderer.preprocess_for_pulse(actual)):
                raise ValueError("single-chain pixel mixing does not equal waveform reference")
        if not torch.equal(before, torch.get_rng_state()) or not torch.equal(cuda_before, torch.cuda.get_rng_state()):
            raise ValueError("pixel admission changed training RNG")
        del expected, actual, wave
        atomic_json(output / "pixel_mixing_admission.json", {
            "single_chain_pixels_exact": True, "single_chain_processor_exact": True,
            "global_torch_rng_unchanged": True, "records": min(4, len(selected)),
            "mixing_before_processor_and_vision_encoder": True, "clean_residual": False})
    vision_cache = FrozenVisionCache(renderer, model.get_base_model(), len(selected), train_vision=visual)
    initial_trainables = trainable_state(model) if visual else {}
    frozen_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad} if visual else {}
    adapter = [p for name, p in model.named_parameters() if p.requires_grad and "lora_" in name]
    projector = [p for name, p in model.named_parameters() if p.requires_grad and ".mm_projector." in name]
    optimizer = torch.optim.AdamW([{"params": adapter, "lr": settings["learning_rate"]},
        {"params": projector, "lr": settings["projector_learning_rate"]}], weight_decay=0.0)
    steps, warmup = settings["optimizer_steps"], settings["warmup_steps"]
    def scale(step):
        return (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.0)
    history, start_step = [], 0
    if config["resume_from"]:
        previous = torch.load(config["resume_from"], map_location="cpu")
        if previous["protocol_identity"] != identity:
            raise ValueError("resume protocol identity differs")
        restore_trainables(model, previous["trainables"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        torch.set_rng_state(previous["rng_torch"])
        torch.cuda.set_rng_state(previous["rng_cuda"])
        np.random.set_state(previous["rng_numpy"])
        random.setstate(previous["rng_python"])
        history, start_step = previous["history"], previous["step"]
        if not 0 < start_step < steps:
            raise ValueError("resume must execute at least one remaining optimizer step")
        del previous
    batch = settings["effective_batch"]
    schedule = sample_schedule(len(selected), steps * batch, seed)
    atomic_json(output / "exposure_schedule.json", [selected[i]["hash_id"] for i in schedule])
    stop_requested = []
    def request_stop(signum, frame):
        stop_requested.append(signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model.train()
    model.get_base_model().get_vision_tower().eval()
    for step in range(start_step, steps):
        step_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        ce_sum, jsd_sum = 0.0, 0.0
        for offset in range(batch):
            exposure = step * batch + offset
            index = schedule[exposure]
            clean = waves[index:index + 1]
            if visual:
                from core.augmix import native500_augmix_jsd_views
                from core.pulse_visual import aligned_answer_logits, answer_jsd_loss
                images, trace = native500_augmix_jsd_views(clean, width=config["width"], profile=profile,
                    seed=seed, identity=f"{selected[index]['hash_id']}|{exposure}", renderer=renderer)
                input_ids, labels, _ = tokens[index]
                input_ids, labels = input_ids.cuda(), labels.cuda()
                logits, answer_targets = [], None
                for image in images:
                    features, pixels = vision_cache.features(image, input_is_image=True)
                    vision_cache.verify_packing(input_ids, labels, features, pixels)
                    with torch.autocast("cuda", dtype=torch.float16):
                        base = model.get_base_model()
                        packed, masked = pack_flat_visual_features(base, input_ids, labels, features)
                        value, target = aligned_answer_logits(base, packed, masked)
                    if answer_targets is not None and not torch.equal(answer_targets, target):
                        raise ValueError("JSD answer positions differ across views")
                    logits.append(value)
                    answer_targets = target
                    del features, pixels, packed, masked
                loss, ce, jsd = answer_jsd_loss(logits, answer_targets)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite visual JSD loss")
                loss_sum += float(loss.detach()) / batch
                ce_sum += float(ce.detach()) / batch
                jsd_sum += float(jsd.detach()) / batch
                scaler.scale(loss / batch).backward()
                del loss, ce, jsd, logits, value, images, image
                continue
            augmented, trace = native500_augmix_one(clean, width=config["width"], profile=profile,
                seed=seed, identity=f"{selected[index]['hash_id']}|{exposure}",
                renderer=renderer if config.get("mixing_domain") == "rendered_rgb" else None)
            input_ids, labels, _ = tokens[index]
            input_ids, labels = input_ids.cuda(), labels.cuda()
            for view_index, view in enumerate((clean, augmented)):
                features, pixels = vision_cache.features(view, key=index if view_index == 0 else None,
                    input_is_image=view_index == 1 and config.get("mixing_domain") == "rendered_rgb")
                vision_cache.verify_packing(input_ids, labels, features, pixels)
                del pixels
                with torch.autocast("cuda", dtype=torch.float16):
                    base = model.get_base_model()
                    packed, masked = pack_flat_visual_features(base, input_ids, labels, features)
                    hidden = base.get_model()(inputs_embeds=packed, use_cache=False, return_dict=True).last_hidden_state
                    loss = answer_only_loss(hidden, masked, base.lm_head) / (2 * batch)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite PULSE SFT loss")
                loss_sum += float(loss.detach())
                scaler.scale(loss).backward()
                del loss, hidden, packed, masked, features
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_([*adapter, *projector], 1.0, error_if_nonfinite=True)
        groups = {"lora": adapter, "projector": projector}
        grad_audit = {name: float(torch.stack([p.grad.detach().float().square().sum() for p in values if p.grad is not None]).sum().sqrt())
                      for name, values in groups.items()}
        if any(value <= 0 or not math.isfinite(value) for value in grad_audit.values()):
            raise ValueError("missing or invalid adapter/projector gradients")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise ValueError("AMP skipped an optimizer update; matched budget cannot be claimed")
        scheduler.step()
        row = {"step": step + 1, "loss": loss_sum, "grad_norm": float(norm), "grad_groups": grad_audit,
            "seconds": time.time() - step_started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "last_augmentation": trace}
        if visual:
            row.update(clean_answer_ce=ce_sum, answer_token_jsd=jsd_sum)
            if any(p.grad is not None or p._version != frozen_versions[name]
                   for name, p in model.named_parameters() if not p.requires_grad):
                raise ValueError("frozen PULSE parameters changed or received gradients")
        history.append(row)
        atomic_json(output / "progress.json", {"pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "step": step + 1, "target_steps": steps, "latest": row, "elapsed_seconds": time.time() - started})
        print(json.dumps(row), flush=True)
        if (step + 1) % settings["save_every"] == 0 or step + 1 == steps or stop_requested:
            save_checkpoint(output / "checkpoint.pt", model, optimizer, scheduler, scaler,
                step=step + 1, protocol_identity=identity, history=history)
            if config["mode"] == "smoke" and step == 0:
                shutil.copyfile(output / "checkpoint.pt", output / "resume_probe.pt")
        if stop_requested and step + 1 < steps:
            atomic_json(output / "interrupted.json", {"step": step + 1, "signal": stop_requested[-1]})
            raise SystemExit(75)
    atomic_json(output / "history.json", history)
    # Validate adapter+projector disk round trip without relying on PEFT export defaults.
    saved = torch.load(output / "checkpoint.pt", map_location="cpu")
    restored = trainable_state(model)
    if set(saved["trainables"]) != set(restored) or any(not torch.equal(saved["trainables"][name], value) for name, value in restored.items()):
        raise ValueError("adapter/projector save round trip mismatch")
    process_resume_verified = False
    max_delta = None
    if config["mode"] == "smoke" and config["resume_from"]:
        reference = torch.load(Path(config["resume_from"]).parent / "checkpoint.pt", map_location="cpu")
        if reference["protocol_identity"] != identity or reference["step"] != steps:
            raise ValueError("resume reference is not the matching uninterrupted smoke")
        max_delta = max(float((reference["trainables"][name] - value).abs().max()) for name, value in restored.items())
        if max_delta > 1e-7:
            raise ValueError(f"resumed next-step trainables differ from uninterrupted run: {max_delta}")
        process_resume_verified = True
    atomic_json(output / "recovery_audit.json", {"disk_trainables_exact": True,
        "optimizer_state_saved": bool(saved["optimizer"]["state"]), "rng_states_saved": True,
        "process_resume_verified": process_resume_verified, "resume_max_trainable_delta": max_delta,
        "optimizer_steps": steps})
    atomic_json(output / "optimization_audit.json", {"author_flat_packing_exact": vision_cache.packing_verified,
        "answer_only_vocab_logits": True, "cached_clean_records": len(vision_cache.clean),
        "cached_clean_bytes": sum(t.numel() * t.element_size() for t in vision_cache.clean.values()),
        "augmented_images_persisted": 0})
    if visual:
        delta = {name: float((value - initial_trainables[name]).abs().max()) for name, value in restored.items()}
        atomic_json(output / "visual_training_admission.json", {
            "frozen_parameter_versions_unchanged": True, "trainable_vision_outputs_cached": len(vision_cache.clean),
            "vision_lora_updated": any(v > 0 for n, v in delta.items() if "lora_" in n),
            "projector_updated": any(v > 0 for n, v in delta.items() if ".mm_projector." in n),
            "trainable_max_deltas": delta, "author_flat_packing_exact": vision_cache.packing_verified})
    atomic_json(output / "runtime.json", {"seconds": time.time() - started, "gpu_hours": (time.time() - started) / 3600,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "python": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}})
    _finalize(output, config, protocol, steps)


def _finalize(output, config, protocol, step):
    files = {path.relative_to(output).as_posix(): sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"artifact_type": "pulse_train_result", "schema_version": 1, "status": "complete",
        "mode": config["mode"], "protocol": protocol, "optimizer_steps": step, "files": files}
    validate_result(result, output / "train_result.json")
    atomic_json(output / "train_result.json", result)
