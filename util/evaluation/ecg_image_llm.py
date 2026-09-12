"""Bounded-RAM, resumable native500 image-LLM inference; no model training.

Backend names are code-owned. The user-requested RAM progress survives process
restart, not machine restart; final shards are copied to the managed run.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from util.ecg_image_renderer import PulseECGTensorRenderer
from util.evaluation.ecg_image_data import (
    CLASS_ORDER, CENTERS, QUESTION, audit_inputs, corrupt_native500_one, native500_waveform,
)
from util.evaluation.ecg_image_artifact import parse_response
from util.pn2021_artifact_contract import sha256_file

MODEL_REPO = "convaiinnovations/ECG-Instruct-Llama-3.2-11B-Vision"
MODEL_REVISION = "9bf04f4fc65bd78e38899c6100956be7f4f9c042"
MODEL_IDENTITIES = {
    "mllama": (MODEL_REPO, MODEL_REVISION),
    "ecg_r1_image": ("PKUDigitalHealth/ECG-R1-8B-RL", "f9257759e2e3d6b1864c2e1af4dfaa8e96eb84ea"),
}
PROMPTS = {
    "pulse_exact": QUESTION,
    "strict_codes_v1": (
        "This is a MULTILABEL CLASSIFICATION task, NOT an ECG report-writing task. "
        "Classify the 12-lead ECG image into all applicable categories: "
        "CD = Conduction Disturbance; HYP = Hypertrophy; MI = Myocardial Infarction; "
        "NORM = Normal ECG; STTC = ST/T Change. "
        "Return ONLY the category CODES, separated by semicolons. "
        "Do not output words other than CD, HYP, MI, NORM, STTC. "
        "Do not expand the codes, explain, or write an ECG report."
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def validate_config(config: dict) -> None:
    expected = {"schema_version", "stage", "references", "baseline", "cohort", "model", "runtime", "paths"}
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError("invalid image-LLM config schema")
    if config["stage"] not in {"smoke", "evaluate"}:
        raise ValueError("unsupported image-LLM stage")
    model = config["model"]
    if MODEL_IDENTITIES.get(model["backend"]) != (model["repo_id"], model["revision"]):
        raise ValueError("model is not the frozen code-owned image-LLM backend")
    if model["precision"] not in ({"nf4_fp16", "float16"} if model["backend"] == "mllama" else {"bfloat16"}):
        raise ValueError("unsupported precision")
    strategy = model.get("device_strategy", "single_gpu")
    if strategy not in {"single_gpu", "two_gpu_balanced"} or (strategy == "two_gpu_balanced" and (
        model["backend"] != "mllama" or model["precision"] != "float16" or config["stage"] != "smoke"
        or config["runtime"]["world_size"] != 1
    )):
        raise ValueError("two-GPU placement is limited to the FP16 Llama admission probe")
    variant = model.get("prompt_variant", "pulse_exact")
    if variant not in PROMPTS or (model["backend"] != "mllama" and variant != "pulse_exact"):
        raise ValueError("unsupported code-owned prompt variant")
    runtime = config["runtime"]
    if not 1 <= runtime["batch_size"] <= 8 or not 1 <= runtime["loader_workers"] <= 4:
        raise ValueError("unsafe inference batch/worker request")
    if not 0 <= runtime["rank"] < runtime["world_size"] <= 4:
        raise ValueError("invalid shard topology")
    if runtime["max_new_tokens"] not in ({64, 256} if model["backend"] == "mllama" else {64, 512}):
        raise ValueError("unsupported frozen generation budget")
    if not 1 <= config["cohort"]["per_center"] <= 500:
        raise ValueError("invalid cohort count")
    working = Path(config["paths"]["working_root"]).resolve()
    if working != Path("/dev/shm/ecg_image_llm_10h_20260907"):
        raise ValueError("working outputs must use the authorized task RAM directory")
    for name in ("raw_root", "toolkit_dir"):
        Path(config["paths"][name]).resolve().relative_to(Path("/home/linbinhao"))
    Path(model["directory"]).resolve().relative_to(Path("/home/linbinhao"))


def audit_model(directory: Path, revision: str, repo_id: str) -> dict:
    # Replicas share the read-only weights and this small verification cache.
    # Serialize its update to avoid racing on the atomic writer's temporary file.
    with (directory / ".asset_audit.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _audit_model_locked(directory, revision, repo_id)


def _audit_model_locked(directory: Path, revision: str, repo_id: str) -> dict:
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    audit_path = directory / "local_verified_assets.json"
    previous = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    verified = []
    for name in shards:
        path = directory / name
        stat = path.stat()
        metadata = directory / ".cache/huggingface/download" / (name + ".metadata")
        lines = metadata.read_text().splitlines()
        if lines[0] != revision or len(lines[1]) != 64:
            raise ValueError(f"download identity mismatch: {name}")
        expected = lines[1]
        prior = next((r for r in previous.get("shards", []) if r["name"] == name), {})
        if (prior.get("sha256"), prior.get("size"), prior.get("mtime_ns")) == (expected, stat.st_size, stat.st_mtime_ns):
            actual = expected
            verification = "previous_stream_sha256_with_unchanged_size_mtime"
        else:
            actual = sha256_file(path)
            verification = "stream_sha256"
        if actual != expected:
            raise ValueError(f"downloaded weight SHA256 mismatch: {name}")
        verified.append({"name": name, "sha256": actual, "size": stat.st_size,
                         "mtime_ns": stat.st_mtime_ns, "verification": verification})
    small_files = {p.name: sha256_file(p) for p in sorted(directory.glob("*.json"))
                   if p.name != "local_verified_assets.json"}
    result = {"repo_id": repo_id, "revision": revision, "shards": verified, "config_files": small_files}
    write_json(audit_path, result)
    return result


class LlamaBackend:
    def __init__(self, config: dict):
        from transformers import AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration

        path = config["directory"]
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        kwargs = {"torch_dtype": torch.float16, "device_map": {"": 0},
                  "local_files_only": True, "low_cpu_mem_usage": True, "attn_implementation": "sdpa"}
        parallel = config.get("device_strategy", "single_gpu") == "two_gpu_balanced"
        if parallel:
            kwargs.update(device_map="balanced", max_memory={0: "14GiB", 1: "14GiB"})
        if config["precision"] == "nf4_fp16":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.model, loading = MllamaForConditionalGeneration.from_pretrained(path, output_loading_info=True, **kwargs)
        self.loading = loading
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"checkpoint architecture mismatch: {loading}")
        if parallel and {str(value) for value in self.model.hf_device_map.values()} != {"0", "1"}:
            raise ValueError("two-GPU Llama must use both selected GPUs, with no CPU/disk offload")
        self.model.eval()
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        question = PROMPTS[config.get("prompt_variant", "pulse_exact")]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        self.prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    @torch.inference_mode()
    def generate(self, rgb: torch.Tensor, *, max_new_tokens: int) -> tuple[list[str], list[int]]:
        # Same float GPU-rendered page; this backend's standard image processor
        # consumes RGB uint8, unlike PULSE's direct floating-point CLIP tiling.
        pixels = rgb.mul(255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        images = [Image.fromarray(p) for p in pixels]
        inputs = self.processor(images=images, text=[self.prompt] * len(images), padding=True, return_tensors="pt")
        inputs = {key: value.to("cuda", dtype=torch.float16 if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        width = inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )[:, width:]
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
        pad = self.processor.tokenizer.pad_token_id
        lengths = [(row != pad).sum().item() for row in outputs]
        return decoded, lengths


class ECG_R1_ImageBackend:
    """Checkpoint's image-only Qwen3-VL path, with no waveform inputs.

Official ECG-R1 5008b8f9 mounts an ECG tower/projector and conditionally invokes
them only when ecg_features is present. Here those inactive weights are omitted;
all image/text weights must load exactly. This is an explicitly labelled
image-only adapter, not the author's joint waveform+image evaluation.
"""
    def __init__(self, config: dict):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if config["precision"] != "bfloat16":
            raise ValueError("ECG-R1 image-only currently requires unquantized bfloat16")
        path = config["directory"]
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.image_processing = config.get("image_processing", "pil_cpu")
        if self.image_processing not in {"pil_cpu", "cpu_uint8_equivalent_cuda_v1"}:
            raise ValueError("unsupported R1 image processing implementation")
        # Matches IMAGE_MAX_TOKEN_NUM=768 and 32-pixel factor in official infer.sh.
        self.processor.image_processor.size = {"shortest_edge": 4 * 32**2, "longest_edge": 768 * 32**2}
        self.model, self.loading = Qwen3VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
            low_cpu_mem_usage=True, attn_implementation="sdpa", output_loading_info=True,
        )
        unexpected = self.loading.get("unexpected_keys", [])
        if any(self.loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")) or any(
            not key.startswith(("model.ecg_tower.", "model.ecg_projector.")) for key in unexpected
        ):
            raise ValueError(f"ECG-R1 active image/text weight mismatch: {self.loading}")
        self.model.eval()
        self.model.config.text_config.use_cache = True
        self.processor.tokenizer.padding_side = "left"
        messages = [
            {"role": "system", "content": "You are a helpful, harmless clinical ECG assistant. Provide concise, evidence-based interpretations."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]},
        ]
        self.prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    @torch.inference_mode()
    def generate(self, rgb: torch.Tensor, *, max_new_tokens: int) -> tuple[list[str], list[int]]:
        pixels = rgb.mul(255).round().clamp(0, 255).to(torch.uint8)
        if self.image_processing == "cpu_uint8_equivalent_cuda_v1":
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
            from util.evaluation.ecg_image_processor import resize_uint8_cpu_equivalent
            if not pixels.is_cuda or tuple(pixels.shape[1:]) != (3, 1700, 2200):
                raise ValueError("R1 optimized input must be the frozen GPU-rendered ECG page")
            size = smart_resize(1700, 2200, factor=32, min_pixels=4 * 32**2, max_pixels=768 * 32**2)
            images = resize_uint8_cpu_equivalent(pixels, size)
            inputs = self.processor(images=images, text=[self.prompt] * len(images), padding=True,
                                    return_tensors="pt", images_kwargs={"do_resize": False,
                                    "input_data_format": "channels_first", "device": str(rgb.device)})
        else:
            images = [Image.fromarray(p) for p in pixels.permute(0, 2, 3, 1).cpu().numpy()]
            inputs = self.processor(images=images, text=[self.prompt] * len(images), padding=True, return_tensors="pt")
        if "ecg_features" in inputs or torch.any(inputs["input_ids"] == self.model.config.ecg_token_id):
            raise ValueError("waveform tokens/features leaked into image-only inference")
        inputs = {key: value.to("cuda", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                      use_cache=True, pad_token_id=self.processor.tokenizer.pad_token_id)
        outputs = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
        lengths = [(row != self.processor.tokenizer.pad_token_id).sum().item() for row in outputs]
        return decoded, lengths


def run(config_path: Path, config_root: Path, output_dir: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    ram = Path(config["paths"]["working_root"])
    batch_parity = ram / f"renderer_batch_parity_torch_{torch.__version__.replace('+', '_')}.json"
    if json.loads(batch_parity.read_text()).get("status") != "passed":
        raise ValueError("the active environment requires a renderer batch parity check")
    parity_files = [batch_parity]
    if config["model"]["backend"] == "ecg_r1_image":
        for name in (f"input_parity_torch_{torch.__version__.replace('+', '_')}.json", "r1_image_only_forward_parity.json"):
            path = ram / name
            evidence = json.loads(path.read_text())
            if evidence.get("status") != "passed" or evidence.get("torch") != torch.__version__:
                raise ValueError("ECG-R1 requires matching input and image-only adapter parity checks")
            parity_files.append(path)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    strategy = config["model"].get("device_strategy", "single_gpu")
    device_count = 2 if strategy == "two_gpu_balanced" else 1
    if (len(set(visible.split(","))) != device_count or not visible or not torch.cuda.is_available()
            or torch.cuda.device_count() != device_count):
        raise RuntimeError(f"exactly {device_count} explicitly bound CUDA devices required for this placement")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260501)
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = config["runtime"]
    rank, world = runtime["rank"], runtime["world_size"]
    task_name = f"{config['model']['backend']}_{config['model']['precision']}_{config['model'].get('prompt_variant', 'pulse_exact')}_{config['stage']}_n{config['cohort']['per_center']}_w{world}_r{rank}_b{runtime['batch_size']}_t{runtime['max_new_tokens']}"
    if strategy != "single_gpu":
        task_name += f"_{strategy}"
    working = Path(config["paths"]["working_root"]) / task_name
    working.mkdir(parents=True, exist_ok=True)
    lock = (working / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha = sha256_file(config_path)
    implementation = {str(path.relative_to(Path(__file__).resolve().parents[2])): sha256_file(path)
                      for path in (Path(__file__).resolve(), Path(__file__).with_name("ecg_image_data.py"),
                                   Path(__file__).with_name("ecg_image_artifact.py"),
                                   Path(__file__).resolve().parents[1] / "ecg_image_renderer.py")}
    identity_path = working / "identity.json"
    identity = {"config_sha256": config_sha, "implementation_sha256": implementation}
    if identity_path.exists():
        previous_identity = json.loads(identity_path.read_text())
        if any(previous_identity.get(k) != v for k, v in identity.items()):
            raise ValueError("RAM progress belongs to different frozen configuration or code")
    else:
        write_json(identity_path, {**identity, "created_at": now()})
    source_snapshot = working / "source_snapshot"
    for relative, digest in implementation.items():
        destination = source_snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(Path(__file__).resolve().parents[2] / relative, destination)
        if sha256_file(destination) != digest:
            raise ValueError("RAM source snapshot does not match frozen implementation")
    started = time.monotonic()
    print(f"{now()} auditing inputs rank={rank}/{world} GPU={visible}", flush=True)
    rows, conditions, profile, baseline_protocol, audit = audit_inputs(config, config_path, config_root)
    all_rows = rows
    # Assign each center independently so every shard remains center-balanced.
    rows = [row for index, row in enumerate(rows) if (index // len(CENTERS)) % world == rank]
    if config["stage"] == "smoke":
        conditions = [conditions[i] for i in (0, 1, 11, 20)]
    expected = {(r["sample_key"], c["condition_id"]) for r in rows for c in conditions}
    write_json(working / "data_audit.json", audit)
    with (working / "cohort.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    assets = audit_model(Path(config["model"]["directory"]), config["model"]["revision"], config["model"]["repo_id"])
    write_json(working / "model_audit.json", assets)
    print(f"{now()} loading model", flush=True)
    backend_class = {"mllama": LlamaBackend, "ecg_r1_image": ECG_R1_ImageBackend}[config["model"]["backend"]]
    backend = backend_class(config["model"])
    write_json(working / "device_placement.json", {
        "strategy": strategy, "cuda_visible_devices": visible,
        "hf_device_map": {name: str(value) for name, value in getattr(backend.model, "hf_device_map", {}).items()},
    })
    if hasattr(backend, "loading"):
        write_json(working / "checkpoint_loading.json", backend.loading)
    renderer, render_audit = PulseECGTensorRenderer.from_ecg_image_kit(
        config["paths"]["toolkit_dir"], device="cuda:0", tmp_root=config["paths"]["working_root"],
    )
    write_json(working / "renderer.json", render_audit)
    predictions_path = working / "predictions.jsonl"
    done: dict[tuple[str, str], dict] = {}
    if predictions_path.exists():
        with predictions_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["sample_key"], row["condition_id"])
                if key in done or key not in expected:
                    raise ValueError("duplicate or foreign RAM prediction")
                done[key] = row
    initial_count = len(done)
    inference_start = time.monotonic()
    raw_root = Path(config["paths"]["raw_root"])
    stop_at = datetime.fromisoformat(runtime["inference_deadline_utc"]).timestamp()
    progress_path = working / "progress.json"
    batch_size = runtime["batch_size"]
    with ThreadPoolExecutor(max_workers=runtime["loader_workers"]) as pool, predictions_path.open("a", buffering=1) as handle:
        for offset in range(0, len(rows), batch_size):
            samples = rows[offset:offset + batch_size]
            if all((s["sample_key"], c["condition_id"]) in done for s in samples for c in conditions):
                continue
            if time.time() >= stop_at:
                raise TimeoutError("inference timebox reached; incomplete RAM progress retained")
            t0 = time.monotonic()
            waveforms = list(pool.map(lambda s: native500_waveform(s, raw_root)[0], samples))
            clean_hashes = [hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest() for w in waveforms]
            clean = torch.from_numpy(np.stack(waveforms)).to("cuda")
            for condition in conditions:
                pending = [i for i, s in enumerate(samples) if (s["sample_key"], condition["condition_id"]) not in done]
                if not pending:
                    continue
                selected = [samples[i] for i in pending]
                views = torch.cat([
                    corrupt_native500_one(clean[i:i+1], source_hash=samples[i]["hash_id"],
                                          condition=condition, profile=profile,
                                          base_seed=baseline_protocol["corruption"]["seed_config"]["base_seed"])
                    for i in pending
                ])
                view_hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in views.cpu().numpy()]
                rgb = renderer.render(views)
                responses, lengths = backend.generate(rgb, max_new_tokens=runtime["max_new_tokens"])
                for sample, response, length, source_index, view_hash in zip(selected, responses, lengths, pending, view_hashes, strict=True):
                    row = {"schema_version": 1, "sample_key": sample["sample_key"],
                           "record_id": sample["record_id"], "hash_id": sample["hash_id"],
                           "logical_center": sample["logical_center"], **condition,
                           "true_labels": sample["label_names"], "response": response,
                           "clean_waveform_sha256": clean_hashes[source_index], "input_waveform_sha256": view_hash,
                           "generated_tokens": int(length), "hit_token_limit": length >= runtime["max_new_tokens"],
                           **parse_response(response)}
                    handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                    done[(sample["sample_key"], condition["condition_id"])] = row
                del rgb, views
            elapsed = time.monotonic() - inference_start
            speed = (len(done) - initial_count) / max(elapsed, 1e-9)
            progress = {"status": "running", "updated_at": now(), "pid": os.getpid(),
                        "gpu": visible, "completed": len(done), "expected": len(expected),
                        "images_per_second": speed, "elapsed_inference_seconds": elapsed,
                        "last_record_block_seconds": time.monotonic() - t0,
                        "eta_seconds": (len(expected) - len(done)) / max(speed, 1e-9),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "parse_failures": sum(not r["parse_valid"] for r in done.values()),
                        "token_limit_hits": sum(r["hit_token_limit"] for r in done.values())}
            write_json(progress_path, progress)
            print(json.dumps(progress), flush=True)
    if set(done) != expected:
        raise ValueError("incomplete prediction grid")
    elapsed = time.monotonic() - inference_start
    progress = {"status": "complete", "updated_at": now(), "gpu": visible,
                "completed": len(done), "expected": len(expected),
                "resumed_from_predictions": initial_count, "new_predictions_this_process": len(done) - initial_count,
                "images_per_second": (len(done) - initial_count) / max(elapsed, 1e-9),
                "inference_seconds": elapsed, "total_seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "parse_failures": sum(not r["parse_valid"] for r in done.values()),
                "strict_list_failures": sum(not r["strict_label_list"] for r in done.values()),
                "token_limit_hits": sum(r["hit_token_limit"] for r in done.values())}
    write_json(progress_path, progress)
    write_json(working / "gpu_memory.json", [
        {"logical_device": index, "physical_device": visible.split(",")[index],
         "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
         "peak_reserved_bytes": torch.cuda.max_memory_reserved(index)} for index in range(device_count)
    ])
    environment = {name: importlib.metadata.version(name) for name in
                   ("torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "numpy", "Pillow")}
    write_json(working / "environment.json", {
        "python": sys.executable, "python_version": sys.version, "packages": environment,
        "all_visible_packages": {name: importlib.metadata.version(name) for name in sorted({
            distribution.metadata["Name"] for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")})},
        "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(), "torch_cpu_threads": torch.get_num_threads(),
        "environment_note": "isolated_user_venv_with_project_system_site_packages",
    })
    # Only completed shard artifacts are persisted to SSD, as requested.
    for name in ("identity.json", "cohort.jsonl", "data_audit.json", "model_audit.json",
                 "renderer.json", "predictions.jsonl", "progress.json", "environment.json",
                 "device_placement.json", "gpu_memory.json"):
        destination = output_dir / name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite finalized artifact: {destination}")
        shutil.copyfile(working / name, destination)
    if (working / "checkpoint_loading.json").exists():
        shutil.copyfile(working / "checkpoint_loading.json", output_dir / "checkpoint_loading.json")
    for path in parity_files:
        shutil.copyfile(path, output_dir / path.name)
    shutil.copytree(source_snapshot, output_dir / "source_snapshot")
    result = {"schema_version": 1, "artifact_type": "ecg_image_evaluation_result", "status": "complete",
              "model": {**config["model"], "assets": assets, "prompt": backend.prompt,
                        "generation": {"do_sample": False, "max_new_tokens": runtime["max_new_tokens"],
                                       "batch_size": batch_size, "padding_side": "left", "use_cache": True},
                        "precision_note": ("quantized inference is not a full-precision replication"
                                           if config["model"]["precision"].startswith("nf4") else f"unquantized_{config['model']['precision']}")},
              "protocol": {"mapping_hash": "555ec85d5b51", "class_order": list(CLASS_ORDER),
                           "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                           "metric_view": "drop_all_zero", "k500_ref_excluded": True,
                           "stage": config["stage"], "cohort_sha256": sha256_file(output_dir / "cohort.jsonl"),
                           "baseline_protocol_sha256": audit["baseline_protocol_sha256"],
                           "implementation_sha256": implementation,
                           "source_snapshot": True,
                           "inference_packages": environment,
                           "conditions": conditions, "rank": rank, "world_size": world,
                           "backend_image_conversion": "shared_GPU_float32_page_to_RGB_uint8_then_official_processor",
                           "environment_adapter_parity": {path.name: sha256_file(path) for path in parity_files},
                           "evidence_level": "development_frozen_inference_not_paper_final"},
              "expected_predictions": len(expected), "completed_predictions": len(done),
              "prediction_artifact": {"path": "predictions.jsonl", "sha256": sha256_file(output_dir / "predictions.jsonl")},
              "performance": progress}
    write_json(output_dir / "evaluation_result.json", result)
    print(f"{now()} complete {output_dir}", flush=True)
