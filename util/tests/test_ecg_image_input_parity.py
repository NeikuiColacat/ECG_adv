"""Opt-in CUDA input parity check across isolated image-model environments."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from util.ecg_image_renderer import PulseECGTensorRenderer
from util.evaluation.ecg_image_data import audit_inputs, corrupt_native500_one, native500_waveform


@pytest.mark.parametrize("shape,size", [((2, 3, 41, 67), (17, 31)), ((1, 3, 17, 29), (32, 48)),
                                       ((2, 3, 37, 49), (37, 19)), ((1, 3, 61, 47), (23, 47)),
                                       ((1, 3, 7, 9), (7, 9)), ((1, 3, 101, 133), (21, 27))])
def test_tensor_uint8_resize_matches_cpu_reference(shape, size):
    from util.evaluation.ecg_image_processor import resize_uint8_cpu_equivalent

    data = torch.randint(0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(42))
    expected = torch.nn.functional.interpolate(data, size=size, mode="bicubic", align_corners=False, antialias=True)
    actual = resize_uint8_cpu_equivalent(data, size)
    assert torch.equal(expected, actual), f"uint8 mismatch count={int(torch.count_nonzero(expected != actual))}"


@pytest.mark.skipif(os.environ.get("ECG_R1_RESIZE_CUDA_PARITY") != "1", reason="explicit full-size CUDA resize parity")
def test_tensor_uint8_resize_cuda_full_size_random_and_extremes():
    from util.evaluation.ecg_image_processor import resize_uint8_cpu_equivalent

    assert os.environ.get("CUDA_VISIBLE_DEVICES") and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    torch.set_num_threads(1)
    data = torch.randint(0, 256, (2, 3, 1700, 2200), dtype=torch.uint8, generator=torch.Generator().manual_seed(20260907))
    data[1, :, :, ::2] = 0
    data[1, :, :, 1::2] = 255
    expected = torch.nn.functional.interpolate(data, size=(768, 992), mode="bicubic", align_corners=False, antialias=True)
    actual = resize_uint8_cpu_equivalent(data.cuda(), (768, 992)).cpu()
    assert torch.equal(expected, actual), f"full-size uint8 mismatch={int(torch.count_nonzero(expected != actual))}"
    print(f"FULL_SIZE_CUDA_RESIZE_PARITY elements={actual.numel()} max_abs_error=0", flush=True)


@pytest.mark.skipif(os.environ.get("ECG_R1_PROCESSOR_PROBE") != "1", reason="explicit R1 GPU processor diagnostic")
def test_r1_processor_gpu_tensor_numeric_probe():
    """Measure the unchanged official CPU/PIL path versus its CUDA tensor path.

    Passing means the diagnostic completed, not that a nonzero error is approved
    for inference or that old predictions may be imported into a new protocol.
    No model weights or correctness scores are loaded.
    """
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from util.evaluation.ecg_image_processor import resize_uint8_cpu_equivalent

    assert os.environ.get("CUDA_VISIBLE_DEVICES") and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/eval/ecg_image_r1_elastic_validation_retry1.yaml"
    config = yaml.safe_load(config_path.read_text())
    rows, conditions, profile, protocol, audit = audit_inputs(config, config_path, root / "configs")
    processor = AutoProcessor.from_pretrained(config["model"]["directory"], local_files_only=True)
    processor.image_processor.size = {"shortest_edge": 4 * 32**2, "longest_edge": 768 * 32**2}
    exact_resize = os.environ.get("ECG_R1_PROCESSOR_PROBE_EXACT") == "1"
    target_size = smart_resize(1700, 2200, factor=32, min_pixels=4 * 32**2, max_pixels=768 * 32**2)
    renderer, render_audit = PulseECGTensorRenderer.from_ecg_image_kit(
        config["paths"]["toolkit_dir"], device="cuda:0", tmp_root=config["paths"]["working_root"])
    selected = rows[:8]
    clean = torch.from_numpy(np.stack([
        native500_waveform(row, Path(config["paths"]["raw_root"]))[0] for row in selected])).cuda()
    checks = []
    for condition in conditions:
        views = torch.cat([corrupt_native500_one(
            clean[i:i+1], source_hash=row["hash_id"], condition=condition,
            profile=profile, base_seed=protocol["corruption"]["seed_config"]["base_seed"])
            for i, row in enumerate(selected)])
        rgb = renderer.render(views)
        pixels = rgb.mul(255).round().clamp(0, 255).to(torch.uint8)
        torch.cuda.synchronize()
        start = time.perf_counter()
        host = pixels.permute(0, 2, 3, 1).cpu().numpy()
        images = [Image.fromarray(page) for page in host]
        reference = processor.image_processor(images=images, return_tensors="pt")
        cpu_seconds = time.perf_counter() - start
        start = time.perf_counter()
        cpu_tensor = processor.image_processor(images=pixels.cpu(), input_data_format="channels_first",
                                               device="cpu", return_tensors="pt")
        cpu_tensor_seconds = time.perf_counter() - start
        assert torch.equal(reference["image_grid_thw"], cpu_tensor["image_grid_thw"])
        assert torch.equal(reference["pixel_values"], cpu_tensor["pixel_values"])
        start = time.perf_counter()
        resized = resize_uint8_cpu_equivalent(pixels, target_size) if exact_resize else pixels
        candidate = processor.image_processor(images=resized, input_data_format="channels_first",
                                              do_resize=not exact_resize, device="cuda:0", return_tensors="pt")
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - start
        assert torch.equal(reference["image_grid_thw"], candidate["image_grid_thw"].cpu())
        before, after = reference["pixel_values"], candidate["pixel_values"].cpu()
        assert before.shape == after.shape
        difference = (before - after).abs()
        bf16_before, bf16_after = before.to(torch.bfloat16), after.to(torch.bfloat16)
        checks.append({"condition_id": condition["condition_id"], "images": len(selected),
                       "elements": before.numel(), "float32_max_abs_error": difference.max().item(),
                       "float32_mean_abs_error": difference.mean().item(),
                       "float32_mismatched_elements": int(torch.count_nonzero(before != after)),
                       "bfloat16_mismatched_elements": int(torch.count_nonzero(bf16_before != bf16_after)),
                       "bfloat16_max_abs_error": (bf16_before.float()-bf16_after.float()).abs().max().item(),
                       "cpu_pil_processor_seconds": cpu_seconds, "gpu_tensor_processor_seconds": gpu_seconds,
                       "cpu_tensor_processor_seconds": cpu_tensor_seconds, "cpu_tensor_bitwise_equal": True})
        print(json.dumps(checks[-1]), flush=True)
        del rgb, views, pixels, resized, host, images, reference, cpu_tensor, candidate, before, after, difference, bf16_before, bf16_after
    destination = Path(os.environ["ECG_R1_PROCESSOR_PROBE_OUTPUT"]).resolve()
    destination.relative_to(Path("/home/linbinhao/ECG_adv_data/runs/ecg_r1_full_elastic_20260907"))
    payload = {"status": "measured_not_approved", "scope": "8 reserved smoke records, all21 conditions, image processor only",
               "torch": torch.__version__, "processor": type(processor.image_processor).__name__,
               "resize_mode": "cpu_equivalent_fixed_point" if exact_resize else "native_fast_cuda",
               "helper_sha256": hashlib.sha256((root / "util/evaluation/ecg_image_processor.py").read_bytes()).hexdigest(),
               "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "renderer": render_audit,
               "data_audit": audit, "sample_keys": [row["sample_key"] for row in selected], "checks": checks,
               "all_bfloat16_inputs_bitwise_equal": all(row["bfloat16_mismatched_elements"] == 0 for row in checks),
               "all_float32_inputs_bitwise_equal": all(row["float32_mismatched_elements"] == 0 for row in checks)}
    with destination.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"R1_PROCESSOR_PROBE {destination}", flush=True)


@pytest.mark.skipif(os.environ.get("ECG_IMAGE_PARITY_RUN") != "1", reason="explicit isolated CUDA input parity opt-in")
def test_frozen_native500_cuda_input_parity():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/eval/ecg_image_llama_smoke.yaml"
    config = yaml.safe_load(config_path.read_text())
    rows, conditions, profile, protocol, _ = audit_inputs(config, config_path, root / "configs")
    ram = Path("/dev/shm/ecg_image_llm_10h_20260907")
    ram.mkdir(exist_ok=True)
    renderer, _ = PulseECGTensorRenderer.from_ecg_image_kit(config["paths"]["toolkit_dir"], device="cuda:0", tmp_root=ram)
    fingerprints = []
    for sample in rows[:4]:
        waveform, _ = native500_waveform(sample, Path(config["paths"]["raw_root"]))
        clean = torch.from_numpy(waveform).unsqueeze(0).cuda()
        for condition in conditions:
            view = corrupt_native500_one(clean, source_hash=sample["hash_id"], condition=condition,
                                         profile=profile, base_seed=protocol["corruption"]["seed_config"]["base_seed"])
            rgb = renderer.render(view)
            pixels = rgb.mul(255).round().clamp(0, 255).to(torch.uint8)
            fingerprints.append({"sample_key": sample["sample_key"], "condition_id": condition["condition_id"],
                                 "waveform_sha256": hashlib.sha256(view.cpu().numpy().tobytes()).hexdigest(),
                                 "rgb_float32_sha256": hashlib.sha256(rgb.cpu().numpy().tobytes()).hexdigest(),
                                 "rgb_uint8_sha256": hashlib.sha256(pixels.cpu().numpy().tobytes()).hexdigest()})
            del rgb, pixels, view
    output = ram / f"input_parity_torch_{torch.__version__.replace('+', '_')}.json"
    payload = {"torch": torch.__version__, "status": "generated", "fingerprints": fingerprints}
    reference_path = os.environ.get("ECG_IMAGE_PARITY_REFERENCE")
    if reference_path:
        reference = json.loads(Path(reference_path).read_text())
        mismatches = [(a["sample_key"], a["condition_id"]) for a, b in zip(reference["fingerprints"], fingerprints, strict=True) if a != b]
        payload.update({"reference_torch": reference["torch"], "mismatches": mismatches,
                        "status": "passed" if not mismatches else "failed"})
    output.write_text(json.dumps(payload, indent=2) + "\n")
    assert not payload.get("mismatches"), "input parity differs; do not reuse PULSE blindly"
    print(f"INPUT_PARITY {output} count={len(fingerprints)} status={payload['status']}")


@pytest.mark.skipif(os.environ.get("ECG_IMAGE_PARITY_RUN") != "1", reason="explicit isolated CUDA batch parity opt-in")
def test_frozen_native500_renderer_batch_parity():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/eval/ecg_image_llama_smoke.yaml"
    config = yaml.safe_load(config_path.read_text())
    rows, conditions, profile, protocol, _ = audit_inputs(config, config_path, root / "configs")
    ram = Path(config["paths"]["working_root"])
    renderer, _ = PulseECGTensorRenderer.from_ecg_image_kit(config["paths"]["toolkit_dir"], device="cuda:0", tmp_root=ram)
    selected = rows[:8]
    clean = torch.from_numpy(np.stack([native500_waveform(r, Path(config["paths"]["raw_root"]))[0] for r in selected])).cuda()
    checks = []
    for index in (0, 1, 11, 20):
        condition = conditions[index]
        views = torch.cat([corrupt_native500_one(clean[i:i+1], source_hash=r["hash_id"], condition=condition,
                          profile=profile, base_seed=protocol["corruption"]["seed_config"]["base_seed"])
                           for i, r in enumerate(selected)])
        reference = torch.cat([renderer.render(view.unsqueeze(0)) for view in views])
        for batch in (2, 4, 8):
            candidate = torch.cat([renderer.render(views[i:i+batch]) for i in range(0, len(selected), batch)])
            difference = float((candidate - reference).abs().max())
            checks.append({"condition": condition["condition_id"], "batch": batch, "max_abs_error": difference})
            assert difference == 0
            del candidate
        del reference, views
    output = ram / f"renderer_batch_parity_torch_{torch.__version__.replace('+', '_')}.json"
    output.write_text(json.dumps({"status": "passed", "torch": torch.__version__, "checks": checks}, indent=2) + "\n")
    print(f"RENDERER_BATCH_PARITY {output} checks={len(checks)}")
