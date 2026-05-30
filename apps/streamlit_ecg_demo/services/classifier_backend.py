from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
from apps.streamlit_ecg_demo.services.paths import APP_DATA_ROOT, DATA_ROOT, MODEL_ROOT

DEEPECG_NB = MODEL_ROOT / "DeepECG/notebooks"
for p in [REPO, DEEPECG_NB, REPO / "model" / "DeepECG" / "notebooks"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from apps.streamlit_ecg_demo.services.preprocessing import CLASS_NAMES, classifier_input  # noqa: E402


DEFAULT_CKPT = (
    str(DATA_ROOT / "triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt")
)
DEFAULT_ONNX = str(APP_DATA_ROOT / "models/efficientnetv2_super5.onnx")
DEFAULT_TRT_ENGINE = str(APP_DATA_ROOT / "models/efficientnetv2_super5_fp16.engine")
DEFAULT_TRT_VENDOR = str(APP_DATA_ROOT / "python_pkgs/tensorrt_cu12")


def _add_tensorrt_vendor_path() -> None:
    vendor = Path(os.environ.get("TENSORRT_VENDOR_PATH", DEFAULT_TRT_VENDOR))
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def _import_tensorrt():
    try:
        import tensorrt as trt
    except ModuleNotFoundError:
        import tensorrt_bindings as trt
    return trt


def _load_efficientnet1dv2() -> Any:
    try:
        from EfficientNetv2 import EfficientNet1DV2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DeepECG EfficientNet implementation is unavailable. Restore external model repos with "
            "`bash scripts/bootstrap_model_repos.sh`, or restore the packaged artifacts before "
            "running PyTorch inference."
        ) from exc
    return EfficientNet1DV2


def build_efficientnet_super5() -> torch.nn.Module:
    EfficientNet1DV2 = _load_efficientnet1dv2()
    return EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=len(CLASS_NAMES),
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    )


class PyTorchClassifierBackend:
    def __init__(self, ckpt_path: str = DEFAULT_CKPT, device: str = "cuda"):
        self.ckpt_path = str(ckpt_path)
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = build_efficientnet_super5()
        state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {str(k).removeprefix("_orig_mod."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, signal) -> dict:
        x_np = classifier_input(np.asarray(signal, dtype=np.float32), target_len=1000)
        x = torch.from_numpy(x_np).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logits_np = logits.detach().cpu().numpy()[0]
        probs_np = probs.detach().cpu().numpy()[0]
        return {
            "backend": "torch",
            "device": str(self.device),
            "checkpoint": self.ckpt_path,
            "latency_ms": float(latency_ms),
            "class_names": list(CLASS_NAMES),
            "logits": logits_np.tolist(),
            "probabilities": probs_np.tolist(),
            "predicted_labels": [
                CLASS_NAMES[i] for i, p in enumerate(probs_np) if float(p) >= 0.5
            ],
        }


class UnavailableBackend:
    def __init__(self, reason: str):
        self.reason = reason

    def predict(self, signal) -> dict:
        raise RuntimeError(self.reason)


def _torch_dtype_from_trt(dtype):
    trt = _import_tensorrt()

    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int8:
        return torch.int8
    raise TypeError(f"unsupported TensorRT dtype: {dtype}")


class TensorRTClassifierBackend:
    def __init__(self, engine_path: str = DEFAULT_TRT_ENGINE):
        _add_tensorrt_vendor_path()
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires CUDA")
        self.engine_path = str(engine_path)
        if not Path(self.engine_path).exists():
            raise FileNotFoundError(
                f"missing TensorRT engine: {self.engine_path}; run scripts/deploy/build_tensorrt_engine.py"
            )

        trt = _import_tensorrt()

        self.trt = trt
        self.device = torch.device("cuda")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(self.engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if not inputs or not outputs:
            raise RuntimeError(f"TensorRT engine has no usable inputs/outputs: {names}")
        self.input_name = "ecg" if "ecg" in inputs else inputs[0]
        self.output_name = "logits" if "logits" in outputs else outputs[0]
        self.predict_batch_array(np.zeros((1, 12, 1000), dtype=np.float32))

    def predict_batch_array(self, x_np: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device).contiguous()
        if x.ndim != 3 or x.shape[1:] != (12, 1000):
            raise ValueError(f"expected input shape (B, 12, 1000), got {tuple(x.shape)}")
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        out_shape = tuple(int(v) for v in self.context.get_tensor_shape(self.output_name))
        out_dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(self.output_name))
        y = torch.empty(out_shape, dtype=out_dtype, device=self.device)
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(y.data_ptr()))
        stream = torch.cuda.current_stream()
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return y.detach().float().cpu().numpy()

    def predict(self, signal) -> dict:
        x_np = classifier_input(np.asarray(signal, dtype=np.float32), target_len=1000)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits_np = self.predict_batch_array(x_np)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs_np = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50, 50)))
        return {
            "backend": "tensorrt",
            "device": "cuda",
            "checkpoint": self.engine_path,
            "latency_ms": float(latency_ms),
            "class_names": list(CLASS_NAMES),
            "logits": logits_np.astype(float).tolist(),
            "probabilities": probs_np.astype(float).tolist(),
            "predicted_labels": [
                CLASS_NAMES[i] for i, p in enumerate(probs_np) if float(p) >= 0.5
            ],
        }


def load_classifier_backend(
    backend: str,
    ckpt_path: str = DEFAULT_CKPT,
    device: str = "cuda",
):
    backend = {"pytorch": "torch", "torch": "torch", "tensorrt": "tensorrt"}.get(
        str(backend).strip().lower(),
        str(backend).strip().lower(),
    )
    if backend == "torch":
        return PyTorchClassifierBackend(ckpt_path=ckpt_path, device=device)
    if backend == "onnxruntime":
        onnx_path = ckpt_path if str(ckpt_path).endswith(".onnx") else DEFAULT_ONNX
        return ONNXRuntimeClassifierBackend(onnx_path=onnx_path)
    if backend == "tensorrt":
        engine_path = ckpt_path if str(ckpt_path).endswith(".engine") else DEFAULT_TRT_ENGINE
        return TensorRTClassifierBackend(engine_path=engine_path)
    raise ValueError(f"unknown backend={backend!r}")


class ONNXRuntimeClassifierBackend:
    def __init__(self, onnx_path: str = DEFAULT_ONNX):
        self.onnx_path = str(onnx_path)
        if not Path(self.onnx_path).exists():
            raise FileNotFoundError(
                f"missing ONNX model: {self.onnx_path}; run scripts/deploy/export_efficientnetv2_onnx.py"
            )
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, signal) -> dict:
        x_np = classifier_input(np.asarray(signal, dtype=np.float32), target_len=1000)
        t0 = time.perf_counter()
        logits_np = self.session.run(None, {self.input_name: x_np})[0][0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs_np = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50, 50)))
        return {
            "backend": "onnxruntime",
            "device": self.provider,
            "checkpoint": self.onnx_path,
            "latency_ms": float(latency_ms),
            "class_names": list(CLASS_NAMES),
            "logits": logits_np.astype(float).tolist(),
            "probabilities": probs_np.astype(float).tolist(),
            "predicted_labels": [
                CLASS_NAMES[i] for i, p in enumerate(probs_np) if float(p) >= 0.5
            ],
        }
