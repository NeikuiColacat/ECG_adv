from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
DEEPECG_NB = Path("/root/autodl-tmp/models/DeepECG/notebooks")
for p in [REPO, DEEPECG_NB, REPO / "model" / "DeepECG" / "notebooks"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from apps.streamlit_ecg_demo.services.preprocessing import CLASS_NAMES, classifier_input  # noqa: E402


DEFAULT_CKPT = (
    "/root/autodl-tmp/triple_labels/"
    "super5_minresample_full10_perglobal_20260503/best_model.pt"
)
DEFAULT_ONNX = "/root/autodl-tmp/streamlit_ecg_demo/models/efficientnetv2_super5.onnx"


def build_efficientnet_super5() -> EfficientNet1DV2:
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
            "backend": "pytorch",
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


def load_classifier_backend(
    backend: str,
    ckpt_path: str = DEFAULT_CKPT,
    device: str = "cuda",
):
    if backend == "pytorch":
        return PyTorchClassifierBackend(ckpt_path=ckpt_path, device=device)
    if backend == "onnxruntime":
        return ONNXRuntimeClassifierBackend(onnx_path=DEFAULT_ONNX)
    if backend == "tensorrt":
        return UnavailableBackend(
            f"{backend} runtime is planned in scripts/deploy; use PyTorch fallback for now"
        )
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
