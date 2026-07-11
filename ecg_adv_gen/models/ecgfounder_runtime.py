"""ECGFounder checkpoint/model runtime helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ecg_adv_gen.models.ecgfounder_heads import (
    FeatureAdapterHead,
    OperatorConditionedLogitAdapter,
    ResidualAdapterHead,
)


def fullft_model_path(run_dir: str | Path) -> Path:
    """Return a recorded selection, otherwise preserve legacy last-model semantics."""

    run_dir = Path(run_dir)
    for record_name in ("eval_result.json", "selection.json"):
        record_path = run_dir / record_name
        if not record_path.exists():
            continue
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        selected = payload.get("selected_checkpoint")
        if not selected and isinstance(payload.get("selection"), dict):
            selected = payload["selection"].get("checkpoint")
        if not selected:
            continue
        selected_name = str(selected)
        if Path(selected_name).name != selected_name:
            raise ValueError(f"selected checkpoint must be a run-local filename: {selected_name!r}")
        selected_path = run_dir / selected_name
        if not selected_path.exists():
            raise FileNotFoundError(f"recorded selected checkpoint does not exist: {selected_path}")
        return selected_path
    last_path = run_dir / "last_model.pt"
    if last_path.exists():
        return last_path
    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        return best_path
    return best_path


def detect_eval_mode(run_dir: str | Path) -> str:
    """Return how an ECGFounder run should be evaluated."""

    run_dir = Path(run_dir)
    if fullft_model_path(run_dir).exists():
        return "fullft_model"
    if (run_dir / "best_head.pt").exists():
        return "feature_head"
    raise FileNotFoundError(
        f"expected best_head.pt, last_model.pt, or best_model.pt under {run_dir}"
    )


def infer_feature_dim(head_state: dict[str, torch.Tensor]) -> int:
    for key in ("weight", "base_head.weight"):
        if key in head_state:
            return int(head_state[key].shape[1])
    raise RuntimeError("cannot infer ECGFounder feature dim from best_head.pt")


def build_ecgfounder_feature_model(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    from net1d import Net1D

    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
        return_features=True,
    )
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)


def build_ecgfounder_head(
    run_dir: str | Path,
    result: dict[str, Any],
    device: torch.device,
    *,
    num_classes: int,
) -> nn.Module:
    head_path = Path(run_dir) / "best_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(head_path)
    state = torch.load(head_path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported head checkpoint payload: {head_path}")
    feature_dim = infer_feature_dim(state)
    cfg = dict(result.get("config") or {})
    head_type = str(result.get("head_type") or cfg.get("head_type") or "linear")
    base_head = nn.Linear(feature_dim, int(num_classes)).to(device)
    if head_type == "linear":
        head: nn.Module = base_head
    elif head_type == "residual_adapter":
        head = ResidualAdapterHead(
            base_head=base_head,
            hidden_dim=int(cfg.get("adapter_hidden", 128)),
            dropout=float(cfg.get("adapter_dropout", 0.0)),
            scale=float(cfg.get("adapter_scale", 1.0)),
            freeze_base=bool(cfg.get("freeze_base_head", False)),
        ).to(device)
    elif head_type == "feature_adapter":
        head = FeatureAdapterHead(
            base_head=base_head,
            hidden_dim=int(cfg.get("adapter_hidden", 128)),
            dropout=float(cfg.get("adapter_dropout", 0.0)),
            scale=float(cfg.get("adapter_scale", 1.0)),
            freeze_base=bool(cfg.get("freeze_base_head", False)),
        ).to(device)
    else:
        raise ValueError(f"unsupported ECGFounder head_type={head_type!r}")
    head.load_state_dict(state)
    head.eval()
    return head


def extract_fullft_state_dict(payload: Any, model_path: str | Path) -> dict[str, Any]:
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported fullFT checkpoint payload: {Path(model_path)}")
    return {str(k).removeprefix("_orig_mod."): v for k, v in state.items()}


def build_ecgfounder_fullft_model(
    run_dir: str | Path,
    checkpoint: str | Path,
    device: torch.device,
    *,
    num_classes: int,
) -> nn.Module:
    from finetune_model import ft_12lead_ECGFounder

    model = ft_12lead_ECGFounder(device, str(checkpoint), int(num_classes), linear_prob=False)
    model_path = fullft_model_path(run_dir)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    payload = torch.load(model_path, map_location=device)
    model.load_state_dict(extract_fullft_state_dict(payload, model_path))
    model.eval()
    model = model.to(device)
    if isinstance(payload, dict) and "op_adapter_state_dict" in payload:
        cfg = dict(payload.get("op_adapter_config") or {})
        if not cfg:
            raise RuntimeError(f"{model_path} has op_adapter_state_dict but no op_adapter_config")
        op_adapter = OperatorConditionedLogitAdapter(
            feature_dim=int(cfg["feature_dim"]),
            num_classes=int(cfg["num_classes"]),
            op_names=list(cfg["op_names"]),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            dropout=float(cfg.get("dropout", 0.0)),
            scale=float(cfg.get("scale", 1.0)),
        ).to(device)
        op_adapter.load_state_dict(payload["op_adapter_state_dict"])
        op_adapter.eval()
        return OperatorConditionedFullFTModel(model, op_adapter).to(device)
    return model


def _forward_logits_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model, "return_features"):
        raise RuntimeError("op-conditioned ECGFounder eval requires model.return_features support")
    previous = bool(getattr(model, "return_features"))
    try:
        setattr(model, "return_features", True)
        output = model(x)
    finally:
        setattr(model, "return_features", previous)
    if not (isinstance(output, tuple) and len(output) == 2):
        raise RuntimeError("op-conditioned ECGFounder eval expected model(x) to return (logits, features)")
    logits, features = output
    return logits, features


class OperatorConditionedFullFTModel(nn.Module):
    def __init__(self, base_model: nn.Module, op_adapter: OperatorConditionedLogitAdapter) -> None:
        super().__init__()
        self.base_model = base_model
        self.op_adapter = op_adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def forward_with_operator(self, x: torch.Tensor, operator_name: str) -> torch.Tensor:
        logits, features = _forward_logits_features(self.base_model, x)
        op_ids = self.op_adapter.op_ids_from_names(
            [str(operator_name)] * int(x.shape[0]),
            device=x.device,
        )
        return logits + self.op_adapter(features, op_ids)


__all__ = [
    "OperatorConditionedFullFTModel",
    "build_ecgfounder_feature_model",
    "build_ecgfounder_fullft_model",
    "build_ecgfounder_head",
    "detect_eval_mode",
    "extract_fullft_state_dict",
    "fullft_model_path",
    "infer_feature_dim",
]
