from __future__ import annotations

import re
from pathlib import Path

from apps.streamlit_ecg_demo.services.paths import APP_DATA_ROOT, DATA_ROOT, PYTHON_EXECUTABLE

HOSPITAL_ROOT = APP_DATA_ROOT / "hospitals"
DEFAULT_CLASSIC_SAMPLE_PATH = APP_DATA_ROOT / "demo" / "classic_ecg_samples.npz"
DEFAULT_TRAINING_SIGNAL_PATH = APP_DATA_ROOT / "demo" / "default_hospital_signals.npy"
DEFAULT_TRAINING_LABEL_PATH = APP_DATA_ROOT / "demo" / "default_hospital_labels.csv"
DEFAULT_SYNTH_PATHS = [
    DEFAULT_CLASSIC_SAMPLE_PATH,
    DATA_ROOT / "ecgtwin_prompt_token_super5"
    / "effectiveness_pilot_v42_task1gate_ningbo_token_scale_large_20260504"
    / "target_token_s05/ningbo/gated/gated_samples.npz",
    DATA_ROOT / "ecgtwin_prompt_token_super5"
    / "effectiveness_pilot_v42_task1gate_ningbo_no_token_large_20260504"
    / "no_token/ningbo/gated/gated_samples.npz",
    DATA_ROOT / "ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced"
    / "ningbo/gated/gated_samples.npz",
]
DEFAULT_PROMPT_CACHE_ROOT = str(DATA_ROOT / "ecgtwin_prompt_token_super5/cache_v1")
DEFAULT_PROMPT_BANK = f"{DEFAULT_PROMPT_CACHE_ROOT}/text_prompt_bank.pt"
DEFAULT_PYTHON = PYTHON_EXECUTABLE


def sanitize_project_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    text = text.strip("._-")
    if not text:
        raise ValueError("project id cannot be empty")
    return text[:80]


def project_dir(project_id: str) -> Path:
    return HOSPITAL_ROOT / sanitize_project_id(project_id)


def ensure_project(project_id: str) -> Path:
    root = project_dir(project_id)
    for name in [
        "uploads",
        "preprocessed",
        "cache",
        "center_tokens",
        "synthetic_pools",
        "classifier_runs",
        "exports",
        "reports",
        "jobs",
    ]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def list_projects() -> list[str]:
    HOSPITAL_ROOT.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in HOSPITAL_ROOT.iterdir() if p.is_dir())


def canonical_dataset_path(root: Path) -> Path:
    return root / "preprocessed" / "hospital_dataset.npz"


def canonical_audit_path(root: Path) -> Path:
    return root / "reports" / "dataset_audit.json"


def prompt_cache_root(root: Path) -> Path:
    return root / "cache" / "prompt_token_cache"


def _existing_unique(paths: list[Path]) -> list[Path]:
    out = []
    seen = set()
    for path in paths:
        if path.exists() and str(path) not in seen:
            out.append(path)
            seen.add(str(path))
    return out


def list_token_banks(root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if root is not None:
        candidates.extend(sorted((root / "center_tokens").glob("**/prompt_token_bank*.pt")))
    candidates.extend(sorted((DATA_ROOT / "ecgtwin_prompt_token_super5").glob("**/prompt_token_bank*.pt")))
    seen = set()
    out = []
    for path in candidates:
        key = str(path)
        if key not in seen and path.exists():
            seen.add(key)
            out.append(path)
    return out


def list_synthetic_pools(root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if root is not None:
        candidates.extend(sorted((root / "synthetic_pools").glob("**/gated_samples.npz")))
        candidates.extend(sorted((root / "synthetic_pools").glob("**/samples.npz")))
    candidates.extend(Path(p) for p in DEFAULT_SYNTH_PATHS)
    return _existing_unique(candidates)


def list_ecg_sample_pools(root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if root is not None:
        candidates.append(canonical_dataset_path(root))
        candidates.extend(sorted((root / "synthetic_pools").glob("**/gated_samples.npz")))
        candidates.extend(sorted((root / "synthetic_pools").glob("**/samples.npz")))
    candidates.extend(Path(p) for p in DEFAULT_SYNTH_PATHS)
    return _existing_unique(candidates)


def list_training_signal_files(root: Path | None = None) -> list[Path]:
    candidates: list[Path] = [DEFAULT_TRAINING_SIGNAL_PATH, DEFAULT_CLASSIC_SAMPLE_PATH]
    if root is not None:
        candidates.append(canonical_dataset_path(root))
    return _existing_unique(candidates)


def list_training_label_files(root: Path | None = None) -> list[Path]:
    candidates: list[Path] = [DEFAULT_TRAINING_LABEL_PATH]
    if root is not None:
        candidates.extend(sorted((root / "uploads").glob("**/*.csv")))
    return _existing_unique(candidates)


def list_classifier_artifacts(root: Path | None = None) -> list[Path]:
    candidates = [
        DATA_ROOT / "triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt",
    ]
    if root is not None:
        candidates.extend(sorted((root / "classifier_runs").glob("**/best_model.pt")))
        candidates.extend(sorted((root / "classifier_runs").glob("**/*.engine")))
        candidates.extend(sorted((root / "classifier_runs").glob("**/*.onnx")))
        candidates.extend(sorted((root / "exports").glob("**/*.engine")))
        candidates.extend(sorted((root / "exports").glob("**/*.onnx")))
    out = []
    seen = set()
    for path in candidates:
        if path.exists() and str(path) not in seen:
            out.append(path)
            seen.add(str(path))
    return out
