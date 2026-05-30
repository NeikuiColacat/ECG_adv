from __future__ import annotations

import os
import sys
from pathlib import Path


def _path_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def _default_data_root() -> Path:
    return Path.home() / "autodl-tmp"


DATA_ROOT = _path_env("ECG_ADV_DATA_ROOT", _default_data_root())
APP_DATA_ROOT = _path_env("ECG_ADV_APP_DATA_ROOT", DATA_ROOT / "streamlit_ecg_demo")
MODEL_ROOT = _path_env("ECG_ADV_MODEL_ROOT", DATA_ROOT / "models")
TMP_DIR = _path_env("ECG_ADV_TMPDIR", DATA_ROOT / "tmp")
CACHE_HOME = _path_env("ECG_ADV_CACHE_HOME", DATA_ROOT / "cache")
PYTHON_EXECUTABLE = os.environ.get("ECG_ADV_PYTHON", sys.executable)
