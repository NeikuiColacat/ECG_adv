from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_streamlit_app_renders_default_workflow(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    app_root = data_root / "streamlit_ecg_demo"
    data_root.mkdir()
    app_root.mkdir()
    monkeypatch.setenv("ECG_ADV_DATA_ROOT", str(data_root))
    monkeypatch.setenv("ECG_ADV_APP_DATA_ROOT", str(app_root))

    at = AppTest.from_file(str(REPO_ROOT / "apps/streamlit_ecg_demo/app.py"), default_timeout=15)
    at.run()

    assert not at.exception
    assert [tab.label for tab in at.tabs] == [
        "异常检测 Detection",
        "ECG 生成 Generation",
        "目标医院训练 Training",
    ]
    backend_select = next(select for select in at.selectbox if select.label == "分类器后端")
    assert backend_select.options == ["torch", "TensorRT"]
