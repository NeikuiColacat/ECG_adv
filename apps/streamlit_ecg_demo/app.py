from __future__ import annotations

import json
import sys
import time
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DETECTION_PREVIEW_COUNT = 8

from apps.streamlit_ecg_demo.components.ecg_plot import make_ecg_figure
from apps.streamlit_ecg_demo.services.artifact_registry import (
    APP_DATA_ROOT,
    DEFAULT_CLASSIC_SAMPLE_PATH,
    DEFAULT_PROMPT_BANK,
    DEFAULT_PROMPT_CACHE_ROOT,
    DEFAULT_PYTHON,
    canonical_audit_path,
    canonical_dataset_path,
    ensure_project,
    list_classifier_artifacts,
    list_ecg_sample_pools,
    list_projects,
    list_synthetic_pools,
    list_token_banks,
    list_training_label_files,
    list_training_signal_files,
    prompt_cache_root,
    sanitize_project_id,
)
from apps.streamlit_ecg_demo.services.classifier_backend import DEFAULT_CKPT, DEFAULT_TRT_ENGINE, load_classifier_backend
from apps.streamlit_ecg_demo.services.dataset_intake import load_audit, save_uploaded_dataset
from apps.streamlit_ecg_demo.services.job_manager import JobSpec, list_jobs, read_job, start_job, stop_job, tail_log
from apps.streamlit_ecg_demo.services.preprocessing import (
    CLASS_NAMES,
    DEFAULT_SAMPLE_LIMIT_PER_POOL,
    load_ecg_samples,
    load_npz_samples,
    to_signal_ct,
)
from util.ecg_digital_features import extract_digital_features


st.set_page_config(page_title="ECG Target-Center Adaptation", layout="wide")


I18N = {
    "zh": {
        "app_title": "目标中心 ECG 生成与异常检测系统",
        "app_caption": "上传医院 ECG，适配 ECGTwin center token，生成目标中心 ECG，并训练可部署的检测模型。",
        "language": "语言 / Language",
        "project": "项目",
        "hospital_project_id": "医院项目 ID",
        "open_existing_project": "打开已有项目",
        "inference": "推理设置",
        "classifier_backend": "分类器后端",
        "device": "设备",
        "default_model_checkpoint": "默认模型 checkpoint",
        "tab_detection": "异常检测 Detection",
        "tab_generation": "ECG 生成 Generation",
        "tab_training": "目标医院训练 Training",
        "class_col": "类别",
        "probability_col": "概率",
        "label_col": "阈值@0.5",
        "inference_unavailable": "推理不可用",
        "no_sample_pools": "没有找到默认或项目 ECG 样本。",
        "job_history": "任务历史",
        "status": "状态",
        "pid": "PID",
        "stop_selected_job": "停止选中任务",
        "log_tail": "日志尾部",
        "no_jobs": "暂无后台任务。",
        "detection": "异常检测",
        "detection_caption": "选择 ECG 文件，点击开始推理，然后逐条查看波形和检测结果。",
        "deployment_model": "部署模型 / engine",
        "upload_ecg": "上传推理 ECG 文件 (.npy/.npz)",
        "choose_demo_sample": "选择查看的 ECG 记录",
        "choose_uploaded_sample": "选择上传文件中的 ECG",
        "inference_file": "待推理 ECG 文件",
        "start_inference": "开始推理",
        "auto_demo_inference": "已自动完成默认 demo 推理",
        "inference_finished": "推理完成",
        "inference_file_required": "请选择或上传一个 ECG 文件。",
        "inference_records": "推理记录数",
        "select_or_upload": "请选择推理文件并点击开始推理。",
        "input_ecg": "输入 ECG",
        "prediction": "检测结果",
        "quality_gate": "质量门控",
        "source_file": "来源",
        "selected_record": "当前记录",
        "leads": "导联数",
        "duration_s": "时长",
        "sample_rate": "采样率",
        "heart_rate": "心率",
        "signal_range": "幅度范围",
        "mean_std": "均值 / 标准差",
        "finite": "数值有效",
        "label_unknown": "未标注",
        "record_prefix": "记录",
        "lead_order": "导联顺序",
        "collection_loaded": "集合文件：共 {total} 条，当前载入前 {loaded} 条用于推理。",
        "download_report": "下载推理报告 JSON",
        "generation": "ECG 生成",
        "generation_caption": "选择 center token，指定每类合格 ECG 数量，后台生成并自动完成质量过滤。",
        "target_center": "目标中心",
        "generation_mode": "生成模式",
        "prompt_token_bank": "Center token 文件",
        "token_center": "Token center",
        "prompt_cache_root": "Prompt-token cache root",
        "classes": "类别",
        "samples_per_class": "每类合格生成数量",
        "steps": "DDIM/DDPM steps",
        "token_scale": "Token scale",
        "seed": "随机种子",
        "output_directory": "输出目录",
        "no_token_bank": "没有找到 center token 文件。请先在训练页完成 center-token 训练，或检查默认 token 路径。",
        "queue_generation": "开始生成任务",
        "queued_generation": "已开始生成任务",
        "gate_generated": "门控已生成样本",
        "quality_gate_caption": "质量过滤已集成到生成任务中。",
        "queue_gate": "开始质量门控任务",
        "queued_gate": "已开始质量门控任务",
        "existing_synth_sample": "已生成样本预览",
        "synthetic_ecg": "Synthetic ECG",
        "sample_metadata": "样本元数据",
        "plausibility_proxy": "医学合理性 proxy",
        "background_jobs": "后台任务",
        "training": "目标医院训练",
        "training_caption": "上传目标中心 ECG，训练 center prompt token，生成 ECG，再训练可导出的医院模型。",
        "upload_hospital_ecg": "1. 上传医院 ECG",
        "default_ecg_file": "默认 ECG 文件",
        "default_label_file": "默认标签 CSV",
        "ecg_signals": "上传 ECG signals (.npy/.npz)",
        "labels_csv": "上传标签 CSV（如果 NPZ 已含 labels，可不上传）",
        "save_audit": "保存并审计数据集",
        "saved_dataset": "已保存数据集",
        "dataset_intake_failed": "数据接入失败",
        "dataset_audit": "数据集审计",
        "samples": "样本数",
        "class_count": "数量",
        "non_finite": "非有限值",
        "flatline": "平直线",
        "shape": "形状",
        "no_audit": "还没有已审计的上传数据。",
        "center_token_jobs": "2. Center Token 适配任务",
        "queue_token": "开始 center-token 训练任务",
        "queued_token": "已开始 center-token 训练任务",
        "train_hospital_classifier": "3. 训练医院分类器",
        "synth_pool_training": "训练样本 pool",
        "initialize_checkpoint": "初始化权重文件",
        "epochs": "训练轮数",
        "classifier_export_dir": "训练导出目录",
        "batch_size": "Batch size",
        "synth_ratio": "Synthetic:real 比例上限",
        "lr": "学习率",
        "queue_model_training": "开始分类器训练任务",
        "queued_classifier": "已开始分类器训练任务",
        "source_summary": "输入摘要",
        "simple_token_mode": "生成模式",
        "center_token": "center-token",
        "use_synthetic": "使用 synthetic augmentation",
        "no_synth_pool": "当前项目没有可用 synthetic pool，将只使用真实 ECG 训练。",
        "quick_actions": "流程操作",
        "train_center_token_action": "训练 center token",
        "train_model_action": "训练医院模型",
        "export_model_action": "导出部署模型",
    },
    "en": {
        "app_title": "Target-Center ECG Generation And Detection System",
        "app_caption": "Upload hospital ECG, adapt ECGTwin center tokens, synthesize target-center ECG, and train a deployable detector.",
        "language": "Language",
        "project": "Project",
        "hospital_project_id": "Hospital project id",
        "open_existing_project": "Open existing project",
        "inference": "Inference",
        "classifier_backend": "Classifier backend",
        "device": "Device",
        "default_model_checkpoint": "Default model checkpoint",
        "tab_detection": "Detection",
        "tab_generation": "Generation",
        "tab_training": "Training",
        "class_col": "class",
        "probability_col": "probability",
        "label_col": "label@0.5",
        "inference_unavailable": "Inference unavailable",
        "no_sample_pools": "No default or project ECG samples found.",
        "job_history": "Job history",
        "status": "Status",
        "pid": "PID",
        "stop_selected_job": "Stop selected job",
        "log_tail": "Log tail",
        "no_jobs": "No background jobs yet.",
        "detection": "Detection",
        "detection_caption": "Select an ECG file, run inference, then inspect each record and prediction.",
        "deployment_model": "Deployment model / engine",
        "upload_ecg": "Upload inference ECG file (.npy/.npz)",
        "choose_demo_sample": "Choose ECG record to inspect",
        "choose_uploaded_sample": "Choose ECG from uploaded file",
        "inference_file": "ECG file for inference",
        "start_inference": "Start inference",
        "auto_demo_inference": "Default demo inference finished",
        "inference_finished": "Inference finished",
        "inference_file_required": "Select or upload an ECG file.",
        "inference_records": "Inference records",
        "select_or_upload": "Select an ECG file and start inference.",
        "input_ecg": "Input ECG",
        "prediction": "Prediction",
        "quality_gate": "Quality Gate",
        "source_file": "Source",
        "selected_record": "Selected record",
        "leads": "Leads",
        "duration_s": "Duration",
        "sample_rate": "Sample rate",
        "heart_rate": "Heart rate",
        "signal_range": "Signal range",
        "mean_std": "Mean / std",
        "finite": "Finite",
        "label_unknown": "unlabeled",
        "record_prefix": "Record",
        "lead_order": "Lead order",
        "collection_loaded": "Collection file: {total} records, first {loaded} loaded for inference.",
        "download_report": "Download inference report JSON",
        "generation": "Generation",
        "generation_caption": "Choose a center token, request accepted ECGs per class, and run generation with automatic quality filtering.",
        "target_center": "Target center",
        "generation_mode": "Generation mode",
        "prompt_token_bank": "Center token file",
        "token_center": "Token center",
        "prompt_cache_root": "Prompt-token cache root",
        "classes": "Classes",
        "samples_per_class": "Accepted samples per class",
        "steps": "DDIM/DDPM steps",
        "token_scale": "Token scale",
        "seed": "Seed",
        "output_directory": "Output directory",
        "no_token_bank": "No center token file found. Train center tokens first or check the default token path.",
        "queue_generation": "Start generation job",
        "queued_generation": "Started generation job",
        "gate_generated": "Gate generated samples",
        "quality_gate_caption": "Quality filtering is built into the generation job.",
        "queue_gate": "Start quality gate job",
        "queued_gate": "Started gate job",
        "existing_synth_sample": "Generated sample preview",
        "synthetic_ecg": "Synthetic ECG",
        "sample_metadata": "Sample metadata",
        "plausibility_proxy": "Plausibility proxy",
        "background_jobs": "Background Jobs",
        "training": "Training",
        "training_caption": "Upload target-center ECG, train center prompt tokens, synthesize ECG, then train an exportable hospital model.",
        "upload_hospital_ecg": "1. Upload Hospital ECG",
        "default_ecg_file": "Default ECG file",
        "default_label_file": "Default label CSV",
        "ecg_signals": "Upload ECG signals (.npy/.npz)",
        "labels_csv": "Upload labels CSV (optional if NPZ contains labels)",
        "save_audit": "Save and audit dataset",
        "saved_dataset": "Saved dataset",
        "dataset_intake_failed": "Dataset intake failed",
        "dataset_audit": "Dataset Audit",
        "samples": "Samples",
        "class_count": "Count",
        "non_finite": "Non-finite",
        "flatline": "Flatline",
        "shape": "Shape",
        "no_audit": "No audited upload yet.",
        "center_token_jobs": "2. Center Token Adaptation Jobs",
        "queue_token": "Start center-token training",
        "queued_token": "Started center-token training job",
        "train_hospital_classifier": "3. Train Hospital Classifier",
        "synth_pool_training": "Training sample pool",
        "initialize_checkpoint": "Initial weight file",
        "epochs": "Epochs",
        "classifier_export_dir": "Training export directory",
        "batch_size": "Batch size",
        "synth_ratio": "Synthetic:real ratio cap",
        "lr": "LR",
        "queue_model_training": "Start classifier training",
        "queued_classifier": "Started classifier job",
        "source_summary": "Input summary",
        "simple_token_mode": "Generation mode",
        "center_token": "center-token",
        "use_synthetic": "Use synthetic augmentation",
        "no_synth_pool": "No synthetic pool is available for this project; training will use real ECG only.",
        "quick_actions": "Workflow actions",
        "train_center_token_action": "Train center token",
        "train_model_action": "Train hospital model",
        "export_model_action": "Export deployment model",
    },
}


def lang_code() -> str:
    return st.session_state.get("ui_lang", "zh")


def tr(key: str) -> str:
    lang = lang_code()
    return I18N.get(lang, I18N["zh"]).get(key, I18N["en"].get(key, key))


@st.cache_resource(show_spinner=False)
def get_backend(backend_name: str, ckpt_path: str, device: str):
    return load_classifier_backend(backend_name, ckpt_path=ckpt_path, device=device)


@st.cache_data(show_spinner=False)
def get_samples(paths: tuple[str, ...]):
    return load_ecg_samples(paths)


def _path_options(paths: list[Path], fallback: str = "") -> list[str]:
    out = [str(p) for p in paths]
    if fallback and fallback not in out:
        out.insert(0, fallback)
    return out or ([fallback] if fallback else [])


def _classifier_artifact_options(root: Path | None, backend_name: str) -> list[str]:
    paths = list_classifier_artifacts(root)
    if backend_name == "tensorrt":
        return _path_options([p for p in paths if p.suffix == ".engine"], fallback=DEFAULT_TRT_ENGINE)
    return _path_options([p for p in paths if p.suffix in {".pt", ".pth"}], fallback=DEFAULT_CKPT)


def _save_dataset_from_paths(
    project_root: Path,
    signal_path: str | Path,
    label_path: str | Path | None,
) -> tuple[Path, Path, dict]:
    signal_path = Path(signal_path)
    label_path = Path(label_path) if label_path else None
    if signal_path.suffix == ".npz":
        try:
            with np.load(signal_path, allow_pickle=True) as data:
                if "labels" in data.files:
                    label_path = None
        except Exception:
            pass
    with open(signal_path, "rb") as signal_f:
        if label_path and label_path.exists():
            with open(label_path, "rb") as label_f:
                return save_uploaded_dataset(
                    project_root,
                    signal_f,
                    signal_path.name,
                    label_f,
                    label_path.name,
                )
        return save_uploaded_dataset(project_root, signal_f, signal_path.name, None, None)


def _render_dataset_audit(audit: dict) -> None:
    shape = audit.get("signal_shape") or []
    lead_count = audit.get("lead_count")
    if lead_count is None and len(shape) >= 2:
        lead_count = shape[1]
    sample_rate = audit.get("sample_rate_hz", 100.0)
    cols = st.columns(3)
    cols[0].metric(tr("samples"), int(audit.get("n_samples", 0)))
    cols[1].metric(tr("leads"), int(lead_count or 0))
    cols[2].metric(tr("sample_rate"), f"{float(sample_rate):.0f} Hz")
    st.dataframe(
        pd.DataFrame([
            {tr("class_col"): cls, tr("class_count"): int(audit.get("class_counts", {}).get(cls, 0))}
            for cls in CLASS_NAMES
        ]),
        width="stretch",
        hide_index=True,
    )


def _newest_path(paths: list[Path]) -> str:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return ""
    return str(max(existing, key=lambda p: p.stat().st_mtime))


def probability_table(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        tr("class_col"): CLASS_NAMES,
        tr("probability_col"): probabilities.astype(float),
        tr("label_col"): probabilities >= 0.5,
    })


def render_prediction_result(pred: dict) -> None:
    probs = np.asarray(pred["probabilities"], dtype=np.float32)
    df = probability_table(probs)
    st.bar_chart(df.set_index(tr("class_col"))[tr("probability_col")])
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(
        f"backend={pred['backend']} device={pred['device']} "
        f"latency={pred['latency_ms']:.2f} ms"
    )


def predict_panel(signal_ct: np.ndarray, backend_name: str, ckpt_path: str, device: str):
    try:
        backend = get_backend(backend_name, ckpt_path, device)
        pred = backend.predict(signal_ct)
    except Exception as exc:
        st.error(f"{tr('inference_unavailable')}: {exc}")
        return None
    render_prediction_result(pred)
    return pred


def sample_selector(samples: list[dict], key: str, label: str = "ECG sample") -> dict | None:
    if not samples:
        st.info(tr("no_sample_pools"))
        return None
    labels = [
        f"{i:03d} | {s.get('center', 'unknown')} | {','.join(s.get('label_names') or ['unknown'])}"
        for i, s in enumerate(samples)
    ]
    idx = st.selectbox(label, range(len(samples)), format_func=lambda i: labels[i], key=key)
    return samples[int(idx)]


def _label_names_from_row(label: np.ndarray | None, class_names: list[str]) -> list[str]:
    if label is None:
        return []
    return [class_names[j] for j, v in enumerate(label) if float(v) > 0.5]


def _samples_from_array(
    signals: np.ndarray,
    labels: np.ndarray | None,
    class_names: list[str],
    source: str,
    center: str,
) -> list[dict]:
    arr = np.asarray(signals, dtype=np.float32)
    if arr.ndim == 2:
        sample_iter = [arr]
        total = 1
    elif arr.ndim == 3:
        total = int(arr.shape[0])
        sample_iter = [arr[i] for i in range(min(total, DEFAULT_SAMPLE_LIMIT_PER_POOL))]
    else:
        raise ValueError(f"ECG upload must have shape (12,L), (L,12), (N,12,L), or (N,L,12); got {arr.shape}")

    out = []
    loaded = len(sample_iter)
    for i, sample in enumerate(sample_iter):
        label = None
        if labels is not None:
            labels_arr = np.asarray(labels, dtype=np.float32)
            label = labels_arr if labels_arr.ndim == 1 else labels_arr[i]
        label_names = _label_names_from_row(label, class_names)
        out.append({
            "id": f"{Path(source).stem}:{i}",
            "source": source,
            "center": center,
            "signal": to_signal_ct(sample),
            "label": label.astype(np.float32).tolist() if label is not None else None,
            "label_names": label_names,
            "record_index": i,
            "record_count": total,
            "loaded_count": loaded,
            "raw_file_shape": list(arr.shape),
        })
    return out


def uploaded_samples(uploaded) -> list[dict]:
    if uploaded is None:
        return []
    name = str(uploaded.name)
    uploaded.seek(0)
    if name.endswith(".npy"):
        signals = np.load(uploaded)
        labels = None
        class_names = CLASS_NAMES
        center = "uploaded"
    else:
        data = np.load(uploaded, allow_pickle=True)
        key = "signals" if "signals" in data.files else data.files[0]
        signals = data[key]
        labels = data["labels"] if "labels" in data.files else None
        class_names = [str(x) for x in data["class_names"]] if "class_names" in data.files else CLASS_NAMES
        center = str(data["center_name"]) if "center_name" in data.files else "uploaded"
    return _samples_from_array(signals, labels, class_names, name, center)


def path_samples(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".npy":
        signals = np.load(path)
        return _samples_from_array(signals, None, CLASS_NAMES, str(path), path.parent.name)
    if path.suffix == ".npz":
        return load_npz_samples(path)
    raise ValueError(f"unsupported ECG file type: {path}")


def estimate_heart_rate_bpm(signal_ct: np.ndarray) -> float | None:
    try:
        feats = extract_digital_features(np.asarray(signal_ct, dtype=np.float32), fs=100.0, lead_order="ptbxl")
    except Exception:
        return None
    hr = feats.get("hr_bpm")
    if hr is None:
        return None
    return float(hr)


def signal_summary(signal_ct: np.ndarray, meta: dict) -> dict:
    arr = np.asarray(signal_ct, dtype=np.float32)
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    if finite_values.size:
        mean = float(finite_values.mean())
        std = float(finite_values.std())
        p2p = float(finite_values.max() - finite_values.min())
    else:
        mean = std = p2p = float("nan")
    return {
        "source": str(meta.get("source") or meta.get("id") or "sample"),
        "center": str(meta.get("center") or "unknown"),
        "shape": list(arr.shape),
        "raw_file_shape": meta.get("raw_file_shape"),
        "record_index": meta.get("record_index"),
        "record_count": meta.get("record_count"),
        "loaded_count": meta.get("loaded_count"),
        "leads": int(arr.shape[0]),
        "samples": int(arr.shape[1]),
        "duration_s": float(arr.shape[1] / 100.0),
        "heart_rate_bpm": estimate_heart_rate_bpm(arr),
        "mean": mean,
        "std": std,
        "p2p": p2p,
        "finite": bool(finite.all()),
        "lead_order": "PTB-XL",
        "sampling_rate_hz": 100,
    }


def _format_metric(value: float, suffix: str = "", precision: int = 2) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def _render_label_chips(labels: list[str]) -> None:
    chips = labels or [tr("label_unknown")]
    html = " ".join(
        f"<span style='display:inline-block;padding:0.18rem 0.5rem;margin:0.08rem 0.18rem 0.08rem 0;"
        f"border-radius:999px;border:1px solid rgba(49,51,63,0.20);background:rgba(49,51,63,0.06);"
        f"font-size:0.78rem;'>{escape(str(label))}</span>"
        for label in chips
    )
    st.markdown(html, unsafe_allow_html=True)


def render_input_summary(signal_ct: np.ndarray, meta: dict) -> dict:
    summary = signal_summary(signal_ct, meta)
    labels = [str(x) for x in meta.get("label_names") or []]
    with st.container(border=True):
        st.caption(f"{tr('source_file')}: {summary['source']}")
        _render_label_chips(labels)
        cols = st.columns(4)
        cols[0].metric(tr("leads"), summary["leads"])
        cols[1].metric(tr("samples"), summary["samples"])
        cols[2].metric(tr("duration_s"), _format_metric(summary["duration_s"], " s", precision=1))
        hr = summary.get("heart_rate_bpm")
        cols[3].metric(tr("heart_rate"), "N/A" if hr is None else _format_metric(float(hr), " bpm", precision=1))

        if summary.get("record_count") and int(summary["record_count"]) > 1:
            st.caption(
                tr("collection_loaded").format(
                    total=int(summary["record_count"]),
                    loaded=int(summary.get("loaded_count") or 0),
                )
            )
        if summary.get("record_index") is not None:
            st.caption(f"{tr('selected_record')}: {tr('record_prefix')} {int(summary['record_index']) + 1:03d}")
    return summary


def render_jobs(project_root: Path, key_prefix: str = "jobs"):
    jobs = list_jobs(project_root)
    if not jobs:
        st.caption(tr("no_jobs"))
        return
    selected = st.selectbox(
        tr("job_history"),
        jobs,
        format_func=lambda p: p.name,
        key=f"{key_prefix}_job_history",
    )
    meta = read_job(Path(selected))
    cols = st.columns([0.22, 0.22, 0.56])
    cols[0].metric(tr("status"), str(meta.get("status", "unknown")))
    cols[1].metric(tr("pid"), str(meta.get("pid", "")))
    cols[2].code(" ".join(meta.get("command", [])), language="bash")
    if meta.get("status") == "running" and st.button(tr("stop_selected_job"), key=f"{key_prefix}_stop_job"):
        stop_job(Path(selected))
        st.rerun()
    st.text_area(tr("log_tail"), tail_log(Path(selected)), height=260, key=f"{key_prefix}_log_tail")


def _sample_result_label(i: int, sample: dict, pred: dict | None = None) -> str:
    return f"{tr('record_prefix')} {i + 1:03d}"


def _run_inference_batch(
    samples: list[dict],
    backend_name: str,
    model_path: str,
    device: str,
    *,
    show_progress: bool = True,
) -> list[dict]:
    backend = get_backend(backend_name, model_path, device)
    results = []
    progress = st.progress(0.0) if show_progress else None
    status = st.empty() if show_progress else None
    total = len(samples)
    for i, sample in enumerate(samples, start=1):
        if status is not None:
            status.caption(f"{tr('inference_records')}: {i}/{total}")
        pred = backend.predict(sample["signal"])
        results.append(pred)
        if progress is not None:
            progress.progress(i / max(1, total))
    if status is not None:
        status.empty()
    if progress is not None:
        progress.empty()
    return results


def _store_detection_batch(source: str, backend_name: str, model_path: str, samples: list[dict], predictions: list[dict]) -> None:
    st.session_state["detection_batch"] = {
        "source": source,
        "backend": backend_name,
        "model": model_path,
        "samples": samples,
        "predictions": predictions,
    }


def detection_page(project_root: Path | None, sample_paths: list[str], backend_name: str, ckpt_path: str, device: str):
    st.subheader(tr("detection"))
    st.caption(tr("detection_caption"))

    ckpt_options = _classifier_artifact_options(project_root, backend_name)
    if ckpt_path and ckpt_path not in ckpt_options:
        ckpt_options.insert(0, ckpt_path)
    selected_model = st.selectbox(tr("deployment_model"), ckpt_options, index=0)

    left, right = st.columns([0.34, 0.66])
    with left:
        file_options = [str(p) for p in sample_paths]
        default_idx = file_options.index(str(DEFAULT_CLASSIC_SAMPLE_PATH)) if str(DEFAULT_CLASSIC_SAMPLE_PATH) in file_options else 0
        selected_file = (
            st.selectbox(tr("inference_file"), file_options, index=default_idx, format_func=lambda p: Path(p).name)
            if file_options
            else ""
        )
        uploaded = st.file_uploader(tr("upload_ecg"), type=["npy", "npz"], key="detect_upload")
        source_name = str(uploaded.name) if uploaded is not None else selected_file
        auto_key = ("auto_demo_detection", backend_name, selected_model, str(DEFAULT_CLASSIC_SAMPLE_PATH))
        batch = st.session_state.get("detection_batch")
        if uploaded is None and selected_file == str(DEFAULT_CLASSIC_SAMPLE_PATH) and batch is None:
            try:
                demo_samples = path_samples(DEFAULT_CLASSIC_SAMPLE_PATH)[:DEFAULT_DETECTION_PREVIEW_COUNT]
                predictions = _run_inference_batch(demo_samples, backend_name, selected_model, device, show_progress=False)
                _store_detection_batch(str(DEFAULT_CLASSIC_SAMPLE_PATH), backend_name, selected_model, demo_samples, predictions)
                st.session_state["detection_auto_key"] = auto_key
                st.success(f"{tr('auto_demo_inference')}: {len(demo_samples)}")
            except Exception as exc:
                st.warning(f"{tr('inference_unavailable')}: {exc}")
        if st.button(tr("start_inference"), disabled=not source_name):
            try:
                selected_samples = uploaded_samples(uploaded) if uploaded is not None else path_samples(selected_file)
                if not selected_samples:
                    raise ValueError("no ECG records found")
                predictions = _run_inference_batch(selected_samples, backend_name, selected_model, device)
                _store_detection_batch(source_name, backend_name, selected_model, selected_samples, predictions)
                st.success(f"{tr('inference_finished')}: {len(selected_samples)}")
            except Exception as exc:
                st.error(f"{tr('inference_unavailable')}: {exc}")

        batch = st.session_state.get("detection_batch")
        if batch:
            st.metric(tr("inference_records"), len(batch["samples"]))
            idx = st.selectbox(
                tr("choose_demo_sample"),
                range(len(batch["samples"])),
                format_func=lambda i: _sample_result_label(i, batch["samples"][i], batch["predictions"][i]),
                key="detect_result_sample",
            )
            selected = batch["samples"][int(idx)]
            pred = batch["predictions"][int(idx)]
            signal_ct = selected["signal"]
            meta = {k: v for k, v in selected.items() if k != "signal"}
        else:
            signal_ct, meta = None, {}
            pred = None
        if signal_ct is not None:
            st.markdown(f"**{tr('source_summary')}**")
            summary = render_input_summary(signal_ct, meta)
    with right:
        if signal_ct is None:
            st.info(tr("select_or_upload") if sample_paths else tr("inference_file_required"))
            return
        st.pyplot(make_ecg_figure(signal_ct))
        st.markdown(f"**{tr('prediction')}**")
        if pred:
            render_prediction_result(pred)
            report = {
                "metadata": {**meta, "summary": summary},
                "prediction": {k: v for k, v in pred.items() if k not in {"logits"}},
            }
            st.download_button(
                tr("download_report"),
                data=json.dumps(report, indent=2),
                file_name="ecg_inference_report.json",
                mime="application/json",
            )


def generation_command(
    center: str,
    token_bank: str,
    cache_root: str,
    out_dir: str,
    classes: list[str],
    n_per_class: int,
    steps: int,
    seed: int,
    device: str,
    token_center: str,
    token_scale: float,
) -> list[str]:
    cmd = [
        DEFAULT_PYTHON,
        "scripts/streamlit_demo/generate_gated_center_token_pool.py",
        "--center",
        center,
        "--classes",
        *classes,
        "--cache_root",
        cache_root,
        "--prompt_bank",
        DEFAULT_PROMPT_BANK,
        "--out_dir",
        out_dir,
        "--target_per_class",
        str(n_per_class),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--device",
        device,
        "--victim_ckpt",
        DEFAULT_CKPT,
        "--token_scale",
        str(token_scale),
        "--token_bank",
        token_bank,
        "--token_center",
        token_center,
        "--max_rounds",
        "5",
        "--oversample_factor",
        "3.0",
        "--min_candidates_per_class",
        "8",
        "--min_target_prob",
        "0.30",
    ]
    return cmd


def generation_page(
    project_root: Path | None,
    project_id: str | None,
    samples: list[dict],
    device: str,
):
    st.subheader(tr("generation"))
    st.caption(tr("generation_caption"))

    left, right = st.columns([0.34, 0.66])
    with left:
        center = st.text_input(tr("target_center"), value=project_id or "ningbo")
        selected_classes = st.multiselect(tr("classes"), CLASS_NAMES, default=["NORM", "MI", "STTC"])
        n_per_class = st.number_input(tr("samples_per_class"), min_value=1, max_value=200, value=8)
        token_banks = list_token_banks(project_root)
        cache_root = str(prompt_cache_root(project_root)) if project_root and (prompt_cache_root(project_root) / "center_full_latents").exists() else DEFAULT_PROMPT_CACHE_ROOT
        token_options = _path_options(token_banks)
        token_default = _newest_path(token_banks)
        if token_default in token_options:
            token_index = token_options.index(token_default)
        else:
            token_index = 0
        token_bank = st.selectbox(tr("prompt_token_bank"), token_options or [""], index=token_index)
        token_center = center
        steps = 25
        token_scale = 1.0
        seed = 42
        out_dir = str((project_root or APP_DATA_ROOT) / "synthetic_pools" / f"gen_{int(time.time())}")
        if not token_bank:
            st.warning(tr("no_token_bank"))
        if project_root and st.button(tr("queue_generation"), disabled=(not selected_classes or not token_bank)):
            cmd = generation_command(
                center=center,
                token_bank=token_bank,
                cache_root=cache_root,
                out_dir=out_dir,
                classes=selected_classes,
                n_per_class=int(n_per_class),
                steps=int(steps),
                seed=int(seed),
                device=device,
                token_center=token_center,
                token_scale=float(token_scale),
            )
            job_dir = start_job(project_root, JobSpec("generate_synthetic_pool", cmd, str(REPO_ROOT)))
            st.success(f"{tr('queued_generation')}: {job_dir.name}")

    with right:
        selected = sample_selector(samples, key="generation_sample", label=tr("existing_synth_sample"))
        if selected is not None:
            signal_ct = selected["signal"]
            st.pyplot(make_ecg_figure(signal_ct))
        if project_root:
            st.markdown(f"**{tr('background_jobs')}**")
            render_jobs(project_root, key_prefix="generation")


def training_page(project_root: Path, project_id: str, device: str):
    st.subheader(tr("training"))
    st.caption(tr("training_caption"))

    audit_path = canonical_audit_path(project_root)
    dataset_path = canonical_dataset_path(project_root)

    step1, step2 = st.columns([0.38, 0.62])
    with step1:
        st.markdown(f"**{tr('upload_hospital_ecg')}**")
        default_signals = list_training_signal_files(project_root)
        default_labels = list_training_label_files(project_root)
        selected_signal_path = st.selectbox(
            tr("default_ecg_file"),
            [str(p) for p in default_signals] or [""],
            format_func=lambda p: Path(p).name if p else "",
        )
        selected_label_path = st.selectbox(
            tr("default_label_file"),
            [str(p) for p in default_labels] or [""],
            format_func=lambda p: Path(p).name if p else "",
        )
        sig_upload = st.file_uploader(tr("ecg_signals"), type=["npy", "npz"], key="train_signal_upload")
        label_upload = st.file_uploader(tr("labels_csv"), type=["csv"], key="train_label_upload")
        can_save_dataset = sig_upload is not None or bool(selected_signal_path)

        auto_key = f"{project_id}:{selected_signal_path}:{selected_label_path}"
        should_auto_intake = (
            sig_upload is None
            and bool(selected_signal_path)
            and (
                st.session_state.get("training_auto_intake_key") != auto_key
                or not dataset_path.exists()
                or not audit_path.exists()
            )
        )
        if should_auto_intake:
            try:
                _save_dataset_from_paths(project_root, selected_signal_path, selected_label_path)
                st.session_state["training_auto_intake_key"] = auto_key
            except Exception as exc:
                st.warning(f"{tr('dataset_intake_failed')}: {exc}")

        if st.button(tr("save_audit"), disabled=not can_save_dataset):
            try:
                if sig_upload is not None:
                    sig_upload.seek(0)
                    if label_upload is not None:
                        label_upload.seek(0)
                    saved_npz, saved_audit, audit = save_uploaded_dataset(
                        project_root,
                        sig_upload,
                        sig_upload.name,
                        label_upload,
                        label_upload.name if label_upload else None,
                    )
                else:
                    saved_npz, saved_audit, audit = _save_dataset_from_paths(
                        project_root,
                        selected_signal_path,
                        selected_label_path,
                    )
                st.success(f"{tr('saved_dataset')}: {saved_npz}")
            except Exception as exc:
                st.error(f"{tr('dataset_intake_failed')}: {exc}")

    with step2:
        st.markdown(f"**{tr('dataset_audit')}**")
        audit = load_audit(audit_path)
        if audit:
            _render_dataset_audit(audit)
        else:
            st.info(tr("no_audit"))

    st.divider()
    st.markdown(f"**{tr('center_token_jobs')}**")
    K, floor, steps, batch_size = 500, 10, 2000, 16
    cache_root = str(prompt_cache_root(project_root))
    token_save_dir = str(project_root / "center_tokens" / f"token_{int(time.time())}")
    if st.button(tr("queue_token"), disabled=not dataset_path.exists()):
        cmd = [
            DEFAULT_PYTHON,
            "scripts/streamlit_demo/train_center_token_pipeline.py",
            "--dataset_npz",
            str(dataset_path),
            "--center",
            project_id,
            "--cache_root",
            cache_root,
            "--prompt_bank",
            DEFAULT_PROMPT_BANK,
            "--save_dir",
            token_save_dir,
            "--K",
            str(int(K)),
            "--floor_per_class",
            str(int(floor)),
            "--total_steps",
            str(int(steps)),
            "--batch_size",
            str(int(batch_size)),
            "--num_workers",
            "2",
            "--device",
            device,
        ]
        job_dir = start_job(project_root, JobSpec("train_center_token", cmd, str(REPO_ROOT)))
        st.success(f"{tr('queued_token')}: {job_dir.name}")

    st.divider()
    st.markdown(f"**{tr('train_hospital_classifier')}**")
    synth_options = _path_options(list_synthetic_pools(project_root)) or [""]
    model_options = _classifier_artifact_options(project_root, "torch") or [""]
    synth_npz = st.selectbox(
        tr("synth_pool_training"),
        synth_options,
        index=0,
        format_func=lambda p: Path(p).name if p else "",
    )
    init_ckpt = st.selectbox(tr("initialize_checkpoint"), model_options, index=0)
    epochs = st.selectbox(tr("epochs"), [10, 20, 50], index=1)
    model_batch = 32
    synth_ratio = 1.0
    lr = 1e-4
    if not synth_npz:
        st.warning(tr("no_synth_pool"))
    export_key = f"classifier_export_dir::{project_id}"
    if export_key not in st.session_state:
        st.session_state[export_key] = str(project_root / "classifier_runs" / f"hospital_effnet_{int(time.time())}")
    model_out = st.text_input(tr("classifier_export_dir"), st.session_state[export_key])
    st.session_state[export_key] = model_out
    model_out_path = Path(model_out).expanduser()
    if model_out and not model_out_path.is_absolute():
        model_out_path = project_root / model_out_path
    disable_train = not dataset_path.exists() or not synth_npz or not init_ckpt or not model_out
    if st.button(tr("queue_model_training"), disabled=disable_train):
        cmd = [
            DEFAULT_PYTHON,
            "scripts/streamlit_demo/train_and_export_hospital_classifier.py",
            "--dataset_npz",
            str(dataset_path),
            "--output_dir",
            str(model_out_path),
            "--synth_npz",
            synth_npz,
            "--init_ckpt",
            init_ckpt,
            "--epochs",
            str(int(epochs)),
            "--batch_size",
            str(int(model_batch)),
            "--synth_ratio",
            str(float(synth_ratio)),
            "--lr",
            str(float(lr)),
            "--device",
            "cuda" if device.startswith("cuda") else "cpu",
        ]
        job_dir = start_job(project_root, JobSpec("train_hospital_classifier", cmd, str(REPO_ROOT)))
        st.success(f"{tr('queued_classifier')}: {job_dir.name}")

    st.divider()
    render_jobs(project_root, key_prefix="training")


def main():
    existing = list_projects()
    with st.sidebar:
        language_label = st.selectbox(tr("language"), ["中文", "English"], index=0 if lang_code() == "zh" else 1)
        st.session_state["ui_lang"] = "zh" if language_label == "中文" else "en"

        st.header(tr("project"))
        default_project = existing[0] if existing else "demo_hospital"
        project_id_input = st.text_input(tr("hospital_project_id"), default_project)
        try:
            project_id = sanitize_project_id(project_id_input)
            active_root = ensure_project(project_id)
        except Exception as exc:
            st.error(str(exc))
            project_id = "demo_hospital"
            active_root = ensure_project(project_id)
        if existing:
            chosen = st.selectbox(tr("open_existing_project"), [project_id] + [p for p in existing if p != project_id])
            if chosen != project_id:
                project_id = chosen
                active_root = ensure_project(project_id)
        st.caption(str(active_root))

        st.header(tr("inference"))
        backend_label = st.selectbox(tr("classifier_backend"), ["torch", "TensorRT"])
        backend_name = "tensorrt" if backend_label == "TensorRT" else "torch"
        device = st.selectbox(tr("device"), ["cuda:0", "cuda", "cpu"])
        default_model = _classifier_artifact_options(active_root, backend_name)[0]
        ckpt_path = st.text_input(tr("default_model_checkpoint"), default_model)

    st.title(tr("app_title"))
    st.caption(tr("app_caption"))

    sample_paths = [str(p) for p in list_ecg_sample_pools(active_root)]
    generated_sample_paths = [str(p) for p in list_synthetic_pools(active_root)]
    generated_samples = get_samples(tuple(generated_sample_paths))
    page_detection, page_generation, page_training = st.tabs([
        tr("tab_detection"),
        tr("tab_generation"),
        tr("tab_training"),
    ])
    with page_detection:
        detection_page(active_root, sample_paths, backend_name, ckpt_path, device)
    with page_generation:
        generation_page(active_root, project_id, generated_samples, device)
    with page_training:
        training_page(active_root, project_id, device)


if __name__ == "__main__":
    main()
