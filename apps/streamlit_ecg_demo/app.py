from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.streamlit_ecg_demo.components.ecg_plot import make_ecg_figure
from apps.streamlit_ecg_demo.services.artifact_registry import (
    DEFAULT_PROMPT_BANK,
    DEFAULT_PYTHON,
    canonical_audit_path,
    canonical_dataset_path,
    ensure_project,
    list_classifier_artifacts,
    list_projects,
    list_synthetic_pools,
    list_token_banks,
    prompt_cache_root,
    sanitize_project_id,
)
from apps.streamlit_ecg_demo.services.classifier_backend import DEFAULT_CKPT, load_classifier_backend
from apps.streamlit_ecg_demo.services.dataset_intake import load_audit, save_uploaded_dataset
from apps.streamlit_ecg_demo.services.job_manager import JobSpec, list_jobs, read_job, start_job, stop_job, tail_log
from apps.streamlit_ecg_demo.services.preprocessing import CLASS_NAMES, load_demo_samples, to_signal_ct
from apps.streamlit_ecg_demo.services.quality_gate import run_quality_gate


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
        "samples_per_pool": "每个样本池载入数量",
        "tab_detection": "异常检测 Detection",
        "tab_generation": "ECG 生成 Generation",
        "tab_training": "目标医院训练 Training",
        "class_col": "类别",
        "probability_col": "概率",
        "label_col": "阈值@0.5",
        "inference_unavailable": "推理不可用",
        "no_sample_pools": "没有找到样本池，请检查 demo/project 路径。",
        "signal_sanity": "信号质量",
        "job_history": "任务历史",
        "status": "状态",
        "pid": "PID",
        "stop_selected_job": "停止选中任务",
        "log_tail": "日志尾部",
        "no_jobs": "暂无后台任务。",
        "pass": "通过",
        "fail": "失败",
        "warning": "警告",
        "detection": "异常检测",
        "detection_caption": "上传或选择一条 12 导联 ECG，并使用选定部署后端进行检测。",
        "deployment_model": "部署模型 / engine",
        "upload_ecg": "上传 ECG (.npy/.npz)",
        "choose_demo_sample": "或选择 demo/project ECG",
        "select_or_upload": "请选择或上传一条 ECG。",
        "input_ecg": "输入 ECG",
        "prediction": "检测结果",
        "quality_gate": "质量门控",
        "download_report": "下载推理报告 JSON",
        "generation": "ECG 生成",
        "generation_caption": "管理目标中心 synthetic ECG pool；实时生成以后台任务方式排队。",
        "target_center": "目标中心",
        "generation_mode": "生成模式",
        "prompt_token_bank": "Prompt-token bank",
        "token_center": "Token center",
        "prompt_cache_root": "Prompt-token cache root",
        "classes": "类别",
        "samples_per_class": "每类生成数量",
        "steps": "DDIM/DDPM steps",
        "token_scale": "Token scale",
        "seed": "随机种子",
        "output_directory": "输出目录",
        "no_token_bank": "没有找到 prompt-token bank。请切换 no-token 模式，或先训练/选择 token bank。",
        "queue_generation": "排队生成任务",
        "queued_generation": "已排队生成任务",
        "gate_generated": "门控已生成样本",
        "queue_gate": "排队质量门控任务",
        "queued_gate": "已排队质量门控任务",
        "existing_synth_sample": "已有 synthetic ECG 样本",
        "synthetic_ecg": "Synthetic ECG",
        "sample_metadata": "样本元数据",
        "plausibility_proxy": "医学合理性 proxy",
        "background_jobs": "后台任务",
        "training": "目标医院训练",
        "training_caption": "上传目标中心 ECG，训练 center prompt token，生成 ECG，再训练可导出的医院模型。",
        "upload_hospital_ecg": "1. 上传医院 ECG",
        "ecg_signals": "ECG signals (.npy/.npz)",
        "labels_csv": "标签 CSV（如果 NPZ 已含 labels，可不上传）",
        "save_audit": "保存并审计数据集",
        "saved_dataset": "已保存数据集",
        "dataset_intake_failed": "数据接入失败",
        "dataset_audit": "数据集审计",
        "samples": "样本数",
        "non_finite": "非有限值",
        "flatline": "平直线",
        "shape": "形状",
        "no_audit": "还没有已审计的上传数据。",
        "center_token_jobs": "2. Center Token 适配任务",
        "k_refs": "K refs",
        "class_floor": "类别下限",
        "token_steps": "Token steps",
        "token_batch": "Token batch",
        "queue_cache": "排队 cache 准备任务",
        "queued_cache": "已排队 cache 任务",
        "queue_token": "排队 center-token 训练任务",
        "queued_token": "已排队 token 任务",
        "train_hospital_classifier": "3. 训练医院分类器",
        "synth_pool_training": "用于训练的 synthetic pool",
        "initialize_checkpoint": "初始化 checkpoint",
        "epochs": "训练轮数",
        "batch_size": "Batch size",
        "synth_ratio": "Synthetic:real 比例上限",
        "lr": "学习率",
        "queue_model_training": "排队医院模型训练",
        "queued_classifier": "已排队分类器训练任务",
        "export_artifact": "4. 导出部署产物",
        "model_to_export": "待导出模型",
        "queue_onnx": "排队 ONNX 导出",
        "queued_export": "已排队导出任务",
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
        "samples_per_pool": "Samples per pool",
        "tab_detection": "Detection",
        "tab_generation": "Generation",
        "tab_training": "Training",
        "class_col": "class",
        "probability_col": "probability",
        "label_col": "label@0.5",
        "inference_unavailable": "Inference unavailable",
        "no_sample_pools": "No sample pools found under configured demo/project paths.",
        "signal_sanity": "Signal sanity",
        "job_history": "Job history",
        "status": "Status",
        "pid": "PID",
        "stop_selected_job": "Stop selected job",
        "log_tail": "Log tail",
        "no_jobs": "No background jobs yet.",
        "pass": "PASS",
        "fail": "FAIL",
        "warning": "WARNING",
        "detection": "Detection",
        "detection_caption": "Upload or select a 12-lead ECG, then run the selected deployment backend.",
        "deployment_model": "Deployment model / engine",
        "upload_ecg": "Upload ECG (.npy/.npz)",
        "choose_demo_sample": "Or choose demo/project sample",
        "select_or_upload": "Select or upload an ECG to begin.",
        "input_ecg": "Input ECG",
        "prediction": "Prediction",
        "quality_gate": "Quality Gate",
        "download_report": "Download inference report JSON",
        "generation": "Generation",
        "generation_caption": "Manage target-center synthetic ECG pools. Live generation is queued as a background job.",
        "target_center": "Target center",
        "generation_mode": "Generation mode",
        "prompt_token_bank": "Prompt-token bank",
        "token_center": "Token center",
        "prompt_cache_root": "Prompt-token cache root",
        "classes": "Classes",
        "samples_per_class": "Samples per class",
        "steps": "DDIM/DDPM steps",
        "token_scale": "Token scale",
        "seed": "Seed",
        "output_directory": "Output directory",
        "no_token_bank": "No prompt-token bank found. Use no-token mode or train/select a token bank first.",
        "queue_generation": "Queue generation job",
        "queued_generation": "Queued generation job",
        "gate_generated": "Gate generated samples",
        "queue_gate": "Queue quality gate job",
        "queued_gate": "Queued gate job",
        "existing_synth_sample": "Existing synthetic sample",
        "synthetic_ecg": "Synthetic ECG",
        "sample_metadata": "Sample metadata",
        "plausibility_proxy": "Plausibility proxy",
        "background_jobs": "Background Jobs",
        "training": "Training",
        "training_caption": "Upload target-center ECG, train center prompt tokens, synthesize ECG, then train an exportable hospital model.",
        "upload_hospital_ecg": "1. Upload Hospital ECG",
        "ecg_signals": "ECG signals (.npy/.npz)",
        "labels_csv": "Labels CSV (optional if NPZ contains labels)",
        "save_audit": "Save and audit dataset",
        "saved_dataset": "Saved dataset",
        "dataset_intake_failed": "Dataset intake failed",
        "dataset_audit": "Dataset Audit",
        "samples": "Samples",
        "non_finite": "Non-finite",
        "flatline": "Flatline",
        "shape": "Shape",
        "no_audit": "No audited upload yet.",
        "center_token_jobs": "2. Center Token Adaptation Jobs",
        "k_refs": "K refs",
        "class_floor": "Class floor",
        "token_steps": "Token steps",
        "token_batch": "Token batch",
        "queue_cache": "Queue cache preparation",
        "queued_cache": "Queued cache job",
        "queue_token": "Queue center-token training",
        "queued_token": "Queued token job",
        "train_hospital_classifier": "3. Train Hospital Classifier",
        "synth_pool_training": "Synthetic pool for training",
        "initialize_checkpoint": "Initialize from checkpoint",
        "epochs": "Epochs",
        "batch_size": "Batch size",
        "synth_ratio": "Synthetic:real ratio cap",
        "lr": "LR",
        "queue_model_training": "Queue hospital model training",
        "queued_classifier": "Queued classifier job",
        "export_artifact": "4. Export Deployment Artifact",
        "model_to_export": "Model to export",
        "queue_onnx": "Queue ONNX export",
        "queued_export": "Queued export job",
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
def get_samples(paths: tuple[str, ...], max_items_per_file: int):
    return load_demo_samples(paths, max_items_per_file=max_items_per_file)


def _path_options(paths: list[Path], fallback: str = "") -> list[str]:
    out = [str(p) for p in paths]
    if fallback and fallback not in out:
        out.insert(0, fallback)
    return out or ([fallback] if fallback else [])


def _status_chip(status: str) -> str:
    if status == "pass":
        return tr("pass")
    if status == "fail":
        return tr("fail")
    return tr("warning")


def probability_table(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        tr("class_col"): CLASS_NAMES,
        tr("probability_col"): probabilities.astype(float),
        tr("label_col"): probabilities >= 0.5,
    })


def predict_panel(signal_ct: np.ndarray, backend_name: str, ckpt_path: str, device: str):
    try:
        backend = get_backend(backend_name, ckpt_path, device)
        pred = backend.predict(signal_ct)
    except Exception as exc:
        st.error(f"{tr('inference_unavailable')}: {exc}")
        return None
    probs = np.asarray(pred["probabilities"], dtype=np.float32)
    df = probability_table(probs)
    st.bar_chart(df.set_index(tr("class_col"))[tr("probability_col")])
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(
        f"backend={pred['backend']} device={pred['device']} "
        f"latency={pred['latency_ms']:.2f} ms"
    )
    return pred


def sample_selector(samples: list[dict], key: str, label: str = "ECG sample") -> dict | None:
    if not samples:
        st.warning(tr("no_sample_pools"))
        return None
    labels = [
        f"{i:03d} | {s.get('center', 'unknown')} | {','.join(s.get('label_names') or ['unknown'])}"
        for i, s in enumerate(samples)
    ]
    idx = st.selectbox(label, range(len(samples)), format_func=lambda i: labels[i], key=key)
    return samples[int(idx)]


def uploaded_signal(uploaded) -> tuple[np.ndarray | None, dict]:
    if uploaded is None:
        return None, {}
    if uploaded.name.endswith(".npy"):
        signal = np.load(uploaded)
    else:
        data = np.load(uploaded, allow_pickle=True)
        key = "signals" if "signals" in data.files else data.files[0]
        arr = data[key]
        signal = arr[0] if np.asarray(arr).ndim == 3 else arr
    return to_signal_ct(signal), {"source": uploaded.name}


def render_quality(signal_ct: np.ndarray):
    gate = run_quality_gate(signal_ct)
    st.metric(tr("signal_sanity"), _status_chip(str(gate["status"])))
    st.json(gate)
    return gate


def render_jobs(project_root: Path):
    jobs = list_jobs(project_root)
    if not jobs:
        st.caption(tr("no_jobs"))
        return
    selected = st.selectbox(tr("job_history"), jobs, format_func=lambda p: p.name)
    meta = read_job(Path(selected))
    cols = st.columns([0.22, 0.22, 0.56])
    cols[0].metric(tr("status"), str(meta.get("status", "unknown")))
    cols[1].metric(tr("pid"), str(meta.get("pid", "")))
    cols[2].code(" ".join(meta.get("command", [])), language="bash")
    if meta.get("status") == "running" and st.button(tr("stop_selected_job")):
        stop_job(Path(selected))
        st.rerun()
    st.text_area(tr("log_tail"), tail_log(Path(selected)), height=260)


def detection_page(project_root: Path | None, samples: list[dict], backend_name: str, ckpt_path: str, device: str):
    st.subheader(tr("detection"))
    st.caption(tr("detection_caption"))

    classifier_paths = list_classifier_artifacts(project_root)
    ckpt_options = _path_options(classifier_paths, fallback=ckpt_path)
    selected_model = st.selectbox(tr("deployment_model"), ckpt_options, index=0)

    left, right = st.columns([0.34, 0.66])
    with left:
        uploaded = st.file_uploader(tr("upload_ecg"), type=["npy", "npz"], key="detect_upload")
        selected = sample_selector(samples, key="detect_sample", label=tr("choose_demo_sample")) if uploaded is None else None
        if uploaded is not None:
            signal_ct, meta = uploaded_signal(uploaded)
        elif selected is not None:
            signal_ct = selected["signal"]
            meta = {k: v for k, v in selected.items() if k != "signal"}
        else:
            signal_ct, meta = None, {}
        st.json(meta)
    with right:
        if signal_ct is None:
            st.info(tr("select_or_upload"))
            return
        st.pyplot(make_ecg_figure(signal_ct, title=tr("input_ecg")))
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**{tr('prediction')}**")
            pred = predict_panel(signal_ct, backend_name, selected_model, device)
        with cols[1]:
            st.markdown(f"**{tr('quality_gate')}**")
            gate = render_quality(signal_ct)
        if pred:
            report = {
                "metadata": meta,
                "prediction": {k: v for k, v in pred.items() if k not in {"logits"}},
                "quality_gate": gate,
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
    no_token: bool,
    token_center: str,
    token_scale: float,
) -> list[str]:
    cmd = [
        DEFAULT_PYTHON,
        "scripts/ecgtwin_gen/generate_center_prompt_token_synth.py",
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
        "--n_per_class",
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
    ]
    if no_token:
        cmd.append("--no_token")
    else:
        cmd.extend(["--token_bank", token_bank, "--token_center", token_center])
    return cmd


def generation_page(project_root: Path | None, project_id: str | None, samples: list[dict], device: str):
    st.subheader(tr("generation"))
    st.caption(tr("generation_caption"))

    left, right = st.columns([0.34, 0.66])
    with left:
        center = st.text_input(tr("target_center"), value=project_id or "ningbo")
        mode = st.radio(tr("generation_mode"), ["target-token", "no-token", "wrong-token control"], horizontal=False)
        token_banks = list_token_banks(project_root)
        token_bank_options = _path_options(token_banks) or [""]
        token_bank = st.selectbox(tr("prompt_token_bank"), token_bank_options)
        token_center = st.text_input(tr("token_center"), value=center if mode != "wrong-token control" else "georgia")
        cache_root = str(prompt_cache_root(project_root)) if project_root and (prompt_cache_root(project_root) / "center_full_latents").exists() else "/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1"
        cache_root = st.text_input(tr("prompt_cache_root"), cache_root)
        selected_classes = st.multiselect(tr("classes"), CLASS_NAMES, default=["NORM", "MI", "STTC"])
        n_per_class = st.number_input(tr("samples_per_class"), min_value=1, max_value=200, value=8)
        steps = st.slider(tr("steps"), 5, 100, 25)
        token_scale = st.slider(tr("token_scale"), 0.0, 2.0, 1.0, step=0.05)
        seed = st.number_input(tr("seed"), min_value=0, value=42)
        out_base = str((project_root or Path("/root/autodl-tmp/streamlit_ecg_demo")) / "synthetic_pools" / f"gen_{int(time.time())}")
        out_dir = st.text_input(tr("output_directory"), out_base)
        needs_token = mode != "no-token"
        if needs_token and not token_bank:
            st.warning(tr("no_token_bank"))
        if project_root and st.button(tr("queue_generation"), disabled=(not selected_classes or (needs_token and not token_bank))):
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
                no_token=(mode == "no-token"),
                token_center=token_center,
                token_scale=float(token_scale),
            )
            job_dir = start_job(project_root, JobSpec("generate_synthetic_pool", cmd, str(REPO_ROOT)))
            st.success(f"{tr('queued_generation')}: {job_dir.name}")

        candidate_inputs = []
        if project_root:
            candidate_inputs.extend(sorted((project_root / "synthetic_pools").glob("**/samples.npz")))
        gate_input = st.selectbox(tr("gate_generated"), [str(p.parent) for p in candidate_inputs] or [""])
        if project_root and gate_input and st.button(tr("queue_gate")):
            cmd = [
                DEFAULT_PYTHON,
                "scripts/ecgtwin_gen/gate_prompt_token_synth.py",
                "--input_dir",
                gate_input,
                "--out_dir",
                str(Path(gate_input) / "gated"),
                "--classes",
                *selected_classes,
                "--min_target_prob",
                "0.30",
            ]
            job_dir = start_job(project_root, JobSpec("gate_synthetic_pool", cmd, str(REPO_ROOT)))
            st.success(f"{tr('queued_gate')}: {job_dir.name}")

    with right:
        selected = sample_selector(samples, key="generation_sample", label=tr("existing_synth_sample"))
        if selected is not None:
            signal_ct = selected["signal"]
            st.pyplot(make_ecg_figure(signal_ct, title=tr("synthetic_ecg")))
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**{tr('sample_metadata')}**")
                st.json({k: v for k, v in selected.items() if k != "signal"})
            with cols[1]:
                st.markdown(f"**{tr('plausibility_proxy')}**")
                render_quality(signal_ct)
        if project_root:
            st.markdown(f"**{tr('background_jobs')}**")
            render_jobs(project_root)


def training_page(project_root: Path, project_id: str, device: str):
    st.subheader(tr("training"))
    st.caption(tr("training_caption"))

    audit_path = canonical_audit_path(project_root)
    dataset_path = canonical_dataset_path(project_root)

    step1, step2 = st.columns([0.38, 0.62])
    with step1:
        st.markdown(f"**{tr('upload_hospital_ecg')}**")
        sig_upload = st.file_uploader(tr("ecg_signals"), type=["npy", "npz"], key="train_signal_upload")
        label_upload = st.file_uploader(tr("labels_csv"), type=["csv"], key="train_label_upload")
        if st.button(tr("save_audit"), disabled=sig_upload is None):
            try:
                saved_npz, saved_audit, audit = save_uploaded_dataset(
                    project_root,
                    sig_upload,
                    sig_upload.name,
                    label_upload,
                    label_upload.name if label_upload else None,
                )
                st.success(f"{tr('saved_dataset')}: {saved_npz}")
                st.json(audit)
            except Exception as exc:
                st.error(f"{tr('dataset_intake_failed')}: {exc}")

    with step2:
        st.markdown(f"**{tr('dataset_audit')}**")
        audit = load_audit(audit_path)
        if audit:
            cols = st.columns(4)
            cols[0].metric(tr("samples"), audit["n_samples"])
            cols[1].metric(tr("non_finite"), audit["nonfinite_count"])
            cols[2].metric(tr("flatline"), audit["flatline_count"])
            cols[3].metric(tr("shape"), "x".join(map(str, audit["signal_shape"])))
            st.dataframe(pd.DataFrame([
                {tr("class_col"): cls, "count": audit["class_counts"].get(cls, 0)}
                for cls in CLASS_NAMES
            ]), width="stretch", hide_index=True)
        else:
            st.info(tr("no_audit"))

    st.divider()
    st.markdown(f"**{tr('center_token_jobs')}**")
    c1, c2, c3, c4 = st.columns(4)
    K = c1.number_input(tr("k_refs"), min_value=1, max_value=5000, value=500)
    floor = c2.number_input(tr("class_floor"), min_value=0, max_value=200, value=10)
    steps = c3.number_input(tr("token_steps"), min_value=20, max_value=10000, value=800)
    batch_size = c4.number_input(tr("token_batch"), min_value=1, max_value=128, value=16)
    cache_root = str(prompt_cache_root(project_root))
    token_save_dir = str(project_root / "center_tokens" / f"token_{int(time.time())}")
    if st.button(tr("queue_cache"), disabled=not dataset_path.exists()):
        cmd = [
            DEFAULT_PYTHON,
            "scripts/streamlit_demo/prepare_hospital_prompt_cache.py",
            "--dataset_npz",
            str(dataset_path),
            "--center",
            project_id,
            "--out_root",
            cache_root,
            "--K",
            str(int(K)),
            "--floor_per_class",
            str(int(floor)),
            "--device",
            device,
        ]
        job_dir = start_job(project_root, JobSpec("prepare_prompt_cache", cmd, str(REPO_ROOT)))
        st.success(f"{tr('queued_cache')}: {job_dir.name}")
    if st.button(tr("queue_token")):
        cmd = [
            DEFAULT_PYTHON,
            "scripts/ecgtwin_gen/train_center_prompt_tokens.py",
            "--centers",
            project_id,
            "--cache_root",
            cache_root,
            "--prompt_bank",
            DEFAULT_PROMPT_BANK,
            "--save_dir",
            token_save_dir,
            "--K",
            str(int(K)),
            "--total_steps",
            str(int(steps)),
            "--batch_size",
            str(int(batch_size)),
            "--num_workers",
            "2",
            "--device",
            device,
            "--amp_dtype",
            "bf16",
            "--token_mode",
            "direct",
            "--n_token_vectors",
            "4",
            "--sample_strategy",
            "center_class_balanced",
            "--log_every",
            "20",
        ]
        job_dir = start_job(project_root, JobSpec("train_center_token", cmd, str(REPO_ROOT)))
        st.success(f"{tr('queued_token')}: {job_dir.name}")

    st.divider()
    st.markdown(f"**{tr('train_hospital_classifier')}**")
    synth_options = _path_options(list_synthetic_pools(project_root)) or [""]
    model_options = _path_options(list_classifier_artifacts(project_root), fallback=DEFAULT_CKPT)
    synth_npz = st.selectbox(tr("synth_pool_training"), synth_options)
    init_ckpt = st.selectbox(tr("initialize_checkpoint"), model_options)
    m1, m2, m3, m4 = st.columns(4)
    epochs = m1.number_input(tr("epochs"), min_value=1, max_value=200, value=20)
    model_batch = m2.number_input(tr("batch_size"), min_value=1, max_value=256, value=32)
    synth_ratio = m3.slider(tr("synth_ratio"), 0.0, 5.0, 1.0, step=0.25)
    lr = m4.number_input(tr("lr"), min_value=1e-6, max_value=1e-2, value=1e-4, format="%g")
    model_out = str(project_root / "classifier_runs" / f"hospital_effnet_{int(time.time())}")
    if st.button(tr("queue_model_training"), disabled=not dataset_path.exists()):
        cmd = [
            DEFAULT_PYTHON,
            "scripts/streamlit_demo/train_hospital_classifier.py",
            "--dataset_npz",
            str(dataset_path),
            "--output_dir",
            model_out,
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
        if synth_npz:
            cmd.extend(["--synth_npz", synth_npz])
        job_dir = start_job(project_root, JobSpec("train_hospital_classifier", cmd, str(REPO_ROOT)))
        st.success(f"{tr('queued_classifier')}: {job_dir.name}")

    st.markdown(f"**{tr('export_artifact')}**")
    export_model = st.selectbox(tr("model_to_export"), _path_options(list_classifier_artifacts(project_root), fallback=DEFAULT_CKPT), key="export_model")
    export_out = str(project_root / "exports" / "efficientnetv2_super5.onnx")
    if st.button(tr("queue_onnx")):
        cmd = [
            DEFAULT_PYTHON,
            "scripts/deploy/export_efficientnetv2_onnx.py",
            "--ckpt",
            export_model,
            "--out",
            export_out,
            "--device",
            "cuda" if device.startswith("cuda") else "cpu",
        ]
        job_dir = start_job(project_root, JobSpec("export_onnx", cmd, str(REPO_ROOT)))
        st.success(f"{tr('queued_export')}: {job_dir.name}")

    st.divider()
    render_jobs(project_root)


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
        backend_name = st.selectbox(tr("classifier_backend"), ["pytorch", "onnxruntime", "tensorrt"])
        device = st.selectbox(tr("device"), ["cuda:0", "cuda", "cpu"])
        default_model = str(list_classifier_artifacts(active_root)[0]) if list_classifier_artifacts(active_root) else DEFAULT_CKPT
        ckpt_path = st.text_input(tr("default_model_checkpoint"), default_model)
        max_items = st.slider(tr("samples_per_pool"), 4, 64, 16)

    st.title(tr("app_title"))
    st.caption(tr("app_caption"))

    sample_paths = [str(p) for p in list_synthetic_pools(active_root)]
    samples = get_samples(tuple(sample_paths), max_items)
    page_detection, page_generation, page_training = st.tabs([
        tr("tab_detection"),
        tr("tab_generation"),
        tr("tab_training"),
    ])
    with page_detection:
        detection_page(active_root, samples, backend_name, ckpt_path, device)
    with page_generation:
        generation_page(active_root, project_id, samples, device)
    with page_training:
        training_page(active_root, project_id, device)


if __name__ == "__main__":
    main()
