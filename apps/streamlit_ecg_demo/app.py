from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from apps.streamlit_ecg_demo.components.ecg_plot import make_ecg_figure
from apps.streamlit_ecg_demo.services.classifier_backend import DEFAULT_CKPT, load_classifier_backend
from apps.streamlit_ecg_demo.services.preprocessing import (
    CLASS_NAMES,
    apply_corruption,
    load_demo_samples,
    to_signal_ct,
)
from apps.streamlit_ecg_demo.services.quality_gate import run_quality_gate


DEFAULT_SYNTH_PATHS = [
    "/root/autodl-tmp/ecgtwin_prompt_token_super5/"
    "effectiveness_pilot_v42_task1gate_ningbo_token_scale_large_20260504/"
    "target_token_s05/ningbo/gated/gated_samples.npz",
    "/root/autodl-tmp/ecgtwin_prompt_token_super5/"
    "effectiveness_pilot_v42_task1gate_ningbo_no_token_large_20260504/"
    "no_token/ningbo/gated/gated_samples.npz",
    "/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced/"
    "ningbo/gated/gated_samples.npz",
]


st.set_page_config(page_title="ECG LDM Demo", layout="wide")


@st.cache_resource(show_spinner=False)
def get_backend(backend_name: str, ckpt_path: str, device: str):
    return load_classifier_backend(backend_name, ckpt_path=ckpt_path, device=device)


@st.cache_data(show_spinner=False)
def get_samples(paths: tuple[str, ...], max_items_per_file: int):
    return load_demo_samples(paths, max_items_per_file=max_items_per_file)


def predict_panel(signal_ct: np.ndarray, backend_name: str, ckpt_path: str, device: str):
    try:
        backend = get_backend(backend_name, ckpt_path, device)
        pred = backend.predict(signal_ct)
    except Exception as exc:
        st.error(f"Inference unavailable: {exc}")
        return None
    probs = np.asarray(pred["probabilities"], dtype=np.float32)
    df = pd.DataFrame({"class": CLASS_NAMES, "probability": probs})
    st.bar_chart(df.set_index("class"))
    st.caption(
        f"backend={pred['backend']} device={pred['device']} "
        f"latency={pred['latency_ms']:.2f} ms"
    )
    st.json({
        "predicted_labels_at_0.5": pred["predicted_labels"],
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
    })
    return pred


def sample_selector(samples: list[dict], key: str) -> dict | None:
    if not samples:
        st.warning("No demo samples found. Check /root/autodl-tmp generated pools.")
        return None
    labels = [
        f"{i:03d} | {s['center']} | {','.join(s.get('label_names') or ['unknown'])}"
        for i, s in enumerate(samples)
    ]
    idx = st.selectbox(
        "Demo sample",
        range(len(samples)),
        format_func=lambda i: labels[i],
        key=key,
    )
    return samples[int(idx)]


with st.sidebar:
    st.header("Backend")
    backend_name = st.selectbox("Classifier backend", ["pytorch", "onnxruntime", "tensorrt"])
    device = st.selectbox("Device", ["cuda", "cpu"])
    ckpt_path = st.text_input("EfficientNet1DV2 checkpoint", DEFAULT_CKPT)
    st.header("Samples")
    max_items = st.slider("Samples per generated pool", 4, 64, 16)

st.title("Latent Diffusion ECG Generation And Super5 Detection")
st.caption("PTB-XL order, 100Hz, 10s. Probabilities are model outputs, not clinical diagnosis.")

samples = get_samples(tuple(DEFAULT_SYNTH_PATHS), max_items)
tab_detect, tab_generate, tab_robust, tab_bench = st.tabs([
    "Detection", "Generation", "Robustness", "Benchmark"
])

with tab_detect:
    col_a, col_b = st.columns([0.38, 0.62])
    with col_a:
        uploaded = st.file_uploader("Upload .npy or .npz ECG", type=["npy", "npz"])
        selected = sample_selector(samples, key="detect_demo_sample") if uploaded is None else None
        if uploaded is not None:
            if uploaded.name.endswith(".npy"):
                signal = np.load(uploaded)
            else:
                data = np.load(uploaded, allow_pickle=True)
                key = "signals" if "signals" in data else data.files[0]
                signal = data[key][0] if np.asarray(data[key]).ndim == 3 else data[key]
            signal_ct = to_signal_ct(signal)
            meta = {"source": uploaded.name}
        elif selected is not None:
            signal_ct = selected["signal"]
            meta = {k: v for k, v in selected.items() if k != "signal"}
        else:
            signal_ct, meta = None, {}
        st.json(meta)
    with col_b:
        if signal_ct is not None:
            st.pyplot(make_ecg_figure(signal_ct, title="Input ECG"))
            st.subheader("Prediction")
            predict_panel(signal_ct, backend_name, ckpt_path, device)
            st.subheader("Quality Gate")
            st.json(run_quality_gate(signal_ct))

with tab_generate:
    st.info("MVP uses precomputed ECGTwin generated pools for stable defense demo; live generation can be added later.")
    selected = sample_selector(samples, key="generation_demo_sample")
    if selected is not None:
        signal_ct = selected["signal"]
        st.pyplot(make_ecg_figure(signal_ct, title="Precomputed ECGTwin Sample"))
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Generator Metadata")
            st.json({k: v for k, v in selected.items() if k != "signal"})
        with cols[1]:
            st.subheader("Detection On Generated ECG")
            predict_panel(signal_ct, backend_name, ckpt_path, device)
        st.subheader("Medical Plausibility Proxy")
        st.json(run_quality_gate(signal_ct))

with tab_robust:
    selected = sample_selector(samples, key="robustness_demo_sample")
    corruption = st.selectbox(
        "Corruption",
        ["none", "powerline_noise", "emg_noise", "baseline_wander", "baseline_shift", "random_leads_masking"],
    )
    severity = st.slider("Severity", 1, 5, 3)
    if selected is not None:
        clean = selected["signal"]
        corrupt = apply_corruption(clean, corruption, severity)
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(make_ecg_figure(clean, title="Clean"))
            st.write("Clean prediction")
            clean_pred = predict_panel(clean, backend_name, ckpt_path, device)
        with c2:
            st.pyplot(make_ecg_figure(corrupt, title=f"{corruption} severity={severity}"))
            st.write("Corrupted prediction")
            corrupt_pred = predict_panel(corrupt, backend_name, ckpt_path, device)
        if clean_pred and corrupt_pred:
            delta = np.asarray(corrupt_pred["probabilities"]) - np.asarray(clean_pred["probabilities"])
            st.write(pd.DataFrame({"class": CLASS_NAMES, "prob_delta": delta}))

with tab_bench:
    st.write("Use deployment scripts for reproducible latency numbers.")
    st.code(
        "python scripts/deploy/export_efficientnetv2_onnx.py --ckpt " + ckpt_path + "\n"
        "python scripts/deploy/benchmark_inference_backends.py --ckpt " + ckpt_path,
        language="bash",
    )
    report_path = Path("/root/autodl-tmp/streamlit_ecg_demo/reports/inference_benchmark.json")
    if report_path.exists():
        st.json(json.loads(report_path.read_text()))
    else:
        st.warning("No benchmark report found yet.")
