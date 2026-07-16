"""
PTBXL 数据预处理：将 PTBXL 数据集转为 ECGTwin 可用的 .pt 格式

流程：
1. 加载 raw100.npy (100Hz, 1000 samples) → 线性插值到 1024
2. 导联重排 (PTBXL → ECGTwin: swap aVL/aVF at positions 4,5)
3. SCP code → 自然语言文本
4. VAE 编码 → latent (4, 128)
5. Nomic 文本嵌入 (split mode)
6. 心率估算 (R-peak detection)
7. 诊断类别标注 (diagnostic_class)
8. 保存为 .pt

输出格式:
[{
    'data': tensor(4, 128),        # VAE latent
    'label': {
        'text': "sinus rhythm|normal ecg",
        'text_embed': tensor(L, 768),
        'hr': 72.5,
        'age': 56.0,
        'sex': 'F',
        'ecg_id': 1,
        'patient_id': 15709,
        'diagnostic_class': 'NORM',
        'strat_fold': 1,
    }
}, ...]
"""

import sys
import os
import ast
import argparse
from pathlib import Path

# ——— TensorFlow segfault workaround ———
# 步骤1：先把 tensorflow 设为 None，让 transformers' find_spec 认为 TF 不可用（不会崩溃）
# 步骤2：import transformers 之后再删掉这个 None 条目，让 einops 在运行时跳过 TF backend
# 原理：sys.modules['tensorflow'] = None → importlib.find_spec 返回 None → TF 被标为不可用
#       del sys.modules['tensorflow'] → einops._backends 的 'tensorflow' in sys.modules 为 False
import sys as _sys
_sys.modules['tensorflow'] = None  # type: ignore  # 步骤1：阻止 TF 初始化（防止 segfault）

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal
from tqdm import tqdm

# 添加 ECGTwin 路径
PROJECT_ROOT = Path(__file__).parent.parent
ECGTWIN_ROOT = PROJECT_ROOT / "model" / "ECGTwin"
sys.path.insert(0, str(ECGTWIN_ROOT))

from module.vae_model import VAE_Encoder
from transformers import AutoTokenizer, AutoModel

# 步骤2：transformers 已经 import 完毕，现在删掉 None 条目让 einops 正常工作
import sys as _sys2
if _sys2.modules.get('tensorflow') is None:
    del _sys2.modules['tensorflow']

# 导联重排: PTBXL [I,II,III,AVR,AVL,AVF,...] → ECGTwin [I,II,III,aVR,aVF,aVL,...]
PTBXL_TO_ECGTWIN_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]


def _linear_interpolate_time_batch(
    signals_ntc: np.ndarray,
    target_num_samples: int,
) -> np.ndarray:
    """Resize ``(N,T,C)`` signals with aligned-corner linear interpolation."""

    signals_ntc = np.asarray(signals_ntc)
    if signals_ntc.ndim != 3 or signals_ntc.shape[1] < 2:
        raise ValueError(
            "Expected batched time-channel signals with at least two time points, "
            f"got {signals_ntc.shape}"
        )
    if target_num_samples < 2:
        raise ValueError(
            f"target_num_samples must be at least 2, got {target_num_samples}"
        )
    source = torch.from_numpy(np.array(signals_ntc, copy=True, order="C")).permute(0, 2, 1)
    with torch.no_grad():
        resized = F.interpolate(
            source,
            size=int(target_num_samples),
            mode="linear",
            align_corners=True,
        )
    return resized.permute(0, 2, 1).contiguous().numpy()


def load_scp_mapping(scp_statements_path: str) -> dict:
    """从 scp_statements.csv 加载 SCP code → description 映射"""
    scp_df = pd.read_csv(scp_statements_path, index_col=0)
    mapping = {}
    for code, row in scp_df.iterrows():
        desc = row.get("description", "")
        if pd.notna(desc) and desc.strip():
            mapping[code] = desc.strip().lower()
    return mapping


def load_scp_diagnostic_class(scp_statements_path: str) -> dict:
    """从 scp_statements.csv 加载 SCP code → diagnostic_class 映射"""
    scp_df = pd.read_csv(scp_statements_path, index_col=0)
    mapping = {}
    for code, row in scp_df.iterrows():
        dc = row.get("diagnostic_class", "")
        if pd.notna(dc) and str(dc).strip():
            mapping[code] = str(dc).strip()
    return mapping


def scp_to_text(scp_codes: dict, scp_mapping: dict, threshold: float = 50.0) -> str:
    """
    将 SCP codes 字典转换为 ECGTwin 格式的文本描述
    多个诊断用 | 分隔
    """
    # 按 likelihood 降序排列
    sorted_codes = sorted(scp_codes.items(), key=lambda x: x[1], reverse=True)
    descriptions = []
    for code, likelihood in sorted_codes:
        if likelihood < threshold:
            continue
        if code in scp_mapping:
            descriptions.append(scp_mapping[code])
    if not descriptions:
        # fallback: 取 likelihood 最高的
        if sorted_codes:
            code = sorted_codes[0][0]
            if code in scp_mapping:
                descriptions.append(scp_mapping[code])
    if not descriptions:
        descriptions = ["unknown"]
    return "|".join(descriptions)


def get_diagnostic_class(scp_codes: dict, dc_mapping: dict, threshold: float = 50.0) -> str:
    """获取样本的 superdiagnostic class (CD/HYP/MI/NORM/STTC)"""
    sorted_codes = sorted(scp_codes.items(), key=lambda x: x[1], reverse=True)
    for code, likelihood in sorted_codes:
        if likelihood >= threshold and code in dc_mapping:
            return dc_mapping[code]
    # fallback: 取最高 likelihood 的
    for code, _ in sorted_codes:
        if code in dc_mapping:
            return dc_mapping[code]
    return "UNKNOWN"


def estimate_heart_rate(ecg_signal: np.ndarray, fs: float = 100.0) -> float:
    """
    从 ECG 信号估算心率
    使用 lead II (index 1) 的 R-peak 检测
    """
    try:
        lead_ii = ecg_signal[:, 1]
        lead_ii = np.nan_to_num(lead_ii)

        # 简单的 R-peak 检测: 带通滤波 + 峰值检测
        # 带通滤波 5-25Hz
        sos = scipy_signal.butter(4, [5, 25], btype='bandpass', fs=fs, output='sos')
        filtered = scipy_signal.sosfilt(sos, lead_ii)

        # 峰值检测
        min_distance = int(0.4 * fs)  # 至少 0.4s 间隔 (最高 150bpm)
        height_threshold = np.std(filtered) * 0.5
        peaks, _ = scipy_signal.find_peaks(
            filtered, distance=min_distance, height=height_threshold
        )

        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / fs  # 转为秒
            mean_rr = np.mean(rr_intervals)
            hr = 60.0 / mean_rr
            if 30 <= hr <= 200:
                return round(hr, 1)
    except Exception:
        pass

    return 75.0  # fallback


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@torch.no_grad()
def compute_nomic_embeddings(
    texts: list, tokenizer, model, device: str, batch_size: int = 64
) -> list:
    """
    计算 nomic text embeddings (split mode)
    每个 text 按 | 分隔后分别嵌入
    返回 list of tensor, 每个 tensor shape (num_diagnoses, 768)
    """
    embeddings = []
    for text in tqdm(texts, desc="Computing text embeddings"):
        reports = text.split("|")
        encoded = tokenizer(
            reports, padding=True, truncation=True, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = model(**encoded)
        emb = mean_pooling(output, encoded["attention_mask"])
        emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))
        emb = emb[:, :768]
        emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu())
    return embeddings


@torch.no_grad()
def vae_encode_batch(
    ecg_signals: np.ndarray, encoder: VAE_Encoder, device: str, batch_size: int = 64
) -> list:
    """
    批量 VAE 编码
    ecg_signals: (N, 1024, 12) numpy array
    返回 list of tensor, 每个 shape (4, 128)
    """
    latents = []
    n = len(ecg_signals)
    for i in tqdm(range(0, n, batch_size), desc="VAE encoding"):
        batch = ecg_signals[i : i + batch_size]
        batch_tensor = torch.from_numpy(batch).float().to(device)
        # VAE_Encoder expects (B, L, 12)
        latent, _, _ = encoder(batch_tensor)
        for j in range(latent.shape[0]):
            latents.append(latent[j].cpu())
    return latents


def prepare_ptbxl(
    ptbxl_path: str,
    output_path: str,
    vae_checkpoint_path: str,
    device: str = "cuda:0",
    scp_threshold: float = 50.0,
    vae_batch_size: int = 64,
):
    """主函数: 准备 PTBXL 数据"""
    ptbxl_path = Path(ptbxl_path)

    # 1. 加载元数据
    print("Loading PTBXL metadata...")
    db = pd.read_csv(ptbxl_path / "ptbxl_database.csv")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

    scp_mapping = load_scp_mapping(ptbxl_path / "scp_statements.csv")
    dc_mapping = load_scp_diagnostic_class(ptbxl_path / "scp_statements.csv")

    # 2. 加载 ECG 信号 (100Hz, 1000 samples)
    print("Loading ECG signals from raw100.npy...")
    raw_signals = np.load(ptbxl_path / "raw100.npy", allow_pickle=True)
    raw_signals = np.nan_to_num(raw_signals)
    print(f"  Shape: {raw_signals.shape}")  # (21799, 1000, 12)

    # 3. 线性插值 1000 → 1024 并重排导联
    print("Linearly interpolating to 1024 and reordering leads...")
    resampled = np.empty(
        (len(raw_signals), 1024, raw_signals.shape[2]),
        dtype=raw_signals.dtype,
    )
    interpolation_batch_size = 256
    for start in range(0, len(raw_signals), interpolation_batch_size):
        end = min(start + interpolation_batch_size, len(raw_signals))
        resampled[start:end] = _linear_interpolate_time_batch(
            raw_signals[start:end],
            1024,
        )
    # 导联重排: PTBXL → ECGTwin
    resampled = resampled[:, :, PTBXL_TO_ECGTWIN_INDICES]

    # 4. 心率估算 (使用原始 100Hz 信号)
    print("Estimating heart rates...")
    heart_rates = []
    for i in tqdm(range(len(raw_signals)), desc="HR estimation"):
        hr = estimate_heart_rate(raw_signals[i], fs=100.0)
        heart_rates.append(hr)

    # 5. SCP → text + diagnostic class
    print("Converting SCP codes to text...")
    texts = []
    diag_classes = []
    for _, row in db.iterrows():
        text = scp_to_text(row["scp_codes"], scp_mapping, scp_threshold)
        dc = get_diagnostic_class(row["scp_codes"], dc_mapping, scp_threshold)
        texts.append(text)
        diag_classes.append(dc)

    # 6. VAE 编码
    print("Loading VAE encoder...")
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
    encoder = VAE_Encoder()
    encoder.load_state_dict(vae_checkpoint["encoder"])
    encoder.to(device)
    encoder.eval()

    print("VAE encoding...")
    latents = vae_encode_batch(resampled, encoder, device, vae_batch_size)
    del encoder
    torch.cuda.empty_cache()

    # 7. Nomic text embedding
    print("Loading nomic text embedding model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedding_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True,
    )
    embedding_model.to(device)
    embedding_model.eval()

    print("Computing text embeddings...")
    text_embeds = compute_nomic_embeddings(texts, tokenizer, embedding_model, device)
    del embedding_model
    torch.cuda.empty_cache()

    # 8. 组装数据集
    print("Assembling dataset...")
    dataset = []
    for i in tqdm(range(len(db)), desc="Assembling"):
        row = db.iloc[i]
        # PTBXL sex: 0=male→'M', 1=female→'F'
        sex = "F" if row["sex"] == 1 else "M"
        age = float(row["age"]) if pd.notna(row["age"]) else 60.0

        entry = {
            "data": latents[i],  # (4, 128)
            "label": {
                "text": texts[i],
                "text_embed": text_embeds[i],  # (num_reports, 768)
                "hr": heart_rates[i],
                "age": age,
                "sex": sex,
                "ecg_id": int(row["ecg_id"]),
                "patient_id": int(row["patient_id"]),
                "diagnostic_class": diag_classes[i],
                "strat_fold": int(row["strat_fold"]),
            },
        }
        dataset.append(entry)

    # 9. 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset ({len(dataset)} samples) to {output_path}...")
    torch.save(dataset, output_path)

    # 10. 打印统计
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Latent shape: {dataset[0]['data'].shape}")
    print(f"Text embed shape example: {dataset[0]['label']['text_embed'].shape}")

    dc_counts = {}
    for entry in dataset:
        dc = entry["label"]["diagnostic_class"]
        dc_counts[dc] = dc_counts.get(dc, 0) + 1
    print(f"Diagnostic class distribution:")
    for dc, count in sorted(dc_counts.items()):
        print(f"  {dc}: {count}")

    fold_counts = {}
    for entry in dataset:
        fold = entry["label"]["strat_fold"]
        fold_counts[fold] = fold_counts.get(fold, 0) + 1
    print(f"Fold distribution:")
    for fold, count in sorted(fold_counts.items()):
        print(f"  Fold {fold}: {count}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PTBXL for ECGTwin")
    parser.add_argument(
        "--ptbxl_path",
        type=str,
        default="datasets/PTBXL",
        help="Path to PTBXL dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/PTBXL/PTBXL_vae_multi_nomic.pt",
        help="Output .pt file path",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="model/ECGTwin/checkpoints/vae_model.pth",
        help="VAE checkpoint path",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scp_threshold", type=float, default=50.0)
    parser.add_argument("--vae_batch_size", type=int, default=64)
    args = parser.parse_args()

    prepare_ptbxl(
        ptbxl_path=args.ptbxl_path,
        output_path=args.output_path,
        vae_checkpoint_path=args.vae_path,
        device=args.device,
        scp_threshold=args.scp_threshold,
        vae_batch_size=args.vae_batch_size,
    )
