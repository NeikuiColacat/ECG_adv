"""
微调训练循环

训练数据：
  1. 生成的困难样本 (~6400 条)，来自 adv_generate.py
  2. PTBXL 真实数据（fold 1-8），resample 1000→2500，SCP → 77-dim 标签

只训练 EfficientNetAdapter 的 adapter 参数，JIT backbone 始终冻结。
"""

import sys
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
_PTBXL_CODE_ROOT = _PROJECT_ROOT / "model" / "ecg_ptbxl_benchmarking" / "code"

for p in [str(_PROJECT_ROOT), str(_PTBXL_CODE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from adversarial.efficientnet_adapter import EfficientNetAdapter
from adversarial.label_mapping import scp_codes_to_77_labels, batch_scp_to_77_labels


# ——— 训练默认配置 ———
DEFAULT_TRAIN_CONFIG = {
    "epochs": 30,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "generated_weight": 2.0,   # 困难样本的损失权重倍数
    "val_fold": 9,
    "hidden_dim": 128,
    "dropout": 0.3,
    "grad_clip": 1.0,
}

EFFICIENTNET_INPUT_LENGTH = 2500


def load_ptbxl_for_finetune(
    ptbxl_data_root: str,
    val_fold: int = 9,
    n_per_superdiag: Optional[int] = None,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    加载 PTBXL 数据集用于微调

    使用 fold 1-8（排除 val_fold）作为训练集，val_fold 作为验证集。
    ECG: resample 1000→2500, 转换为 (B, 12, 2500)
    标签: SCP codes → 77-dim binary

    Args:
        ptbxl_data_root: PTBXL 数据集根目录
        val_fold: 验证集 fold 编号（默认 9）
        n_per_superdiag: 每个 superdiagnostic class 最多取多少训练样本
                         None = 全部数据；设置小值（如 50）模拟少样本场景
        seed: 随机种子

    Returns:
        train_dataset, val_dataset
    """
    import pandas as pd
    from scipy.signal import resample

    data_root = Path(ptbxl_data_root)
    raw_npy_path = data_root / "raw100.npy"
    csv_path = data_root / "ptbxl_database.csv"

    print(f"Loading PTBXL from {data_root}")
    X_all = np.load(str(raw_npy_path), allow_pickle=True)  # (21799, 1000, 12)
    df = pd.read_csv(str(csv_path), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # 从 scp_statements.csv 推导 diagnostic_class（NORM/MI/CD/HYP/STTC）
    # ptbxl_database.csv 本身没有这列，需要从 SCP code 映射
    scp_statements_path = data_root / "scp_statements.csv"
    if scp_statements_path.exists():
        scp_stmts = pd.read_csv(str(scp_statements_path), index_col=0)
        dc_map = {
            code: str(row.get("diagnostic_class", "")).strip()
            for code, row in scp_stmts.iterrows()
            if pd.notna(row.get("diagnostic_class", "")) and str(row.get("diagnostic_class", "")).strip()
        }

        def _get_diag_class(scp_dict):
            sorted_codes = sorted(scp_dict.items(), key=lambda x: x[1], reverse=True)
            for code, likelihood in sorted_codes:
                if likelihood >= 50.0 and code in dc_map:
                    return dc_map[code]
            for code, _ in sorted_codes:
                if code in dc_map:
                    return dc_map[code]
            return "UNKNOWN"

        df["diagnostic_class"] = df["scp_codes"].apply(_get_diag_class)
    else:
        df["diagnostic_class"] = "UNKNOWN"

    # 用 reset_index 获取位置索引（0-based），避免 ecg_id 间隙问题
    df_reset = df.reset_index()

    train_folds = [f for f in range(1, 9) if f != val_fold]
    train_df = df_reset[df_reset["strat_fold"].isin(train_folds)].copy()
    val_df = df_reset[df_reset["strat_fold"] == val_fold].copy()

    # 少样本采样（按 diagnostic_class 分层，只对训练集采样）
    if n_per_superdiag is not None:
        rng = np.random.RandomState(seed)
        sampled_parts = []
        for cls, grp in train_df.groupby("diagnostic_class"):
            n = min(len(grp), n_per_superdiag)
            sampled_parts.append(grp.sample(n=n, random_state=rng))
        train_df = pd.concat(sampled_parts).sort_index()
        dc_dist = train_df["diagnostic_class"].value_counts().to_dict()
        print(f"  Few-shot {n_per_superdiag}/class → {len(train_df)} train samples: {dc_dist}")

    def prepare_split(split_df):
        pos_idx = split_df.index.values
        X = X_all[pos_idx]                                              # (N, 1000, 12)
        scp_list = split_df["scp_codes"].tolist()
        X_resampled = resample(X, EFFICIENTNET_INPUT_LENGTH, axis=1)    # (N, 2500, 12)
        X_tensor = torch.from_numpy(X_resampled).float().transpose(1, 2)  # (N, 12, 2500)
        labels = batch_scp_to_77_labels(scp_list, threshold=0.0)
        return TensorDataset(X_tensor, labels)

    train_ds = prepare_split(train_df)
    val_ds = prepare_split(val_df)
    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    return train_ds, val_ds


def load_generated_samples(generated_path: str) -> TensorDataset:
    """
    加载生成的困难样本

    Args:
        generated_path: all_hard_samples.pt 的路径

    Returns:
        TensorDataset(ecg, labels_77)
    """
    data = torch.load(generated_path, map_location="cpu")
    ecg = data["ecg"]           # (N, 12, 2500)
    labels_77 = data["labels_77"]  # (N, 77)
    print(f"Loaded {ecg.shape[0]} generated hard samples from {generated_path}")
    return TensorDataset(ecg, labels_77)


def compute_pos_weight(labels: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    """
    计算 BCEWithLogitsLoss 的 pos_weight（处理类别不平衡）

    pos_weight[i] = (N - pos[i]) / (pos[i] + epsilon)

    Args:
        labels: (N, 77) 二值标签
        epsilon: 平滑系数，避免除零

    Returns:
        pos_weight: (77,)
    """
    N = labels.shape[0]
    pos = labels.sum(dim=0).clamp(min=epsilon)
    neg = N - pos
    return (neg / pos).clamp(max=100.0)


def train_one_epoch(
    adapter: EfficientNetAdapter,
    train_loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    grad_clip: float = 1.0,
    ewa_params: Optional[List[torch.Tensor]] = None,
    anchor_lambda: float = 0.1,
    ewa_decay: float = 0.999,
) -> float:
    """
    训练 adapter 一个 epoch（供 per-epoch 在线训练使用）

    Args:
        adapter: EfficientNetAdapter
        train_loader: 混合 DataLoader（real + generated）
        optimizer: 优化器
        criterion: BCEWithLogitsLoss
        device: 设备
        grad_clip: 梯度裁剪
        ewa_params: EWA anchor 参数列表（None 则不使用 anchor 正则化）
        anchor_lambda: anchor 正则化强度
        ewa_decay: EWA 衰减系数

    Returns:
        平均 train loss
    """
    adapter.train()
    train_losses = []

    for ecg, labels in train_loader:
        ecg = ecg.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        adapted_logits = adapter(ecg)
        bce_loss = criterion(adapted_logits, labels)

        # EWA anchor 正则化（SA-AET 启发：防止灾难性遗忘）
        if ewa_params is not None:
            anchor_loss = sum(
                (p - p_ewa.detach()).pow(2).sum()
                for p, p_ewa in zip(adapter.adapter_parameters(), ewa_params)
            )
            loss = bce_loss + anchor_lambda * anchor_loss
        else:
            loss = bce_loss

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(adapter.adapter_parameters(), grad_clip)

        optimizer.step()

        # EWA 更新
        if ewa_params is not None:
            with torch.no_grad():
                for p, p_ewa in zip(adapter.adapter_parameters(), ewa_params):
                    p_ewa.mul_(ewa_decay).add_(p.data, alpha=1 - ewa_decay)

        train_losses.append(bce_loss.item())

    return float(np.mean(train_losses))


def validate_one_epoch(
    adapter: EfficientNetAdapter,
    val_loader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    """验证一个 epoch，返回平均 val loss"""
    adapter.eval()
    val_losses = []
    with torch.no_grad():
        for ecg, labels in val_loader:
            ecg = ecg.to(device)
            labels = labels.to(device)
            adapted_logits = adapter(ecg)
            val_loss = criterion(adapted_logits, labels)
            val_losses.append(val_loss.item())
    return float(np.mean(val_losses))


def train_adapter(
    adapter: EfficientNetAdapter,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    generated_ds: Optional[TensorDataset] = None,
    config: Dict = None,
    save_path: str = None,
    device: str = "cuda",
    initial_adapter_path: Optional[str] = None,
) -> Dict:
    """
    训练 adapter

    Args:
        adapter: EfficientNetAdapter 实例
        train_ds: PTBXL 训练集 (ecg, labels_77)
        val_ds: PTBXL 验证集
        generated_ds: 生成的困难样本（可选，如有会以更高权重混入）
        config: 训练配置
        save_path: 最优 adapter 权重保存路径
        device: 设备

    Returns:
        训练历史 dict
    """
    cfg = {**DEFAULT_TRAIN_CONFIG, **(config or {})}
    device = torch.device(device)
    adapter = adapter.to(device)

    # Warm-start: 加载之前轮次的 adapter 权重
    if initial_adapter_path and Path(initial_adapter_path).exists():
        adapter.load_adapter(initial_adapter_path)
        print(f"Warm-start from {initial_adapter_path}")

    # 构建训练数据集（混合真实 + 生成）
    if generated_ds is not None:
        # 用 WeightedRandomSampler 给生成样本更高权重
        n_real = len(train_ds)
        n_gen = len(generated_ds)
        real_weight = 1.0
        gen_weight = cfg["generated_weight"]
        weights = [real_weight] * n_real + [gen_weight] * n_gen
        sampler = WeightedRandomSampler(weights, num_samples=n_real + n_gen, replacement=True)
        combined_ds = ConcatDataset([train_ds, generated_ds])
        train_loader = DataLoader(
            combined_ds,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        print(f"Combined dataset: {n_real} real + {n_gen} generated")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 计算 pos_weight
    all_labels = torch.cat([labels for _, labels in train_ds], dim=0)
    if generated_ds is not None:
        gen_labels = torch.cat([labels for _, labels in generated_ds], dim=0)
        all_labels = torch.cat([all_labels, gen_labels], dim=0)
    pos_weight = compute_pos_weight(all_labels).to(device)

    # Optimizer & Scheduler（只优化 adapter 参数）
    optimizer = AdamW(adapter.adapter_parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print(f"\nTraining adapter ({adapter.count_adapter_params():,} params) for {cfg['epochs']} epochs")
    print(f"  lr={cfg['lr']}, batch_size={cfg['batch_size']}, generated_weight={cfg['generated_weight']}")

    for epoch in range(cfg["epochs"]):
        # ——— Training ———
        adapter.train()
        train_losses = []

        for ecg, labels in train_loader:
            ecg = ecg.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            adapted_logits = adapter(ecg)
            loss = criterion(adapted_logits, labels)
            loss.backward()

            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(adapter.adapter_parameters(), cfg["grad_clip"])

            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ——— Validation ———
        adapter.eval()
        val_losses = []

        with torch.no_grad():
            for ecg, labels in val_loader:
                ecg = ecg.to(device)
                labels = labels.to(device)
                adapted_logits = adapter(ecg)
                val_loss = criterion(adapted_logits, labels)
                val_losses.append(val_loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1:3d}/{cfg['epochs']} | train={train_loss:.4f} | val={val_loss:.4f}")

        # 保存最优
        if val_loss < best_val_loss and save_path is not None:
            best_val_loss = val_loss
            adapter.save(save_path)
            print(f"  → Saved best adapter to {save_path}")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    return history
