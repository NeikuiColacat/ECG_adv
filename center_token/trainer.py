"""
Center Token 训练器

通过 hook 将 center token 注入 DiT_ECGTwin 的 AdaX 调制路径，
使用 diffusion 重建损失 + 疾病不变性损失 + 正则化损失进行训练。

只有 center token (256 参数) 有梯度，其余模型参数全部冻结。
"""

import sys
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import DDPMScheduler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
ECGTWIN_ROOT = PROJECT_ROOT / "model" / "ECGTwin"
sys.path.insert(0, str(ECGTWIN_ROOT))

from module.IBExtractor import IBExtractor
from utils.model_utils import build_noise_predictor
from utils.data_utils import (
    ListDataset,
    process_pat_info,
    sex_transform,
    _pad_text_embed,
)

from center_token.model import CenterToken


# ─────────────────────────────────────────────
# Collate function for PTBXL dataset
# ─────────────────────────────────────────────

def center_token_collate_fn(batch):
    """
    Collate function，处理变长 text_embed 并传递 diagnostic_class。
    batch: list of (data_tensor, label_dict)
    """
    data_list = [entry[0] for entry in batch]
    label_list = [entry[1] for entry in batch]

    ecg_data = torch.stack(data_list)  # (B, 4, 128)

    ecg_label = {}

    # 数值型字段
    ecg_label["hr"] = torch.tensor([l["hr"] for l in label_list], dtype=torch.float32)
    ecg_label["age"] = torch.tensor([l["age"] for l in label_list], dtype=torch.float32)
    ecg_label["sex"] = torch.tensor(
        [sex_transform(l["sex"]) for l in label_list], dtype=torch.float32
    )

    # text_embed: 变长，需要 padding
    embed_list = [l["text_embed"] for l in label_list]
    padded_embed, embed_mask = _pad_text_embed(embed_list)
    ecg_label["text_embed"] = padded_embed      # (B, max_L, 768)
    ecg_label["text_embed_mask"] = embed_mask    # (B, max_L)

    # diagnostic_class: list of str (不转 tensor)
    ecg_label["diagnostic_class"] = [l["diagnostic_class"] for l in label_list]

    # strat_fold
    ecg_label["strat_fold"] = [l["strat_fold"] for l in label_list]

    return ecg_data, ecg_label


# ─────────────────────────────────────────────
# CenterTokenTrainer
# ─────────────────────────────────────────────

class CenterTokenTrainer:

    def __init__(
        self,
        config: dict,
        ecgtwin_config: dict,
        device: str = "cuda:0",
    ):
        self.config = config
        self.device = torch.device(device)
        self.ecgtwin_config = ecgtwin_config
        h_ = ecgtwin_config["hyper_para"]

        # ── 加载冻结模型 ──
        model_type = ecgtwin_config["meta"]["model_type"]
        n_channels = 4

        # Noise predictor (DiT)
        self.noise_predictor = build_noise_predictor(model_type, n_channels, h_)
        np_path = ECGTWIN_ROOT / ecgtwin_config["inference_setting"]["noise_predictor_path"]
        self.noise_predictor.load_state_dict(torch.load(np_path, map_location="cpu"))
        self.noise_predictor.to(self.device)
        self.noise_predictor.eval()
        for p in self.noise_predictor.parameters():
            p.requires_grad = False

        # IBExtractor
        self.ibe_model = IBExtractor(
            embed_dim=h_["ibe"]["embed_dim"],
            num_heads=h_["ibe"]["num_heads"],
            ff_hidden_size=h_["ibe"]["ff_hidden_size"],
            num_layers=h_["ibe"]["num_layers"],
            text_embed_dim=h_["ibe"]["text_embed_dim"],
            patient_info_size=h_["ibe"]["patient_info_size"],
        )
        ibe_path = ECGTWIN_ROOT / ecgtwin_config["dependencies"]["ibe_path"]
        self.ibe_model.load_state_dict(torch.load(ibe_path, map_location="cpu"))
        self.ibe_model.to(self.device)
        self.ibe_model.eval()
        for p in self.ibe_model.parameters():
            p.requires_grad = False

        # DDPM Scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=h_["ddpm"]["num_train_steps"],
            beta_start=h_["ddpm"]["beta_start"],
            beta_end=h_["ddpm"]["beta_end"],
        )

        # ── Center Token ──
        ct_config = config["center_token"]
        self.center_token = CenterToken(dim=ct_config["dim"])
        self.center_token.to(self.device)

        # ── 优化器 & 调度器 ──
        train_cfg = config["training"]
        self.optimizer = torch.optim.AdamW(
            self.center_token.parameters(), lr=train_cfg["lr"]
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg["epochs"],
            eta_min=train_cfg["scheduler"]["eta_min"],
        )

        # ── 超参数 ──
        self.lambda_inv = train_cfg["lambda_inv"]
        self.lambda_reg = train_cfg["lambda_reg"]
        self.grad_clip = train_cfg["grad_clip"]
        self.epochs = train_cfg["epochs"]
        self.batch_size = train_cfg["batch_size"]

        # ── Hook 管理 ──
        self.hooks = []
        self.hooks_enabled = True

    # ─────────────────────────────────────────
    # Hook registration
    # ─────────────────────────────────────────

    def _register_hooks(self):
        """在每个 DiTBlock 上注册 forward_pre_hook，注入 center token 到 c 路径"""
        self._remove_hooks()
        for block in self.noise_predictor.blocks:
            hook = block.register_forward_pre_hook(self._make_hook())
            self.hooks.append(hook)

    def _make_hook(self):
        trainer = self

        def hook_fn(module, args):
            if not trainer.hooks_enabled:
                return args
            # DiTBlock_ECGTwin.forward(x, c, c2, mask, ...)
            x, c, c2 = args[0], args[1], args[2]
            rest = args[3:]
            c = c + trainer.center_token(c.size(0))
            return (x, c, c2) + rest

        return hook_fn

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    # ─────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────

    def create_dataloaders(self, dataset_path: str):
        """创建训练和验证 DataLoader"""
        dataset = ListDataset(dataset_path)

        val_fold = self.config["training"]["val_fold"]
        test_fold = self.config["training"]["test_fold"]

        train_indices = []
        val_indices = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            fold = label.get("strat_fold", 1)
            if fold == val_fold:
                val_indices.append(i)
            elif fold == test_fold:
                continue  # 跳过测试集
            else:
                train_indices.append(i)

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=center_token_collate_fn,
            num_workers=0,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=center_token_collate_fn,
            num_workers=0,
            drop_last=False,
        )

        print(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
        return train_loader, val_loader

    # ─────────────────────────────────────────
    # Loss computation
    # ─────────────────────────────────────────

    def _compute_losses(
        self,
        noise_pred_with: torch.Tensor,
        noise: torch.Tensor,
        diagnostic_classes: list,
        latent: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor,
        text_embed_mask: torch.Tensor,
        pat_info: torch.Tensor,
        base_vector: torch.Tensor,
    ):
        B = noise.size(0)

        # L_recon: 主重建损失
        l_recon = F.mse_loss(noise_pred_with, noise, reduction="sum") / B

        # L_inv: 疾病不变性损失
        # 需要无 center token 的 baseline prediction
        self.hooks_enabled = False
        with torch.no_grad():
            noise_pred_base = self.noise_predictor(
                latent, t, text_embed, text_embed_mask, pat_info, base_vector
            )
        self.hooks_enabled = True

        # 每个样本的 loss
        loss_with = F.mse_loss(noise_pred_with, noise, reduction="none").mean(dim=[1, 2])  # (B,)
        loss_base = F.mse_loss(noise_pred_base, noise, reduction="none").mean(dim=[1, 2])  # (B,)
        delta = (loss_base - loss_with).detach()  # positive = center helped

        # 按 diagnostic_class 分组，计算 group mean delta 的方差
        unique_classes = list(set(diagnostic_classes))
        group_deltas = []
        for cls in unique_classes:
            mask = torch.tensor(
                [1.0 if dc == cls else 0.0 for dc in diagnostic_classes],
                device=delta.device,
            )
            count = mask.sum()
            if count >= 2:
                group_mean = (delta * mask).sum() / count
                group_deltas.append(group_mean)

        if len(group_deltas) >= 2:
            l_inv = torch.stack(group_deltas).var()
        else:
            l_inv = torch.tensor(0.0, device=self.device)

        # L_reg: 正则化
        l_reg = self.center_token.embedding.norm(p=2)

        # 总损失
        l_total = l_recon + self.lambda_inv * l_inv + self.lambda_reg * l_reg

        return {
            "total": l_total,
            "recon": l_recon.item(),
            "inv": l_inv.item(),
            "reg": l_reg.item(),
            "delta_mean": delta.mean().item(),
        }

    # ─────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────

    def _train_step(self, ecg_data, ecg_label):
        """单步训练"""
        B = ecg_data.size(0)
        latent = ecg_data.to(self.device)  # (B, 4, 128)

        # 准备条件
        text_embed = ecg_label["text_embed"].to(self.device)       # (B, max_L, 768)
        text_embed_mask = ecg_label["text_embed_mask"].to(self.device)  # (B, max_L)
        diagnostic_classes = ecg_label["diagnostic_class"]

        pat_info = process_pat_info(
            normalize=True,
            hr=ecg_label["hr"],
            age=ecg_label["age"],
            sex=ecg_label["sex"],
        ).to(self.device)

        # 自配对: IBExtractor 使用自身作为参考
        latent_for_ibe = latent.transpose(2, 1)  # (B, 128, 4) for IBE
        with torch.no_grad():
            base_vector = self.ibe_model.extract_features(
                latent_for_ibe, text_embed, text_embed_mask, pat_info, reduce=True
            )

        pat_info_tar = process_pat_info(
            normalize=True,
            add_token=False,
            hr=ecg_label["hr"],
            age=ecg_label["age"],
            sex=ecg_label["sex"],
        ).to(self.device)

        # Diffusion forward
        noise = torch.randn_like(latent)
        t = torch.randint(1, 1000, (B,), device=self.device)
        noisy_latent = self.scheduler.add_noise(latent, noise, t)

        # 带 center token 的噪声预测
        self.hooks_enabled = True
        noise_pred = self.noise_predictor(
            noisy_latent, t, text_embed, text_embed_mask, pat_info_tar, base_vector
        )

        # 计算损失
        losses = self._compute_losses(
            noise_pred_with=noise_pred,
            noise=noise,
            diagnostic_classes=diagnostic_classes,
            latent=noisy_latent,
            t=t,
            text_embed=text_embed,
            text_embed_mask=text_embed_mask,
            pat_info=pat_info_tar,
            base_vector=base_vector,
        )

        # 反向传播 (只有 center_token.embedding 有梯度)
        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.center_token.parameters(), self.grad_clip)
        self.optimizer.step()

        return losses

    # ─────────────────────────────────────────
    # Validation step
    # ─────────────────────────────────────────

    @torch.no_grad()
    def _val_step(self, ecg_data, ecg_label):
        B = ecg_data.size(0)
        latent = ecg_data.to(self.device)

        text_embed = ecg_label["text_embed"].to(self.device)
        text_embed_mask = ecg_label["text_embed_mask"].to(self.device)

        pat_info = process_pat_info(
            normalize=True,
            hr=ecg_label["hr"],
            age=ecg_label["age"],
            sex=ecg_label["sex"],
        ).to(self.device)

        latent_for_ibe = latent.transpose(2, 1)
        base_vector = self.ibe_model.extract_features(
            latent_for_ibe, text_embed, text_embed_mask, pat_info, reduce=True
        )

        pat_info_tar = process_pat_info(
            normalize=True,
            add_token=False,
            hr=ecg_label["hr"],
            age=ecg_label["age"],
            sex=ecg_label["sex"],
        ).to(self.device)

        noise = torch.randn_like(latent)
        t = torch.randint(1, 1000, (B,), device=self.device)
        noisy_latent = self.scheduler.add_noise(latent, noise, t)

        self.hooks_enabled = True
        noise_pred = self.noise_predictor(
            noisy_latent, t, text_embed, text_embed_mask, pat_info_tar, base_vector
        )

        l_recon = F.mse_loss(noise_pred, noise, reduction="sum") / B
        return l_recon.item()

    # ─────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────

    def train(self, dataset_path: str, save_dir: str):
        """完整训练循环"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader = self.create_dataloaders(dataset_path)
        self._register_hooks()

        best_val_loss = float("inf")
        log_lines = []

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            self.center_token.train()
            train_metrics = {"recon": 0, "inv": 0, "reg": 0, "delta_mean": 0}
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
            for ecg_data, ecg_label in pbar:
                losses = self._train_step(ecg_data, ecg_label)
                for k in train_metrics:
                    train_metrics[k] += losses[k]
                n_batches += 1
                pbar.set_postfix(
                    recon=f"{losses['recon']:.4f}",
                    delta=f"{losses['delta_mean']:.4f}",
                    norm=f"{self.center_token.norm:.4f}",
                )

            for k in train_metrics:
                train_metrics[k] /= max(n_batches, 1)

            # ── Validate ──
            self.center_token.eval()
            val_loss = 0
            n_val = 0
            for ecg_data, ecg_label in val_loader:
                val_loss += self._val_step(ecg_data, ecg_label)
                n_val += 1
            val_loss /= max(n_val, 1)

            self.lr_scheduler.step()

            # ── Logging ──
            log = (
                f"Epoch {epoch:3d} | "
                f"Train L_recon: {train_metrics['recon']:.4f} | "
                f"L_inv: {train_metrics['inv']:.6f} | "
                f"L_reg: {train_metrics['reg']:.4f} | "
                f"delta_mean: {train_metrics['delta_mean']:.4f} | "
                f"center_norm: {self.center_token.norm:.4f} | "
                f"Val L_recon: {val_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            print(log)
            log_lines.append(log)

            # ── Save best ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "center_token": self.center_token.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "center_norm": self.center_token.norm,
                    },
                    save_dir / "center_token_best.pth",
                )
                print(f"  => Saved best model (val_loss={val_loss:.4f})")

            # 每个 epoch 也保存最新
            torch.save(
                {
                    "center_token": self.center_token.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_dir / "center_token_latest.pth",
            )

        # 保存训练日志
        with open(save_dir / "training_log.txt", "w") as f:
            f.write("\n".join(log_lines))

        self._remove_hooks()
        print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
        print(f"Center token norm: {self.center_token.norm:.4f}")
        return best_val_loss
