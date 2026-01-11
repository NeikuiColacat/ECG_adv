"""
PGD (Projected Gradient Descent) 对抗攻击实现

在 ECG 信号空间直接进行扰动，生成对抗样本
与 AdvDiff 不同，PGD 是经典的输入空间攻击方法
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目路径
_PROJECT_ROOT = Path(__file__).parent.parent
_PTBXL_CODE_ROOT = _PROJECT_ROOT / "model" / "ecg_ptbxl_benchmarking" / "code"

sys.path.insert(0, str(_PROJECT_ROOT))

from util.ecgtwin_utils import load_ecgtwin
from util.lead_utils import ecgtwin_to_ptbxl, resample_ecg
from util.save_tool import save_adversarial_sample, save_adversarial_batch, CLASS_NAMES

import pickle


def load_classifier(device: str = "cuda:0") -> nn.Module:
    """加载 PTB-XL ResNet1D 分类器"""
    old_path = sys.path.copy()
    sys.path.insert(0, str(_PTBXL_CODE_ROOT))
    
    try:
        from models.resnet1d import resnet1d_wang
        
        model = resnet1d_wang(
            num_classes=5,
            input_channels=12,
            inplanes=128,
            kernel_size=[5, 3],
            ps_head=0.5,
            lin_ftrs_head=[128],
        )
        
        weight_path = _PROJECT_ROOT / "model" / "ecg_ptbxl_benchmarking" / "output" / "exp1.1.1" / "models" / "fastai_resnet1d_wang" / "models" / "fastai_resnet1d_wang.pth"
        checkpoint = torch.load(str(weight_path), map_location=device)
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=True)
        print(f"✓ ResNet1D 权重已加载")
        
    finally:
        sys.path = old_path
    
    model = model.to(device)
    model.eval()
    return model


def load_normalization_params(device: str = "cuda:0") -> Tuple[torch.Tensor, torch.Tensor]:
    """加载 PTB-XL 数据集的标准化参数"""
    scaler_path = _PROJECT_ROOT / "model" / "ecg_ptbxl_benchmarking" / "output" / "exp1.1.1" / "data" / "standard_scaler.pkl"
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
    std = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)
    
    return mean, std


class PGDAttacker:
    """
    PGD 对抗攻击器
    
    在 ECG 信号空间直接进行对抗扰动
    """
    
    def __init__(
        self,
        model: nn.Module = None,
        device: str = "cuda:0",
        eps: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 40,
        random_start: bool = True,
    ):
        """
        初始化 PGD 攻击器
        
        Args:
            model: 受害分类器
            device: 设备
            eps: 扰动范围 (L∞ norm)
            alpha: 每步步长
            num_steps: 迭代次数
            random_start: 是否随机初始化
        """
        self.device = torch.device(device)
        self.eps = eps
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        
        # 加载分类器
        if model is None:
            print("Loading classifier...")
            self.model = load_classifier(device)
        else:
            self.model = model.to(self.device)
        
        self.model.eval()
        
        # 加载标准化参数
        self.mean, self.std = load_normalization_params(device)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """标准化 ECG 信号"""
        # x: (B, 12, L), mean/std: (12,)
        mean = self.mean.view(1, -1, 1)
        std = self.std.view(1, -1, 1)
        return (x - mean) / (std + 1e-8)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """反标准化 ECG 信号"""
        mean = self.mean.view(1, -1, 1)
        std = self.std.view(1, -1, 1)
        return x * std + mean
    
    def attack(
        self,
        x: torch.Tensor,
        target_label: int,
        attack_type: str = "targeted",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        执行 PGD 攻击
        
        Args:
            x: 原始 ECG 信号，shape (B, 12, L) 或 (B, L, 12)
            target_label: 目标攻击标签
            attack_type: "targeted" 或 "untargeted"
            verbose: 是否显示进度
            
        Returns:
            results: 包含对抗样本和攻击结果的字典
        """
        # 确保输入格式正确 (B, 12, L)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] == 12:
            x = x.transpose(-1, -2)
        
        x = x.to(self.device)
        batch_size = x.shape[0]
        
        # 保存原始信号（用于计算扰动范围）
        x_orig = x.clone()
        
        # 重采样到 1000 点（PTB-XL 格式）
        if x.shape[-1] != 1000:
            x = resample_ecg(x, 1000)
            x_orig = x.clone()
        
        # 标准化
        x_normalized = self.normalize(x).detach()
        
        # 初始化扰动
        if self.random_start:
            delta = torch.zeros_like(x_normalized).uniform_(-self.eps, self.eps)
        else:
            delta = torch.zeros_like(x_normalized)
        
        # 目标标签
        target = torch.full((batch_size,), target_label, dtype=torch.long, device=self.device)
        
        # PGD 迭代
        loop = range(self.num_steps)
        if verbose:
            loop = tqdm(loop, desc="PGD Attack")
        
        for step in loop:
            # 设置扰动需要梯度
            delta_var = delta.clone().requires_grad_(True)
            
            # 计算对抗样本
            x_adv = x_normalized + delta_var
            
            # 前向传播
            logits = self.model(x_adv)
            
            # 计算损失
            if attack_type == "targeted":
                # 最小化目标类别的负对数概率
                loss = F.cross_entropy(logits, target)
            else:
                # 最大化真实类别的损失
                loss = -F.cross_entropy(logits, target)
            
            # 反向传播
            loss.backward()
            
            # 获取梯度
            grad = delta_var.grad.data
            
            # 更新扰动
            with torch.no_grad():
                if attack_type == "targeted":
                    # 沿着负梯度方向（减小 loss）
                    delta = delta - self.alpha * grad.sign()
                else:
                    # 沿着梯度方向（增大 loss）
                    delta = delta + self.alpha * grad.sign()
                
                # 投影到 epsilon 球内
                delta = torch.clamp(delta, -self.eps, self.eps)
            
            # 打印中间结果
            if verbose and (step + 1) % 10 == 0:
                with torch.no_grad():
                    x_adv_check = x_normalized + delta
                    preds = self.model(x_adv_check).argmax(dim=1)
                    success = (preds == target_label).float().mean().item()
                    tqdm.write(f"  Step {step+1}: success rate = {success:.0%}")
        
        # 最终对抗样本
        x_adv = x_normalized + delta
        
        # 最终预测
        with torch.no_grad():
            final_logits = self.model(x_adv)
            final_preds = final_logits.argmax(dim=1)
            
            if attack_type == "targeted":
                success_mask = (final_preds == target_label)
            else:
                success_mask = (final_preds != target_label)
        
        # 反标准化得到实际 ECG 信号
        x_adv_denorm = self.denormalize(x_adv.detach())
        
        # 计算扰动（在原始空间）
        x_orig_denorm = self.denormalize(x_normalized)
        perturbation = x_adv_denorm - x_orig_denorm
        
        # 转换为 (B, L, 12) 格式用于保存
        x_adv_save = x_adv_denorm.transpose(-1, -2)  # (B, 1000, 12)
        
        results = {
            "adv_samples": x_adv_save,
            "original_samples": x_orig.transpose(-1, -2),
            "perturbation": perturbation,
            "final_predictions": final_preds,
            "target_label": target_label,
            "success_mask": success_mask,
            "eps": self.eps,
            "alpha": self.alpha,
            "num_steps": self.num_steps,
        }
        
        return results


def generate_and_attack(
    target_label: int = 2,  # MI
    batch_size: int = 8,
    eps: float = 0.1,
    alpha: float = 0.01,
    num_steps: int = 50,
    save_dir: str = "./result/pgd",
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    生成正常 ECG 并使用 PGD 攻击
    
    Args:
        target_label: 攻击目标 (default: 2 = MI)
        batch_size: 批次大小
        eps: 扰动范围
        alpha: 步长
        num_steps: 迭代次数
        save_dir: 保存目录
        device: 设备
        
    Returns:
        results: 攻击结果
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("PGD Attack on ECG Classifier")
    print("=" * 60)
    print(f"Target: {CLASS_NAMES[target_label]} (label={target_label})")
    print(f"Epsilon: {eps}")
    print(f"Alpha: {alpha}")
    print(f"Steps: {num_steps}")
    print("=" * 60)
    
    # 1. 加载 ECGTwin 生成正常 ECG
    print("\n[1/3] Generating normal ECG samples using ECGTwin...")
    ecgtwin = load_ecgtwin(device=device, load_text_model=False)
    
    ref_data = torch.load(
        str(_PROJECT_ROOT / "model" / "ECGTwin" / "data" / "prepared_input" / "normal_1.pt"),
        map_location="cpu"
    )
    
    conditions = ecgtwin.prepare_conditions(
        ref_latent=ref_data["data"],
        ref_label=ref_data["label"],
        batch_size=batch_size,
    )
    
    # 生成正常 ECG
    latent = ecgtwin.ddpm_sample(conditions, batch_size=batch_size, num_inference_steps=50)
    ecg_generated = ecgtwin.decode_latent(latent)  # (B, 1024, 12)
    
    print(f"  Generated ECG shape: {ecg_generated.shape}")
    print(f"  ECG range: [{ecg_generated.min().item():.3f}, {ecg_generated.max().item():.3f}]")
    
    # 转换为分类器输入格式
    ecg_input = ecgtwin_to_ptbxl(ecg_generated)  # (B, 12, 1024)
    ecg_input = resample_ecg(ecg_input, 1000)    # (B, 12, 1000)
    
    # 2. 验证原始预测
    print("\n[2/3] Verifying original predictions...")
    attacker = PGDAttacker(
        device=device,
        eps=eps,
        alpha=alpha,
        num_steps=num_steps,
    )
    
    with torch.no_grad():
        ecg_normalized = attacker.normalize(ecg_input.to(device))
        orig_logits = attacker.model(ecg_normalized)
        orig_preds = orig_logits.argmax(dim=1)
    
    print(f"  Original predictions: {[CLASS_NAMES[p.item()] for p in orig_preds]}")
    norm_count = (orig_preds == 3).sum().item()
    print(f"  NORM samples: {norm_count}/{batch_size}")
    
    # 只攻击预测为 NORM 的样本
    norm_mask = (orig_preds == 3)
    if norm_mask.sum() == 0:
        print("  Warning: No NORM samples found, attacking all samples anyway")
        norm_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    # 3. PGD 攻击
    print(f"\n[3/3] Running PGD attack (target={CLASS_NAMES[target_label]})...")
    results = attacker.attack(
        x=ecg_input,
        target_label=target_label,
        attack_type="targeted",
        verbose=True,
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Attack Results:")
    print("=" * 60)
    print(f"Final predictions: {[CLASS_NAMES[p.item()] for p in results['final_predictions']]}")
    print(f"Success mask: {results['success_mask'].tolist()}")
    print(f"Success rate: {results['success_mask'].float().mean().item():.0%}")
    
    # 计算扰动统计
    perturbation = results['perturbation']
    print(f"\nPerturbation stats:")
    print(f"  L∞ norm: {perturbation.abs().max().item():.4f}")
    print(f"  L2 norm (mean): {perturbation.norm(p=2, dim=(1,2)).mean().item():.4f}")
    
    # 4. 保存对抗样本
    print(f"\n[4/4] Saving adversarial samples to {save_dir}...")
    gt_label = 3  # NORM (原始标签)
    
    saved_paths = save_adversarial_batch(
        ecg_batch=results['adv_samples'],
        gt_label=gt_label,
        target_label=target_label,
        pred_labels=results['final_predictions'],
        save_dir=save_dir,
        add_title=True,
        verbose=True,
    )
    
    print(f"\nSaved {len(saved_paths)} images to {save_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PGD Attack on ECG Classifier")
    parser.add_argument("--target", type=int, default=2, help="Target label (0:CD, 1:HYP, 2:MI, 3:NORM, 4:STTC)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--eps", type=float, default=0.1, help="Epsilon (L∞ bound)")
    parser.add_argument("--alpha", type=float, default=0.01, help="Step size")
    parser.add_argument("--steps", type=int, default=50, help="Number of PGD steps")
    parser.add_argument("--save_dir", type=str, default="./result/pgd", help="Save directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    
    args = parser.parse_args()
    
    generate_and_attack(
        target_label=args.target,
        batch_size=args.batch_size,
        eps=args.eps,
        alpha=args.alpha,
        num_steps=args.steps,
        save_dir=args.save_dir,
        device=args.device,
    )
