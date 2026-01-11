"""
AdvDiff 对抗攻击算法实现

基于论文：Adversarial Diffusion Attacks via Gradients (arXiv:2307.12499)

Algorithm 1: DDPM Adversarial Diffusion Sampling
- 使用 ECGTwin diffusion 模型在 latent space 生成对抗样本
- 攻击 PTB-XL 上训练的 ResNet1D 分类器
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm
import pickle

# 添加必要路径（注意顺序，避免冲突）
_PROJECT_ROOT = Path(__file__).parent.parent
_ECGTWIN_ROOT = _PROJECT_ROOT / "model" / "ECGTwin"
_PTBXL_CODE_ROOT = _PROJECT_ROOT / "model" / "ecg_ptbxl_benchmarking" / "code"

# 先添加项目根目录
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ECGTwin 路径
if str(_ECGTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_ECGTWIN_ROOT))

from util.ecgtwin_utils import ECGTwinWrapper, load_ecgtwin
from util.lead_utils import prepare_ecg_for_classifier, ECGTWIN_TO_PTBXL_INDICES


def get_resnet1d_wang(
    num_classes: int = 5,
    input_channels: int = 12,
    weight_path: str = None,
    device: str = "cuda",
) -> nn.Module:
    """
    加载 PTB-XL 上训练的 ResNet1D 分类器
    
    避免 utils 命名冲突的独立实现
    """
    # 临时添加路径
    old_path = sys.path.copy()
    sys.path.insert(0, str(_PTBXL_CODE_ROOT))
    
    try:
        from models.resnet1d import resnet1d_wang
        
        model = resnet1d_wang(
            num_classes=num_classes,
            input_channels=input_channels,
            inplanes=128,
            kernel_size=[5, 3],
            ps_head=0.5,
            lin_ftrs_head=[128],
        )
        
        if weight_path is None:
            weight_path = _PROJECT_ROOT / "model" / "ecg_ptbxl_benchmarking" / "output" / "exp1.1.1" / "models" / "fastai_resnet1d_wang" / "models" / "fastai_resnet1d_wang.pth"
        
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")
        
        checkpoint = torch.load(str(weight_path), map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=True)
        print(f"✓ ResNet1D 权重已加载: {weight_path}")
        
    finally:
        sys.path = old_path
    
    model = model.to(device)
    model.eval()
    return model


def get_data_normalization_params(device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取 PTB-XL 数据集的标准化参数
    """
    scaler_path = _PROJECT_ROOT / "model" / "ecg_ptbxl_benchmarking" / "output" / "exp1.1.1" / "data" / "standard_scaler.pkl"
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
    std = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)
    
    return mean, std


class AdvDiffAttacker:
    """
    AdvDiff 对抗攻击器
    
    实现论文 Algorithm 1: DDPM Adversarial Diffusion Sampling
    
    攻击流程：
    1. 从 xT ~ N(0, I) 开始
    2. 对每个 timestep:
       - Classifier-free sampling 得到 x_{t-1}
       - 计算对抗梯度 ∇_{x_{t-1}} log p_f(y_a | x_{t-1})
       - 应用 Adversarial Guidance: x*_{t-1} = x_{t-1} + σ²_t * s * ∇
    3. Noise Sampling Guidance:
       - 根据最终分类结果更新 xT
       - xT = xT + σ²_T * a * ∇_{x_0} log p_f(y_a | x_0)
    """
    
    def __init__(
        self,
        ecgtwin: ECGTwinWrapper = None,
        victim_model: nn.Module = None,
        device: str = "cuda:0",
        num_classes: int = 5,
    ):
        """
        初始化攻击器
        
        Args:
            ecgtwin: ECGTwin 模型包装器
            victim_model: 受害分类模型
            device: 设备
            num_classes: 分类类别数
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        # 加载 ECGTwin（不加载 text model，使用预计算的 embedding）
        if ecgtwin is None:
            print("Loading ECGTwin model...")
            self.ecgtwin = load_ecgtwin(
                device=device, 
                load_encoder=True,
                load_text_model=False,  # 不加载 nomic 模型
            )
        else:
            self.ecgtwin = ecgtwin
        
        # 加载受害模型
        if victim_model is None:
            print("Loading victim classifier (ResNet1D)...")
            self.victim_model = get_resnet1d_wang(
                num_classes=num_classes,
                device=device,
            )
        else:
            self.victim_model = victim_model
        
        self.victim_model.eval()
        
        # 加载数据集的标准化参数
        self.data_mean, self.data_std = get_data_normalization_params(device)
    
    def _latent_to_classifier_input(
        self,
        latent: torch.Tensor,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """
        将 diffusion latent 转换为分类器输入
        
        Args:
            latent: VAE latent，shape (B, 4, 128)
            enable_grad: 是否保留梯度
            
        Returns:
            classifier_input: 分类器输入，shape (B, 12, 1000)
        """
        # 1. VAE 解码
        # 注意：decoder 内部有 /= 0.18215 的 in-place 操作
        # 需要 clone 来避免修改原始 latent
        latent_clone = latent / 0.18215  # 预先做除法，避免 in-place
        latent_clone = latent_clone * 0.18215  # 恢复，让 decoder 内部的除法正确工作
        
        if enable_grad:
            # 手动实现 decoder forward，避免 in-place 操作
            x = latent.clone()
            x = x / 0.18215  # out-of-place 除法
            
            for module in self.ecgtwin.decoder:
                x = module(x)
            
            # (B, 12, L) -> (B, L, 12)
            ecg = x.transpose(1, 2)
        else:
            with torch.no_grad():
                ecg = self.ecgtwin.decoder(latent.clone())
        
        # 2. 准备分类器输入（导联对齐 + 重采样 + 标准化）
        ecg_input = prepare_ecg_for_classifier(
            ecg=ecg,
            source="ecgtwin",
            target_length=1000,
            normalize=True,
            mean=self.data_mean,
            std=self.data_std,
        )
        
        return ecg_input
    
    def _compute_adversarial_gradient(
        self,
        latent: torch.Tensor,
        target_label: int,
        attack_type: str = "targeted",
    ) -> torch.Tensor:
        """
        计算对抗梯度 ∇_latent log p_f(y_a | latent)
        
        Args:
            latent: VAE latent，shape (B, 4, 128)
            target_label: 目标攻击标签
            attack_type: "targeted" 或 "untargeted"
            
        Returns:
            gradient: 对抗梯度，shape (B, 4, 128)
        """
        latent = latent.detach().requires_grad_(True)
        
        # 转换为分类器输入
        classifier_input = self._latent_to_classifier_input(latent, enable_grad=True)
        
        # 前向传播
        logits = self.victim_model(classifier_input)
        
        # 计算损失
        batch_size = latent.shape[0]
        target = torch.full((batch_size,), target_label, dtype=torch.long, device=self.device)
        
        if attack_type == "targeted":
            # 最大化目标类别的概率 -> 最小化负对数概率
            # ∇ log p(y_a | x) 的方向是增加 y_a 概率的方向
            log_probs = F.log_softmax(logits, dim=1)
            loss = -log_probs[range(batch_size), target].mean()
        else:
            # 非定向攻击：最小化真实类别的概率
            log_probs = F.log_softmax(logits, dim=1)
            loss = log_probs[range(batch_size), target].mean()
        
        # 反向传播计算梯度
        loss.backward()
        
        # 对于 targeted 攻击：
        # loss = -log p(y_a | x)，我们想最小化 loss（最大化 log p）
        # 所以应该沿着负梯度方向走：-∇loss = ∇log p(y_a | x)
        # 因为 adversarial_sampling 中做的是 x + s * grad，所以这里返回 -grad
        gradient = -latent.grad.clone()
        
        return gradient
    
    def _compute_sigma_t(self, t: int) -> float:
        """
        计算 timestep t 对应的 sigma_t
        
        Args:
            t: timestep
            
        Returns:
            sigma_t: 噪声标准差
        """
        # 从 scheduler 获取 beta 值
        betas = self.ecgtwin.scheduler.betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        if t >= len(alphas_cumprod):
            t = len(alphas_cumprod) - 1
        
        alpha_t = alphas_cumprod[t]
        sigma_t = ((1 - alpha_t) / alpha_t).sqrt()
        
        return sigma_t.item()
    
    def adversarial_sampling(
        self,
        conditions: Dict[str, torch.Tensor],
        target_label: int,
        batch_size: int = 1,
        num_inference_steps: int = 100,
        adversarial_guidance_scale: float = 1.0,
        noise_sampling_guidance_scale: float = 0.1,
        num_noise_sampling_steps: int = 3,
        classifier_free_guidance_scale: float = 1.0,
        attack_type: str = "targeted",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Algorithm 1: DDPM Adversarial Diffusion Sampling
        
        Args:
            conditions: 生成条件
            target_label: 目标攻击标签 y_a
            batch_size: 批次大小
            num_inference_steps: 反向生成步数 T
            adversarial_guidance_scale: 对抗引导强度 s
            noise_sampling_guidance_scale: 噪声采样引导强度 a
            num_noise_sampling_steps: 噪声采样引导迭代次数 N
            classifier_free_guidance_scale: classifier-free guidance 强度 w
            attack_type: "targeted" 或 "untargeted"
            verbose: 是否显示进度
            
        Returns:
            results: 包含对抗样本和中间结果的字典
        """
        self.ecgtwin.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.ecgtwin.scheduler.timesteps
        
        # 初始化 xT ~ N(0, I)
        x_T = torch.randn(batch_size, 4, 128, device=self.device)
        
        # 存储成功的对抗样本
        adv_samples = []
        adv_latents = []
        
        outer_loop = range(num_noise_sampling_steps)
        if verbose:
            outer_loop = tqdm(outer_loop, desc="Noise Sampling Guidance")
        
        for n in outer_loop:
            x_t = x_T.clone()
            
            # 反向生成过程
            inner_loop = enumerate(timesteps)
            if verbose:
                inner_loop = tqdm(inner_loop, desc=f"Iteration {n+1}/{num_noise_sampling_steps}", leave=False)
            
            for step_idx, t in inner_loop:
                t_batch = t * torch.ones(batch_size, dtype=torch.long, device=self.device)
                
                # Step 1: Classifier-free sampling
                # ε̃_t = (1 + w) * ε_θ(x_t, y) - w * ε_θ(x_t)
                # 这里 ECGTwin 是条件模型，我们直接用条件预测
                with torch.no_grad():
                    noise_pred = self.ecgtwin.noise_predictor(
                        x_t, t_batch,
                        conditions["text_embed"],
                        conditions["text_embed_mask"],
                        conditions["pat_info"],
                        conditions["base_vector"],
                    )
                
                # Step 2: 标准采样得到 x_{t-1}
                scheduler_output = self.ecgtwin.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=x_t,
                )
                x_prev = scheduler_output["prev_sample"]
                
                # Step 3: Adversarial Guidance
                # x*_{t-1} = x_{t-1} + σ²_t * s * ∇_{x_{t-1}} log p_f(y_a | x_{t-1})
                # 
                # 修复：原始实现中 sigma² * scale * grad 会导致更新量过大
                # 使用归一化梯度 + 自适应步长来控制更新幅度
                # 
                # 策略：只在后半部分的 timesteps 应用对抗引导
                # 因为早期（t 接近 T）的引导会被后续去噪覆盖
                
                # 只在后 50% 的步骤中应用对抗引导
                apply_guidance = (step_idx >= num_inference_steps // 2)
                
                if apply_guidance:
                    adv_grad = self._compute_adversarial_gradient(
                        latent=x_prev,
                        target_label=target_label,
                        attack_type=attack_type,
                    )
                    
                    # 梯度 L2 归一化（按样本）
                    grad_norm = adv_grad.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
                    normalized_grad = adv_grad / grad_norm
                    
                    # 使用固定步长，越接近 x_0 步长可以越大
                    # 进度：从 0.5 到 1.0
                    progress = (step_idx - num_inference_steps // 2) / (num_inference_steps // 2)
                    step_size = adversarial_guidance_scale * 0.01 * (1 + progress)  # 步长逐渐增大
                    
                    # 梯度裁剪，防止极端值
                    guidance = normalized_grad.clamp(-1, 1) * step_size
                    
                    # 应用对抗引导
                    x_prev = x_prev + guidance
                
                x_t = x_prev
            
            # 最终样本 x_0
            x_0 = x_t
            
            # 检查攻击是否成功
            with torch.no_grad():
                classifier_input = self._latent_to_classifier_input(x_0, enable_grad=False)
                logits = self.victim_model(classifier_input)
                preds = logits.argmax(dim=1)
            
            # 收集成功的对抗样本
            if attack_type == "targeted":
                success_mask = (preds == target_label)
            else:
                success_mask = (preds != target_label)
            
            if success_mask.any():
                successful_latents = x_0[success_mask]
                successful_ecg = self.ecgtwin.decode_latent(successful_latents)
                adv_samples.append(successful_ecg)
                adv_latents.append(successful_latents)
            
            # Noise Sampling Guidance (更新 xT)
            # x_T = x_T + σ̄²_T * a * ∇_{x_0} log p_f(y_a | x_0)
            # 
            # 同样使用归一化梯度 + 固定步长
            noise_grad = self._compute_adversarial_gradient(
                latent=x_0,
                target_label=target_label,
                attack_type=attack_type,
            )
            
            # 梯度归一化
            noise_grad_norm = noise_grad.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
            normalized_noise_grad = noise_grad / noise_grad_norm
            
            # 使用固定步长更新 xT
            noise_step_size = noise_sampling_guidance_scale * 0.1  # 默认 scale=1.0 时，步长=0.1
            noise_guidance = normalized_noise_grad.clamp(-1, 1) * noise_step_size
            
            x_T = x_T + noise_guidance
            
            if verbose:
                success_rate = success_mask.float().mean().item()
                print(f"  Iteration {n+1}: Success rate = {success_rate:.2%}")
        
        # 整理结果
        results = {
            "final_latent": x_0,
            "final_ecg": self.ecgtwin.decode_latent(x_0),
            "final_predictions": preds,
            "target_label": target_label,
            "success_mask": success_mask,
        }
        
        if adv_samples:
            results["adv_samples"] = torch.cat(adv_samples, dim=0)
            results["adv_latents"] = torch.cat(adv_latents, dim=0)
        
        return results
    
    def attack(
        self,
        ref_data_path: str = None,
        ref_latent: torch.Tensor = None,
        ref_label: Dict[str, Any] = None,
        target_label: int = 0,
        target_text: str = None,
        batch_size: int = 1,
        num_inference_steps: int = 100,
        adversarial_guidance_scale: float = 1.0,
        noise_sampling_guidance_scale: float = 0.1,
        num_noise_sampling_steps: int = 3,
        attack_type: str = "targeted",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        执行对抗攻击的主接口
        
        Args:
            ref_data_path: 参考数据路径
            ref_latent: 参考 latent
            ref_label: 参考标签
            target_label: 目标攻击标签
            target_text: 目标诊断文本
            batch_size: 批次大小
            num_inference_steps: 推理步数
            adversarial_guidance_scale: 对抗引导强度 s
            noise_sampling_guidance_scale: 噪声采样引导强度 a
            num_noise_sampling_steps: 外层迭代次数 N
            attack_type: "targeted" 或 "untargeted"
            verbose: 是否显示进度
            
        Returns:
            attack_results: 攻击结果
        """
        # 加载参考数据
        if ref_data_path is not None:
            ref_data = torch.load(ref_data_path, map_location="cpu")
            ref_latent = ref_data["data"]
            ref_label = ref_data["label"]
        
        if ref_latent is None or ref_label is None:
            # 使用默认参考数据
            ecgtwin_root = Path(__file__).parent.parent / "model" / "ECGTwin"
            ref_data_path = ecgtwin_root / "data" / "prepared_input" / "normal_1.pt"
            ref_data = torch.load(ref_data_path, map_location="cpu")
            ref_latent = ref_data["data"]
            ref_label = ref_data["label"]
        
        # 获取目标 text_embed（使用预计算的，或者从 ref_label 继承）
        target_text_embed = None
        if "text_embed" in ref_label and torch.is_tensor(ref_label["text_embed"]):
            target_text_embed = ref_label["text_embed"]
        
        # 准备条件（不传 target_text，使用 target_text_embed）
        conditions = self.ecgtwin.prepare_conditions(
            ref_latent=ref_latent,
            ref_label=ref_label,
            batch_size=batch_size,
            target_text_embed=target_text_embed,  # 使用预计算的 embedding
        )
        
        # 执行对抗采样
        results = self.adversarial_sampling(
            conditions=conditions,
            target_label=target_label,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            adversarial_guidance_scale=adversarial_guidance_scale,
            noise_sampling_guidance_scale=noise_sampling_guidance_scale,
            num_noise_sampling_steps=num_noise_sampling_steps,
            attack_type=attack_type,
            verbose=verbose,
        )
        
        return results


def run_advdiff_attack(
    target_label: int = 0,
    batch_size: int = 4,
    num_inference_steps: int = 100,
    adversarial_guidance_scale: float = 2.0,
    noise_sampling_guidance_scale: float = 0.5,
    num_noise_sampling_steps: int = 5,
    device: str = "cuda:0",
    save_path: str = None,
) -> Dict[str, Any]:
    """
    运行 AdvDiff 对抗攻击的便捷函数
    
    PTB-XL Superdiagnostic 类别：
        0: CD (Conduction Disturbance)
        1: HYP (Hypertrophy)
        2: MI (Myocardial Infarction)
        3: NORM (Normal)
        4: STTC (ST/T Change)
    
    Args:
        target_label: 目标攻击标签
        batch_size: 批次大小
        num_inference_steps: 推理步数
        adversarial_guidance_scale: 对抗引导强度
        noise_sampling_guidance_scale: 噪声采样引导强度
        num_noise_sampling_steps: 外层迭代次数
        device: 设备
        save_path: 结果保存路径
        
    Returns:
        results: 攻击结果
    """
    print("=" * 60)
    print("AdvDiff Attack on ECGTwin + PTB-XL ResNet1D")
    print("=" * 60)
    print(f"Target label: {target_label}")
    print(f"Batch size: {batch_size}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Adversarial guidance scale (s): {adversarial_guidance_scale}")
    print(f"Noise sampling guidance scale (a): {noise_sampling_guidance_scale}")
    print(f"Noise sampling iterations (N): {num_noise_sampling_steps}")
    print("=" * 60)
    
    # 创建攻击器
    attacker = AdvDiffAttacker(device=device)
    
    # 执行攻击
    results = attacker.attack(
        target_label=target_label,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        adversarial_guidance_scale=adversarial_guidance_scale,
        noise_sampling_guidance_scale=noise_sampling_guidance_scale,
        num_noise_sampling_steps=num_noise_sampling_steps,
        attack_type="targeted",
        verbose=True,
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Attack Results:")
    print("=" * 60)
    print(f"Final predictions: {results['final_predictions'].tolist()}")
    print(f"Success mask: {results['success_mask'].tolist()}")
    print(f"Success rate: {results['success_mask'].float().mean().item():.2%}")
    
    if "adv_samples" in results:
        print(f"Total successful adversarial samples: {len(results['adv_samples'])}")
    
    # 保存结果
    if save_path is not None:
        save_data = {
            "final_latent": results["final_latent"].cpu(),
            "final_ecg": results["final_ecg"].cpu(),
            "final_predictions": results["final_predictions"].cpu(),
            "target_label": results["target_label"],
            "success_mask": results["success_mask"].cpu(),
        }
        if "adv_samples" in results:
            save_data["adv_samples"] = results["adv_samples"].cpu()
            save_data["adv_latents"] = results["adv_latents"].cpu()
        
        torch.save(save_data, save_path)
        print(f"\nResults saved to: {save_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AdvDiff Attack")
    parser.add_argument("--target_label", type=int, default=0, help="Target attack label")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=100, help="Inference steps")
    parser.add_argument("--adv_scale", type=float, default=2.0, help="Adversarial guidance scale")
    parser.add_argument("--noise_scale", type=float, default=0.5, help="Noise sampling guidance scale")
    parser.add_argument("--num_iterations", type=int, default=5, help="Noise sampling iterations")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--save_path", type=str, default=None, help="Save path")
    
    args = parser.parse_args()
    
    run_advdiff_attack(
        target_label=args.target_label,
        batch_size=args.batch_size,
        num_inference_steps=args.num_steps,
        adversarial_guidance_scale=args.adv_scale,
        noise_sampling_guidance_scale=args.noise_scale,
        num_noise_sampling_steps=args.num_iterations,
        device=args.device,
        save_path=args.save_path,
    )
