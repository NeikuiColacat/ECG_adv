"""
AdvDiff 边界引导生成困难样本

与原版 AdvDiff 的区别：
  - 原版：目标是误分类（targeted attack）
  - 本版：目标是将置信度推向 0.5（决策边界），生成"困难样本"供微调

核心损失：
    boundary_loss = mean(logit²)   # logit 空间，避免 sigmoid 饱和区梯度消失
    梯度方向 = -∇(boundary_loss)，使 logit → 0（即 prob → 0.5）

生成的样本满足 0.3 < target_prob < 0.7，是模型最不确定的区域。
这类样本用于微调时，迫使模型在困难决策边界区域改善泛化。
"""

import sys
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
_ECGTWIN_ROOT = _PROJECT_ROOT / "model" / "ECGTwin"

for p in [str(_PROJECT_ROOT), str(_ECGTWIN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from util.ecgtwin_utils import ECGTwinWrapper, load_ecgtwin
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES
from adversarial.label_mapping import (
    PROMPT_TO_PATTERN_INDICES,
    SUPERDIAG_TO_PROMPTS,
    get_target_indices,
)
from adversarial.efficientnet_victim import EfficientNetVictim, EFFICIENTNET_INPUT_LENGTH


# ——— 生成计划 ———
# prompt_key → (ECGTwin text prompt, PTBXL superdiag class, 目标数量)
GENERATION_PLAN: List[Dict] = [
    {
        "prompt_key": "MI_inferior",
        "text_prompt": "inferior myocardial infarction|pathological q waves",
        "superdiag": "MI",
        "target_count": 1000,
    },
    {
        "prompt_key": "MI_anterior",
        "text_prompt": "anterior myocardial infarction|st elevation",
        "superdiag": "MI",
        "target_count": 1000,
    },
    {
        "prompt_key": "MI_lateral",
        "text_prompt": "lateral myocardial infarction",
        "superdiag": "MI",
        "target_count": 500,
    },
    {
        "prompt_key": "MI_acute",
        "text_prompt": "acute myocardial infarction|st elevation",
        "superdiag": "MI",
        "target_count": 500,
    },
    {
        "prompt_key": "ISCHEMIA_ant",
        "text_prompt": "anterior ischemia|st depression",
        "superdiag": "STTC",
        "target_count": 500,
    },
    {
        "prompt_key": "ISCHEMIA_inf",
        "text_prompt": "inferior ischemia|st depression",
        "superdiag": "STTC",
        "target_count": 500,
    },
    {
        "prompt_key": "NORM",
        "text_prompt": "sinus rhythm|normal ecg",
        "superdiag": "NORM",
        "target_count": 1000,
    },
    {
        "prompt_key": "LBBB",
        "text_prompt": "left bundle branch block",
        "superdiag": "CD",
        "target_count": 300,
    },
    {
        "prompt_key": "RBBB",
        "text_prompt": "right bundle branch block",
        "superdiag": "CD",
        "target_count": 300,
    },
    {
        "prompt_key": "LVH",
        "text_prompt": "left ventricular hypertrophy",
        "superdiag": "HYP",
        "target_count": 300,
    },
]

# 默认超参数
DEFAULT_HYPERPARAMS = {
    "num_inference_steps": 100,
    "adversarial_guidance_scale": 0.02,  # 每步扰动 = latent_norm 的 2%（latent-relative step）
    "noise_sampling_guidance_scale": 0.01,  # NSG 扰动 = x_T norm 的 1%
    "num_noise_sampling_steps": 3,
    "guidance_start_fraction": 0.8,    # 只在最后 20% timestep 应用引导（AdvDiff: last 20%）
    "acceptance_range": [0.3, 0.7],    # 接受概率范围
    "batch_size": 4,
}


def boundary_loss(logits: torch.Tensor, target_indices: List[int]) -> torch.Tensor:
    """
    边界引导损失：将目标 pattern 的 logit 推向 0（对应 sigmoid(0)=0.5，即决策边界）

    在 logit 空间操作而非 probability 空间，避免 sigmoid 饱和区的梯度消失：
    - 当 prob ≈ 0（logit << 0）时：sigmoid'(logit) ≈ 0，prob-space 梯度消失
    - 在 logit 空间：loss = logit²，梯度 = 2*logit，无论 logit 多小都有非零梯度

    Args:
        logits: (B, 77) raw logits（sigmoid 之前）
        target_indices: 目标 pattern 索引列表

    Returns:
        loss: scalar，越小表示 logit 越接近 0（越接近边界）
    """
    target_logits = logits[:, target_indices]  # (B, K)
    return (target_logits ** 2).mean()


def sample_triangle_weights() -> Tuple[float, float, float]:
    """
    SA-AET 启发的三角形插值权重采样。

    返回 (w_adv, w_clean, w_hist)，约束 w_clean < w_hist < w_adv。
    确保结果偏向当前对抗样本（w_adv 最大），但混入 clean 和 historical 增加多样性。
    """
    for _ in range(1000):
        w_adv = random.randint(40, 80) / 100.0
        remaining = 100 - int(w_adv * 100)
        if remaining < 2:
            continue
        w_clean = random.randint(1, remaining - 1) / 100.0
        w_hist = 1.0 - w_adv - w_clean
        if 0.01 <= w_clean < w_hist < w_adv:
            return w_adv, w_clean, w_hist
    return 0.6, 0.15, 0.25  # fallback


class BoundaryAdvDiffGenerator:
    """
    边界引导 AdvDiff 生成器

    基于 AdvDiffAttacker 框架，修改梯度计算为 boundary guidance。
    复用 ECGTwinWrapper 的 DDPM 采样流程。
    """

    def __init__(
        self,
        ecgtwin_wrapper: ECGTwinWrapper,
        victim: EfficientNetVictim,
        device: str = "cuda",
        hyperparams: Dict = None,
    ):
        self.ecgtwin = ecgtwin_wrapper
        self.victim = victim
        self.device = torch.device(device)
        self.hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}

        # 将 victim 设置为 eval
        self.victim.eval()

    def _compute_boundary_gradient(
        self,
        latent: torch.Tensor,
        target_indices: List[int],
    ) -> torch.Tensor:
        """
        计算边界引导梯度 -∇_latent boundary_loss

        在 logit 空间计算损失，避免 prob 接近 0 时的梯度消失问题。
        梯度方向使 target logit 趋向 0（对应 prob → 0.5）。

        Args:
            latent: (B, 4, 128)，需要梯度
            target_indices: 目标 pattern 索引

        Returns:
            gradient: (B, 4, 128)，与 latent 同形，已乘以 -1（使 loss 减小）
        """
        latent_req = latent.detach().requires_grad_(True)

        # 全可微分推理链路，取 logits（sigmoid 之前）
        logits = self.victim.forward_from_latent_to_logits(latent_req)

        loss = boundary_loss(logits, target_indices)
        loss.backward()

        # -∇loss 方向使 loss 减小（即 logit 趋向 0）
        gradient = -latent_req.grad.clone()
        return gradient

    def _ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """DDPM 去噪一步"""
        t_batch = t * torch.ones(x_t.shape[0], dtype=torch.long, device=self.device)

        with torch.no_grad():
            noise_pred = self.ecgtwin.noise_predictor(
                x_t,
                t_batch,
                conditions["text_embed"],
                conditions["text_embed_mask"],
                conditions["pat_info"],
                conditions["base_vector"],
            )

        output = self.ecgtwin.scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=x_t,
        )
        return output["prev_sample"]

    def _select_best_of_k(
        self,
        z_adv: torch.Tensor,
        z_clean: torch.Tensor,
        z_hist: torch.Tensor,
        target_indices: List[int],
        K: int = 5,
    ) -> torch.Tensor:
        """
        SA-AET 启发的 Best-of-K 选择：生成 K 个三角形插值候选，选最接近决策边界的。

        Args:
            z_adv: (B, 4, 128) 当前 DDPM 输出
            z_clean: (B, 4, 128) 参考 ECG 的 VAE latent
            z_hist: (B, 4, 128) 上一轮的最佳 latent
            target_indices: 目标 pattern 索引
            K: 候选数量

        Returns:
            best_latents: (B, 4, 128)
        """
        B = z_adv.shape[0]
        best_latents = z_adv.clone()
        best_losses = torch.full((B,), float('inf'), device=self.device)

        for k in range(K):
            w_adv, w_clean, w_hist = sample_triangle_weights()
            z_k = w_adv * z_adv + w_clean * z_clean + w_hist * z_hist

            with torch.no_grad():
                logits = self.victim.forward_from_latent_to_logits(z_k)
                # boundary_loss per sample: 越小 = 越接近决策边界
                per_sample_loss = (logits[:, target_indices] ** 2).mean(dim=1)

            better = per_sample_loss < best_losses
            best_latents[better] = z_k[better]
            best_losses[better] = per_sample_loss[better]

        return best_latents

    def _check_acceptance(
        self,
        latent: torch.Tensor,
        target_indices: List[int],
        acceptance_range: Tuple[float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检查生成的 latent 是否满足接受条件（概率在边界附近）

        Returns:
            mask: (B,) bool，满足条件的样本
            probs: (B, 77) 概率
        """
        with torch.no_grad():
            probs = self.victim.forward_from_latent(latent, enable_grad=False)

        lo, hi = acceptance_range
        target_probs = probs[:, target_indices].mean(dim=1)  # (B,) 目标 pattern 均值
        mask = (target_probs > lo) & (target_probs < hi)
        return mask, probs

    def generate_batch(
        self,
        conditions: Dict[str, torch.Tensor],
        target_indices: List[int],
        batch_size: int = 4,
        historical_latent: Optional[torch.Tensor] = None,
        clean_latent: Optional[torch.Tensor] = None,
        best_of_k: int = 0,
    ) -> Dict[str, Any]:
        """
        生成一个 batch 的边界样本

        Args:
            conditions: ECGTwin 生成条件（text_embed, pat_info, base_vector 等）
            target_indices: 目标 EfficientNet pattern 索引
            batch_size: batch 大小
            historical_latent: (4, 128) 上一轮的最佳 latent（用于 SA-AET Evolution Triangle）
            clean_latent: (4, 128) 参考 ECG 的 VAE latent（用于 SA-AET Evolution Triangle）
            best_of_k: K 候选数（0 = 不使用三角形采样）

        Returns:
            dict with:
                ecg: (N_accept, 12, 2500) 接受的 ECG 样本（PTB-XL 格式）
                probs: (N_accept, 77) 对应的 EfficientNet 概率
                latents: (N_accept, 4, 128) 对应的 latent
        """
        hp = self.hp
        self.ecgtwin.scheduler.set_timesteps(hp["num_inference_steps"])
        timesteps = self.ecgtwin.scheduler.timesteps

        acceptance_range = tuple(hp["acceptance_range"])
        guidance_start = int(hp["num_inference_steps"] * hp["guidance_start_fraction"])

        # 初始噪声
        x_T = torch.randn(batch_size, 4, 128, device=self.device)

        all_accepted_ecg = []
        all_accepted_probs = []
        all_accepted_latents = []

        for n in range(hp["num_noise_sampling_steps"]):
            x_t = x_T.clone()

            # DDPM 反向过程 + 边界引导
            for step_idx, t in enumerate(timesteps):
                x_prev = self._ddpm_step(x_t, t, conditions)

                # 只在最后 20% 步应用边界引导（AdvDiff 原版策略）
                if step_idx >= guidance_start:
                    adv_grad = self._compute_boundary_gradient(x_prev, target_indices)

                    # Latent-relative step sizing:
                    # 归一化梯度方向，步长 = guidance_scale × ||latent||
                    # 这保证每步扰动是 latent norm 的固定比例（如 2%），
                    # 与 VAE/classifier 链路的梯度放大倍数无关。
                    # 50 步 × 20% guidance_start → 10 步引导，
                    # 0.02/step × 10 steps = ~20% 总扰动，足够影响分类但不破坏 ECG
                    grad_norm = adv_grad.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
                    latent_norm = x_prev.detach().norm(p=2, dim=(1, 2), keepdim=True)
                    guidance = hp["adversarial_guidance_scale"] * latent_norm * (adv_grad / grad_norm)
                    x_prev = x_prev + guidance

                x_t = x_prev

            x_0 = x_t

            # SA-AET Evolution Triangle + Best-of-K 选择
            if best_of_k > 0 and historical_latent is not None and clean_latent is not None:
                z_clean = clean_latent.unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                z_hist = historical_latent.unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
                x_0 = self._select_best_of_k(x_0, z_clean, z_hist, target_indices, K=best_of_k)

            # 筛选满足接受条件的样本
            accept_mask, probs = self._check_acceptance(x_0, target_indices, acceptance_range)

            if accept_mask.any():
                accepted_latents = x_0[accept_mask]

                # 解码为 ECG，并转换为 EfficientNet 输入格式 (B, 12, 2500)
                with torch.no_grad():
                    ecg_raw = self.ecgtwin.decoder(accepted_latents.clone())  # (B, 1024, 12)
                    if ecg_raw.shape[-1] == 12:
                        ecg_raw = ecg_raw.transpose(-1, -2)  # (B, 12, 1024)
                    ecg_raw = ecg_raw[:, ECGTWIN_TO_PTBXL_INDICES, :]  # lead reorder
                    ecg_ptbxl = F.interpolate(
                        ecg_raw, size=EFFICIENTNET_INPUT_LENGTH, mode="linear", align_corners=True
                    )  # (B, 12, 2500)

                all_accepted_ecg.append(ecg_ptbxl.cpu())
                all_accepted_probs.append(probs[accept_mask].cpu())
                all_accepted_latents.append(accepted_latents.cpu())

            # Noise Sampling Guidance：用 x_0 的梯度更新 x_T（AdvDiff NSG 机制）
            # 同样使用 latent-relative step sizing
            noise_grad = self._compute_boundary_gradient(x_0, target_indices)
            ng_norm = noise_grad.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
            xt_norm = x_T.detach().norm(p=2, dim=(1, 2), keepdim=True)
            x_T = x_T + hp["noise_sampling_guidance_scale"] * xt_norm * (noise_grad / ng_norm)

        result = {}
        if all_accepted_ecg:
            result["ecg"] = torch.cat(all_accepted_ecg, dim=0)
            result["probs"] = torch.cat(all_accepted_probs, dim=0)
            result["latents"] = torch.cat(all_accepted_latents, dim=0)
        else:
            result["ecg"] = torch.zeros(0, 12, EFFICIENTNET_INPUT_LENGTH)
            result["probs"] = torch.zeros(0, 77)
            result["latents"] = torch.zeros(0, 4, 128)

        return result


def prepare_conditions_for_prompt(
    ecgtwin: ECGTwinWrapper,
    text_prompt: str,
    reference_items: List[Dict],  # list of {"data": latent(4,128), "label": {...}}
    batch_size: int,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    为给定 prompt 准备 ECGTwin 生成条件

    Args:
        ecgtwin: ECGTwinWrapper 实例
        text_prompt: ECGTwin 目标文本 prompt（用 | 分隔）
        reference_items: PTBXL 同类样本的编码数据列表（每项含 'data' 和 'label'）
        batch_size: batch 大小
        device: 设备

    Returns:
        conditions dict 供 DDPM 采样使用
    """
    # 随机选取一个参考样本
    idx = torch.randint(0, len(reference_items), (1,)).item()
    ref_item = reference_items[idx]
    ref_latent = ref_item["data"]   # (4, 128)
    ref_label = ref_item["label"]   # dict with hr, age, sex, text_embed, etc.

    # 用 ECGTwinWrapper 的 prepare_conditions
    conditions = ecgtwin.prepare_conditions(
        ref_latent=ref_latent,
        ref_label=ref_label,
        batch_size=batch_size,
        target_text=text_prompt,
    )
    # SA-AET: 传递 clean latent 用于 Evolution Triangle
    conditions["ref_latent_raw"] = ref_latent  # (4, 128) CPU tensor
    return conditions


def run_generation(
    output_dir: str,
    ptbxl_encoded_path: str,
    device: str = "cuda",
    hyperparams: Dict = None,
    plan_override: List[Dict] = None,
    resume: bool = True,
):
    """
    运行完整生成流程，按 GENERATION_PLAN 依次生成各类别困难样本

    Args:
        output_dir: 输出目录（保存生成的 ECG 和标签）
        ptbxl_encoded_path: PTBXL VAE 编码数据的路径（prepare_ptbxl_for_ecgtwin.py 的输出）
        device: 设备
        hyperparams: 超参数覆盖
        plan_override: 覆盖默认的 GENERATION_PLAN
        resume: 是否从已有进度恢复（跳过已生成的 prompt_key）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_override or GENERATION_PLAN
    hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}

    # 加载 PTBXL 编码数据
    # 格式：List[{"data": tensor(4,128), "label": {"hr":..., "age":..., "sex":...,
    #              "text":..., "text_embed":..., "diagnostic_class":..., "strat_fold":...}}]
    print(f"Loading PTBXL encoded data from {ptbxl_encoded_path}...")
    ptbxl_items = torch.load(ptbxl_encoded_path, map_location="cpu")
    all_superdiag = [item["label"]["diagnostic_class"] for item in ptbxl_items]
    print(f"Loaded {len(ptbxl_items)} encoded PTBXL ECGs")

    # 加载 ECGTwin
    print("Loading ECGTwin...")
    ecgtwin = ECGTwinWrapper(device=device, load_encoder=False, load_text_model=True)

    # 加载 EfficientNet
    print("Loading EfficientNet...")
    victim = EfficientNetVictim(device=device, ecgtwin_wrapper=ecgtwin)

    generator = BoundaryAdvDiffGenerator(
        ecgtwin_wrapper=ecgtwin,
        victim=victim,
        device=device,
        hyperparams=hp,
    )

    all_results = []

    for item in plan:
        prompt_key = item["prompt_key"]
        text_prompt = item["text_prompt"]
        superdiag = item["superdiag"]
        target_count = item["target_count"]
        target_indices = get_target_indices(prompt_key)

        if not target_indices:
            print(f"[SKIP] {prompt_key}: no target indices defined")
            continue

        # 检查是否已生成
        save_path = output_dir / f"{prompt_key}.pt"
        if resume and save_path.exists():
            existing = torch.load(save_path, map_location="cpu")
            n_existing = existing["ecg"].shape[0]
            if n_existing >= target_count:
                print(f"[SKIP] {prompt_key}: already have {n_existing}/{target_count} samples")
                all_results.append({"prompt_key": prompt_key, **existing})
                continue
            else:
                print(f"[RESUME] {prompt_key}: have {n_existing}/{target_count}, generating more")

        print(f"\n{'='*60}")
        print(f"Generating: {prompt_key} ({target_count} samples)")
        print(f"  Text: {text_prompt}")
        print(f"  Target indices: {target_indices}")
        print(f"  Superdiag class: {superdiag}")

        # 选取对应 superdiag 的参考样本
        ref_items = [item for item, s in zip(ptbxl_items, all_superdiag) if s == superdiag]
        if len(ref_items) == 0:
            # fallback: 随机选取
            print(f"  [WARN] No {superdiag} samples found, using random reference")
            indices = torch.randperm(len(ptbxl_items))[:100].tolist()
            ref_items = [ptbxl_items[i] for i in indices]
        print(f"  Reference pool: {len(ref_items)} samples")

        accumulated_ecg = []
        accumulated_probs = []
        accumulated_latents = []

        # 已有进度
        if resume and save_path.exists():
            existing = torch.load(save_path, map_location="cpu")
            accumulated_ecg.append(existing["ecg"])
            accumulated_probs.append(existing["probs"])
            accumulated_latents.append(existing["latents"])

        batch_size = hp["batch_size"]
        n_collected = sum(e.shape[0] for e in accumulated_ecg)

        pbar = tqdm(total=target_count, initial=n_collected, desc=prompt_key)

        # 最大尝试次数：防止 acceptance_range 过严导致无限循环
        # 每次 generate_batch 最多产出 batch_size * num_noise_sampling_steps 个候选
        # 若持续无接受样本（如某些 pattern EfficientNet 永远输出极端概率），则放宽标准后退出
        max_attempts = target_count * 20   # 超过此次数未收集到足够样本则退出
        n_attempts = 0
        n_zero_attempts = 0               # 连续 0-接受 次数

        while n_collected < target_count:
            n_attempts += 1

            conditions = prepare_conditions_for_prompt(
                ecgtwin=ecgtwin,
                text_prompt=text_prompt,
                reference_items=ref_items,
                batch_size=batch_size,
                device=device,
            )

            batch_result = generator.generate_batch(
                conditions=conditions,
                target_indices=target_indices,
                batch_size=batch_size,
            )

            n_new = batch_result["ecg"].shape[0]
            if n_new > 0:
                accumulated_ecg.append(batch_result["ecg"])
                accumulated_probs.append(batch_result["probs"])
                accumulated_latents.append(batch_result["latents"])
                n_collected += n_new
                pbar.update(n_new)
                n_zero_attempts = 0

                # 定期保存
                if n_collected % 100 == 0 or n_collected >= target_count:
                    _save_intermediate(
                        save_path, accumulated_ecg, accumulated_probs, accumulated_latents, prompt_key
                    )
            else:
                n_zero_attempts += 1

            # 连续 50 次无接受样本：放宽 acceptance_range（两级：→(0.3,0.7)→(0.1,0.9)）
            if n_zero_attempts >= 50 and hp["acceptance_range"] != [0.1, 0.9]:
                old_range = hp["acceptance_range"]
                # 两步放宽：第一次 →(0.3,0.7)，第二次 →(0.1,0.9)
                if hp["acceptance_range"] != [0.3, 0.7]:
                    new_range = [0.3, 0.7]
                else:
                    new_range = [0.1, 0.9]
                print(f"\n  [WARN] {prompt_key}: 50 consecutive zero-accept batches. "
                      f"Widening acceptance_range {old_range} → {new_range}")
                hp = dict(hp, acceptance_range=new_range)
                generator.hp = hp
                n_zero_attempts = 0

            # 超过最大尝试次数：以现有样本保存并跳出
            if n_attempts >= max_attempts:
                print(f"\n  [WARN] {prompt_key}: reached max_attempts={max_attempts}, "
                      f"collected {n_collected}/{target_count}. Saving partial results.")
                break

        pbar.close()

        # 最终保存（截断到 target_count，允许部分结果）
        if not accumulated_ecg:
            print(f"  [SKIP] {prompt_key}: no accepted samples, skipping.")
            continue
        ecg_all = torch.cat(accumulated_ecg, dim=0)[:target_count]
        probs_all = torch.cat(accumulated_probs, dim=0)[:target_count]
        latents_all = torch.cat(accumulated_latents, dim=0)[:target_count]

        result = {
            "prompt_key": prompt_key,
            "ecg": ecg_all,
            "probs": probs_all,
            "latents": latents_all,
            "target_indices": target_indices,
        }
        torch.save(result, save_path)
        all_results.append(result)
        print(f"  Saved {ecg_all.shape[0]} samples to {save_path}")

    # 合并所有生成数据
    if all_results:
        _merge_and_save(all_results, output_dir)
        print(f"\nAll done. Results saved to {output_dir}")

    return all_results


def _save_intermediate(save_path, ecg_list, probs_list, latents_list, prompt_key):
    ecg = torch.cat(ecg_list, dim=0)
    probs = torch.cat(probs_list, dim=0)
    latents = torch.cat(latents_list, dim=0)
    torch.save(
        {"prompt_key": prompt_key, "ecg": ecg, "probs": probs, "latents": latents},
        save_path,
    )


def _make_labels_77(probs: torch.Tensor, prompt_key: str, soft: bool = False) -> torch.Tensor:
    """
    为生成样本构建 77 维标签向量。

    策略：
      1. 用 probs > 0.5 作为基础标签（保留非目标 pattern 的模型预测）
      2. 将 prompt 对应的 target_indices 设为阳性

    Args:
        probs: (N, 77) 模型预测概率
        prompt_key: prompt 名称
        soft: 是否使用 soft labeling（v4 SA-AET 改进）
              True: labels[:, target] = clamp(probs, min=0.6)（减少噪声）
              False: labels[:, target] = 1.0（原始行为）
    """
    labels = (probs > 0.5).float()
    target_indices = PROMPT_TO_PATTERN_INDICES.get(prompt_key, [])
    if target_indices:
        if soft:
            labels[:, target_indices] = torch.clamp(probs[:, target_indices], min=0.6)
        else:
            labels[:, target_indices] = 1.0
    return labels


def _merge_and_save(all_results: List[Dict], output_dir: Path):
    """合并所有 prompt 的生成结果为单个 .pt 文件"""
    ecg_list = [r["ecg"] for r in all_results]
    probs_list = [r["probs"] for r in all_results]
    labels_77_list = [
        _make_labels_77(r["probs"], r["prompt_key"]) for r in all_results
    ]

    merged = {
        "ecg": torch.cat(ecg_list, dim=0),           # (N_total, 12, 2500)
        "probs": torch.cat(probs_list, dim=0),         # (N_total, 77)
        "labels_77": torch.cat(labels_77_list, dim=0), # (N_total, 77)
        "prompt_keys": [
            pk for r in all_results
            for pk in [r["prompt_key"]] * r["ecg"].shape[0]
        ],
    }
    merge_path = output_dir / "all_hard_samples.pt"
    torch.save(merged, merge_path)
    print(f"Merged {merged['ecg'].shape[0]} total samples → {merge_path}")

    # 打印每个 prompt 的标签阳性率，便于核查
    offset = 0
    for r in all_results:
        n = r["ecg"].shape[0]
        pk = r["prompt_key"]
        tgt = PROMPT_TO_PATTERN_INDICES.get(pk, [])
        if tgt:
            pos_rate = merged["labels_77"][offset:offset+n, tgt].mean().item()
            print(f"  {pk}: {n} samples, target pattern positive rate = {pos_rate:.3f}")
        offset += n
