#!/usr/bin/env python
"""
在线对抗训练 v4 — SA-AET 启发改进

4 项核心改进（vs v2/v3）：
  1. Latent Evolution Triangle + Best-of-K：三角形插值 + 自适应选择最难样本
  2. EWA Anchor 正则化：防止灾难性遗忘
  3. Quality-Aware Buffer：质量淘汰替代 FIFO
  4. Soft Labeling：减少边界样本的噪声标签

用法:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/run_online_training_v4.py
    /root/miniforge3/envs/ECGTwin/bin/python scripts/run_online_training_v4.py --n_epochs 80
"""

import argparse
import json
import shutil
import sys
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from adversarial.adv_generate import (
    BoundaryAdvDiffGenerator,
    prepare_conditions_for_prompt,
    _make_labels_77,
    DEFAULT_HYPERPARAMS,
)
from adversarial.efficientnet_adapter import EfficientNetAdapter
from adversarial.efficientnet_victim import EfficientNetVictim, EFFICIENTNET_INPUT_LENGTH
from adversarial.finetune import (
    load_ptbxl_for_finetune,
    compute_pos_weight,
    train_one_epoch,
    validate_one_epoch,
    DEFAULT_TRAIN_CONFIG,
)
from adversarial.evaluate import evaluate as run_evaluation
from adversarial.label_mapping import get_target_indices, PROMPT_TO_PATTERN_INDICES
from util.ecgtwin_utils import ECGTwinWrapper

# ——— 路径配置 ———
PTBXL_ENCODED_PATH = "/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt"
PTBXL_ROOT = "/root/ECG_adv_Gen/datasets/PTBXL"
EFFICIENTNET_WEIGHT = (
    "/root/ECG_adv_Gen/model/DeepECG/weights"
    "/efficientnetv2_77_classes/efficientnet_deepecg_unscaled.pt"
)
DEFAULT_WORK_DIR = "/root/autodl-tmp/online_training_v4"

# ——— MI-focused generation plan ———
ONLINE_GENERATION_PLAN = [
    {
        "prompt_key": "MIMIC_MI_inferior",
        "text_prompt": "inferior myocardial infarction|st elevation|abnormal ecg",
        "superdiag": "MI",
    },
    {
        "prompt_key": "MIMIC_MI_acute",
        "text_prompt": "consider acute st elevation mi|inferior st elevation, consider acute infarct|abnormal ecg",
        "superdiag": "MI",
    },
    {
        "prompt_key": "MIMIC_MI_acute_inferior",
        "text_prompt": "acute myocardial infarction|inferior myocardial infarction|st elevation|abnormal ecg",
        "superdiag": "MI",
    },
]

GEN_HYPERPARAMS = {
    "num_inference_steps": 50,
    "adversarial_guidance_scale": 0.03,
    "noise_sampling_guidance_scale": 0.01,
    "num_noise_sampling_steps": 2,
    "guidance_start_fraction": 0.8,
    "acceptance_range": [0.35, 0.70],
    "batch_size": 4,
}


# ============================================================
# Quality-Aware Buffer (改动 3)
# ============================================================

class QualityAwareBuffer:
    """
    质量感知对抗样本缓冲区。

    与 FIFO 的区别：
    - 每个样本携带 informativeness score
    - 淘汰 score 最低的（模型已"解决"的），而非最旧的
    - 定期用当前 adapter 重新评分
    """

    def __init__(self, max_size: int = 1500):
        self.max_size = max_size
        self.ecg_list = []       # list of (12, 2500) tensors
        self.label_list = []     # list of (77,) tensors
        self.score_list = []     # list of float

    def __len__(self):
        return len(self.ecg_list)

    def add_batch(self, ecg: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor):
        """添加一批样本 (all CPU tensors)"""
        for i in range(ecg.shape[0]):
            self.ecg_list.append(ecg[i])
            self.label_list.append(labels[i])
            self.score_list.append(scores[i].item())
        self._evict()

    def _evict(self):
        """淘汰 score 最低的样本"""
        while len(self.ecg_list) > self.max_size:
            min_idx = min(range(len(self.score_list)), key=lambda i: self.score_list[i])
            self.ecg_list.pop(min_idx)
            self.label_list.pop(min_idx)
            self.score_list.pop(min_idx)

    def rescore(self, adapter: EfficientNetAdapter, device: str = "cuda"):
        """用当前 adapter 重新评估所有样本的 informativeness"""
        if not self.ecg_list:
            return
        adapter.eval()
        ecg_all = torch.stack(self.ecg_list)
        with torch.no_grad():
            for i in range(0, len(ecg_all), 64):
                batch = ecg_all[i:i+64].to(device)
                logits = adapter(batch)
                # informativeness = 1 / (1 + mean|logit|)，越不确定分越高
                scores = 1.0 / (1.0 + logits.abs().mean(dim=1))
                for j, s in enumerate(scores.cpu().tolist()):
                    if i + j < len(self.score_list):
                        self.score_list[i + j] = s

    def to_dataset(self) -> TensorDataset:
        if not self.ecg_list:
            return None
        return TensorDataset(
            torch.stack(self.ecg_list),
            torch.stack(self.label_list),
        )

    def get_sampling_weights(self):
        """返回 per-sample weights 用于 WeightedRandomSampler"""
        return [max(s, 0.05) for s in self.score_list]


# ============================================================
# 生成函数
# ============================================================

def generate_epoch_samples(
    generator: BoundaryAdvDiffGenerator,
    ecgtwin: ECGTwinWrapper,
    ref_items_by_superdiag: dict,
    plan: list,
    samples_per_prompt: int,
    z_hist: dict,
    device: str,
    best_of_k: int = 5,
    max_attempts: int = 50,
) -> dict:
    """
    Per-epoch 生成：每个 prompt 生成少量样本，使用 SA-AET Evolution Triangle。

    Returns:
        dict with ecg, labels_77, scores, updated z_hist
    """
    all_ecg = []
    all_labels = []
    all_scores = []

    for item in plan:
        prompt_key = item["prompt_key"]
        text_prompt = item["text_prompt"]
        superdiag = item["superdiag"]
        target_indices = get_target_indices(prompt_key)

        if not target_indices:
            continue

        ref_items = ref_items_by_superdiag.get(superdiag, [])
        if not ref_items:
            continue

        n_collected = 0
        n_attempts = 0

        while n_collected < samples_per_prompt and n_attempts < max_attempts:
            n_attempts += 1
            conditions = prepare_conditions_for_prompt(
                ecgtwin=ecgtwin,
                text_prompt=text_prompt,
                reference_items=ref_items,
                batch_size=generator.hp["batch_size"],
                device=device,
            )

            # SA-AET: 传入 historical 和 clean latent
            hist_lat = z_hist.get(prompt_key)
            clean_lat = conditions.get("ref_latent_raw")

            batch_result = generator.generate_batch(
                conditions=conditions,
                target_indices=target_indices,
                batch_size=generator.hp["batch_size"],
                historical_latent=hist_lat,
                clean_latent=clean_lat,
                best_of_k=best_of_k if hist_lat is not None else 0,
            )

            n_new = batch_result["ecg"].shape[0]
            if n_new > 0:
                # Soft labeling (改动 4)
                labels_77 = _make_labels_77(batch_result["probs"], prompt_key, soft=True)

                # Informativeness score: 1 / (1 + mean|logit_target|)
                with torch.no_grad():
                    target_probs = batch_result["probs"][:, target_indices]
                    # 越接近 0.5 → score 越高
                    scores = 1.0 - 2.0 * (target_probs.mean(dim=1) - 0.5).abs()

                all_ecg.append(batch_result["ecg"])
                all_labels.append(labels_77)
                all_scores.append(scores)
                n_collected += n_new

                # 更新 historical latent（用最后一个 accepted latent）
                z_hist[prompt_key] = batch_result["latents"][-1].cpu()

    result = {
        "ecg": torch.cat(all_ecg, dim=0) if all_ecg else torch.zeros(0, 12, EFFICIENTNET_INPUT_LENGTH),
        "labels_77": torch.cat(all_labels, dim=0) if all_labels else torch.zeros(0, 77),
        "scores": torch.cat(all_scores, dim=0) if all_scores else torch.zeros(0),
        "z_hist": z_hist,
    }
    return result


# ============================================================
# ECG 可视化
# ============================================================

def visualize_ecg_samples(ecg: torch.Tensor, probs: torch.Tensor, save_path: str, n_samples: int = 4):
    """可视化生成的 ECG 样本"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from adversarial.evaluate import ECG_PATTERNS

    n_samples = min(n_samples, ecg.shape[0])
    if n_samples == 0:
        return

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        ax = axes[i]
        ecg_i = ecg[i].numpy()
        t = np.arange(2500) / 250.0
        for lead_idx in range(12):
            ax.plot(t, ecg_i[lead_idx] + lead_idx * 2.0, linewidth=0.5, color="black")
            ax.text(-0.3, lead_idx * 2.0, lead_names[lead_idx], fontsize=7, va="center")
        prob_i = probs[i].numpy()
        top5 = np.argsort(prob_i)[-5:][::-1]
        top5_str = ", ".join([f"{ECG_PATTERNS[j]}:{prob_i[j]:.2f}" for j in top5])
        ax.set_title(f"Sample {i}: {top5_str}", fontsize=8)
        ax.set_xlim(-0.5, 10.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SA-AET Online Adversarial Training v4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--samples_per_prompt", type=int, default=10,
                        help="Samples per prompt per epoch")
    parser.add_argument("--buffer_max_size", type=int, default=1500)
    parser.add_argument("--n_per_superdiag", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--anchor_lambda", type=float, default=0.1)
    parser.add_argument("--ewa_decay", type=float, default=0.999)
    parser.add_argument("--best_of_k", type=int, default=5)
    parser.add_argument("--rescore_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=15)
    parser.add_argument("--work_dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print("=" * 70)
    print("SA-AET Online Adversarial Training v4")
    print("=" * 70)
    print(f"Epochs:         {args.n_epochs}")
    print(f"Samples/prompt: {args.samples_per_prompt}")
    print(f"Buffer size:    {args.buffer_max_size}")
    print(f"LR:             {args.lr}")
    print(f"Best-of-K:      {args.best_of_k}")
    print(f"Anchor lambda:  {args.anchor_lambda}")
    print(f"EWA decay:      {args.ewa_decay}")

    # ===== 加载共享资源 =====
    print("\nLoading resources...")
    t0 = time.time()

    ecgtwin = ECGTwinWrapper(device=device, load_encoder=False, load_text_model=True)
    victim = EfficientNetVictim(device=device, ecgtwin_wrapper=ecgtwin)

    ptbxl_items = torch.load(PTBXL_ENCODED_PATH, map_location="cpu")
    ref_items_by_superdiag = {}
    for item in ptbxl_items:
        sd = item["label"].get("diagnostic_class", "UNKNOWN")
        ref_items_by_superdiag.setdefault(sd, []).append(item)

    train_ds, val_ds = load_ptbxl_for_finetune(
        ptbxl_data_root=PTBXL_ROOT, val_fold=9,
        n_per_superdiag=args.n_per_superdiag, seed=args.seed,
    )
    print(f"Resources loaded in {time.time() - t0:.1f}s")

    # ===== 初始化 adapter + 优化器 =====
    adapter = EfficientNetAdapter(backbone=victim.jit_model, device=device).to(device)
    optimizer = AdamW(adapter.adapter_parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # EWA anchor 参数 (改动 2)
    ewa_params = [p.data.clone() for p in adapter.adapter_parameters()]

    # pos_weight for BCE
    all_labels = torch.cat([labels for _, labels in train_ds], dim=0)
    pos_weight = compute_pos_weight(all_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Val loader
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # ===== Buffer & generator =====
    buffer = QualityAwareBuffer(max_size=args.buffer_max_size)
    generator = BoundaryAdvDiffGenerator(
        ecgtwin_wrapper=ecgtwin, victim=victim, device=device, hyperparams=GEN_HYPERPARAMS,
    )
    z_hist = {}  # per-prompt historical latent

    # ===== Training log =====
    log = {"args": vars(args), "epochs": []}
    log_path = work_dir / "training_log.json"
    best_val_loss = float("inf")
    adapter_save_path = str(work_dir / "adapter_best.pt")

    print(f"\nStarting {args.n_epochs}-epoch online training...")

    for epoch in range(args.n_epochs):
        epoch_start = time.time()
        epoch_info = {"epoch": epoch}

        # ===== Phase A: 生成新对抗样本 =====
        victim.set_adapter(adapter)
        gen_result = generate_epoch_samples(
            generator=generator,
            ecgtwin=ecgtwin,
            ref_items_by_superdiag=ref_items_by_superdiag,
            plan=ONLINE_GENERATION_PLAN,
            samples_per_prompt=args.samples_per_prompt,
            z_hist=z_hist,
            device=device,
            best_of_k=args.best_of_k,
        )
        z_hist = gen_result["z_hist"]
        n_generated = gen_result["ecg"].shape[0]

        if n_generated > 0:
            buffer.add_batch(gen_result["ecg"], gen_result["labels_77"], gen_result["scores"])

        epoch_info["n_generated"] = n_generated
        epoch_info["buffer_size"] = len(buffer)

        # ===== Phase B: Buffer 重评分 =====
        if epoch > 0 and epoch % args.rescore_interval == 0:
            buffer.rescore(adapter, device)

        # ===== Phase C: 构建 DataLoader =====
        victim.clear_adapter()

        generated_ds = buffer.to_dataset()
        if generated_ds is not None:
            n_real = len(train_ds)
            n_gen = len(generated_ds)
            # Quality-weighted sampling
            buffer_weights = buffer.get_sampling_weights()
            real_weight = 1.0
            gen_weight = 2.0  # generated_weight
            weights = [real_weight] * n_real + [gen_weight * w for w in buffer_weights]
            sampler = WeightedRandomSampler(weights, num_samples=n_real + n_gen, replacement=True)
            combined_ds = ConcatDataset([train_ds, generated_ds])
            train_loader = DataLoader(combined_ds, batch_size=32, sampler=sampler,
                                      num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                      num_workers=4, pin_memory=True)

        # ===== Phase D: 训练一个 epoch =====
        train_loss = train_one_epoch(
            adapter=adapter,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device(device),
            grad_clip=1.0,
            ewa_params=ewa_params,
            anchor_lambda=args.anchor_lambda,
            ewa_decay=args.ewa_decay,
        )
        scheduler.step()

        # ===== Phase E: 验证 =====
        val_loss = validate_one_epoch(adapter, val_loader, criterion, torch.device(device))

        epoch_info["train_loss"] = train_loss
        epoch_info["val_loss"] = val_loss
        epoch_info["lr"] = optimizer.param_groups[0]["lr"]
        epoch_info["duration"] = time.time() - epoch_start

        # 保存最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            adapter.save(adapter_save_path)

        # 打印进度
        gen_str = f"gen={n_generated}" if n_generated > 0 else "gen=0"
        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"buf={len(buffer)} | {gen_str} | "
              f"{epoch_info['duration']:.0f}s"
              + (" *" if val_loss == best_val_loss else ""))

        # ===== Phase F: 定期 AUPRC 评估 =====
        if epoch > 0 and (epoch % args.eval_interval == 0 or epoch == args.n_epochs - 1):
            print(f"\n--- AUPRC Evaluation at epoch {epoch+1} ---")
            eval_path = str(work_dir / f"eval_epoch_{epoch+1}.json")
            eval_results = run_evaluation(
                ptbxl_data_root=PTBXL_ROOT,
                adapter_path=adapter_save_path,
                efficientnet_weight_path=EFFICIENTNET_WEIGHT,
                device=device,
                output_path=eval_path,
            )
            # Extract key metrics
            baseline_cat = eval_results.get("baseline", {}).get("per_category", {})
            finetuned_cat = eval_results.get("finetuned", {}).get("per_category", {})
            eval_summary = {}
            for cat in baseline_cat:
                b = baseline_cat[cat].get("macro_auprc", float("nan"))
                f = finetuned_cat.get(cat, {}).get("macro_auprc", float("nan"))
                if not (np.isnan(b) or np.isnan(f)):
                    eval_summary[cat] = {"baseline": b, "finetuned": f, "delta": f - b}
                    print(f"  {cat}: {b:.4f} → {f:.4f} ({f-b:+.4f})")
            epoch_info["eval"] = eval_summary

            # ECG 可视化
            if n_generated > 0:
                vis_path = str(work_dir / f"ecg_epoch_{epoch+1}.png")
                visualize_ecg_samples(gen_result["ecg"], gen_result.get("probs", gen_result["labels_77"]),
                                      vis_path)
            print()

        log["epochs"].append(epoch_info)

        # 定期保存 log
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2, default=str)

    # ===== 最终评估 =====
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    final_eval_path = str(work_dir / "eval_final.json")
    final_results = run_evaluation(
        ptbxl_data_root=PTBXL_ROOT,
        adapter_path=adapter_save_path,
        efficientnet_weight_path=EFFICIENTNET_WEIGHT,
        device=device,
        output_path=final_eval_path,
    )

    # 复制最佳 adapter 到项目目录
    final_output = Path("/root/ECG_adv_Gen/outputs/online_training_v4")
    final_output.mkdir(parents=True, exist_ok=True)
    shutil.copy(adapter_save_path, final_output / "adapter_best.pt")
    shutil.copy(log_path, final_output / "training_log.json")

    print(f"\nBest adapter → {final_output / 'adapter_best.pt'}")
    print(f"Training log → {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
