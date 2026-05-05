"""
使用训练好的 Center Token 生成 PTBXL 风格的 ECG

流程：
1. 加载 ECGTwinWrapper + 训练好的 center token
2. 注册 hook 注入 center token
3. 按 generation prompts 生成 ECG
4. 保存生成结果
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ECGTWIN_ROOT = PROJECT_ROOT / "model" / "ECGTwin"
sys.path.insert(0, str(ECGTWIN_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from util.ecgtwin_utils import ECGTwinWrapper
from methods.ecgtwin_gen.center_token.model import CenterToken


# 默认生成 prompts
GENERATION_PROMPTS = [
    {"text": "sinus rhythm|normal ecg", "hr": 70, "age": 60},
    {"text": "sinus rhythm|normal ecg", "hr": 85, "age": 45},
    {"text": "sinus rhythm|normal ecg", "hr": 60, "age": 70},
    {"text": "inferior myocardial infarction|abnormal ecg", "hr": 75, "age": 65},
    {"text": "anterior myocardial infarction|st elevation|abnormal ecg", "hr": 80, "age": 60},
    {"text": "left ventricular hypertrophy|abnormal ecg", "hr": 75, "age": 55},
    {"text": "complete right bundle branch block|abnormal ecg", "hr": 70, "age": 65},
    {"text": "complete left bundle branch block|abnormal ecg", "hr": 72, "age": 60},
    {"text": "atrial fibrillation|abnormal ecg", "hr": 110, "age": 70},
    {"text": "sinus tachycardia|abnormal ecg", "hr": 120, "age": 50},
    {"text": "sinus bradycardia|abnormal ecg", "hr": 50, "age": 60},
    {"text": "first degree atrioventricular block|abnormal ecg", "hr": 68, "age": 65},
]


class CenterTokenGenerator:

    def __init__(
        self,
        center_token_path: str,
        ecgtwin_config_path: Optional[str] = None,
        device: str = "cuda:0",
        center_dim: int = 256,
    ):
        # 加载 ECGTwin
        self.wrapper = ECGTwinWrapper(
            config_path=ecgtwin_config_path,
            device=device,
            load_encoder=False,
            load_text_model=True,
        )
        self.device = torch.device(device)

        # 加载 center token
        self.center_token = CenterToken(dim=center_dim)
        ckpt = torch.load(center_token_path, map_location="cpu")
        self.center_token.load_state_dict(ckpt["center_token"])
        self.center_token.to(self.device)
        self.center_token.eval()
        print(f"Loaded center token (norm={self.center_token.norm:.4f})")

        # 注册 hooks
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        center = self.center_token

        def make_hook():
            def hook_fn(module, args):
                x, c, c2 = args[0], args[1], args[2]
                rest = args[3:]
                c = c + center(c.size(0))
                return (x, c, c2) + rest
            return hook_fn

        for block in self.wrapper.noise_predictor.blocks:
            h = block.register_forward_pre_hook(make_hook())
            self.hooks.append(h)

    def generate(
        self,
        ref_data_path: str,
        prompts: list = None,
        gen_batch: int = 50,
        num_inference_steps: int = 1000,
        save_path: str = "generation_result",
        sex: str = "M",
    ):
        """
        生成 ECG

        Args:
            ref_data_path: 参考数据路径 (.pt)
            prompts: 生成 prompt 列表, 每个 {"text", "hr", "age"}
            gen_batch: 每个 prompt 生成多少条
            num_inference_steps: 推理步数
            save_path: 保存路径
            sex: 参考患者性别
        """
        if prompts is None:
            prompts = GENERATION_PROMPTS

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 加载参考数据
        ref_data = torch.load(ref_data_path, map_location="cpu")
        ref_latent = ref_data["data"]
        ref_label = ref_data["label"]

        all_generated = []

        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Generating: {prompt['text']}")
            prompt_dir = save_path / f"{i:03d}"
            prompt_dir.mkdir(exist_ok=True)

            # 准备条件
            conditions = self.wrapper.prepare_conditions(
                ref_latent=ref_latent,
                ref_label=ref_label,
                batch_size=gen_batch,
                target_text=prompt["text"],
                target_hr=prompt["hr"],
                target_age=prompt["age"],
            )

            # 采样 (hooks 自动注入 center token)
            latent_gen = self.wrapper.ddpm_sample(
                conditions=conditions,
                batch_size=gen_batch,
                num_inference_steps=num_inference_steps,
            )

            # 解码
            ecg_gen = self.wrapper.decode_latent(latent_gen)

            # 保存
            torch.save(latent_gen.cpu(), prompt_dir / "latent_gen.pt")
            torch.save(ecg_gen.cpu(), prompt_dir / "ecg_gen.pt")

            features = {
                "prompt": prompt,
                "gen_batch": gen_batch,
                "num_inference_steps": num_inference_steps,
                "center_token_norm": self.center_token.norm,
                "ref_sex": sex,
            }
            with open(prompt_dir / "features.json", "w") as f:
                json.dump(features, f, indent=4)

            all_generated.append({
                "prompt": prompt,
                "latent": latent_gen.cpu(),
                "ecg": ecg_gen.cpu(),
            })
            print(f"  Saved {gen_batch} ECGs to {prompt_dir}")

        # 保存汇总
        torch.save(all_generated, save_path / "all_generated.pt")
        print(f"\nGeneration complete. Total: {len(prompts) * gen_batch} ECGs")

    def generate_without_center(
        self,
        ref_data_path: str,
        prompts: list = None,
        gen_batch: int = 50,
        num_inference_steps: int = 1000,
        save_path: str = "generation_result_baseline",
    ):
        """生成不带 center token 的 baseline ECG 用于对比"""
        # 移除 hooks
        for h in self.hooks:
            h.remove()
        self.hooks = []

        self.generate(
            ref_data_path=ref_data_path,
            prompts=prompts,
            gen_batch=gen_batch,
            num_inference_steps=num_inference_steps,
            save_path=save_path,
        )

        # 重新注册 hooks
        self._register_hooks()


def main():
    parser = argparse.ArgumentParser(description="Generate ECGs with center token")
    parser.add_argument(
        "--center_token_path",
        type=str,
        required=True,
        help="Path to trained center token checkpoint",
    )
    parser.add_argument(
        "--ref_data_path",
        type=str,
        default="model/ECGTwin/data/prepared_input/normal_1.pt",
        help="Reference ECG data path",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="datasets/PTBXL/generated_with_center_token",
    )
    parser.add_argument(
        "--save_path_baseline",
        type=str,
        default="datasets/PTBXL/generated_baseline",
    )
    parser.add_argument("--gen_batch", type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--generate_baseline",
        action="store_true",
        help="Also generate baseline without center token",
    )
    args = parser.parse_args()

    generator = CenterTokenGenerator(
        center_token_path=args.center_token_path,
        device=args.device,
    )

    generator.generate(
        ref_data_path=args.ref_data_path,
        gen_batch=args.gen_batch,
        num_inference_steps=args.num_inference_steps,
        save_path=args.save_path,
    )

    if args.generate_baseline:
        generator.generate_without_center(
            ref_data_path=args.ref_data_path,
            gen_batch=args.gen_batch,
            num_inference_steps=args.num_inference_steps,
            save_path=args.save_path_baseline,
        )


if __name__ == "__main__":
    main()
