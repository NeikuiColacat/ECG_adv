"""
ECGTwin 推理 batch size 压力测试

在 GPU 上从小到大尝试不同 batch size，找到不 OOM 的最大值。
每个 batch size 跑完整 DDPM 去噪 1000 步，测量显存和耗时。

Usage:
    python util/test_batch_size.py
"""

import sys
import time
import torch
import yaml
from pathlib import Path

# ——— 路径设置 ———
PROJECT_ROOT = Path(__file__).parent.parent
ECGTWIN_ROOT = PROJECT_ROOT / "model" / "ECGTwin"
sys.modules['tensorflow'] = None
sys.path.insert(0, str(ECGTWIN_ROOT))

from diffusers import DDPMScheduler
from transformers import AutoModel, AutoTokenizer

if sys.modules.get('tensorflow') is None:
    del sys.modules['tensorflow']

from module.IBExtractor import IBExtractor
from module.vae_model import VAE_Decoder
from utils.model_utils import build_noise_predictor
from utils.data_utils import process_pat_info, get_text_embedding, sex_transform


def load_all_models(config_path: str, device: str = "cuda:0"):
    """加载所有 ECGTwin 推理组件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    h_ = config["hyper_para"]
    settings = config["inference_setting"]
    model_type = config["meta"]["model_type"]

    # 1. Noise predictor (DiT)
    noise_predictor = build_noise_predictor(model_type, 4, h_)
    noise_predictor.load_state_dict(
        torch.load(ECGTWIN_ROOT / settings["noise_predictor_path"], map_location="cpu")
    )
    noise_predictor.to(device).eval()

    # 2. DDPM Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=h_["ddpm"]["num_train_steps"],
        beta_start=h_["ddpm"]["beta_start"],
        beta_end=h_["ddpm"]["beta_end"],
    )
    scheduler.set_timesteps(settings["inference_timestep"])

    # 3. IBExtractor
    ibe_model = IBExtractor(
        embed_dim=h_["ibe"]["embed_dim"],
        num_heads=h_["ibe"]["num_heads"],
        ff_hidden_size=h_["ibe"]["ff_hidden_size"],
        num_layers=h_["ibe"]["num_layers"],
        text_embed_dim=h_["ibe"]["text_embed_dim"],
        patient_info_size=h_["ibe"]["patient_info_size"],
    )
    ibe_model.load_state_dict(
        torch.load(ECGTWIN_ROOT / config["dependencies"]["ibe_path"], map_location="cpu")
    )
    ibe_model.to(device).eval()

    # 4. VAE Decoder
    decoder = VAE_Decoder()
    vae_ckpt = torch.load(ECGTWIN_ROOT / config["dependencies"]["vae_path"], map_location="cpu")
    decoder.load_state_dict(vae_ckpt["decoder"])
    decoder.to(device).eval()

    # 5. Nomic text encoder
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedding_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True,
    )
    embedding_model.to(device).eval()

    return noise_predictor, scheduler, ibe_model, decoder, tokenizer, embedding_model


def prepare_conditions(ibe_model, tokenizer, embedding_model, batch_size, device):
    """准备推理所需的所有条件向量"""
    # 加载参考数据
    ref_data = torch.load(ECGTWIN_ROOT / "data/prepared_input/normal_1.pt", map_location="cpu")
    ref_latent = ref_data["data"]       # (4, 128)
    ref_label = ref_data["label"]

    # ref 条件
    pat_info_ref = process_pat_info(
        normalize=True,
        hr=torch.tensor([ref_label["hr"]]),
        age=torch.tensor([ref_label["age"]]),
        sex=torch.tensor([sex_transform(ref_label["sex"])]),
    ).repeat(batch_size, 1).to(device)

    text_embed_ref = ref_label["text_embed"].unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    latent_ref = ref_latent.unsqueeze(0).repeat(batch_size, 1, 1).transpose(2, 1).to(device)

    # 提取 base_vector
    with torch.no_grad():
        base_vector = ibe_model.extract_features(latent_ref, text_embed_ref, None, pat_info_ref, reduce=True)

    # tar 条件
    text_embed_tar = get_text_embedding(
        text="sinus rhythm|normal ecg.",
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        mix=False,
    ).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    pat_info_tar = process_pat_info(
        normalize=True,
        add_token=False,
        hr=torch.tensor([70.0]),
        age=torch.tensor([70.0]),
        sex=torch.tensor([sex_transform("F")]),
    ).repeat(batch_size, 1).to(device)

    return text_embed_tar, pat_info_tar, base_vector


@torch.no_grad()
def run_full_generation(noise_predictor, scheduler, decoder, batch_size, device,
                        text_embed, pat_info, base_vector):
    """跑完整 1000 步 DDPM 去噪 + VAE 解码"""
    # DDPM 去噪
    xi = torch.randn(batch_size, 4, 128, device=device)
    for i in scheduler.timesteps:
        t = i * torch.ones(batch_size, dtype=torch.long, device=device)
        predicted_noise = noise_predictor(xi, t, text_embed, None, pat_info, base_vector)
        xi = scheduler.step(model_output=predicted_noise, timestep=i, sample=xi)['prev_sample']

    # VAE 解码
    ecg = decoder(xi)  # (B, 1024, 12)
    return ecg


def get_gpu_memory():
    """返回当前 GPU 已用/总显存 (MB)"""
    used = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
    return used, reserved, total


def main():
    device = "cuda:0"
    config_path = str(ECGTWIN_ROOT / "config" / "DiT_ECGTwin.yaml")

    print("=" * 70)
    print("ECGTwin Batch Size 压力测试")
    print("=" * 70)

    # GPU 信息
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
    print(f"GPU: {gpu_name} | 显存: {total_mem:.0f} MB")
    print()

    # 加载模型
    print("加载模型...")
    noise_predictor, scheduler, ibe_model, decoder, tokenizer, embedding_model = \
        load_all_models(config_path, device)

    used, reserved, total = get_gpu_memory()
    print(f"模型加载后显存: {used:.0f} MB used / {total:.0f} MB total")
    print()

    # 要测试的 batch size 列表
    batch_sizes = [16, 32, 64, 128, 256, 512, 768, 1024]

    print(f"{'Batch':>6} | {'状态':>6} | {'去噪耗时':>10} | {'已用显存':>10} | {'峰值显存':>10} | {'每条耗时':>10}")
    print("-" * 78)

    max_success_batch = 0

    for bs in batch_sizes:
        # 清理显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # 准备条件
            text_embed, pat_info, base_vector = prepare_conditions(
                ibe_model, tokenizer, embedding_model, bs, device
            )

            # 跑完整生成
            start_time = time.time()
            ecg = run_full_generation(
                noise_predictor, scheduler, decoder, bs, device,
                text_embed, pat_info, base_vector
            )
            elapsed = time.time() - start_time

            used, _, total = get_gpu_memory()
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            per_sample = elapsed / bs

            print(f"{bs:>6} | {'OK':>6} | {elapsed:>8.1f}s | {used:>8.0f} MB | {peak:>8.0f} MB | {per_sample:>8.3f}s")
            max_success_batch = bs

            # 清理本轮数据
            del ecg, text_embed, pat_info, base_vector

        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6} | {'OOM':>6} | {'---':>10} | {'---':>10} | {'---':>10} | {'---':>10}")
            torch.cuda.empty_cache()
            break

        except Exception as e:
            print(f"{bs:>6} | {'ERROR':>6} | {str(e)[:40]}")
            break

    print()
    print("=" * 70)
    print(f"最大可用 batch size: {max_success_batch}")
    if max_success_batch > 0:
        print(f"建议生产 batch size: {max_success_batch // 2}  (留 50% 显存余量)")
    print("=" * 70)

    return max_success_batch


def loop_generate(batch_size: int = 64, config_path: str = "", device: str = "cuda:0"):
    """
    死循环持续生成样本，测试长时间运行稳定性。
    Ctrl+C 停止。

    Usage:
        python util/test_batch_size.py --loop --batch 128
    """
    if not config_path:
        config_path = str(ECGTWIN_ROOT / "config" / "DiT_ECGTwin.yaml")

    print("=" * 70)
    print(f"ECGTwin 持续生成模式 | batch_size={batch_size}")
    print("Ctrl+C 停止")
    print("=" * 70)

    # 加载模型
    print("加载模型...")
    noise_predictor, scheduler, ibe_model, decoder, tokenizer, embedding_model = \
        load_all_models(config_path, device)

    # 准备条件（只需做一次）
    print("准备条件...")
    text_embed, pat_info, base_vector = prepare_conditions(
        ibe_model, tokenizer, embedding_model, batch_size, device
    )

    total_samples = 0
    round_num = 0
    start_all = time.time()

    print()
    print(f"{'轮次':>6} | {'本轮耗时':>8} | {'累计样本':>8} | {'吞吐量':>12} | {'已用显存':>10} | {'峰值显存':>10}")
    print("-" * 78)

    try:
        while True:
            round_num += 1
            torch.cuda.reset_peak_memory_stats()

            start = time.time()
            ecg = run_full_generation(
                noise_predictor, scheduler, decoder, batch_size, device,
                text_embed, pat_info, base_vector
            )
            elapsed = time.time() - start

            total_samples += batch_size
            total_elapsed = time.time() - start_all
            throughput = total_samples / total_elapsed

            used = torch.cuda.memory_allocated() / 1024 / 1024
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024

            print(f"{round_num:>6} | {elapsed:>6.1f}s | {total_samples:>8} | {throughput:>8.2f} s/条 | {used:>8.0f} MB | {peak:>8.0f} MB")

            del ecg

    except KeyboardInterrupt:
        total_elapsed = time.time() - start_all
        print()
        print("=" * 70)
        print(f"已停止 | 共 {round_num} 轮 | {total_samples} 条样本 | 总耗时 {total_elapsed:.1f}s")
        if total_samples > 0:
            print(f"平均吞吐: {total_elapsed / total_samples:.3f} s/条")
        print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ECGTwin batch size 测试")
    parser.add_argument("--loop", action="store_true", help="死循环持续生成模式")
    parser.add_argument("--batch", type=int, default=64, help="loop 模式的 batch size")
    args = parser.parse_args()

    if args.loop:
        loop_generate(batch_size=args.batch)
    else:
        best = main()
        print(f"\n提示: 用 --loop --batch {best} 进入持续生成模式")
