"""
ECGTwin 模型加载与采样工具

提供 ECGTwin diffusion 模型的加载、条件编码、采样等功能
支持可微分采样用于对抗攻击
"""

import sys
import torch
import yaml
from typing import Dict, Any, Tuple
from pathlib import Path

# ——— TensorFlow segfault workaround（两步法）———
# 步骤1：先阻止 TF 初始化，让 transformers 认为 TF 不可用
sys.modules['tensorflow'] = None  # type: ignore

# 添加 ECGTwin 模块路径
ECGTWIN_ROOT = Path(__file__).parent.parent / "model" / "ECGTwin"
sys.path.insert(0, str(ECGTWIN_ROOT))

from diffusers import DDPMScheduler
from transformers import AutoModel, AutoTokenizer

# 步骤2：transformers 已 import，删掉 None 条目让 einops 正常工作
if sys.modules.get('tensorflow') is None:
    del sys.modules['tensorflow']

from module.IBExtractor import IBExtractor
from module.vae_model import VAE_Decoder, VAE_Encoder
from utils.model_utils import build_noise_predictor
from utils.data_utils import process_pat_info, get_text_embedding, sex_transform


class ECGTwinWrapper:
    """
    ECGTwin 模型的封装类，用于对抗攻击
    """
    
    def __init__(
        self,
        config_path: str = None,
        device: str = "cuda:0",
        load_encoder: bool = False,
        load_text_model: bool = False,  # 新增：是否加载 text embedding 模型
    ):
        """
        初始化 ECGTwin 模型
        
        Args:
            config_path: 配置文件路径，默认使用 DiT_ECGTwin.yaml
            device: 设备
            load_encoder: 是否加载 VAE Encoder（用于编码真实 ECG）
            load_text_model: 是否加载 text embedding 模型（nomic）
                            如果为 False，需要使用预计算的 text_embed
        """
        if config_path is None:
            config_path = ECGTWIN_ROOT / "config" / "DiT_ECGTwin.yaml"
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device)
        self.h_ = self.config["hyper_para"]
        self.settings = self.config.get("inference_setting", {})
        self.model_type = self.config["meta"]["model_type"]
        self.mix = self.config["meta"]["mix"]
        
        # 加载模型
        self._load_models(load_encoder, load_text_model)
        
    def _load_models(self, load_encoder: bool, load_text_model: bool):
        """加载所有必要的模型"""
        
        # 1. Noise Predictor (DiT/UNet)
        n_channels = 4
        self.noise_predictor = build_noise_predictor(
            self.model_type, n_channels, self.h_
        )
        noise_predictor_path = ECGTWIN_ROOT / self.settings.get(
            "noise_predictor_path", "checkpoints/ECGTwin_DiT.pth"
        )
        self.noise_predictor.load_state_dict(
            torch.load(noise_predictor_path, map_location="cpu")
        )
        self.noise_predictor.to(self.device)
        self.noise_predictor.eval()
        
        # 2. DDPM Scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.h_["ddpm"]["num_train_steps"],
            beta_start=self.h_["ddpm"]["beta_start"],
            beta_end=self.h_["ddpm"]["beta_end"],
        )
        self.inference_timesteps = self.settings.get("inference_timestep", 1000)
        
        # 3. IBExtractor
        self.ibe_model = IBExtractor(
            embed_dim=self.h_["ibe"]["embed_dim"],
            num_heads=self.h_["ibe"]["num_heads"],
            ff_hidden_size=self.h_["ibe"]["ff_hidden_size"],
            num_layers=self.h_["ibe"]["num_layers"],
            text_embed_dim=self.h_["ibe"]["text_embed_dim"],
            patient_info_size=self.h_["ibe"]["patient_info_size"],
        )
        ibe_path = ECGTWIN_ROOT / self.config["dependencies"]["ibe_path"]
        self.ibe_model.load_state_dict(torch.load(ibe_path, map_location="cpu"))
        self.ibe_model.to(self.device)
        self.ibe_model.eval()
        
        # 4. VAE Decoder
        self.decoder = VAE_Decoder()
        vae_path = ECGTWIN_ROOT / self.config["dependencies"]["vae_path"]
        vae_checkpoint = torch.load(vae_path, map_location="cpu")
        self.decoder.load_state_dict(vae_checkpoint["decoder"])
        self.decoder.to(self.device)
        self.decoder.eval()
        
        # 5. VAE Encoder (可选)
        self.encoder = None
        if load_encoder:
            self.encoder = VAE_Encoder()
            self.encoder.load_state_dict(vae_checkpoint["encoder"])
            self.encoder.to(self.device)
            self.encoder.eval()
        
        # 6. Text Embedding Model (可选)
        self.tokenizer = None
        self.embedding_model = None
        if load_text_model:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.embedding_model = AutoModel.from_pretrained(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True,
                safe_serialization=True,
            )
            self.embedding_model.to(self.device)
            self.embedding_model.eval()
    
    def encode_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        将 ECG 信号编码为 latent
        
        Args:
            ecg: ECG 信号，shape (B, L, 12) 或 (B, 12, L)
            
        Returns:
            latent: VAE latent，shape (B, 4, 128)
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded. Set load_encoder=True when initializing.")
        
        # 确保格式为 (B, L, 12)
        if ecg.shape[-1] != 12:
            ecg = ecg.transpose(-1, -2)
        
        ecg = ecg.to(self.device)
        
        with torch.no_grad():
            latent, _, _ = self.encoder(ecg)
        
        return latent
    
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        将 latent 解码为 ECG 信号
        
        Args:
            latent: VAE latent，shape (B, 4, 128)
            
        Returns:
            ecg: ECG 信号，shape (B, 1024, 12)
        """
        latent = latent.to(self.device)
        ecg = self.decoder(latent)
        return ecg
    
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        获取文本嵌入
        
        Args:
            text: 诊断文本，多个诊断用 | 分隔
            
        Returns:
            text_embed: 文本嵌入，shape (num_reports, 768)
        """
        if self.embedding_model is None or self.tokenizer is None:
            raise RuntimeError(
                "Text embedding model not loaded. "
                "Set load_text_model=True when initializing, "
                "or use prepare_conditions with precomputed text_embed in ref_label."
            )
        text_embed = get_text_embedding(
            text=text,
            tokenizer=self.tokenizer,
            embedding_model=self.embedding_model,
            mix=self.mix,
        )
        return text_embed
    
    def prepare_conditions(
        self,
        ref_latent: torch.Tensor,
        ref_label: Dict[str, Any],
        batch_size: int = 1,
        target_text: str = None,
        target_hr: float = None,
        target_age: int = None,
        target_text_embed: torch.Tensor = None,  # 新增：预计算的目标 text embedding
    ) -> Dict[str, torch.Tensor]:
        """
        准备生成所需的条件向量
        
        Args:
            ref_latent: 参考 ECG 的 latent，shape (4, 128) 或 (B, 4, 128)
            ref_label: 参考 ECG 的标签信息
            batch_size: 批次大小
            target_text: 目标诊断文本（如果为 None，使用 ref_label 中的文本）
            target_hr: 目标心率
            target_age: 目标年龄
            target_text_embed: 预计算的目标文本嵌入（如果提供，忽略 target_text）
            
        Returns:
            conditions: 包含所有条件向量的字典
        """
        # 处理 ref_latent
        if ref_latent.dim() == 2:
            ref_latent = ref_latent.unsqueeze(0)
        ref_latent = ref_latent.repeat(batch_size, 1, 1).to(self.device)
        
        # (B, 4, 128) -> (B, 128, 4) for IBE
        ref_latent_ibe = ref_latent.transpose(2, 1)
        
        # 准备患者信息（参考）
        pat_info_ref = process_pat_info(
            normalize=True,
            hr=torch.tensor([ref_label["hr"]]),
            age=torch.tensor([ref_label["age"]]),
            sex=torch.tensor([sex_transform(ref_label["sex"])]),
        )
        pat_info_ref = pat_info_ref.repeat(batch_size, 1).to(self.device)
        
        # 获取参考文本嵌入（优先使用预计算的）
        if "text_embed" in ref_label and torch.is_tensor(ref_label["text_embed"]):
            text_embed_ref = ref_label["text_embed"].to(self.device)
            if text_embed_ref.dim() == 2:
                text_embed_ref = text_embed_ref.unsqueeze(0)
            text_embed_ref = text_embed_ref.repeat(batch_size, 1, 1)
        else:
            text_embed_ref = self.get_text_embedding(ref_label["text"])
            text_embed_ref = text_embed_ref.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        
        # 提取 base_vector
        with torch.no_grad():
            base_vector = self.ibe_model.extract_features(
                ref_latent_ibe, text_embed_ref, None, pat_info_ref, reduce=True
            )
        
        # 准备目标条件
        target_hr = target_hr if target_hr is not None else ref_label["hr"]
        target_age = target_age if target_age is not None else ref_label["age"]
        
        # 获取目标文本嵌入（优先使用预计算的）
        if target_text_embed is not None:
            text_embed_tar = target_text_embed.to(self.device)
            if text_embed_tar.dim() == 2:
                text_embed_tar = text_embed_tar.unsqueeze(0)
            text_embed_tar = text_embed_tar.repeat(batch_size, 1, 1)
        elif target_text is not None:
            text_embed_tar = self.get_text_embedding(target_text)
            text_embed_tar = text_embed_tar.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        else:
            # 使用参考的 text_embed
            text_embed_tar = text_embed_ref
        
        pat_info_tar = process_pat_info(
            normalize=True,
            add_token=False,
            hr=torch.tensor([target_hr]),
            age=torch.tensor([target_age]),
            sex=torch.tensor([sex_transform(ref_label["sex"])]),
        )
        pat_info_tar = pat_info_tar.repeat(batch_size, 1).to(self.device)
        
        return {
            "base_vector": base_vector,
            "text_embed": text_embed_tar,
            "text_embed_mask": None,
            "pat_info": pat_info_tar,
            "ref_latent": ref_latent,
        }
    
    @torch.no_grad()
    def ddpm_sample(
        self,
        conditions: Dict[str, torch.Tensor],
        batch_size: int = 1,
        num_inference_steps: int = None,
        x_init: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        标准 DDPM 采样（无梯度）
        
        Args:
            conditions: 条件向量字典
            batch_size: 批次大小
            num_inference_steps: 推理步数
            x_init: 初始噪声（如果为 None，随机采样）
            
        Returns:
            latent: 生成的 latent，shape (B, 4, 128)
        """
        num_inference_steps = num_inference_steps or self.inference_timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        if x_init is None:
            x_t = torch.randn(batch_size, 4, 128, device=self.device)
        else:
            x_t = x_init.to(self.device)
        
        for t in self.scheduler.timesteps:
            t_batch = t * torch.ones(batch_size, dtype=torch.long, device=self.device)
            
            noise_pred = self.noise_predictor(
                x_t, t_batch,
                conditions["text_embed"],
                conditions["text_embed_mask"],
                conditions["pat_info"],
                conditions["base_vector"],
            )
            
            x_t = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=x_t,
            )["prev_sample"]
        
        return x_t
    
    def ddpm_sample_with_grad(
        self,
        conditions: Dict[str, torch.Tensor],
        batch_size: int = 1,
        num_inference_steps: int = None,
        x_init: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        可微分的 DDPM 采样（保留梯度）
        
        Args:
            conditions: 条件向量字典
            batch_size: 批次大小
            num_inference_steps: 推理步数
            x_init: 初始噪声（需要 requires_grad=True）
            
        Returns:
            latent: 生成的 latent，shape (B, 4, 128)
        """
        num_inference_steps = num_inference_steps or self.inference_timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        if x_init is None:
            x_t = torch.randn(batch_size, 4, 128, device=self.device)
        else:
            x_t = x_init.to(self.device)
        
        for t in self.scheduler.timesteps:
            t_batch = t * torch.ones(batch_size, dtype=torch.long, device=self.device)
            
            noise_pred = self.noise_predictor(
                x_t, t_batch,
                conditions["text_embed"],
                conditions["text_embed_mask"],
                conditions["pat_info"],
                conditions["base_vector"],
            )
            
            x_t = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=x_t,
            )["prev_sample"]
        
        return x_t
    
    def generate_ecg(
        self,
        ref_data_path: str = None,
        ref_latent: torch.Tensor = None,
        ref_label: Dict[str, Any] = None,
        target_text: str = None,
        target_hr: float = None,
        target_age: int = None,
        batch_size: int = 1,
        num_inference_steps: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成 ECG 信号
        
        Args:
            ref_data_path: 参考数据路径（.pt 文件）
            ref_latent: 参考 latent（如果提供了 ref_data_path 则忽略）
            ref_label: 参考标签
            target_text: 目标诊断文本
            target_hr: 目标心率
            target_age: 目标年龄
            batch_size: 批次大小
            num_inference_steps: 推理步数
            
        Returns:
            ecg: 生成的 ECG，shape (B, 1024, 12)
            latent: 生成的 latent，shape (B, 4, 128)
        """
        # 加载参考数据
        if ref_data_path is not None:
            ref_data = torch.load(ref_data_path, map_location="cpu")
            ref_latent = ref_data["data"]
            ref_label = ref_data["label"]
        
        if ref_latent is None or ref_label is None:
            raise ValueError("Must provide either ref_data_path or (ref_latent, ref_label)")
        
        # 准备条件
        conditions = self.prepare_conditions(
            ref_latent=ref_latent,
            ref_label=ref_label,
            batch_size=batch_size,
            target_text=target_text,
            target_hr=target_hr,
            target_age=target_age,
        )
        
        # 采样
        latent = self.ddpm_sample(
            conditions=conditions,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
        )
        
        # 解码
        ecg = self.decode_latent(latent)
        
        return ecg, latent


def load_ecgtwin(
    config_path: str = None,
    device: str = "cuda:0",
    load_encoder: bool = False,
    load_text_model: bool = False,
) -> ECGTwinWrapper:
    """
    加载 ECGTwin 模型的便捷函数
    
    Args:
        config_path: 配置文件路径
        device: 设备
        load_encoder: 是否加载编码器
        load_text_model: 是否加载 text embedding 模型
        
    Returns:
        ECGTwinWrapper 实例
    """
    return ECGTwinWrapper(
        config_path=config_path,
        device=device,
        load_encoder=load_encoder,
        load_text_model=load_text_model,
    )


if __name__ == "__main__":
    # 测试代码（使用预计算的 text_embed，不需要加载 nomic 模型）
    print("Loading ECGTwin (without text model)...")
    model = load_ecgtwin(device="cuda:0", load_text_model=False)
    
    print("Loading reference data...")
    import torch
    ref_data_path = str(ECGTWIN_ROOT / "data" / "prepared_input" / "normal_1.pt")
    ref_data = torch.load(ref_data_path, map_location="cpu")
    ref_latent = ref_data["data"]
    ref_label = ref_data["label"]
    
    print("Preparing conditions...")
    conditions = model.prepare_conditions(
        ref_latent=ref_latent,
        ref_label=ref_label,
        batch_size=2,
    )
    
    print("Generating ECG...")
    latent = model.ddpm_sample(
        conditions=conditions,
        batch_size=2,
        num_inference_steps=50,  # 快速测试
    )
    ecg = model.decode_latent(latent)
    
    print(f"Generated ECG shape: {ecg.shape}")
    print(f"Generated latent shape: {latent.shape}")
