from __future__ import annotations

import json
import sys
import time
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ECGTWIN_ROOT = PROJECT_ROOT / "model" / "ECGTwin"
DEEPECG_NOTEBOOKS = PROJECT_ROOT / "model" / "DeepECG" / "notebooks"
sys.path.insert(0, str(ECGTWIN_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
if DEEPECG_NOTEBOOKS.exists():
    sys.path.insert(0, str(DEEPECG_NOTEBOOKS))

from module.IBExtractor import IBExtractor  # noqa: E402
from module.vae_model import VAE_Decoder  # noqa: E402
from utils.data_utils import _pad_text_embed, process_pat_info  # noqa: E402
from utils.model_utils import build_noise_predictor  # noqa: E402

from ecg_adv_gen.generation.prompt_tokens import load_text_embed_from_prompt_bank  # noqa: E402
from methods.ecgtwin_gen.prompt_token.model import CenterClassPromptTokenBank  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402

try:
    from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
except Exception:  # pragma: no cover - only needed when optional aux losses are enabled.
    EfficientNet1DV2 = None


PN2021_STYLE_CENTERS = [
    "chapman_shaoxing",
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "ningbo",
    "ptb",
    "st_petersburg_incart",
]


def _sex_to_binary(value) -> float:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).upper()
    return 0.0 if text.startswith("F") else 1.0


def _load_text_embed(prompt_bank: Dict[str, object], primary_snomed, primary_class: str) -> torch.Tensor:
    return load_text_embed_from_prompt_bank(prompt_bank, primary_snomed, primary_class)


def _actual_report_text_embed(cache: Dict[str, object], idx: int) -> Optional[torch.Tensor]:
    """Return the per-record clinical-report embedding when present."""
    embeds = cache.get("text_embed")
    if embeds is None:
        return None
    try:
        emb = embeds[idx]
    except Exception:
        return None
    if torch.is_tensor(emb) and emb.dim() == 2 and emb.shape[-1] == 768:
        return emb.detach().float()
    return None


class CenterPromptTokenCacheDataset(Dataset):
    """Selected K PN2021 records from one center's ECGTwin raw-mV latent cache."""

    def __init__(
        self,
        cache_root: str,
        center: str,
        prompt_bank: Dict[str, object],
        center_idx: int,
        k: int = 500,
        seed: int = 42,
        selected_only: bool = True,
        ref_text_mode: str = "class_fallback",
    ):
        self.cache_root = Path(cache_root)
        self.center = center
        self.center_idx = center_idx
        self.prompt_bank = prompt_bank
        self.ref_text_mode = str(ref_text_mode)
        if self.ref_text_mode not in {"class_fallback", "actual_report", "normal"}:
            raise ValueError(
                "ref_text_mode must be class_fallback|actual_report|normal, "
                f"got {ref_text_mode!r}"
            )
        self.full_path = self.cache_root / "center_full_latents" / f"{center}.pt"
        self.selection_path = (
            self.cache_root / "ref_selection" / f"{center}_k{k}_seed{seed}.json"
        )
        if not self.full_path.exists():
            raise FileNotFoundError(self.full_path)
        self.cache = torch.load(self.full_path, map_location="cpu", weights_only=False)
        if selected_only:
            if not self.selection_path.exists():
                raise FileNotFoundError(self.selection_path)
            with open(self.selection_path) as f:
                selection = json.load(f)
            self.indices = list(selection["selected_indices_in_full_cache"])
            self.selection = selection
        else:
            self.indices = list(range(len(self.cache["record_ids"])))
            self.selection = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, object]:
        idx = self.indices[item]
        primary_class = self.cache["primary_class"][idx]
        primary_snomed = self.cache["primary_snomed"][idx]
        target_text_embed = _load_text_embed(self.prompt_bank, primary_snomed, primary_class)
        if self.ref_text_mode == "actual_report":
            ref_text_embed = _actual_report_text_embed(self.cache, idx)
            if ref_text_embed is None:
                ref_text_embed = target_text_embed
        elif self.ref_text_mode == "normal":
            ref_text_embed = self.prompt_bank["by_class"]["NORM"].detach().float()
        else:
            ref_text_embed = target_text_embed
        return {
            "latent": self.cache["latents"][idx].float(),
            "ref_text_embed": ref_text_embed,
            "target_text_embed": target_text_embed,
            "center_idx": int(self.center_idx),
            "class_idx": int(self.cache["primary_class_idx"][idx]),
            "age": float(self.cache["age"][idx]),
            "hr": float(self.cache["hr"][idx]),
            "sex": _sex_to_binary(self.cache["sex"][idx]),
            "record_id": self.cache["record_ids"][idx],
            "center": self.center,
            "primary_class": primary_class,
            "primary_snomed": primary_snomed,
            "multi_hot": self.cache["super5_multi_hot"][idx].float(),
        }


def prompt_token_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    ref_text_embed, ref_text_mask = _pad_text_embed([b["ref_text_embed"] for b in batch])
    target_text_embed, target_text_mask = _pad_text_embed([b["target_text_embed"] for b in batch])
    return {
        "latent": torch.stack([b["latent"] for b in batch], dim=0),
        "ref_text_embed": ref_text_embed,
        "ref_text_embed_mask": ref_text_mask,
        "target_text_embed": target_text_embed,
        "target_text_embed_mask": target_text_mask,
        "center_idx": torch.tensor([b["center_idx"] for b in batch], dtype=torch.long),
        "class_idx": torch.tensor([b["class_idx"] for b in batch], dtype=torch.long),
        "age": torch.tensor([b["age"] for b in batch], dtype=torch.float32),
        "hr": torch.tensor([b["hr"] for b in batch], dtype=torch.float32),
        "sex": torch.tensor([b["sex"] for b in batch], dtype=torch.float32),
        "multi_hot": torch.stack([b["multi_hot"] for b in batch], dim=0),
        "record_id": [b["record_id"] for b in batch],
        "center": [b["center"] for b in batch],
        "primary_class": [b["primary_class"] for b in batch],
        "primary_snomed": [b["primary_snomed"] for b in batch],
    }


def _autocast_context(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype != "none"
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)


class ECGTwinPromptTokenTrainer:
    def __init__(
        self,
        centers: Iterable[str],
        cache_root: str,
        prompt_bank_path: str,
        ecgtwin_config_path: str,
        device: str = "cuda:0",
        k: int = 500,
        seed: int = 42,
        batch_size: int = 16,
        num_workers: int = 4,
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-4,
        reg_init_weight: float = 1.0e-3,
        grad_clip: float = 1.0,
        amp_dtype: str = "bf16",
        init_noise_std: float = 0.01,
        token_repeat: int = 1,
        sample_strategy: str = "shuffle",
        n_token_vectors: int = 1,
        token_mode: str = "direct",
        center_vectors: int = 1,
        class_vectors: int = 1,
        residual_vectors: int = 1,
        residual_init_std: float = 0.0,
        token_orth_weight: float = 0.0,
        ref_text_mode: str = "class_fallback",
        style_ckpt: Optional[str] = None,
        style_loss_weight: float = 0.0,
        style_loss_mode: str = "ce",
        style_target_prob: float = 0.65,
        style_crop_len: int = 1000,
        semantic_ckpt: Optional[str] = None,
        semantic_loss_weight: float = 0.0,
        semantic_crop_len: int = 250,
        contrast_recon_weight: float = 0.0,
        contrast_recon_margin: float = 0.0,
        contrast_style_delta_weight: float = 0.0,
        contrast_style_delta_margin: float = 0.05,
        contrast_semantic_delta_weight: float = 0.0,
        contrast_semantic_delta_margin: float = 0.02,
        feature_loss_weight: float = 0.0,
        feature_crop_len: int = 1000,
        contrast_feature_weight: float = 0.0,
        contrast_feature_margin: float = 0.0,
        feature_mmd_weight: float = 0.0,
        contrast_feature_mmd_weight: float = 0.0,
        feature_mmd_margin: float = 0.0,
        feature_mmd_sigma: float = 1.0,
        aux_timestep_max: int = 0,
        aux_amp_clamp: float = 6.0,
    ):
        self.centers = list(centers)
        self.center_to_idx = {c: i for i, c in enumerate(self.centers)}
        self.cache_root = cache_root
        self.k = k
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.reg_init_weight = reg_init_weight
        self.grad_clip = grad_clip
        self.amp_dtype = amp_dtype
        self.token_repeat = int(token_repeat)
        if self.token_repeat < 1:
            raise ValueError(f"token_repeat must be >= 1, got {token_repeat}")
        self.sample_strategy = sample_strategy
        self.n_token_vectors = int(n_token_vectors)
        self.token_mode = str(token_mode)
        self.center_vectors = int(center_vectors)
        self.class_vectors = int(class_vectors)
        self.residual_vectors = int(residual_vectors)
        self.residual_init_std = float(residual_init_std)
        self.token_orth_weight = float(token_orth_weight)
        self.ref_text_mode = str(ref_text_mode)
        self.style_ckpt = str(style_ckpt) if style_ckpt else None
        self.style_loss_weight = float(style_loss_weight)
        self.style_loss_mode = str(style_loss_mode)
        if self.style_loss_mode not in {"ce", "target_prob"}:
            raise ValueError(f"unknown style_loss_mode={style_loss_mode!r}")
        self.style_target_prob = float(style_target_prob)
        self.style_crop_len = int(style_crop_len)
        self.semantic_ckpt = str(semantic_ckpt) if semantic_ckpt else None
        self.semantic_loss_weight = float(semantic_loss_weight)
        self.semantic_crop_len = int(semantic_crop_len)
        self.contrast_recon_weight = float(contrast_recon_weight)
        self.contrast_recon_margin = float(contrast_recon_margin)
        self.contrast_style_delta_weight = float(contrast_style_delta_weight)
        self.contrast_style_delta_margin = float(contrast_style_delta_margin)
        self.contrast_semantic_delta_weight = float(contrast_semantic_delta_weight)
        self.contrast_semantic_delta_margin = float(contrast_semantic_delta_margin)
        self.feature_loss_weight = float(feature_loss_weight)
        self.feature_crop_len = int(feature_crop_len)
        self.contrast_feature_weight = float(contrast_feature_weight)
        self.contrast_feature_margin = float(contrast_feature_margin)
        self.feature_mmd_weight = float(feature_mmd_weight)
        self.contrast_feature_mmd_weight = float(contrast_feature_mmd_weight)
        self.feature_mmd_margin = float(feature_mmd_margin)
        self.feature_mmd_sigma = float(feature_mmd_sigma)
        self.aux_timestep_max = int(aux_timestep_max)
        self.aux_amp_clamp = float(aux_amp_clamp)
        if self.ref_text_mode not in {"class_fallback", "actual_report", "normal"}:
            raise ValueError(
                "ref_text_mode must be class_fallback|actual_report|normal, "
                f"got {ref_text_mode!r}"
            )
        if self.sample_strategy not in {"shuffle", "class_balanced", "center_class_balanced"}:
            raise ValueError(
                "sample_strategy must be shuffle|class_balanced|center_class_balanced, "
                f"got {sample_strategy!r}"
            )
        self.device = torch.device(device)

        with open(ecgtwin_config_path) as f:
            self.ecgtwin_config = yaml.safe_load(f)
        self.prompt_bank = torch.load(prompt_bank_path, map_location="cpu", weights_only=False)
        self.prompt_bank_path = prompt_bank_path

        self._load_frozen_ecgtwin()
        self._load_optional_aux_models()
        init_by_class = {
            cls: self.prompt_bank["by_class"][cls].float().mean(dim=0)
            for cls in CLASS_NAMES_SUPER5
        }
        self.token_bank = CenterClassPromptTokenBank(
            centers=self.centers,
            class_names=CLASS_NAMES_SUPER5,
            dim=768,
            init_by_class=init_by_class,
            init_noise_std=init_noise_std,
            n_token_vectors=self.n_token_vectors,
            token_mode=self.token_mode,
            center_vectors=self.center_vectors,
            class_vectors=self.class_vectors,
            residual_vectors=self.residual_vectors,
            residual_init_std=self.residual_init_std,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.token_bank.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _load_frozen_ecgtwin(self) -> None:
        h_ = self.ecgtwin_config["hyper_para"]
        model_type = self.ecgtwin_config["meta"]["model_type"]
        self.noise_predictor = build_noise_predictor(model_type, 4, h_)
        np_path = ECGTWIN_ROOT / self.ecgtwin_config["inference_setting"]["noise_predictor_path"]
        self.noise_predictor.load_state_dict(torch.load(np_path, map_location="cpu"))
        self.noise_predictor.to(self.device).eval()
        for p in self.noise_predictor.parameters():
            p.requires_grad = False

        self.ibe_model = IBExtractor(
            embed_dim=h_["ibe"]["embed_dim"],
            num_heads=h_["ibe"]["num_heads"],
            ff_hidden_size=h_["ibe"]["ff_hidden_size"],
            num_layers=h_["ibe"]["num_layers"],
            text_embed_dim=h_["ibe"]["text_embed_dim"],
            patient_info_size=h_["ibe"]["patient_info_size"],
        )
        ibe_path = ECGTWIN_ROOT / self.ecgtwin_config["dependencies"]["ibe_path"]
        self.ibe_model.load_state_dict(torch.load(ibe_path, map_location="cpu"))
        self.ibe_model.to(self.device).eval()
        for p in self.ibe_model.parameters():
            p.requires_grad = False

        self.scheduler = DDPMScheduler(
            num_train_timesteps=h_["ddpm"]["num_train_steps"],
            beta_start=h_["ddpm"]["beta_start"],
            beta_end=h_["ddpm"]["beta_end"],
        )

        self.vae_decoder = VAE_Decoder()
        vae_path = ECGTWIN_ROOT / self.ecgtwin_config["dependencies"]["vae_path"]
        vae_checkpoint = torch.load(vae_path, map_location="cpu")
        self.vae_decoder.load_state_dict(vae_checkpoint["decoder"])
        self.vae_decoder.to(self.device).eval()
        for p in self.vae_decoder.parameters():
            p.requires_grad = False

    @staticmethod
    def _build_efficientnet(num_classes: int) -> nn.Module:
        if EfficientNet1DV2 is None:
            raise RuntimeError(
                "EfficientNet1DV2 is unavailable; check model/DeepECG/notebooks path."
            )
        return EfficientNet1DV2(
            variant="s_v2",
            input_channels=12,
            num_classes=num_classes,
            activation="leaky_relu",
            stochastic_depth_prob=0.304,
            dropout_rate=0.0,
            use_se=True,
            norm_type="batch",
        )

    @staticmethod
    def _load_state_dict_flexible(path: str):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                value = state.get(key)
                if isinstance(value, dict):
                    return value
        return state

    def _load_optional_aux_models(self) -> None:
        self.style_model = None
        self.semantic_model = None
        self.style_target_by_center_idx = None

        if self.style_loss_weight > 0.0 or self.contrast_style_delta_weight > 0.0:
            if not self.style_ckpt:
                raise ValueError(
                    "--style_ckpt is required when style_loss_weight or "
                    "contrast_style_delta_weight is > 0"
                )
            style_centers = list(PN2021_STYLE_CENTERS)
            meta_path = Path(self.style_ckpt).parent / "train_result.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    if meta.get("center_names"):
                        style_centers = list(meta["center_names"])
                except Exception:
                    pass
            style_to_idx = {c: i for i, c in enumerate(style_centers)}
            missing = [c for c in self.centers if c not in style_to_idx]
            if missing:
                raise ValueError(
                    "style classifier cannot supervise centers not present in "
                    f"its label set: {missing}; style_centers={style_centers}"
                )
            self.style_target_by_center_idx = torch.tensor(
                [style_to_idx[c] for c in self.centers],
                dtype=torch.long,
                device=self.device,
            )
            self.style_model = self._build_efficientnet(num_classes=len(style_centers))
            self.style_model.load_state_dict(self._load_state_dict_flexible(self.style_ckpt))
            self.style_model.to(self.device).eval()
            for p in self.style_model.parameters():
                p.requires_grad = False
            print(
                f"[aux] style loss enabled weight={self.style_loss_weight} "
                f"ckpt={self.style_ckpt} centers={style_centers}",
                flush=True,
            )

        if (
            self.semantic_loss_weight > 0.0
            or self.contrast_semantic_delta_weight > 0.0
            or self.feature_loss_weight > 0.0
            or self.contrast_feature_weight > 0.0
            or self.feature_mmd_weight > 0.0
            or self.contrast_feature_mmd_weight > 0.0
        ):
            if not self.semantic_ckpt:
                raise ValueError(
                    "--semantic_ckpt is required when semantic/feature losses "
                    "or their contrastive variants are > 0"
                )
            self.semantic_model = self._build_efficientnet(num_classes=len(CLASS_NAMES_SUPER5))
            self.semantic_model.load_state_dict(self._load_state_dict_flexible(self.semantic_ckpt))
            self.semantic_model.to(self.device).eval()
            for p in self.semantic_model.parameters():
                p.requires_grad = False
            print(
                f"[aux] semantic loss enabled weight={self.semantic_loss_weight} "
                f"ckpt={self.semantic_ckpt}",
                flush=True,
            )


    def _sampler_for(self, datasets: List[CenterPromptTokenCacheDataset]) -> Optional[WeightedRandomSampler]:
        if self.sample_strategy == "shuffle":
            return None

        keys = []
        for ds in datasets:
            for cache_idx in ds.indices:
                cls_idx = int(ds.cache["primary_class_idx"][cache_idx])
                if self.sample_strategy == "center_class_balanced":
                    key = (int(ds.center_idx), cls_idx)
                else:
                    key = cls_idx
                keys.append(key)
        counts = Counter(keys)
        weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
        print(
            f"[data] sample_strategy={self.sample_strategy} cells={len(counts)} "
            f"min_count={min(counts.values())} max_count={max(counts.values())}",
            flush=True,
        )
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    def create_dataloader(self) -> DataLoader:
        datasets = [
            CenterPromptTokenCacheDataset(
                cache_root=self.cache_root,
                center=center,
                prompt_bank=self.prompt_bank,
                center_idx=self.center_to_idx[center],
                k=self.k,
                seed=self.seed,
                ref_text_mode=self.ref_text_mode,
            )
            for center in self.centers
        ]
        dataset = ConcatDataset(datasets)
        sampler = self._sampler_for(datasets)
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": sampler is None,
            "sampler": sampler,
            "collate_fn": prompt_token_collate_fn,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": self.device.type == "cuda",
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        print(
            f"[data] centers={self.centers} records={len(dataset)} "
            f"batch={self.batch_size} workers={self.num_workers} "
            f"strategy={self.sample_strategy} token_repeat={self.token_repeat}",
            flush=True,
        )
        return DataLoader(dataset, **kwargs)

    def _orthogonality_loss(self, token_seq: torch.Tensor) -> torch.Tensor:
        if token_seq.shape[1] <= 1:
            return token_seq.sum() * 0.0
        x = F.normalize(token_seq.float(), dim=-1)
        gram = torch.matmul(x, x.transpose(1, 2))
        eye = torch.eye(token_seq.shape[1], device=token_seq.device, dtype=torch.bool)
        off_diag = gram[:, ~eye].view(token_seq.shape[0], token_seq.shape[1], token_seq.shape[1] - 1)
        return off_diag.pow(2).mean()

    def _predict_x0_from_noise(
        self,
        noisy_latent: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.scheduler.alphas_cumprod.to(noisy_latent.device)[timesteps]
        alpha = alpha.view(-1, 1, 1).float()
        beta = (1.0 - alpha).clamp(min=1e-8)
        return (noisy_latent.float() - beta.sqrt() * noise_pred.float()) / alpha.sqrt()

    def _decode_latent_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode ECGTwin latent while avoiding VAE_Decoder.forward in-place scaling."""
        x = latent / 0.18215
        for module in self.vae_decoder:
            x = module(x)
        return x.transpose(1, 2)

    def _latent_to_classifier_input(self, latent: torch.Tensor, crop_len: int) -> torch.Tensor:
        ecg_tc = self._decode_latent_differentiable(latent)
        ecg_ct = ecg_tc.transpose(1, 2)
        ecg_ct = ecg_ct[:, ECGTWIN_TO_PTBXL_INDICES, :]
        ecg_ct = torch.clamp(ecg_ct, min=-self.aux_amp_clamp, max=self.aux_amp_clamp)
        ecg_ct = F.interpolate(ecg_ct, size=1000, mode="linear", align_corners=True)
        bsz = ecg_ct.shape[0]
        flat = ecg_ct.reshape(bsz, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp(min=1e-8)
        ecg_ct = (ecg_ct - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        if crop_len != 1000:
            if crop_len > 1000:
                ecg_ct = F.pad(ecg_ct, ((crop_len - 1000) // 2, crop_len - 1000 - (crop_len - 1000) // 2))
            else:
                start = (1000 - crop_len) // 2
                ecg_ct = ecg_ct[..., start:start + crop_len]
        return ecg_ct

    def _style_loss(self, pred_x0: torch.Tensor, center_idx: torch.Tensor) -> torch.Tensor:
        if self.style_model is None:
            return pred_x0.sum() * 0.0
        ecg_ct = self._latent_to_classifier_input(pred_x0, crop_len=self.style_crop_len)
        logits = self.style_model(ecg_ct)
        targets = self.style_target_by_center_idx[center_idx]
        logits_f = logits.float()
        if self.style_loss_mode == "target_prob":
            probs = F.softmax(logits_f, dim=1)
            p_target = probs.gather(1, targets.view(-1, 1)).squeeze(1)
            target = torch.full_like(p_target, self.style_target_prob)
            return F.mse_loss(p_target, target)
        return F.cross_entropy(logits_f, targets)

    def _style_target_prob(self, pred_x0: torch.Tensor, center_idx: torch.Tensor) -> torch.Tensor:
        if self.style_model is None:
            return pred_x0.new_zeros((pred_x0.shape[0],), dtype=torch.float32)
        ecg_ct = self._latent_to_classifier_input(pred_x0, crop_len=self.style_crop_len)
        logits = self.style_model(ecg_ct).float()
        targets = self.style_target_by_center_idx[center_idx]
        probs = F.softmax(logits, dim=1)
        return probs.gather(1, targets.view(-1, 1)).squeeze(1)

    def _semantic_loss(self, pred_x0: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.semantic_model is None:
            return pred_x0.sum() * 0.0
        ecg_ct = self._latent_to_classifier_input(pred_x0, crop_len=self.semantic_crop_len)
        logits = self.semantic_model(ecg_ct)
        mask = labels.ge(0.0)
        if not bool(mask.any()):
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            logits.float()[mask],
            labels.to(logits.device).float()[mask],
        )

    def _semantic_primary_prob(self, pred_x0: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        if self.semantic_model is None:
            return pred_x0.new_zeros((pred_x0.shape[0],), dtype=torch.float32)
        ecg_ct = self._latent_to_classifier_input(pred_x0, crop_len=self.semantic_crop_len)
        logits = self.semantic_model(ecg_ct).float()
        probs = torch.sigmoid(logits)
        return probs.gather(1, class_idx.view(-1, 1)).squeeze(1)

    @staticmethod
    def _efficientnet_penultimate(model: nn.Module, ecg_ct: torch.Tensor) -> torch.Tensor:
        x = model.initial_conv(ecg_ct)
        x = model.features(x)
        x = model.final_conv(x)
        x = model.final_norm(x)
        x = F.adaptive_avg_pool1d(x, 1).flatten(1)
        return x

    def _semantic_feature_distance_per(
        self,
        pred_x0: torch.Tensor,
        real_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.semantic_model is None:
            return pred_x0.new_zeros((pred_x0.shape[0],), dtype=torch.float32)
        pred_feat = self._semantic_feature_tensor(pred_x0)
        with torch.no_grad():
            real_feat = self._semantic_feature_tensor(real_latent)
        return 1.0 - (pred_feat * real_feat).sum(dim=1)

    def _semantic_feature_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        if self.semantic_model is None:
            return latent.new_zeros((latent.shape[0], 1), dtype=torch.float32)
        ecg_ct = self._latent_to_classifier_input(latent, crop_len=self.feature_crop_len)
        feat = self._efficientnet_penultimate(self.semantic_model, ecg_ct)
        return F.normalize(feat.float(), dim=1)

    def _feature_mmd_loss(self, pred_feat: torch.Tensor, real_feat: torch.Tensor) -> torch.Tensor:
        if pred_feat.shape[0] <= 1 or real_feat.shape[0] <= 1:
            return pred_feat.sum() * 0.0
        sigma2 = max(self.feature_mmd_sigma * self.feature_mmd_sigma, 1.0e-6)
        dist_xx = torch.cdist(pred_feat, pred_feat, p=2).pow(2)
        dist_yy = torch.cdist(real_feat, real_feat, p=2).pow(2)
        dist_xy = torch.cdist(pred_feat, real_feat, p=2).pow(2)
        k_xx = torch.exp(-dist_xx / (2.0 * sigma2)).mean()
        k_yy = torch.exp(-dist_yy / (2.0 * sigma2)).mean()
        k_xy = torch.exp(-dist_xy / (2.0 * sigma2)).mean()
        return (k_xx + k_yy - 2.0 * k_xy).clamp_min(0.0)

    def _step(self, batch: Dict[str, object]) -> Dict[str, float]:
        latent = batch["latent"].to(self.device, non_blocking=True)
        ref_text_embed = batch["ref_text_embed"].to(self.device, non_blocking=True)
        ref_text_mask = batch["ref_text_embed_mask"].to(self.device, non_blocking=True)
        target_text_embed = batch["target_text_embed"].to(self.device, non_blocking=True)
        target_text_mask = batch["target_text_embed_mask"].to(self.device, non_blocking=True)
        center_idx = batch["center_idx"].to(self.device, non_blocking=True)
        class_idx = batch["class_idx"].to(self.device, non_blocking=True)
        hr = batch["hr"].to(self.device, non_blocking=True)
        age = batch["age"].to(self.device, non_blocking=True)
        sex = batch["sex"].to(self.device, non_blocking=True)
        multi_hot = batch["multi_hot"].to(self.device, non_blocking=True)
        bsz = latent.shape[0]

        token_seq_base = self.token_bank.token_sequence(center_idx, class_idx)
        token_init_base = self.token_bank.init_sequence_for(center_idx, class_idx)
        token_seq = token_seq_base.repeat(1, self.token_repeat, 1)
        text_aug = torch.cat([target_text_embed, token_seq], dim=1)
        mask_aug = torch.cat(
            [
                target_text_mask,
                torch.ones(
                    bsz,
                    token_seq.shape[1],
                    dtype=target_text_mask.dtype,
                    device=self.device,
                ),
            ],
            dim=1,
        )

        pat_info_ref = process_pat_info(normalize=True, hr=hr, age=age, sex=sex).to(self.device)
        pat_info_tar = process_pat_info(
            normalize=True, add_token=False, hr=hr, age=age, sex=sex
        ).to(self.device)

        with torch.no_grad():
            base_vector = self.ibe_model.extract_features(
                latent.transpose(2, 1), ref_text_embed, ref_text_mask, pat_info_ref, reduce=True
            )

        noise = torch.randn_like(latent)
        max_t = int(self.scheduler.config.num_train_timesteps)
        t = torch.randint(
            1,
            max_t,
            (bsz,),
            device=self.device,
            dtype=torch.long,
        )
        noisy_latent = self.scheduler.add_noise(latent, noise, t)

        with _autocast_context(self.device, self.amp_dtype):
            noise_pred = self.noise_predictor(
                noisy_latent, t, text_aug, mask_aug, pat_info_tar, base_vector
            )
            recon_loss_per = F.mse_loss(
                noise_pred.float(), noise.float(), reduction="none"
            ).mean(dim=(1, 2))
            recon_loss = recon_loss_per.mean()
            reg_loss = F.mse_loss(token_seq_base.float(), token_init_base.float(), reduction="mean")
            orth_loss = self._orthogonality_loss(token_seq_base)
            contrast_enabled = (
                self.contrast_recon_weight > 0.0
                or self.contrast_style_delta_weight > 0.0
                or self.contrast_semantic_delta_weight > 0.0
                or self.contrast_feature_weight > 0.0
                or self.contrast_feature_mmd_weight > 0.0
            )
            aux_enabled = (
                self.style_loss_weight > 0.0
                or self.semantic_loss_weight > 0.0
                or self.contrast_style_delta_weight > 0.0
                or self.contrast_semantic_delta_weight > 0.0
                or self.feature_loss_weight > 0.0
                or self.contrast_feature_weight > 0.0
                or self.feature_mmd_weight > 0.0
                or self.contrast_feature_mmd_weight > 0.0
            )
            no_token_noise_pred = None
            if contrast_enabled:
                with torch.no_grad():
                    no_token_noise_pred = self.noise_predictor(
                        noisy_latent,
                        t,
                        target_text_embed,
                        target_text_mask,
                        pat_info_tar,
                        base_vector,
                    )
            if aux_enabled and 1 < self.aux_timestep_max < max_t:
                aux_noise = torch.randn_like(latent)
                aux_t = torch.randint(
                    1,
                    self.aux_timestep_max,
                    (bsz,),
                    device=self.device,
                    dtype=torch.long,
                )
                aux_noisy_latent = self.scheduler.add_noise(latent, aux_noise, aux_t)
                aux_noise_pred = self.noise_predictor(
                    aux_noisy_latent, aux_t, text_aug, mask_aug, pat_info_tar, base_vector
                )
                pred_x0 = self._predict_x0_from_noise(aux_noisy_latent, aux_noise_pred, aux_t)
                if contrast_enabled:
                    with torch.no_grad():
                        aux_no_token_noise_pred = self.noise_predictor(
                            aux_noisy_latent,
                            aux_t,
                            target_text_embed,
                            target_text_mask,
                            pat_info_tar,
                            base_vector,
                        )
                        pred_x0_no_token = self._predict_x0_from_noise(
                            aux_noisy_latent, aux_no_token_noise_pred, aux_t
                        )
                else:
                    pred_x0_no_token = None
            else:
                pred_x0 = self._predict_x0_from_noise(noisy_latent, noise_pred, t)
                if contrast_enabled and no_token_noise_pred is not None:
                    with torch.no_grad():
                        pred_x0_no_token = self._predict_x0_from_noise(
                            noisy_latent, no_token_noise_pred, t
                        )
                else:
                    pred_x0_no_token = None
            style_loss = self._style_loss(pred_x0, center_idx)
            semantic_loss = self._semantic_loss(pred_x0, multi_hot)
            if self.feature_loss_weight > 0.0 or self.contrast_feature_weight > 0.0:
                feature_dist_per = self._semantic_feature_distance_per(pred_x0, latent)
                feature_loss = feature_dist_per.mean()
            else:
                feature_dist_per = None
                feature_loss = recon_loss * 0.0
            if self.feature_mmd_weight > 0.0 or self.contrast_feature_mmd_weight > 0.0:
                pred_feat = self._semantic_feature_tensor(pred_x0)
                with torch.no_grad():
                    real_feat = self._semantic_feature_tensor(latent)
                feature_mmd_loss = self._feature_mmd_loss(pred_feat, real_feat)
            else:
                feature_mmd_loss = recon_loss * 0.0
            if self.contrast_recon_weight > 0.0 and no_token_noise_pred is not None:
                no_token_recon_per = F.mse_loss(
                    no_token_noise_pred.float(), noise.float(), reduction="none"
                ).mean(dim=(1, 2))
                contrast_recon_loss = F.relu(
                    recon_loss_per - no_token_recon_per + self.contrast_recon_margin
                ).mean()
            else:
                contrast_recon_loss = recon_loss * 0.0
            if (
                self.contrast_style_delta_weight > 0.0
                and pred_x0_no_token is not None
                and self.style_model is not None
            ):
                p_token = self._style_target_prob(pred_x0, center_idx)
                with torch.no_grad():
                    p_no_token = self._style_target_prob(pred_x0_no_token, center_idx)
                contrast_style_delta_loss = F.relu(
                    self.contrast_style_delta_margin - (p_token - p_no_token)
                ).mean()
            else:
                contrast_style_delta_loss = recon_loss * 0.0
            if (
                self.contrast_semantic_delta_weight > 0.0
                and pred_x0_no_token is not None
                and self.semantic_model is not None
            ):
                p_token = self._semantic_primary_prob(pred_x0, class_idx)
                with torch.no_grad():
                    p_no_token = self._semantic_primary_prob(pred_x0_no_token, class_idx)
                contrast_semantic_delta_loss = F.relu(
                    self.contrast_semantic_delta_margin - (p_token - p_no_token)
                ).mean()
            else:
                contrast_semantic_delta_loss = recon_loss * 0.0
            if (
                self.contrast_feature_weight > 0.0
                and pred_x0_no_token is not None
                and self.semantic_model is not None
                and feature_dist_per is not None
            ):
                with torch.no_grad():
                    no_token_feature_dist = self._semantic_feature_distance_per(
                        pred_x0_no_token, latent
                    )
                contrast_feature_loss = F.relu(
                    feature_dist_per - no_token_feature_dist + self.contrast_feature_margin
                ).mean()
            else:
                contrast_feature_loss = recon_loss * 0.0
            if (
                self.contrast_feature_mmd_weight > 0.0
                and pred_x0_no_token is not None
                and self.semantic_model is not None
            ):
                with torch.no_grad():
                    no_token_feat = self._semantic_feature_tensor(pred_x0_no_token)
                    no_token_mmd = self._feature_mmd_loss(no_token_feat, real_feat)
                contrast_feature_mmd_loss = F.relu(
                    feature_mmd_loss - no_token_mmd + self.feature_mmd_margin
                )
            else:
                contrast_feature_mmd_loss = recon_loss * 0.0
            loss = (
                recon_loss
                + self.reg_init_weight * reg_loss
                + self.token_orth_weight * orth_loss
                + self.style_loss_weight * style_loss
                + self.semantic_loss_weight * semantic_loss
                + self.contrast_recon_weight * contrast_recon_loss
                + self.contrast_style_delta_weight * contrast_style_delta_loss
                + self.contrast_semantic_delta_weight * contrast_semantic_delta_loss
                + self.feature_loss_weight * feature_loss
                + self.contrast_feature_weight * contrast_feature_loss
                + self.feature_mmd_weight * feature_mmd_loss
                + self.contrast_feature_mmd_weight * contrast_feature_mmd_loss
            )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.token_bank.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "recon_loss": float(recon_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "orth_loss": float(orth_loss.detach().cpu()),
            "style_loss": float(style_loss.detach().cpu()),
            "semantic_loss": float(semantic_loss.detach().cpu()),
            "feature_loss": float(feature_loss.detach().cpu()),
            "feature_mmd_loss": float(feature_mmd_loss.detach().cpu()),
            "contrast_recon_loss": float(contrast_recon_loss.detach().cpu()),
            "contrast_style_delta_loss": float(contrast_style_delta_loss.detach().cpu()),
            "contrast_semantic_delta_loss": float(contrast_semantic_delta_loss.detach().cpu()),
            "contrast_feature_loss": float(contrast_feature_loss.detach().cpu()),
            "contrast_feature_mmd_loss": float(contrast_feature_mmd_loss.detach().cpu()),
            "token_norm": float(token_seq_base.detach().float().norm(dim=-1).mean().cpu()),
            "token_delta_norm": float(
                (token_seq_base.detach() - token_init_base).float().norm(dim=-1).mean().cpu()
            ),
        }

    def train(
        self,
        save_dir: str,
        total_steps: int,
        log_every: int = 20,
        save_every: int = 0,
    ) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        loader = self.create_dataloader()
        iterator = cycle(loader)

        config = {
            "version": "ecgtwin_center_class_prompt_token_train_v1",
            "centers": self.centers,
            "class_names": list(CLASS_NAMES_SUPER5),
            "cache_root": self.cache_root,
            "prompt_bank_path": self.prompt_bank_path,
            "k": self.k,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "total_steps": total_steps,
            "save_every": int(save_every),
            "reg_init_weight": self.reg_init_weight,
            "grad_clip": self.grad_clip,
            "amp_dtype": self.amp_dtype,
            "token_repeat": self.token_repeat,
            "sample_strategy": self.sample_strategy,
            "n_token_vectors": self.n_token_vectors,
            "token_mode": self.token_mode,
            "center_vectors": self.center_vectors,
            "class_vectors": self.class_vectors,
            "residual_vectors": self.residual_vectors,
            "residual_init_std": self.residual_init_std,
            "token_orth_weight": self.token_orth_weight,
            "ref_text_mode": self.ref_text_mode,
            "style_ckpt": self.style_ckpt,
            "style_loss_weight": self.style_loss_weight,
            "style_loss_mode": self.style_loss_mode,
            "style_target_prob": self.style_target_prob,
            "style_crop_len": self.style_crop_len,
            "semantic_ckpt": self.semantic_ckpt,
            "semantic_loss_weight": self.semantic_loss_weight,
            "semantic_crop_len": self.semantic_crop_len,
            "contrast_recon_weight": self.contrast_recon_weight,
            "contrast_recon_margin": self.contrast_recon_margin,
            "contrast_style_delta_weight": self.contrast_style_delta_weight,
            "contrast_style_delta_margin": self.contrast_style_delta_margin,
            "contrast_semantic_delta_weight": self.contrast_semantic_delta_weight,
            "contrast_semantic_delta_margin": self.contrast_semantic_delta_margin,
            "feature_loss_weight": self.feature_loss_weight,
            "feature_crop_len": self.feature_crop_len,
            "contrast_feature_weight": self.contrast_feature_weight,
            "contrast_feature_margin": self.contrast_feature_margin,
            "feature_mmd_weight": self.feature_mmd_weight,
            "contrast_feature_mmd_weight": self.contrast_feature_mmd_weight,
            "feature_mmd_margin": self.feature_mmd_margin,
            "feature_mmd_sigma": self.feature_mmd_sigma,
            "aux_timestep_max": self.aux_timestep_max,
            "aux_amp_clamp": self.aux_amp_clamp,
            "hardware_note": "Targeted for RTX 4090D 24GB, 80GB RAM, 15 CPU cores.",
            "token_semantics": (
                "Append one learnable 768-d center-class token to ECGTwin text_embed; "
                "mask=1; IBE base_vector stays on the original reference ECG/text path."
            ),
        }
        with open(save_path / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)

        metrics_path = save_path / "metrics.jsonl"
        running: Dict[str, float] = {}
        running_n = 0
        t0 = time.time()
        with open(metrics_path, "w") as mf:
            for step in range(1, total_steps + 1):
                metrics = self._step(next(iterator))
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value
                running_n += 1

                if step % log_every == 0 or step == total_steps:
                    avg = {k: v / max(1, running_n) for k, v in running.items()}
                    row = {
                        "step": step,
                        "elapsed_sec": round(time.time() - t0, 2),
                        **avg,
                        "token_norms": self.token_bank.token_norm_table(),
                        "delta_norms": self.token_bank.delta_norm_table(),
                    }
                    mf.write(json.dumps(row) + "\n")
                    mf.flush()
                    print(
                        f"[step {step:5d}/{total_steps}] "
                        f"loss={avg['loss']:.5f} recon={avg['recon_loss']:.5f} "
                        f"orth={avg['orth_loss']:.5f} style={avg['style_loss']:.5f} "
                        f"sem={avg['semantic_loss']:.5f} "
                        f"feat={avg['feature_loss']:.5f} "
                        f"mmd={avg['feature_mmd_loss']:.5f} "
                        f"delta={avg['token_delta_norm']:.4f} "
                        f"elapsed={time.time() - t0:.0f}s",
                        flush=True,
                    )
                    running = {}
                    running_n = 0
                if save_every > 0 and step % save_every == 0 and step != total_steps:
                    ckpt_path = save_path / f"prompt_token_bank_step{step:05d}.pt"
                    torch.save(self.token_bank.export_state(), ckpt_path)
                    print(f"[ckpt] wrote {ckpt_path}", flush=True)

        torch.save(self.token_bank.export_state(), save_path / "prompt_token_bank.pt")
        print(f"[done] wrote {save_path / 'prompt_token_bank.pt'}", flush=True)
