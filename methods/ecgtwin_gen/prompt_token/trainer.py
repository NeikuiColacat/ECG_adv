from __future__ import annotations

import json
import sys
import time
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ECGTWIN_ROOT = PROJECT_ROOT / "model" / "ECGTwin"
sys.path.insert(0, str(ECGTWIN_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from module.IBExtractor import IBExtractor  # noqa: E402
from utils.data_utils import _pad_text_embed, process_pat_info  # noqa: E402
from utils.model_utils import build_noise_predictor  # noqa: E402

from methods.ecgtwin_gen.prompt_token.model import CenterClassPromptTokenBank  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def _sex_to_binary(value) -> float:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).upper()
    return 0.0 if text.startswith("F") else 1.0


def _load_text_embed(prompt_bank: Dict[str, object], primary_snomed, primary_class: str) -> torch.Tensor:
    by_snomed = prompt_bank["by_snomed"]
    by_class = prompt_bank["by_class"]
    candidates = []
    if primary_snomed is not None:
        candidates.append(primary_snomed)
        try:
            candidates.append(int(primary_snomed))
        except Exception:
            pass
        candidates.append(str(primary_snomed))
    for key in candidates:
        if key in by_snomed:
            return by_snomed[key].detach().float()
    return by_class[primary_class].detach().float()


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
        t = torch.randint(
            1,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
            dtype=torch.long,
        )
        noisy_latent = self.scheduler.add_noise(latent, noise, t)

        with _autocast_context(self.device, self.amp_dtype):
            noise_pred = self.noise_predictor(
                noisy_latent, t, text_aug, mask_aug, pat_info_tar, base_vector
            )
            recon_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            reg_loss = F.mse_loss(token_seq_base.float(), token_init_base.float(), reduction="mean")
            orth_loss = self._orthogonality_loss(token_seq_base)
            loss = (
                recon_loss
                + self.reg_init_weight * reg_loss
                + self.token_orth_weight * orth_loss
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
                        f"orth={avg['orth_loss']:.5f} "
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
