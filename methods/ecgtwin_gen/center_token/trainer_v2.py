"""CenterToken v2 trainer — per-block, per-center INDEPENDENT, step-based.

Plan Rev 13 (post-Rev-12 audit) implementation. Five v1 trainer bugs fixed:

  Bug D (.detach() kills L_inv grad)    → REMOVED. No detach in loss path.
  Bug I (best.pth = epoch-1 near-zero)  → REPLACED by EMA-smoothed snapshot.
  Bug I (1 batch/epoch × 30 epochs)     → step-based loop, 5000 grad steps default.
  Bug D (L_inv variance rewards CT=0)   → DROPPED. Replaced by L_isolation.
  Bug H (L_reg ‖CT‖₂ pulls token to 0)  → REPLACED by sphere projection.

Plus arch upgrade: CenterTokenPerBlock (6 × 256-d, TokenVerse SIGGRAPH'25 paradigm).

L_isolation (TokenVerse-style anchor against base ECGTwin output):
  - Pre-cache ~64 (z, t, prompt_emb, pat_info, base_vec, noise, noise_pred_base)
    from a "generic" pool (PTB-XL latents — different vendor than PN2021).
  - 50% of training steps: also compute MSE(hooked_pred, cached_base_pred) on
    a random pool sample. CT is forced to leave generic generation invariant
    while still adapting to per-center recon.

Per-center INDEPENDENT: trainer takes ONE center's K=200 dataset. New centers
can be enrolled later by running the trainer on their data — no inter-center
joint training needed (matches deployment scenario "few-shot adapt to new center").

Usage (driver: scripts/ecgtwin_gen/train_center_token_v2.py):
  trainer = CenterTokenTrainerV2(config, ecgtwin_config, device='cuda:0')
  trainer.train(dataset_path='extra_k200.pt', save_dir='out/extra_k200/',
                total_steps=5000)
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ECGTWIN_ROOT = PROJECT_ROOT / "model" / "ECGTwin"
sys.path.insert(0, str(ECGTWIN_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from module.IBExtractor import IBExtractor  # noqa: E402
from utils.model_utils import build_noise_predictor  # noqa: E402
from utils.data_utils import (  # noqa: E402
    ListDataset, process_pat_info, sex_transform, _pad_text_embed,
)

from methods.ecgtwin_gen.center_token.model import CenterTokenPerBlock  # noqa: E402
from methods.ecgtwin_gen.center_token.trainer import center_token_collate_fn  # noqa: E402


def _infinite_iter(loader):
    """Cycle through DataLoader forever (for step-based training)."""
    while True:
        for batch in loader:
            yield batch


class CenterTokenTrainerV2:

    def __init__(
        self,
        config: dict,
        ecgtwin_config: dict,
        device: str = "cuda:0",
    ):
        self.config = config
        self.ecgtwin_config = ecgtwin_config
        self.device = torch.device(device)
        h_ = ecgtwin_config["hyper_para"]

        # ── Frozen models ────────────────────────────
        model_type = ecgtwin_config["meta"]["model_type"]
        n_channels = 4

        self.noise_predictor = build_noise_predictor(model_type, n_channels, h_)
        np_path = ECGTWIN_ROOT / ecgtwin_config["inference_setting"]["noise_predictor_path"]
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
        ibe_path = ECGTWIN_ROOT / ecgtwin_config["dependencies"]["ibe_path"]
        self.ibe_model.load_state_dict(torch.load(ibe_path, map_location="cpu"))
        self.ibe_model.to(self.device).eval()
        for p in self.ibe_model.parameters():
            p.requires_grad = False

        self.scheduler = DDPMScheduler(
            num_train_timesteps=h_["ddpm"]["num_train_steps"],
            beta_start=h_["ddpm"]["beta_start"],
            beta_end=h_["ddpm"]["beta_end"],
        )

        # ── CenterToken v2 (per-block) ──────────────
        ct_cfg = config["center_token"]
        num_blocks = len(self.noise_predictor.blocks)
        self.center_token = CenterTokenPerBlock(
            dim=ct_cfg["dim"], num_blocks=num_blocks,
        ).to(self.device)
        print(f"[v2] CenterTokenPerBlock dim={ct_cfg['dim']} "
              f"num_blocks={num_blocks} init_norm={self.center_token.total_norm:.4f}")

        # ── EMA buffer ──────────────────────────────
        self.ema_decay = ct_cfg.get("ema_decay", 0.99)
        self._ema_shadow = [emb.data.clone() for emb in self.center_token.embeddings]

        # ── Sphere norm projection target ───────────
        self.sphere_target = ct_cfg.get("sphere_target_norm", 1.0)
        # Per-block target (uniform by default; can be overridden after calibration)
        self.target_norms = [self.sphere_target] * num_blocks

        # ── Optimizer ───────────────────────────────
        train_cfg = config["training_v2"]
        self.lr_phase1 = train_cfg["lr_phase1"]
        self.lr_phase2 = train_cfg["lr_phase2"]
        self.phase_split_frac = train_cfg.get("phase_split_frac", 0.8)
        self.total_steps = train_cfg["total_steps"]
        self.batch_size = train_cfg["batch_size"]
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.iso_freq = train_cfg.get("iso_freq", 0.5)  # fraction of steps with L_iso
        self.iso_pool_size = train_cfg.get("iso_pool_size", 64)
        self.iso_weight = train_cfg.get("iso_weight", 1.0)
        self.log_every = train_cfg.get("log_every", 50)

        self.optimizer = torch.optim.AdamW(
            self.center_token.parameters(), lr=self.lr_phase1,
        )

        # ── Hook management ─────────────────────────
        self.hooks: List = []
        self.hooks_enabled = True

    # ─────────────────────────────────────────────
    # Hook (per-block routing)
    # ─────────────────────────────────────────────

    def _make_hook(self, block_idx: int):
        trainer = self

        def hook_fn(module, args):
            if not trainer.hooks_enabled:
                return args
            x, c, c2 = args[0], args[1], args[2]
            rest = args[3:]
            c = c + trainer.center_token(block_idx, c.size(0))
            return (x, c, c2) + rest

        return hook_fn

    def _register_hooks(self):
        self._remove_hooks()
        for i, block in enumerate(self.noise_predictor.blocks):
            self.hooks.append(block.register_forward_pre_hook(self._make_hook(i)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    # ─────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────

    def create_dataloader(self, dataset_path: str):
        dataset = ListDataset(dataset_path)
        # All records (no train/val split — step-based, EMA snapshot, smoking-gun
        # is the real validation gate)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=center_token_collate_fn,
            num_workers=0,
            drop_last=False,  # use all records; cycle continues regardless
        )
        print(f"[data] {len(dataset)} records, batch={self.batch_size}, "
              f"~{max(1, len(dataset) // self.batch_size)} batches/epoch, "
              f"~{self.total_steps // max(1, len(dataset) // self.batch_size)} epoch-equivalents")
        return loader

    # ─────────────────────────────────────────────
    # Isolation pool (TokenVerse-style anchor)
    # ─────────────────────────────────────────────

    def build_isolation_pool(self, ptbxl_latent_path: str):
        """Pre-cache base ECGTwin outputs on PTBXL latents.

        Each pool entry stores (latent, t, noise, text_embed, text_embed_mask,
        pat_info, base_vector, noise_pred_base). At training time, sample with
        prob iso_freq → run hooked forward → MSE vs cached noise_pred_base.

        This is TokenVerse-style anchoring: when CT is applied to GENERIC
        (non-this-center) content, output should match base ECGTwin → CT
        cannot catastrophically corrupt the base distribution.

        Why PTBXL: it's a DIFFERENT vendor (Schiller) from PN2021 (cpsc/etc),
        so PTBXL serves as a clean "out-of-this-center-distribution" probe.
        """
        print(f"[iso] building isolation pool from {ptbxl_latent_path} "
              f"(n={self.iso_pool_size})")
        ptbxl_data = torch.load(ptbxl_latent_path, map_location="cpu",
                                weights_only=False)
        # ptbxl_data is a list of dicts {data: (4, 128), label: {hr,age,sex,text_embed}}
        rng = torch.Generator().manual_seed(42)
        n_pool = self.iso_pool_size
        n_data = len(ptbxl_data)
        sample_indices = torch.randperm(n_data, generator=rng)[:n_pool].tolist()

        pool = []
        self.hooks_enabled = False  # base path: no hook
        with torch.no_grad():
            for idx in sample_indices:
                entry = ptbxl_data[idx]
                latent = entry["data"]  # (4, 128)
                if latent.dim() == 2:
                    latent = latent.unsqueeze(0)  # (1, 4, 128)
                latent = latent.to(self.device)

                lab = entry["label"]
                text_embed = lab["text_embed"]
                if text_embed.dim() == 2:
                    text_embed = text_embed.unsqueeze(0)  # (1, L, 768)
                text_embed = text_embed.to(self.device)
                # mask: all valid (no padding in single-sample pool entry)
                text_embed_mask = torch.ones(text_embed.shape[:2],
                                             dtype=torch.float32,
                                             device=self.device)

                hr = torch.tensor([float(lab.get("hr", 75.0))],
                                  dtype=torch.float32)
                age = torch.tensor([float(lab.get("age", 60.0))],
                                   dtype=torch.float32)
                sex = torch.tensor([sex_transform(lab.get("sex", "M"))],
                                   dtype=torch.float32)
                pat_info = process_pat_info(
                    normalize=True, hr=hr, age=age, sex=sex,
                ).to(self.device)
                pat_info_tar = process_pat_info(
                    normalize=True, add_token=False, hr=hr, age=age, sex=sex,
                ).to(self.device)

                # IBE base_vector
                latent_for_ibe = latent.transpose(2, 1)  # (1, 128, 4)
                base_vector = self.ibe_model.extract_features(
                    latent_for_ibe, text_embed, text_embed_mask, pat_info,
                    reduce=True,
                )

                # Random t, noise, noisy latent
                t = torch.randint(1, 1000, (1,),
                                  device=self.device, generator=None)
                noise = torch.randn_like(latent)
                noisy_latent = self.scheduler.add_noise(latent, noise, t)

                # Base prediction (no hook)
                noise_pred_base = self.noise_predictor(
                    noisy_latent, t, text_embed, text_embed_mask,
                    pat_info_tar, base_vector,
                )

                pool.append({
                    "noisy_latent": noisy_latent.detach(),
                    "t": t.detach(),
                    "noise": noise.detach(),
                    "text_embed": text_embed.detach(),
                    "text_embed_mask": text_embed_mask.detach(),
                    "pat_info_tar": pat_info_tar.detach(),
                    "base_vector": base_vector.detach(),
                    "noise_pred_base": noise_pred_base.detach(),
                })
        self.hooks_enabled = True
        print(f"[iso] cached {len(pool)} entries")
        self.iso_pool = pool

    # ─────────────────────────────────────────────
    # Recon step (this-center batch with hook)
    # ─────────────────────────────────────────────

    def _recon_step(self, ecg_data, ecg_label):
        B = ecg_data.size(0)
        latent = ecg_data.to(self.device)

        text_embed = ecg_label["text_embed"].to(self.device)
        text_embed_mask = ecg_label["text_embed_mask"].to(self.device)

        pat_info = process_pat_info(
            normalize=True,
            hr=ecg_label["hr"], age=ecg_label["age"], sex=ecg_label["sex"],
        ).to(self.device)
        pat_info_tar = process_pat_info(
            normalize=True, add_token=False,
            hr=ecg_label["hr"], age=ecg_label["age"], sex=ecg_label["sex"],
        ).to(self.device)

        latent_for_ibe = latent.transpose(2, 1)
        with torch.no_grad():
            base_vector = self.ibe_model.extract_features(
                latent_for_ibe, text_embed, text_embed_mask, pat_info,
                reduce=True,
            )

        noise = torch.randn_like(latent)
        t = torch.randint(1, 1000, (B,), device=self.device)
        noisy_latent = self.scheduler.add_noise(latent, noise, t)

        self.hooks_enabled = True  # apply CT
        noise_pred = self.noise_predictor(
            noisy_latent, t, text_embed, text_embed_mask, pat_info_tar,
            base_vector,
        )
        l_recon = F.mse_loss(noise_pred, noise, reduction="sum") / B
        return l_recon

    # ─────────────────────────────────────────────
    # Isolation step (generic pool with hook vs cached base)
    # ─────────────────────────────────────────────

    def _isolation_step(self):
        """Sample 1 random entry from the isolation pool, run hooked forward,
        compute MSE vs cached base_pred. CT is forced to leave this generic
        sample's noise prediction approximately invariant.
        """
        idx = torch.randint(0, len(self.iso_pool), (1,)).item()
        entry = self.iso_pool[idx]

        self.hooks_enabled = True  # apply CT
        noise_pred_hooked = self.noise_predictor(
            entry["noisy_latent"], entry["t"],
            entry["text_embed"], entry["text_embed_mask"],
            entry["pat_info_tar"], entry["base_vector"],
        )
        l_iso = F.mse_loss(noise_pred_hooked, entry["noise_pred_base"],
                           reduction="mean")
        return l_iso

    # ─────────────────────────────────────────────
    # EMA update
    # ─────────────────────────────────────────────

    def _update_ema(self):
        with torch.no_grad():
            for shadow, emb in zip(self._ema_shadow,
                                   self.center_token.embeddings):
                shadow.mul_(self.ema_decay).add_(emb.data,
                                                 alpha=1.0 - self.ema_decay)

    def _save_ema_snapshot(self, save_path):
        """Write EMA shadow as a CT state_dict for downstream loading."""
        sd = {}
        for i, shadow in enumerate(self._ema_shadow):
            sd[f"embeddings.{i}"] = shadow.cpu()
        torch.save({
            "center_token": sd,
            "norms": [s.norm().item() for s in self._ema_shadow],
            "total_norm": torch.cat([s for s in self._ema_shadow]).norm().item(),
            "ema_decay": self.ema_decay,
            "version": "v2_per_block",
        }, save_path)

    def _save_latest(self, save_path, step):
        torch.save({
            "center_token": self.center_token.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "norms": self.center_token.norms,
            "total_norm": self.center_token.total_norm,
            "version": "v2_per_block",
        }, save_path)

    # ─────────────────────────────────────────────
    # Main training loop (step-based)
    # ─────────────────────────────────────────────

    def train(self, dataset_path: str, save_dir: str,
              ptbxl_latent_path: str = "/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        loader = self.create_dataloader(dataset_path)
        loader_iter = _infinite_iter(loader)

        self.build_isolation_pool(ptbxl_latent_path)

        self._register_hooks()

        log_lines = []
        phase_split_step = int(self.phase_split_frac * self.total_steps)
        running = {"recon": 0.0, "iso": 0.0, "iso_count": 0, "n": 0}

        import time
        t0 = time.time()
        for step in range(1, self.total_steps + 1):
            # Phase-separated LR
            cur_lr = self.lr_phase1 if step <= phase_split_step else self.lr_phase2
            for g in self.optimizer.param_groups:
                g["lr"] = cur_lr

            ecg_data, ecg_label = next(loader_iter)
            self.center_token.train()

            l_recon = self._recon_step(ecg_data, ecg_label)
            losses = {"recon": l_recon}

            # Isolation step with prob iso_freq
            if torch.rand(1).item() < self.iso_freq:
                l_iso = self._isolation_step()
                losses["iso"] = l_iso
                l_total = l_recon + self.iso_weight * l_iso
                running["iso"] += l_iso.item()
                running["iso_count"] += 1
            else:
                l_total = l_recon

            self.optimizer.zero_grad()
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.center_token.parameters(), self.grad_clip,
            )
            self.optimizer.step()

            # Sphere projection (per-block, after step)
            self.center_token.project_to_sphere(self.target_norms)

            # EMA update
            self._update_ema()

            running["recon"] += l_recon.item()
            running["n"] += 1

            if step % self.log_every == 0 or step == self.total_steps:
                avg_recon = running["recon"] / max(running["n"], 1)
                avg_iso = (running["iso"] / max(running["iso_count"], 1)
                           if running["iso_count"] > 0 else 0.0)
                norms = self.center_token.norms
                ema_total = torch.cat([s for s in self._ema_shadow]).norm().item()
                line = (
                    f"step {step:5d}/{self.total_steps} "
                    f"lr={cur_lr:.2e} "
                    f"L_rec={avg_recon:.4f} L_iso={avg_iso:.4f} "
                    f"norms=[{','.join(f'{n:.2f}' for n in norms)}] "
                    f"ema_norm={ema_total:.3f} "
                    f"({time.time()-t0:.0f}s)"
                )
                print(line, flush=True)
                log_lines.append(line)
                running = {"recon": 0.0, "iso": 0.0, "iso_count": 0, "n": 0}

        # Final save
        self._save_ema_snapshot(save_dir / "center_token_smoothed.pth")
        self._save_latest(save_dir / "center_token_latest.pth", self.total_steps)
        with open(save_dir / "training_log.txt", "w") as f:
            f.write("\n".join(log_lines))

        self._remove_hooks()
        print(f"\n[done] training {self.total_steps} steps in "
              f"{time.time()-t0:.0f}s")
        print(f"[done] EMA snapshot saved → {save_dir/'center_token_smoothed.pth'}")
        print(f"[done] EMA per-block norms: "
              f"{[round(s.norm().item(), 3) for s in self._ema_shadow]}")
        print(f"[done] EMA total norm: "
              f"{torch.cat([s for s in self._ema_shadow]).norm().item():.3f}")
