"""Style Translator IBE trainer.

Composite loss: L = lambda_sup * L_supcon + lambda_id * L_identity + lambda_style * L_style
  - L_identity: SIBE(x, proto_own) == base_IBE(x)
  - L_style: SIBE(x_A, proto_B) == SIBE(x_B_sameclass, proto_B)
  - L_supcon: center-label SupCon on translated features
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ECGTWIN_ROOT = PROJECT_ROOT / "model" / "ECGTwin"
for p in (str(PROJECT_ROOT), str(ECGTWIN_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from module.IBExtractor import IBExtractor  # noqa: E402
from utils.data_utils import ListDataset, _pad_text_embed, process_pat_info, sex_transform  # noqa: E402

from methods.ecgtwin_gen.style_translator.losses import style_translator_loss  # noqa: E402
from methods.ecgtwin_gen.style_translator.model import StyleTranslatorIBE  # noqa: E402
from methods.ecgtwin_gen.style_translator.prototype_encoder import PrototypeEncoder  # noqa: E402


def style_translator_collate_fn(batch):
    """Collate for multi-center dataset.

    Returns:
        latents: (B, 4, 128)
        meta: dict with text_embed, mask, patient_info, center_id, class_idx
    """
    latents = torch.stack([entry[0] for entry in batch])  # (B, 4, 128)
    labels = [entry[1] for entry in batch]

    hr = torch.tensor([l["hr"] for l in labels], dtype=torch.float32)
    age = torch.tensor([l["age"] for l in labels], dtype=torch.float32)
    sex = torch.tensor([sex_transform(l["sex"]) for l in labels], dtype=torch.float32)

    embed_list = [l["text_embed"] for l in labels]
    text_embed, text_mask = _pad_text_embed(embed_list)

    center_id = torch.tensor([l["center_id"] for l in labels], dtype=torch.long)
    class_idx = torch.tensor([l["class_idx"] for l in labels], dtype=torch.long)

    return latents, {
        "hr": hr, "age": age, "sex": sex,
        "text_embed": text_embed,
        "text_embed_mask": text_mask,
        "center_id": center_id,
        "class_idx": class_idx,
    }


class StyleTranslatorTrainer:

    def __init__(self, cfg: dict, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_centers = len(cfg["data"]["centers"])

        # ── Load frozen base IBE from ECGTwin checkpoint ──
        ecgtwin_cfg_path = ECGTWIN_ROOT / "config" / "DiT_ECGTwin.yaml"
        with open(ecgtwin_cfg_path) as f:
            eccfg = yaml.safe_load(f)
        h_ibe = eccfg["hyper_para"]["ibe"]

        base_ibe = IBExtractor(
            embed_dim=h_ibe["embed_dim"],
            num_heads=h_ibe["num_heads"],
            ff_hidden_size=h_ibe["ff_hidden_size"],
            num_layers=h_ibe["num_layers"],
            text_embed_dim=h_ibe["text_embed_dim"],
            patient_info_size=h_ibe["patient_info_size"],
        )
        ibe_ckpt = ECGTWIN_ROOT / eccfg["dependencies"]["ibe_path"]
        base_ibe.load_state_dict(torch.load(ibe_ckpt, map_location="cpu"))

        # ── Build Style Translator ──
        mcfg = cfg["model"]
        proto_enc = PrototypeEncoder(
            in_channels=mcfg["prototype"]["in_channels"],
            hidden=mcfg["prototype"]["hidden"],
            out_dim=mcfg["prototype"]["out_dim"],
        )
        self.model = StyleTranslatorIBE(
            base_ibe=base_ibe,
            prototype_encoder=proto_enc,
            embed_dim=mcfg["embed_dim"],
        ).to(self.device)

        print(f"  trainable params: {self.model.trainable_params():,}")
        print(f"  frozen params:    {self.model.frozen_params():,}")

        # ── Optim + scheduler ──
        tcfg = cfg["training"]
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=tcfg["epochs"],
            eta_min=tcfg["scheduler"]["eta_min"],
        )
        self.grad_clip = tcfg["grad_clip"]
        self.epochs = tcfg["epochs"]
        self.batch_size = tcfg["batch_size"]

        # ── Loss cfg ──
        lcfg = cfg["loss"]
        self.lambda_sup = lcfg["lambda_sup"]
        self.lambda_id = lcfg["lambda_id"]
        self.lambda_style = lcfg["lambda_style"]
        self.supcon_temp = lcfg["supcon_temp"]
        self.style_pair_fraction = lcfg["style_pair_fraction"]

        # ── Support sampling ──
        scfg = cfg["support"]
        self.K = scfg["K"]
        self.resample_per_batch = scfg["resample_per_batch"]

    # ─────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────

    def create_dataloaders(self, dataset_path: str):
        self.dataset = ListDataset(dataset_path)
        val_fold = self.cfg["data"]["val_fold"]
        test_fold = self.cfg["data"]["test_fold"]

        train_idx, val_idx = [], []
        per_center_train = defaultdict(list)
        per_center_class_train = defaultdict(list)  # (cid, class_idx) -> [idx]
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            fold = label.get("strat_fold", 0)
            cid = int(label["center_id"])
            cls = int(label["class_idx"])
            if fold == val_fold:
                val_idx.append(i)
            elif fold == test_fold:
                continue
            else:
                train_idx.append(i)
                per_center_train[cid].append(i)
                per_center_class_train[(cid, cls)].append(i)

        # store raw dataset indices (not subset-relative) for support sampling
        self.per_center_train = dict(per_center_train)
        self.train_idx_set = set(train_idx)

        train_loader = DataLoader(
            Subset(self.dataset, train_idx),
            batch_size=self.batch_size, shuffle=True,
            collate_fn=style_translator_collate_fn,
            num_workers=self.cfg["training"]["num_workers"],
            drop_last=True, pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(self.dataset, val_idx),
            batch_size=self.batch_size, shuffle=False,
            collate_fn=style_translator_collate_fn,
            num_workers=self.cfg["training"]["num_workers"],
            drop_last=False, pin_memory=True,
        )
        print(f"  train samples: {len(train_idx)}   val samples: {len(val_idx)}")
        print(f"  per-center train counts: {{ {', '.join(f'{c}: {len(v)}' for c, v in self.per_center_train.items())} }}")
        return train_loader, val_loader

    def _sample_support_latents(self) -> torch.Tensor:
        """Sample K latents per center from train pool.

        Returns: (n_centers, K, 4, 128) tensor on device.
        """
        support = []
        for cid in range(self.n_centers):
            pool = self.per_center_train[cid]
            if len(pool) == 0:
                raise RuntimeError(f"no train samples for center {cid}")
            if len(pool) >= self.K:
                idx = random.sample(pool, self.K)
            else:
                idx = [random.choice(pool) for _ in range(self.K)]  # with replacement
            latents = torch.stack([self.dataset[i][0] for i in idx])  # (K, 4, 128)
            support.append(latents)
        return torch.stack(support).to(self.device)  # (n_centers, K, 4, 128)

    # ─────────────────────────────────────────
    # Style assignment + pairing (per batch)
    # ─────────────────────────────────────────

    def _assign_style_and_pairs(self, center_id: torch.Tensor, class_idx: torch.Tensor):
        """For each query, pick target center (own or cross).

        Returns:
            target_center (B,): which center's proto to use
            identity_mask (B,) bool: target == own
            pair_i (P,), pair_j (P,) long: translated[i] should ≈ own-style[j]
              where i translates to some other center B, and j is a same-class sample
              from center B (using own-center proto).
        """
        B = center_id.shape[0]
        device = center_id.device
        target = center_id.clone()
        cross_mask = torch.rand(B, device=device) < self.style_pair_fraction
        cross_indices = torch.where(cross_mask)[0]

        # pick random "other" center for each cross query
        for i in cross_indices.tolist():
            oc = center_id[i].item()
            others = [c for c in range(self.n_centers) if c != oc]
            target[i] = random.choice(others)

        identity_mask = target == center_id

        # For L_style: find same-class-same-target-center matches within the batch
        pair_i_list, pair_j_list = [], []
        # Build a lookup: (center, class) -> list of in-batch indices with identity style
        id_by_cc = defaultdict(list)
        for j in range(B):
            if identity_mask[j].item():
                id_by_cc[(center_id[j].item(), class_idx[j].item())].append(j)

        for i in cross_indices.tolist():
            tc = target[i].item()
            ci = class_idx[i].item()
            candidates = id_by_cc.get((tc, ci), [])
            if candidates:
                pair_i_list.append(i)
                pair_j_list.append(random.choice(candidates))

        pair_i = torch.tensor(pair_i_list, dtype=torch.long, device=device)
        pair_j = torch.tensor(pair_j_list, dtype=torch.long, device=device)
        return target, identity_mask, pair_i, pair_j

    # ─────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────

    def _forward_batch(self, latents, meta, style_batch=None):
        """Forward IBE once with per-query style_vec.

        latents: (B, 4, 128) tensor on device
        meta: dict from collate
        style_batch: (B, D) or None
        Returns: feat (B, D)
        """
        x = latents.transpose(2, 1)  # (B, 128, 4) for IBE
        text_embed = meta["text_embed"].to(self.device)
        text_mask = meta["text_embed_mask"].to(self.device)
        p = process_pat_info(
            normalize=True, hr=meta["hr"], age=meta["age"], sex=meta["sex"]
        ).to(self.device)
        feat = self.model.extract_features(
            x, text_embed, text_mask, p,
            style_vec=style_batch, reduce=True,
        )
        return feat

    def _train_step(self, latents, meta, proto_centers: torch.Tensor):
        """Single train step.

        proto_centers: (n_centers, D) — current-batch prototypes
        """
        B = latents.shape[0]
        latents = latents.to(self.device)
        center_id = meta["center_id"].to(self.device)
        class_idx = meta["class_idx"].to(self.device)

        target, identity_mask, pair_i, pair_j = self._assign_style_and_pairs(center_id, class_idx)

        style_batch = proto_centers[target]  # (B, D)

        feat = self._forward_batch(latents, meta, style_batch=style_batch)

        # base_feat (no style, detached)
        with torch.no_grad():
            base_feat = self._forward_batch(latents, meta, style_batch=None)

        losses = style_translator_loss(
            feat_translated=feat,
            feat_base=base_feat,
            center_labels=target,
            identity_mask=identity_mask,
            pair_i=pair_i, pair_j=pair_j,
            lambda_sup=self.lambda_sup,
            lambda_id=self.lambda_id,
            lambda_style=self.lambda_style,
            supcon_temp=self.supcon_temp,
        )

        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.grad_clip,
        )
        self.optimizer.step()

        # Extra diag: feat drift for cross-style queries
        with torch.no_grad():
            drift_mask = ~identity_mask
            if drift_mask.any():
                d = (feat[drift_mask] - base_feat[drift_mask]).norm(dim=-1).mean()
                b = base_feat[drift_mask].norm(dim=-1).mean().clamp(min=1e-6)
                feat_drift = (d / b).item()
            else:
                feat_drift = 0.0
            n_pairs = pair_i.shape[0]
            n_cross = int(drift_mask.sum().item())

        return {
            "total": losses["total"].item(),
            "supcon": losses["supcon"].item(),
            "identity": losses["identity"].item(),
            "style": losses["style"].item(),
            "feat_drift": feat_drift,
            "n_pairs": n_pairs,
            "n_cross": n_cross,
        }

    # ─────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, val_loader) -> dict:
        self.model.eval()
        proto = self.model.prototype_encoder.forward_batch(self._sample_support_latents())  # (n_centers, D)
        sums = defaultdict(float)
        n_batches = 0
        # Cross-center classification accuracy: for each query, pick best-matching style by cos sim
        correct_center, total = 0, 0

        for latents, meta in val_loader:
            B = latents.shape[0]
            center_id = meta["center_id"].to(self.device)
            class_idx = meta["class_idx"].to(self.device)
            target, id_mask, pair_i, pair_j = self._assign_style_and_pairs(center_id, class_idx)

            latents = latents.to(self.device)
            style_batch = proto[target]
            feat = self._forward_batch(latents, meta, style_batch=style_batch)
            base_feat = self._forward_batch(latents, meta, style_batch=None)

            losses = style_translator_loss(
                feat_translated=feat, feat_base=base_feat,
                center_labels=target, identity_mask=id_mask,
                pair_i=pair_i, pair_j=pair_j,
                lambda_sup=self.lambda_sup, lambda_id=self.lambda_id,
                lambda_style=self.lambda_style, supcon_temp=self.supcon_temp,
            )
            sums["total"] += losses["total"].item()
            sums["supcon"] += losses["supcon"].item()
            sums["identity"] += losses["identity"].item()
            sums["style"] += losses["style"].item()

            # center cls: each query uses own-center style, then NN over feats with proto centroids.
            feat_own = self._forward_batch(latents, meta, style_batch=proto[center_id])
            feat_norm = F.normalize(feat_own, dim=-1)
            proto_norm = F.normalize(proto, dim=-1)
            sim = feat_norm @ proto_norm.t()  # (B, n_centers)
            pred = sim.argmax(dim=-1)
            correct_center += (pred == center_id).sum().item()
            total += B

            n_batches += 1

        metrics = {k: v / max(n_batches, 1) for k, v in sums.items()}
        metrics["center_cls_acc"] = correct_center / max(total, 1)

        # Proto stability: resample K 3× and measure cos-sim between the style vectors
        stabs = []
        for _ in range(3):
            p2 = self.model.prototype_encoder.forward_batch(self._sample_support_latents())
            stabs.append(p2)
        cos_sims = []
        for i in range(len(stabs)):
            for j in range(i + 1, len(stabs)):
                cs = F.cosine_similarity(stabs[i], stabs[j], dim=-1).mean().item()
                cos_sims.append(cs)
        metrics["proto_cos_sim"] = sum(cos_sims) / max(len(cos_sims), 1)

        self.model.train()
        return metrics

    # ─────────────────────────────────────────
    # Train loop
    # ─────────────────────────────────────────

    def train(self, dataset_path: str, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader = self.create_dataloaders(dataset_path)
        best_val_total = float("inf")
        log_lines = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            agg = defaultdict(float)
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
            for latents, meta in pbar:
                # Fresh support per batch
                if self.resample_per_batch or n_batches == 0:
                    support_batch = self._sample_support_latents()  # (n_centers, K, 4, 128)
                    proto_centers = self.model.prototype_encoder.forward_batch(support_batch)
                step_m = self._train_step(latents, meta, proto_centers)
                for k, v in step_m.items():
                    agg[k] += v
                n_batches += 1
                pbar.set_postfix(
                    tot=f"{step_m['total']:.3f}",
                    sup=f"{step_m['supcon']:.3f}",
                    idn=f"{step_m['identity']:.4f}",
                    sty=f"{step_m['style']:.4f}",
                    drf=f"{step_m['feat_drift']:.3f}",
                    np=step_m['n_pairs'],
                )
            train_avg = {k: v / max(n_batches, 1) for k, v in agg.items()}

            val_m = self._validate(val_loader)
            self.lr_scheduler.step()

            log = (
                f"Epoch {epoch:3d} | "
                f"Train tot:{train_avg['total']:.3f} sup:{train_avg['supcon']:.3f} "
                f"id:{train_avg['identity']:.4f} sty:{train_avg['style']:.4f} "
                f"drift:{train_avg['feat_drift']:.3f} | "
                f"Val tot:{val_m['total']:.3f} sup:{val_m['supcon']:.3f} "
                f"id:{val_m['identity']:.4f} sty:{val_m['style']:.4f} "
                f"c-acc:{val_m['center_cls_acc']:.3f} proto-cos:{val_m['proto_cos_sim']:.3f} | "
                f"LR:{self.optimizer.param_groups[0]['lr']:.2e}"
            )
            print(log)
            log_lines.append(log)

            if val_m["total"] < best_val_total:
                best_val_total = val_m["total"]
                torch.save(
                    {
                        "style_fusion": self.model.style_fusion.state_dict(),
                        "prototype_encoder": self.model.prototype_encoder.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_m,
                        "train_metrics": train_avg,
                    },
                    save_dir / "translator_best.pth",
                )
                print(f"  => Saved best (val_total={best_val_total:.3f})")

            torch.save(
                {
                    "style_fusion": self.model.style_fusion.state_dict(),
                    "prototype_encoder": self.model.prototype_encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_m,
                },
                save_dir / "translator_latest.pth",
            )

        with open(save_dir / "training_log.txt", "w") as f:
            f.write("\n".join(log_lines))
        print(f"\nDone. Best val_total: {best_val_total:.3f}")
        return best_val_total


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config_stage0.yaml"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"[trainer] config: {args.config}")
    print(f"[trainer] stage: {cfg['meta']['stage']}")
    print(f"[trainer] centers: {cfg['data']['centers']}")

    trainer = StyleTranslatorTrainer(cfg, device=args.device)
    trainer.train(
        dataset_path=cfg["data"]["dataset_path"],
        save_dir=cfg["logging"]["save_dir"],
    )


if __name__ == "__main__":
    main()
