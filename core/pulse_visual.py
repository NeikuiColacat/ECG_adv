"""Visual-only PULSE adaptation; no changes to the external author's installation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from core.consistency import categorical_jsd

VISION_BLOCKS = (19, 20, 21, 22)  # CLIP has 24 blocks; feature selection is hidden_states[-2].


def visual_lora_targets(model):
    tower = model.get_vision_tower()
    if (tower.config.num_hidden_layers != 24 or tower.select_layer != -2
            or tower.select_feature != "patch"):
        raise ValueError("visual LoRA requires the locked CLIP penultimate patch features")
    prefix = "model.vision_tower.vision_tower.vision_model.encoder.layers."
    targets = [f"{prefix}{i}.self_attn.{name}" for i in VISION_BLOCKS for name in ("q_proj", "v_proj")]
    modules = dict(model.named_modules())
    if any(not isinstance(modules.get(name), torch.nn.Linear) for name in targets):
        raise ValueError("visual LoRA target module identity changed")
    return targets


def vision_features_with_grad(base, pixels):
    # The author's tower.forward has @no_grad. Call the trained inner CLIP directly.
    # No reloading of vanilla CLIP; no caching of any trainable visual output.
    tower = base.get_vision_tower()
    result = tower.vision_tower(pixels, output_hidden_states=True)
    return tower.feature_select(result).to(pixels.dtype)


def aligned_answer_logits(base, packed, labels):
    selected = labels[:, 1:] != -100
    if not bool(selected.any()):
        raise ValueError("JSD requires supervised answer positions")
    hidden = base.get_model()(inputs_embeds=packed, use_cache=False, return_dict=True).last_hidden_state
    return base.lm_head(hidden[:, :-1][selected]).float(), labels[:, 1:][selected]


def answer_jsd_loss(logits, targets, *, jsd_weight=12.0):
    if len(logits) not in (1, 3):
        raise ValueError("PULSE visual objective requires clean or clean plus two AugMix views")
    ce = F.cross_entropy(logits[0].float(), targets)
    jsd = categorical_jsd(logits) if len(logits) == 3 else ce.new_zeros(())
    return ce + jsd_weight * jsd, ce, jsd
