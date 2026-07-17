"""Explicitly registered node callables for the typed method graph."""

from core.methods.nodes.buffers import replay_read, replay_write
from core.methods.nodes.codecs import vae_decode, vae_encode
from core.methods.nodes.latent_ops import lhat_attack, paired_latent_bridge
from core.methods.nodes.mixers import augmix
from core.methods.nodes.selectors import clean_source, select_candidates, strict_pair
from core.methods.nodes.waveform_ops import canonical_corruption


__all__ = [
    "augmix",
    "canonical_corruption",
    "clean_source",
    "lhat_attack",
    "paired_latent_bridge",
    "replay_read",
    "replay_write",
    "select_candidates",
    "strict_pair",
    "vae_decode",
    "vae_encode",
]
