"""Evidence registry and comparison helpers for agent-safe experiments."""

from .comparison import ComparisonBundleError, build_comparison_bundle
from .backfill import build_legacy_vae_lhat_manifest
from .registry import EvidenceAuditError, audit_active_evidence_registry, load_evidence_registry

__all__ = [
    "ComparisonBundleError",
    "EvidenceAuditError",
    "audit_active_evidence_registry",
    "build_comparison_bundle",
    "build_legacy_vae_lhat_manifest",
    "load_evidence_registry",
]
