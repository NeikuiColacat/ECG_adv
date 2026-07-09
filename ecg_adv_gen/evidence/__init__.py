"""Evidence registry and comparison helpers for agent-safe experiments."""

from .comparison import ComparisonBundleError, build_comparison_bundle
from .registry import EvidenceAuditError, audit_active_evidence_registry, load_evidence_registry
from .run_record import (
    RunRecordError,
    finalize_run_record,
    register_run_in_registry,
    verify_run_file_index,
)

__all__ = [
    "ComparisonBundleError",
    "EvidenceAuditError",
    "RunRecordError",
    "audit_active_evidence_registry",
    "build_comparison_bundle",
    "finalize_run_record",
    "load_evidence_registry",
    "register_run_in_registry",
    "verify_run_file_index",
]
