"""Thor/SM110 dense GEMM performance-bound reference model."""

from .model import (
    Capacity,
    EvidenceKind,
    Hardware,
    ModelError,
    ModelResult,
    PrecisionSpec,
    Schedule,
    Workload,
    audit_inputs,
    evaluate,
    evaluate_manifest,
    precision_specs,
)
from .observations import ObservedBest, summarize_observed_csvs

__all__ = [
    "Capacity",
    "EvidenceKind",
    "Hardware",
    "ModelError",
    "ModelResult",
    "PrecisionSpec",
    "Schedule",
    "Workload",
    "audit_inputs",
    "evaluate",
    "evaluate_manifest",
    "precision_specs",
    "ObservedBest",
    "summarize_observed_csvs",
]
