"""Thor/SM110 dense GEMM performance-bound reference model."""

from .model import (
    Capacity,
    EvidenceKind,
    Hardware,
    ModelError,
    ModelResult,
    PipelineProfile,
    PrecisionSpec,
    Schedule,
    Workload,
    audit_inputs,
    audit_pipeline_profiles,
    evaluate,
    evaluate_domain_upper,
    evaluate_manifest,
    predict_pipeline_worker_seconds,
    precision_specs,
)
from .observations import (
    ObservedBest, qualify_observations_for_suite,
    summarize_closure_campaign, summarize_observed_csvs,
)
from .evidence_import import (
    import_component_campaign,
    import_compute_campaign,
    import_memory_duplex_campaign,
    import_tma_payload_campaign,
)

__all__ = [
    "Capacity",
    "EvidenceKind",
    "Hardware",
    "ModelError",
    "ModelResult",
    "PipelineProfile",
    "PrecisionSpec",
    "Schedule",
    "Workload",
    "audit_inputs",
    "audit_pipeline_profiles",
    "evaluate",
    "evaluate_domain_upper",
    "evaluate_manifest",
    "predict_pipeline_worker_seconds",
    "precision_specs",
    "ObservedBest",
    "qualify_observations_for_suite",
    "summarize_closure_campaign",
    "summarize_observed_csvs",
    "import_component_campaign",
    "import_compute_campaign",
    "import_memory_duplex_campaign",
    "import_tma_payload_campaign",
]
