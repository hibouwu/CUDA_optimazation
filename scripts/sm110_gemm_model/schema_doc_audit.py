#!/usr/bin/env python3
"""Audit that every executable model field is defined in the main document."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

from .coverage import (
    PrecisionCoverage,
    ScenarioCoverage,
    WorkloadManifestCoverage,
)
from .completion import TargetCompletionAudit, TargetPrecisionAudit
from .model import (
    Capacity,
    Hardware,
    LayerResult,
    PipelineProfile,
    PrecisionSpec,
    Schedule,
    WorkAccounting,
    Workload,
    WorkloadEnvelope,
)
from .observations import ObservedBest
from .suite import SuiteDeclaredProvenance, SuiteLinkage


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "Docs/blackwell_tensorcore/thor_sm110_gemm_performance_bounds.md"

COVERAGE_TOP_LEVEL_FIELDS = {
    "precision_coverage",
    "scenario_coverage",
    "workload_manifest_coverage",
    "common_resource_coverage",
    "all_precisions_numerically_closed",
    "all_precisions_absolute_three_layer_closed",
    "all_precisions_same_precision_ratio_closed",
    "all_common_resources_closed",
    "all_precisions_workload_manifest_complete",
    "target_completion",
}

SUITE_TOP_LEVEL_FIELDS = {
    "suite_linkage",
    "imported_capacities",
    "closure_observations",
    "capacity_findings",
    "source_paths",
    "source_urls",
    "artifact_paths",
    "coverage",
}


def documented_identifiers(text: str) -> set[str]:
    return set(re.findall(r"`([A-Za-z][A-Za-z0-9_]*)`", text))


def first_identifier_uses(text: str) -> dict[str, tuple[int, str]]:
    first: dict[str, tuple[int, str]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        for name in re.findall(r"`([A-Za-z][A-Za-z0-9_]*)`", line):
            first.setdefault(name, (line_number, line))
    return first


def is_defining_first_use(name: str, line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("|") and re.match(
        rf"^\|[^|]*`{re.escape(name)}`[^|]*\|\s*\S+", stripped
    ):
        return True
    return (
        "定义" in line
        or "表示" in line
        or "分别" in line
        or "记为" in line
        or "对应字段" in line
    )


def required_fields() -> dict[str, set[str]]:
    classes = (
        PrecisionSpec,
        Workload,
        Schedule,
        Hardware,
        Capacity,
        PipelineProfile,
        WorkAccounting,
        LayerResult,
        WorkloadEnvelope,
        ObservedBest,
        ScenarioCoverage,
        WorkloadManifestCoverage,
        PrecisionCoverage,
        SuiteLinkage,
        SuiteDeclaredProvenance,
        TargetPrecisionAudit,
        TargetCompletionAudit,
    )
    result = {cls.__name__: {field.name for field in fields(cls)} for cls in classes}
    result["CoverageOutput"] = COVERAGE_TOP_LEVEL_FIELDS
    result["SuiteOutput"] = SUITE_TOP_LEVEL_FIELDS
    return result


def audit_document(path: Path = DOCUMENT) -> list[str]:
    text = path.read_text(encoding="utf-8")
    identifiers = documented_identifiers(text)
    first_uses = first_identifier_uses(text)
    errors: list[str] = []
    for class_name, names in required_fields().items():
        missing = sorted(names - identifiers)
        if missing:
            errors.append(f"{class_name}: undocumented fields: {', '.join(missing)}")
    for name in sorted({name for names in required_fields().values() for name in names}):
        if name not in first_uses:
            continue
        line_number, line = first_uses[name]
        if not is_defining_first_use(name, line):
            errors.append(
                f"{name}: first use at line {line_number} is not a definition: "
                f"{line.strip()}"
            )
    return errors


def main() -> int:
    errors = audit_document()
    if errors:
        print("SCHEMA_DOC_AUDIT_FAIL")
        for error in errors:
            print(error)
        return 1
    print("SCHEMA_DOC_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
