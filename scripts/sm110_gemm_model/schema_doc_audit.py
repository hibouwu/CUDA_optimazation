#!/usr/bin/env python3
"""Audit current split GEMM documentation against the executable schema."""

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
GEMM_DOC_ROOT = ROOT / "Docs/blackwell_tensorcore/gemm"
DOCUMENT = GEMM_DOC_ROOT / "appendices/schema_reference.md"

CANONICAL_DOCUMENTS = (
    GEMM_DOC_ROOT / "README.md",
    *(GEMM_DOC_ROOT / "model" / name for name in (
        "01_scope_and_claims.md",
        "02_symbols_units_and_workload.md",
        "03_work_accounting.md",
        "04_strict_performance_upper.md",
        "05_empirical_resource_envelope.md",
        "06_causal_pipeline_model.md",
        "07_observed_gemm_and_falsification.md",
        "08_current_coverage_and_gaps.md",
    )),
    *(GEMM_DOC_ROOT / "experiments" / name for name in (
        "EXP-01-compute-surface.md",
        "EXP-02-l2-physical-bounds.md",
        "EXP-03-tma-payload-surface.md",
        "EXP-04-memory-duplex-surface.md",
        "EXP-05-exact-tma-topology.md",
        "EXP-06-tmem-readback-and-scale.md",
        "EXP-07-causal-pipeline.md",
        "EXP-08-full-gemm-validation.md",
    )),
    *(GEMM_DOC_ROOT / "appendices" / name for name in (
        "schema_reference.md",
        "microbenchmark_sources.md",
        "current_model_replay.md",
        "historical_results.md",
        "audit_and_reproduction.md",
    )),
    *(GEMM_DOC_ROOT / "tutorial" / name for name in (
        "README.md",
        "01_fp16_n2048_worked_example.md",
        "02_common_failure_modes.md",
    )),
)

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


def audit_canonical_layout() -> list[str]:
    errors: list[str] = []
    for path in CANONICAL_DOCUMENTS:
        if not path.is_file():
            errors.append(f"missing canonical document: {path.relative_to(ROOT)}")

    experiment_dir = GEMM_DOC_ROOT / "experiments"
    for path in sorted(experiment_dir.glob("EXP-*.md")):
        text = path.read_text(encoding="utf-8")
        for token in ("研究问题", "对应模型", "不能证明什么", "源码与工件"):
            if token not in text:
                errors.append(
                    f"experiment document lacks required section {token}: "
                    f"{path.name}"
                )

    model_requirements = {
        "03_work_accounting.md": (
            "W_{\\mathrm{use}}",
            "Q_{\\mathrm{TMA,issued}}",
            "tma_a_scale_bytes",
            "512 B",
            "1024 B",
        ),
        "04_strict_performance_upper.md": (
            "T_r^{\\mathrm{LB}}",
            "P_{\\mathrm{ub}}",
            "1024\\ \\mathrm{B/cycle/GPU}",
            "512\\ \\mathrm{B/cycle/GPU}",
            "domain_conditional_upper",
        ),
        "06_causal_pipeline_model.md": (
            "\\lambda_J",
            "\\iota_J",
            "T_{\\mathrm{worker}}",
            "empirical_ideal_envelope",
        ),
        "07_observed_gemm_and_falsification.md": (
            "P_{\\mathrm{obs}}",
            "independent_same_contract",
            "absolute_three_layer_closure",
        ),
    }
    for name, tokens in model_requirements.items():
        path = GEMM_DOC_ROOT / "model" / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(f"model document {name} lacks contract token: {token}")

    empirical_path = GEMM_DOC_ROOT / "model/05_empirical_resource_envelope.md"
    if empirical_path.is_file():
        empirical = empirical_path.read_text(encoding="utf-8")
        for token in (
            'resource="hbm.duplex"',
            "`hbm.duplex.proxy`",
            "`l2.duplex`",
            "精确 read:write ratio",
            "insufficient_evidence",
        ):
            if token not in empirical:
                errors.append(
                    f"empirical model document lacks current semantic token: {token}"
                )
        for stale in (
            "经验层同时使用方向独立的 `hbm.read`/`hbm.write`",
            "empirical `hbm.read`",
            "empirical `l2.read`",
        ):
            if stale in empirical:
                errors.append(
                    f"empirical model document retains legacy semantic: {stale}"
                )

    replay_path = GEMM_DOC_ROOT / "appendices/current_model_replay.md"
    if replay_path.is_file():
        replay = replay_path.read_text(encoding="utf-8")
        for token in (
            "f06f2cd917a4cb23806b5e1be06120be9152ed7b",
            "aa845dd9e70e2c541ae3a7d5293bf8de4bd55092",
            "physical HBM duplex",
            "target completion",
        ):
            if token not in replay:
                errors.append(f"current replay lacks identity/status token: {token}")

    legacy_documents = (
        ROOT / "Docs/blackwell_tensorcore/thor_sm110_gemm_performance_bounds.md",
        ROOT / "Docs/blackwell_tensorcore/thor_sm110_gemm_performance_model_tutorial.md",
        ROOT / "Docs/blackwell_tensorcore/thor_sm110_current_model_replay.md",
    )
    for path in legacy_documents:
        if path.is_file() and "notice（2026-08-18）" not in path.read_text(
            encoding="utf-8"
        ):
            errors.append(f"legacy document lacks migration notice: {path.name}")
    return errors


def audit_local_links() -> list[str]:
    errors: list[str] = []
    for document in CANONICAL_DOCUMENTS:
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8")
        for raw_target in re.findall(r"\]\(([^)\n]+)\)", text):
            target = raw_target.strip().strip("<>")
            if (
                not target
                or target.startswith(("http://", "https://", "#"))
                or "$" in target
                or "<" in target
            ):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                errors.append(
                    f"broken local link: {document.relative_to(ROOT)} -> {target}"
                )
    return errors


def main() -> int:
    errors = [
        *audit_document(),
        *audit_canonical_layout(),
        *audit_local_links(),
    ]
    if errors:
        print("SCHEMA_DOC_AUDIT_FAIL")
        for error in errors:
            print(error)
        return 1
    print("SCHEMA_DOC_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
