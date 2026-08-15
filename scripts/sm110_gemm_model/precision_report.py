from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .coverage import PrecisionCoverage, precision_coverage
from .model import (
    CAUSAL_PIPELINE_DAG_IMPLEMENTED,
    Capacity,
    EvidenceKind,
    Hardware,
    ModelError,
    PipelineProfile,
    Schedule,
    Workload,
    evaluate_manifest,
    precision_specs,
)
from .observations import ObservedBest


SUPPORT_READY = "ready_for_closure_campaign"
REQUIRED_RESIDENCIES = ("hot_l2", "cold_hbm")


def _empirical_capacity_selection_is_closure_qualified(layer: Any) -> bool:
    empirical_resources = [
        resource
        for resource, kind in layer.selected_capacity_evidence_kinds.items()
        if kind in {
            EvidenceKind.MEASURED_SUSTAINED.value,
            EvidenceKind.MEASURED_JOINT.value,
        }
    ]
    return bool(empirical_resources) and all(
        layer.selected_capacity_qualifications.get(resource)
        == "closure_qualified"
        for resource in empirical_resources
    )


def _existing_source_paths(
    row: dict[str, Any], *, repo_root: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    implementation = row.get("implementation") or {}
    declared = tuple(str(path) for path in implementation.get("source_paths", ()))
    missing = tuple(path for path in declared if not (repo_root / path).is_file())
    return declared, missing


def build_precision_evidence_analysis(
    *,
    capacities: Iterable[Capacity],
    observations: Iterable[ObservedBest],
    support_manifest: dict[str, Any],
    repo_root: Path,
    hardware: Hardware | None = None,
    schedules: Iterable[Schedule] = (),
    pipeline_profiles: Iterable[PipelineProfile] = (),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge executable numeric coverage with the implementation support map.

    ``precision_coverage`` answers whether the required evidence exists.  The
    support manifest answers whether a reproducible full-GEMM implementation,
    correctness reference, and same-contract performance denominator are ready
    to produce that evidence.  Neither source is allowed to silently stand in
    for the other.
    """

    metadata = dict(metadata or {})
    capacities = list(capacities)
    observations = list(observations)
    specs = precision_specs()
    support_rows = support_manifest.get("precisions")
    if not isinstance(support_rows, list):
        raise ModelError("support manifest precisions must be a list")
    support_by_id: dict[str, dict[str, Any]] = {}
    for raw in support_rows:
        if not isinstance(raw, dict):
            raise ModelError("support manifest precision row must be an object")
        precision_id = str(raw.get("precision_id", ""))
        if not precision_id or precision_id in support_by_id:
            raise ModelError(
                f"support manifest has empty or duplicate precision ID: "
                f"{precision_id!r}"
            )
        support_by_id[precision_id] = raw
    if set(support_by_id) != set(specs):
        raise ModelError(
            "support manifest precision IDs differ from model contract: "
            f"have={sorted(support_by_id)} expected={sorted(specs)}"
        )

    numeric_by_id = {
        row.precision_id: row
        for row in precision_coverage(capacities, observations)
    }
    schedules = list(schedules)
    pipeline_profiles = list(pipeline_profiles)
    rows: list[dict[str, Any]] = []
    for precision_id in specs:
        numeric: PrecisionCoverage = numeric_by_id[precision_id]
        support = support_by_id[precision_id]
        implementation = support.get("implementation") or {}
        candidate_backend_ids = tuple(
            str(value)
            for value in implementation.get("closure_candidate_backend_ids", ())
        )
        source_paths, missing_source_paths = _existing_source_paths(
            support, repo_root=repo_root
        )
        reference = support.get("numerical_reference") or {}
        denominator = support.get("performance_denominator") or {}
        reference_source_path = reference.get("source_path")
        reference_source_exists = bool(
            reference_source_path
            and (repo_root / str(reference_source_path)).is_file()
        )
        numerical_reference_ready = bool(
            reference.get("same_input_precision") is True
            and reference.get("same_output_type") is True
            and reference_source_exists
        )
        performance_denominator_ready = bool(
            denominator.get("same_precision") is True
            and denominator.get("status") == "ready"
            and denominator.get("backend_id")
        )
        implementation_ready = bool(
            support.get("status") == SUPPORT_READY
            and support.get("native_mainloop") is True
            and candidate_backend_ids
            and source_paths
            and not missing_source_paths
            and numerical_reference_ready
            and performance_denominator_ready
            and not support.get("blockers")
        )
        support_gaps: list[str] = []
        if support.get("status") != SUPPORT_READY:
            support_gaps.append("implementation_status")
        if support.get("native_mainloop") is not True:
            support_gaps.append("native_mainloop")
        if not candidate_backend_ids:
            support_gaps.append("closure_candidate_backend")
        if not source_paths:
            support_gaps.append("implementation_source")
        if missing_source_paths:
            support_gaps.append("missing_implementation_source")
        if not numerical_reference_ready:
            support_gaps.append("same_contract_numerical_reference")
        if not performance_denominator_ready:
            support_gaps.append("same_precision_performance_denominator_impl")

        envelope_scenarios: list[dict[str, Any]] = []
        for n in numeric.required_full_gemm_shapes:
            for residency in REQUIRED_RESIDENCIES:
                scenario_id = f"n{n}.{residency}"
                if hardware is None or not schedules:
                    envelope_scenarios.append({
                        "scenario_id": scenario_id,
                        "n": n,
                        "residency": residency,
                        "status": "model_inputs_missing",
                        "schedule_id": None,
                        "performance_per_second": None,
                        "selected_capacity_ids": {},
                        "selected_capacity_qualifications": {},
                        "missing_resources": [
                            "hardware_or_schedule_manifest"
                        ],
                        "closure_qualified": False,
                        "causal_status": "model_inputs_missing",
                        "causal_schedule_id": None,
                        "causal_closure_qualified": False,
                        "ideal_status": "model_inputs_missing",
                        "ideal_schedule_id": None,
                        "ideal_performance_per_second": None,
                        "ideal_closure_qualified": False,
                    })
                    continue
                envelope = evaluate_manifest(
                    Workload(
                        workload_id=f"{precision_id}.{scenario_id}",
                        m=n,
                        n=n,
                        k=n,
                        precision_id=precision_id,
                        residency=residency,
                    ),
                    schedules,
                    hardware,
                    capacities,
                    pipeline_profiles,
                )
                resource_layer = (
                    envelope.manifest_empirical_resource_envelope
                )
                causal_layer = envelope.causal_pipeline_envelope
                ideal_layer = envelope.empirical_ideal_envelope
                resource_qualified = bool(
                    resource_layer.status == "ok"
                    and resource_layer.performance_per_second is not None
                    and _empirical_capacity_selection_is_closure_qualified(
                        resource_layer
                    )
                )
                causal_qualified = bool(
                    causal_layer.status == "ok"
                    and causal_layer.performance_per_second is not None
                    and _empirical_capacity_selection_is_closure_qualified(
                        causal_layer
                    )
                )
                ideal_qualified = bool(
                    ideal_layer.status == "ok"
                    and ideal_layer.performance_per_second is not None
                    and _empirical_capacity_selection_is_closure_qualified(
                        ideal_layer
                    )
                )
                envelope_scenarios.append({
                    "scenario_id": scenario_id,
                    "n": n,
                    "residency": residency,
                    "status": resource_layer.status,
                    "schedule_id": envelope.empirical_resource_schedule_id,
                    "performance_per_second":
                        resource_layer.performance_per_second,
                    "selected_capacity_ids":
                        resource_layer.selected_capacity_ids,
                    "selected_capacity_qualifications":
                        resource_layer.selected_capacity_qualifications,
                    "missing_resources": resource_layer.missing_resources,
                    "closure_qualified": resource_qualified,
                    "causal_status": causal_layer.status,
                    "causal_schedule_id":
                        envelope.causal_pipeline_schedule_id,
                    "causal_performance_per_second":
                        causal_layer.performance_per_second,
                    "causal_selected_profile_ids":
                        causal_layer.selected_capacity_ids,
                    "causal_missing_resources":
                        causal_layer.missing_resources,
                    "causal_closure_qualified": causal_qualified,
                    "ideal_status": ideal_layer.status,
                    "ideal_schedule_id": envelope.empirical_schedule_id,
                    "ideal_performance_per_second":
                        ideal_layer.performance_per_second,
                    "ideal_missing_resources": ideal_layer.missing_resources,
                    "ideal_closure_qualified": ideal_qualified,
                })
        qualified_envelope_scenarios = tuple(
            row["scenario_id"]
            for row in envelope_scenarios
            if row["closure_qualified"]
        )
        missing_envelope_scenarios = tuple(
            row["scenario_id"]
            for row in envelope_scenarios
            if not row["closure_qualified"]
        )
        resource_envelope_matrix_complete = (
            bool(envelope_scenarios) and not missing_envelope_scenarios
        )
        qualified_causal_scenarios = tuple(
            row["scenario_id"]
            for row in envelope_scenarios
            if row["causal_closure_qualified"]
        )
        missing_causal_scenarios = tuple(
            row["scenario_id"]
            for row in envelope_scenarios
            if not row["causal_closure_qualified"]
        )
        causal_pipeline_model_complete = bool(
            CAUSAL_PIPELINE_DAG_IMPLEMENTED
            and envelope_scenarios
            and not missing_causal_scenarios
        )
        qualified_ideal_scenarios = tuple(
            row["scenario_id"]
            for row in envelope_scenarios
            if row["ideal_closure_qualified"]
        )
        missing_ideal_scenarios = tuple(
            row["scenario_id"]
            for row in envelope_scenarios
            if not row["ideal_closure_qualified"]
        )
        empirical_ideal_matrix_complete = bool(
            envelope_scenarios and not missing_ideal_scenarios
        )
        model_gaps: list[str] = []
        if not resource_envelope_matrix_complete:
            model_gaps.append(
                "closure_qualified_empirical_envelope_matrix"
            )
        if not causal_pipeline_model_complete:
            model_gaps.append("closure_qualified_causal_pipeline_profile_matrix")
        if not empirical_ideal_matrix_complete:
            model_gaps.append("integrated_empirical_ideal_envelope_matrix")
        end_to_end_closed = bool(
            implementation_ready
            and numeric.numeric_closure
            and resource_envelope_matrix_complete
            and causal_pipeline_model_complete
            and empirical_ideal_matrix_complete
        )
        rows.append({
            "precision_id": precision_id,
            "input_type": support.get("input_type"),
            "accumulator_type": support.get("accumulator_type"),
            "output_type": support.get("output_type"),
            "work_unit": support.get("work_unit"),
            "implementation_status": support.get("status"),
            "native_mainloop": support.get("native_mainloop") is True,
            "candidate_backend_ids": list(candidate_backend_ids),
            "source_paths": list(source_paths),
            "missing_source_paths": list(missing_source_paths),
            "numerical_reference_backend": reference.get("backend_id"),
            "numerical_reference_ready": numerical_reference_ready,
            "performance_denominator_backend": denominator.get("backend_id"),
            "performance_denominator_ready": performance_denominator_ready,
            "implementation_ready": implementation_ready,
            "support_gaps": support_gaps,
            "support_blockers": list(support.get("blockers") or ()),
            "numeric_evidence": numeric.to_dict(),
            "numeric_closure": numeric.numeric_closure,
            "required_empirical_envelope_scenarios": [
                f"n{n}.{residency}"
                for n in numeric.required_full_gemm_shapes
                for residency in REQUIRED_RESIDENCIES
            ],
            "qualified_empirical_envelope_scenarios":
                list(qualified_envelope_scenarios),
            "missing_empirical_envelope_scenarios":
                list(missing_envelope_scenarios),
            "empirical_envelope_scenarios": envelope_scenarios,
            "resource_envelope_matrix_complete":
                resource_envelope_matrix_complete,
            "causal_pipeline_model_complete":
                causal_pipeline_model_complete,
            "qualified_causal_pipeline_scenarios":
                list(qualified_causal_scenarios),
            "missing_causal_pipeline_scenarios":
                list(missing_causal_scenarios),
            "empirical_ideal_matrix_complete":
                empirical_ideal_matrix_complete,
            "qualified_empirical_ideal_scenarios":
                list(qualified_ideal_scenarios),
            "missing_empirical_ideal_scenarios":
                list(missing_ideal_scenarios),
            "model_gaps": model_gaps,
            "end_to_end_closed": end_to_end_closed,
        })

    return {
        "schema_version": 3,
        "suite_id": metadata.get("suite_id"),
        "expected_commit": metadata.get("expected_commit"),
        "composition": metadata.get("composition"),
        "campaign_sources": metadata.get("campaign_sources", {}),
        "qualification": metadata.get("qualification"),
        "support_manifest_schema_version": support_manifest.get("schema_version"),
        "precision_count": len(rows),
        "implementation_ready_count": sum(
            bool(row["implementation_ready"]) for row in rows
        ),
        "numeric_closed_count": sum(bool(row["numeric_closure"]) for row in rows),
        "resource_envelope_closed_count": sum(
            bool(row["resource_envelope_matrix_complete"]) for row in rows
        ),
        "causal_pipeline_closed_count": sum(
            bool(row["causal_pipeline_model_complete"]) for row in rows
        ),
        "empirical_ideal_closed_count": sum(
            bool(row["empirical_ideal_matrix_complete"]) for row in rows
        ),
        "end_to_end_closed_count": sum(
            bool(row["end_to_end_closed"]) for row in rows
        ),
        "all_precision_implementations_ready": all(
            bool(row["implementation_ready"]) for row in rows
        ),
        "all_precision_numeric_evidence_closed": all(
            bool(row["numeric_closure"]) for row in rows
        ),
        "all_precision_resource_envelopes_closed": all(
            bool(row["resource_envelope_matrix_complete"]) for row in rows
        ),
        "causal_pipeline_dag_implemented":
            CAUSAL_PIPELINE_DAG_IMPLEMENTED,
        "all_precisions_end_to_end_closed": all(
            bool(row["end_to_end_closed"]) for row in rows
        ),
        "precisions": rows,
    }


def _mark(value: bool) -> str:
    return "yes" if value else "NO"


def render_precision_evidence_markdown(analysis: dict[str, Any]) -> str:
    campaign_sources = analysis.get("campaign_sources") or {}
    base_source = campaign_sources.get("base") or {}
    component_source = campaign_sources.get("component_supplement") or {}
    lines = [
        "# Thor/SM110 全精度 GEMM 证据矩阵",
        "",
        "本表由可执行模型的 numeric coverage 与 full-GEMM support manifest 合并生成。",
        "`implementation_ready` 只表示实现、数值参考和同精度性能 denominator 已经",
        "具备采集条件；`numeric_closure` 只表示所需 Thor 数值证据已经回传并通过",
        "审计。最终 `end_to_end_closed` 还要求六个 residency/shape resource",
        "envelope 只使用精确合同的 closure-qualified capacity，并要求 causal",
        "pipeline DAG 已实现和闭环。",
        "",
        f"- closure suite：`{analysis.get('suite_id')}`",
        f"- composition：`{analysis.get('composition')}`",
        "- base compute/full-GEMM："
        f"`{base_source.get('suite_id')}` @ "
        f"`{base_source.get('expected_commit')}`",
        "- component supplement："
        f"`{component_source.get('suite_id')}` @ "
        f"`{component_source.get('expected_commit')}`",
        f"- composite qualification：`{analysis.get('qualification')}`",
        f"- precision count：`{analysis.get('precision_count')}`",
        f"- implementation ready：`{analysis.get('implementation_ready_count')}`",
        f"- numeric closed：`{analysis.get('numeric_closed_count')}`",
        "- closure-qualified resource envelopes："
        f"`{analysis.get('resource_envelope_closed_count')}`",
        "- causal pipeline closed："
        f"`{analysis.get('causal_pipeline_closed_count')}`",
        "- integrated empirical ideal envelopes："
        f"`{analysis.get('empirical_ideal_closed_count')}`",
        f"- end-to-end closed：`{analysis.get('end_to_end_closed_count')}`",
        "- all precisions end-to-end closed："
        f"`{str(bool(analysis.get('all_precisions_end_to_end_closed'))).lower()}`",
        "",
        "| precision | strict upper | compute shapes | implementation | full-GEMM shapes | numerical | denominator | resource envelope | causal DAG | integrated ideal | end-to-end |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis.get("precisions", []):
        numeric = row["numeric_evidence"]
        compute_shapes = (
            f"{len(numeric['closure_qualified_compute_shapes'])}/"
            f"{len(numeric['required_compute_shapes'])}"
        )
        full_shapes = (
            f"{len(numeric['closure_qualified_full_gemm_shapes'])}/"
            f"{len(numeric['required_full_gemm_shapes'])}"
        )
        lines.append(
            "| "
            f"`{row['precision_id']}` | "
            f"{_mark(numeric['strict_compute_upper'])} | "
            f"{compute_shapes} | "
            f"{_mark(row['implementation_ready'])} | "
            f"{full_shapes} | "
            f"{_mark(numeric['full_gemm_numerical_validation_complete'])} | "
            f"{_mark(numeric['same_precision_performance_denominator'])} | "
            f"{_mark(row['resource_envelope_matrix_complete'])} | "
            f"{_mark(row['causal_pipeline_model_complete'])} | "
            f"{_mark(row['empirical_ideal_matrix_complete'])} | "
            f"{_mark(row['end_to_end_closed'])} |"
        )

    lines.extend(["", "## 未闭环项", ""])
    incomplete = [
        row for row in analysis.get("precisions", [])
        if not row["end_to_end_closed"]
    ]
    if not incomplete:
        lines.append("全部声明精度均已端到端闭环。")
    for row in incomplete:
        numeric = row["numeric_evidence"]
        lines.extend([
            f"### `{row['precision_id']}`",
            "",
            f"- support gaps：`{', '.join(row['support_gaps']) or 'none'}`",
            f"- numeric gaps：`{', '.join(numeric['missing']) or 'none'}`",
            f"- model gaps：`{', '.join(row['model_gaps']) or 'none'}`",
            "- missing compute shapes："
            f"`{', '.join(numeric['missing_compute_shapes']) or 'none'}`",
            "- missing full-GEMM shapes："
            f"`{', '.join(str(v) for v in numeric['missing_full_gemm_shapes']) or 'none'}`",
            "- missing empirical envelope scenarios："
            f"`{', '.join(row['missing_empirical_envelope_scenarios']) or 'none'}`",
            "- missing causal profile scenarios："
            f"`{', '.join(row['missing_causal_pipeline_scenarios']) or 'none'}`",
            "- missing integrated ideal scenarios："
            f"`{', '.join(row['missing_empirical_ideal_scenarios']) or 'none'}`",
        ])
        for blocker in row.get("support_blockers", []):
            lines.append(f"- blocker：{blocker}")
        lines.append("")

    lines.extend([
        "## 关闭条件",
        "",
        "每个 precision 必须同时满足：",
        "",
        "1. 有条件可证明的 compute rate upper；",
        "2. M128N64、M128N128、M128N256 三个 closure-qualified compute 点；",
        "3. 有仓库内可复现的 native full-GEMM candidate；",
        "4. N=1024、2048、4096 三个完整输出数值验证；",
        "5. 三个 shape 都有同输入精度、同输出类型的 performance denominator；",
        "6. hot-L2/cold-HBM × N=1024/2048/4096 六个 resource envelope 都只选择",
        "   closure-qualified 且与 schedule 显式匹配的 capacity；",
        "7. latency、initiation interval、TMA/MMA/TMEM 依赖和 startup/drain 的",
        "   causal pipeline DAG 已实现，并且每个选中 schedule 有独立审计通过的",
        "   closure-qualified joint profile；",
        "8. trial、源码、编译命令、binary hash、function-scoped SASS、NCU、环境和",
        "   硬件身份通过独立 auditor。",
        "",
    ])
    return "\n".join(lines)
