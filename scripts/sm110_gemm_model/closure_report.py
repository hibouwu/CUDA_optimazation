from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .model import (
    Capacity,
    Hardware,
    Schedule,
    Workload,
    evaluate_manifest,
    precision_specs,
)
from .observations import ObservedBest
from .coverage import (
    CAMPAIGN_COMPONENT_CASE_RESOURCES,
    CAMPAIGN_COMPONENT_RESOURCE_COUNTS,
    CAMPAIGN_COMPUTE_SELECTION,
    CAMPAIGN_FULL_PRECISIONS,
    CAMPAIGN_FULL_SHAPES,
    campaign_measurement_coverage,
    common_resource_coverage,
    precision_coverage,
)


def _rate_text(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "flop/s":
        return f"{value / 1e12:.3f} TFLOP/s"
    if unit == "operation/s":
        return f"{value / 1e12:.3f} TOP/s"
    if unit == "byte/s":
        return f"{value / 1e9:.3f} GB/s"
    if unit == "element/s":
        return f"{value / 1e9:.3f} Gelement/s"
    return f"{value:.6g} {unit}"


def _ratio_text(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


MODELED_RESIDENCIES = ("hot_l2", "cold_hbm")


def _finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values
            if value is not None and math.isfinite(value)]


def _capacity_row(capacity: Capacity) -> dict[str, Any]:
    return {
        "capacity_id": capacity.capacity_id,
        "resource": capacity.resource,
        "rate_per_second": capacity.rate_per_second,
        "rate_unit": f"{capacity.work_unit}/s",
        "evidence_kind": capacity.evidence_kind.value,
        "qualification": capacity.qualification,
        "trial_count": capacity.trial_count,
        "condition": capacity.condition,
        "source_id": capacity.source_id,
        "source_path": capacity.source_path,
        "source_locator": capacity.source_locator,
        "source_url": capacity.source_url,
        "artifact_paths": list(capacity.artifact_paths),
    }


def _deduplicate_findings(
    findings: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in findings:
        normalized = {
            "severity": str(finding["severity"]),
            "code": str(finding["code"]),
            "message": str(finding["message"]),
        }
        key = (
            normalized["severity"],
            normalized["code"],
            normalized["message"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def build_closure_analysis(
    *,
    metadata: dict[str, Any],
    base_capacities: Iterable[Capacity],
    closure_capacities: Iterable[Capacity],
    observations: Iterable[ObservedBest],
    hardware: Hardware,
    schedules: Iterable[Schedule],
    upper_tolerance: float = 0.02,
    empirical_tolerance: float = 0.02,
    require_complete_contract: bool = True,
    input_findings: Iterable[dict[str, str]] = (),
) -> dict[str, Any]:
    if not 0.0 <= upper_tolerance < 1.0:
        raise ValueError("upper_tolerance must be in [0, 1)")
    if not 0.0 <= empirical_tolerance < 1.0:
        raise ValueError("empirical_tolerance must be in [0, 1)")
    base_capacities = list(base_capacities)
    closure_capacities = list(closure_capacities)
    observations = list(observations)
    capacities = [*base_capacities, *closure_capacities]
    schedules = list(schedules)
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = [dict(row) for row in input_findings]
    if require_complete_contract:
        if (metadata.get("schema_version") != 1
                or metadata.get("qualification") != "closure_qualified"
                or metadata.get("closure_qualified") is not True):
            findings.append({
                "severity": "error",
                "code": "closure_metadata_not_qualified",
                "message": "closure import metadata is not schema-1 closure-qualified",
            })
        platform = metadata.get("platform_evidence", {})
        if (platform.get("maxn") is not True
                or platform.get("gpu_clock_locked_hz") != 1_575_000_000):
            findings.append({
                "severity": "error",
                "code": "platform_contract_incomplete",
                "message": "MAXN and 1.575-GHz locked-clock evidence are required",
            })
        model_audit = metadata.get("model_input_audit", {})
        if model_audit.get("pass") is not True:
            findings.append({
                "severity": "error",
                "code": "model_input_audit_not_passed",
                "message": "closure import model_input_audit.pass is not true",
            })
        for finding in model_audit.get("findings", []):
            if isinstance(finding, dict) and {
                    "severity", "code", "message"}.issubset(finding):
                findings.append({
                    "severity": str(finding["severity"]),
                    "code": str(finding["code"]),
                    "message": str(finding["message"]),
                })
        expected_campaign_contract = {
            "compute_selection": CAMPAIGN_COMPUTE_SELECTION,
            "compute_precision_count": len(precision_specs()),
            "compute_case_count": (
                len(precision_specs())
                * len(CAMPAIGN_COMPUTE_SELECTION["n_values"])
            ),
            "component_case_count": sum(
                CAMPAIGN_COMPONENT_RESOURCE_COUNTS.values()),
            "full_gemm_precisions": sorted(CAMPAIGN_FULL_PRECISIONS),
            "full_gemm_shapes": list(CAMPAIGN_FULL_SHAPES),
            "full_gemm_observation_count": (
                len(CAMPAIGN_FULL_PRECISIONS) * len(CAMPAIGN_FULL_SHAPES)),
        }
        if metadata.get("campaign_contract") != expected_campaign_contract:
            findings.append({
                "severity": "error",
                "code": "campaign_contract_mismatch",
                "message": (
                    f"expected={expected_campaign_contract}; "
                    f"actual={metadata.get('campaign_contract')}"),
            })
        independent = metadata.get("independent_audits", {})
        if (set(independent) != {"compute", "component", "full_gemm"}
                or any(row.get("pass") is not True
                       for row in independent.values()
                       if isinstance(row, dict))
                or any(not isinstance(row, dict) for row in independent.values())):
            findings.append({
                "severity": "error",
                "code": "independent_audit_set_incomplete",
                "message": "compute/component/full_gemm independent audits must all pass",
            })

        suite_id = str(metadata.get("suite_id", ""))
        expected_capacity_contract = {
            (
                f"{suite_id}.compute.{precision_id}."
                f"m{CAMPAIGN_COMPUTE_SELECTION['m']}n{n}"
            ): (
                f"{spec.compute_resource}."
                f"m{CAMPAIGN_COMPUTE_SELECTION['m']}n{n}"
            )
            for precision_id, spec in precision_specs().items()
            for n in CAMPAIGN_COMPUTE_SELECTION["n_values"]
        }
        expected_capacity_contract.update({
            f"{suite_id}.component.{case_id}": resource
            for case_id, resource
            in CAMPAIGN_COMPONENT_CASE_RESOURCES.items()
        })
        actual_capacity_contract = {
            row.capacity_id: row.resource for row in closure_capacities
        }
        if (actual_capacity_contract != expected_capacity_contract
                or len(closure_capacities) != len(expected_capacity_contract)):
            findings.append({
                "severity": "error",
                "code": "closure_capacity_matrix_incomplete",
                "message": (
                    f"expected={dict(sorted(expected_capacity_contract.items()))}; "
                    f"actual={dict(sorted(actual_capacity_contract.items()))}"),
            })
        if any(not row.is_closure_qualified for row in closure_capacities):
            findings.append({
                "severity": "error",
                "code": "unqualified_closure_capacity",
                "message": "every closure capacity must be closure_qualified",
            })
        capacity_ids = [row.capacity_id for row in closure_capacities]
        if len(capacity_ids) != len(set(capacity_ids)):
            findings.append({
                "severity": "error",
                "code": "duplicate_closure_capacity_id",
                "message": "closure capacity IDs are not unique",
            })

        expected_pairs = {
            (precision_id, n)
            for precision_id in CAMPAIGN_FULL_PRECISIONS
            for n in CAMPAIGN_FULL_SHAPES
        }
        actual_pairs = {(row.precision_id, row.n) for row in observations}
        if (actual_pairs != expected_pairs or len(observations) != len(expected_pairs)
                or any(row.m != row.n or row.k != row.n for row in observations)):
            findings.append({
                "severity": "error",
                "code": "full_gemm_observation_matrix_incomplete",
                "message": (
                    f"expected={sorted(expected_pairs)}; actual={sorted(actual_pairs)}"),
            })
        if any(not row.is_closure_qualified for row in observations):
            findings.append({
                "severity": "error",
                "code": "unqualified_full_gemm_observation",
                "message": "every full-GEMM observation must be closure_qualified",
            })
        observation_ids = [row.observation_id for row in observations]
        if len(observation_ids) != len(set(observation_ids)):
            findings.append({
                "severity": "error",
                "code": "duplicate_full_gemm_observation_id",
                "message": "full-GEMM observation IDs are not unique",
            })
    for observation in sorted(
        observations, key=lambda row: (row.precision_id, row.n)
    ):
        scenarios: dict[str, dict[str, Any]] = {}
        for residency in MODELED_RESIDENCIES:
            envelope = evaluate_manifest(
                Workload(
                    workload_id=f"{observation.observation_id}.{residency}",
                    m=observation.m,
                    n=observation.n,
                    k=observation.k,
                    precision_id=observation.precision_id,
                    residency=residency,
                ),
                schedules,
                hardware,
                capacities,
            )
            strict = envelope.manifest_conditional_upper
            empirical = envelope.empirical_ideal_envelope
            scenarios[residency] = {
                "conditional_upper_status": strict.status,
                "conditional_upper_per_second": strict.performance_per_second,
                "conditional_bottlenecks": strict.bottlenecks,
                "conditional_resource_seconds": strict.resource_seconds,
                "conditional_conditions": strict.conditions,
                "conditional_schedule_id": envelope.conditional_schedule_id,
                "empirical_status": empirical.status,
                "empirical_ideal_per_second": empirical.performance_per_second,
                "empirical_bottlenecks": empirical.bottlenecks,
                "empirical_resource_seconds": empirical.resource_seconds,
                "empirical_conditions": empirical.conditions,
                "empirical_schedule_id": envelope.empirical_schedule_id,
            }
        strict_rates = _finite(
            scenario["conditional_upper_per_second"]
            for scenario in scenarios.values())
        empirical_rates = _finite(
            scenario["empirical_ideal_per_second"]
            for scenario in scenarios.values())
        if len(strict_rates) != len(MODELED_RESIDENCIES):
            findings.append({
                "severity": "error",
                "code": "residency_conditional_upper_incomplete",
                "message": f"{observation.observation_id}: missing scenario upper",
            })
        if len(empirical_rates) != len(MODELED_RESIDENCIES):
            findings.append({
                "severity": "error",
                "code": "residency_empirical_prediction_incomplete",
                "message": f"{observation.observation_id}: missing scenario prediction",
            })
        for residency, scenario in scenarios.items():
            strict_rate = scenario["conditional_upper_per_second"]
            empirical_rate = scenario["empirical_ideal_per_second"]
            if (strict_rate is not None and empirical_rate is not None
                    and math.isfinite(strict_rate)
                    and math.isfinite(empirical_rate)
                    and empirical_rate > strict_rate * (1.0 + upper_tolerance)):
                findings.append({
                    "severity": "error",
                    "code": "empirical_exceeds_conditional_upper",
                    "message": (
                        f"{observation.observation_id}.{residency}: empirical/upper="
                        f"{empirical_rate / strict_rate:.6f}; capacity semantics, "
                        "work accounting, or upper applicability are inconsistent"),
                })
        # The captured full-GEMM timing is warm-repeated but does not prove
        # whether every matrix access is L2-resident or HBM-served. A safe
        # conditional upper for this union of scenarios is therefore the
        # larger performance upper. The empirical model is reported as a
        # sensitivity interval, not collapsed to an invented residency.
        strict_rate_min = min(strict_rates) if strict_rates else None
        strict_rate_max = max(strict_rates) if strict_rates else None
        empirical_rate_min = min(empirical_rates) if empirical_rates else None
        empirical_rate_max = max(empirical_rates) if empirical_rates else None
        candidate_median_rate = observation.median_per_second
        reference_median_rate = observation.reference_median_per_second
        observed_best_median_rate = max(
            candidate_median_rate,
            (reference_median_rate
             if reference_median_rate is not None else candidate_median_rate),
        )
        candidate_maximum_rate = observation.maximum_per_second
        reference_maximum_rate = observation.reference_maximum_per_second
        observed_best_maximum_rate = max(
            candidate_maximum_rate,
            (reference_maximum_rate
             if reference_maximum_rate is not None else candidate_maximum_rate),
        )
        candidate_median_to_upper = (
            candidate_median_rate / strict_rate_max
            if strict_rate_max is not None else None)
        candidate_maximum_to_upper = (
            candidate_maximum_rate / strict_rate_max
            if strict_rate_max is not None else None)
        observed_best_median_to_upper = (
            observed_best_median_rate / strict_rate_max
            if strict_rate_max is not None else None)
        observed_best_maximum_to_upper = (
            observed_best_maximum_rate / strict_rate_max
            if strict_rate_max is not None else None)
        candidate_to_empirical_min = (
            candidate_median_rate / empirical_rate_max
            if empirical_rate_max is not None else None)
        candidate_to_empirical_max = (
            candidate_median_rate / empirical_rate_min
            if empirical_rate_min is not None else None)
        observed_best_to_empirical_min = (
            observed_best_median_rate / empirical_rate_max
            if empirical_rate_max is not None else None)
        observed_best_to_empirical_max = (
            observed_best_median_rate / empirical_rate_min
            if empirical_rate_min is not None else None)
        if (observed_best_maximum_to_upper is not None
                and observed_best_maximum_to_upper > 1.0 + upper_tolerance):
            findings.append({
                "severity": "error",
                "code": "observed_exceeds_conditional_upper",
                "message": (
                    f"{observation.observation_id}: maximum-trial/upper="
                    f"{observed_best_maximum_to_upper:.6f} above the maximum of "
                    f"hot-L2/cold-HBM conditional uppers"),
            })
        if (observed_best_to_empirical_min is not None
                and observed_best_to_empirical_min > 1.0 + empirical_tolerance):
            findings.append({
                "severity": "warning",
                "code": "observed_exceeds_empirical_envelope",
                "message": (
                    f"{observation.observation_id}: candidate/reference best "
                    "median exceeds both "
                    f"hot-L2/cold-HBM empirical predictions; minimum ratio="
                    f"{observed_best_to_empirical_min:.6f}"),
            })
        rows.append({
            "observation_id": observation.observation_id,
            "precision_id": observation.precision_id,
            "n": observation.n,
            "split": (
                "holdout" if observation.n == 4096
                else "calibration" if observation.n in {1024, 2048}
                else "unexpected"),
            "captured_residency": observation.residency,
            "modeled_residencies": list(MODELED_RESIDENCIES),
            "performance_unit": observation.performance_unit,
            "candidate_backend": observation.backend_id,
            "reference_backend": observation.reference,
            # Preserve the established candidate-only fields while adding
            # explicit observed-best fields that also include the reference.
            "observed_median_per_second": candidate_median_rate,
            "observed_maximum_per_second": candidate_maximum_rate,
            "reference_median_per_second":
                reference_median_rate,
            "reference_maximum_per_second": reference_maximum_rate,
            "observed_best_median_per_second": observed_best_median_rate,
            "observed_best_maximum_per_second": observed_best_maximum_rate,
            "observed_best_backend": (
                observation.backend_id
                if (reference_median_rate is None
                    or candidate_median_rate >= reference_median_rate)
                else observation.reference),
            "observed_to_reference": observation.ratio_of_paired_medians,
            "conditional_upper_min_per_second": strict_rate_min,
            "conditional_upper_max_per_second": strict_rate_max,
            "observed_median_to_conditional_upper": candidate_median_to_upper,
            "observed_maximum_to_conditional_upper": candidate_maximum_to_upper,
            "observed_best_median_to_conditional_upper":
                observed_best_median_to_upper,
            "observed_best_maximum_to_conditional_upper":
                observed_best_maximum_to_upper,
            "empirical_ideal_min_per_second": empirical_rate_min,
            "empirical_ideal_max_per_second": empirical_rate_max,
            "observed_to_empirical_ideal_min": candidate_to_empirical_min,
            "observed_to_empirical_ideal_max": candidate_to_empirical_max,
            "observed_best_to_empirical_ideal_min":
                observed_best_to_empirical_min,
            "observed_best_to_empirical_ideal_max":
                observed_best_to_empirical_max,
            "residency_scenarios": scenarios,
            "conditional_statuses": {
                residency: scenario["conditional_upper_status"]
                for residency, scenario in scenarios.items()
            },
            "empirical_statuses": {
                residency: scenario["empirical_status"]
                for residency, scenario in scenarios.items()
            },
        })

    capacity_rows = [
        _capacity_row(capacity)
        for capacity in sorted(
            closure_capacities, key=lambda row: (row.resource, row.capacity_id))
    ]
    base_capacity_rows = [
        _capacity_row(capacity)
        for capacity in sorted(
            base_capacities, key=lambda row: (row.resource, row.capacity_id))
    ]

    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    finite_holdout = [
        row for row in holdout_rows
        if row["observed_to_empirical_ideal_min"] is not None
        and math.isfinite(row["observed_to_empirical_ideal_min"])
    ]
    findings = _deduplicate_findings(findings)
    precision_rows = precision_coverage(capacities, observations)
    common_coverage = common_resource_coverage(capacities)
    campaign_coverage = campaign_measurement_coverage(
        closure_capacities, observations)
    return {
        "schema_version": 1,
        "pass": not any(
            finding["severity"] == "error" for finding in findings),
        "suite_id": metadata.get("suite_id"),
        "expected_commit": metadata.get("expected_commit"),
        "composition": metadata.get("composition", "single_suite"),
        "campaign_sources": metadata.get("campaign_sources", {}),
        "qualification": metadata.get("qualification"),
        "platform_evidence": metadata.get("platform_evidence", {}),
        "model_input_audit": metadata.get("model_input_audit", {}),
        "capacity_count": len(capacity_rows),
        "base_capacity_count": len(base_capacity_rows),
        "observation_count": len(rows),
        "holdout_count": len(holdout_rows),
        "holdout_with_empirical_prediction_count": len(finite_holdout),
        "upper_tolerance_fraction": upper_tolerance,
        "empirical_tolerance_fraction": empirical_tolerance,
        "precision_coverage": [row.to_dict() for row in precision_rows],
        "common_resource_coverage": common_coverage,
        "campaign_measurement_coverage": campaign_coverage,
        "all_precisions_closed": all(
            row.numeric_closure for row in precision_rows),
        "all_common_resources_closed": all(common_coverage.values()),
        "findings": findings,
        "capacities": capacity_rows,
        "base_capacities": base_capacity_rows,
        "observations": rows,
    }


def render_closure_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Thor/SM110 GEMM closure 数值摘要",
        "",
        f"- suite：`{analysis.get('suite_id')}`",
        f"- commit：`{analysis.get('expected_commit')}`",
        f"- composition：`{analysis.get('composition')}`",
        "- campaign sources："
        f"`{json.dumps(analysis.get('campaign_sources', {}), sort_keys=True)}`",
        f"- qualification：`{analysis.get('qualification')}`",
        f"- audit pass：`{analysis.get('pass')}`",
        "- campaign measurement closed："
        f"`{analysis.get('campaign_measurement_coverage', {}).get('all_campaign_measurements_closed')}`",
        f"- all precisions closed：`{analysis.get('all_precisions_closed')}`",
        "- all common resources closed："
        f"`{analysis.get('all_common_resources_closed')}`",
        f"- capacity：{analysis.get('capacity_count')} 项",
        f"- base/profile capacity：{analysis.get('base_capacity_count')} 项",
        f"- full-GEMM observation：{analysis.get('observation_count')} 项",
        f"- overcurrent delta：`{json.dumps(analysis.get('platform_evidence', {}).get('overcurrent_deltas', {}), sort_keys=True)}`",
        "",
        "## Closure-qualified compute/component capacities",
        "",
        "| Resource | Case | Median rate | Trials | Evidence | Qualification | Source |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in analysis.get("capacities", []):
        lines.append(
            f"| `{row['resource']}` | `{row['capacity_id']}` | "
            f"{_rate_text(row['rate_per_second'], row['rate_unit'])} | "
            f"{row['trial_count']} | `{row['evidence_kind']}` | "
            f"`{row['qualification']}` | `{row['source_path']}` |")

    lines.extend([
        "",
        "## Base/profile capacities",
        "",
        "这些参数参与严格上界或 HBM/L2 经验场景，但没有因本次 closure 自动升级；"
        "其 `snapshot_only`/`profiler_model_peak` 等证据等级必须保留。",
        "",
        "| Resource | Case | Rate | Evidence | Qualification | Source |",
        "| --- | --- | ---: | --- | --- | --- |",
    ])
    for row in analysis.get("base_capacities", []):
        lines.append(
            f"| `{row['resource']}` | `{row['capacity_id']}` | "
            f"{_rate_text(row['rate_per_second'], row['rate_unit'])} | "
            f"`{row['evidence_kind']}` | `{row['qualification']}` | "
            f"`{row['source_path']}` |")

    lines.extend([
        "",
        "## Full-GEMM 与模型",
        "",
        "1024/2048 是预声明的 calibration，4096 是 holdout；该划分不证明 cache "
        "residency。报告同时计算 hot-L2 和 cold-HBM：严格上界采用两者中更松的 "
        "performance upper，经验包络保留两场景区间。",
        f"条件上界反证容差为 {100.0 * analysis.get('upper_tolerance_fraction', 0):.2f}%，"
        f"经验重校准容差为 {100.0 * analysis.get('empirical_tolerance_fraction', 0):.2f}%。",
        "",
        "| Precision | N | Split | Candidate | Candidate median | Reference | "
        "Reference median | Observed-best backend | Cand/ref | "
        "Upper status (L2/HBM) | Conditional upper range | "
        "Candidate median/max upper | Observed-best max trial/max upper | "
        "Empirical range | Candidate/empirical | Observed-best/empirical |",
        "| --- | ---: | --- | --- | ---: | --- | ---: | --- | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in analysis.get("observations", []):
        unit = row["performance_unit"]
        lines.append(
            f"| `{row['precision_id']}` | {row['n']} | "
            f"{row['split']} | "
            f"`{row['candidate_backend']}` | "
            f"{_rate_text(row['observed_median_per_second'], unit)} | "
            f"`{row['reference_backend']}` | "
            f"{_rate_text(row['reference_median_per_second'], unit)} | "
            f"`{row['observed_best_backend']}` | "
            f"{_ratio_text(row['observed_to_reference'])} | "
            f"`{row['conditional_statuses']['hot_l2']}/"
            f"{row['conditional_statuses']['cold_hbm']}` | "
            f"{_rate_text(row['conditional_upper_min_per_second'], unit)}–"
            f"{_rate_text(row['conditional_upper_max_per_second'], unit)} | "
            f"{_ratio_text(row['observed_median_to_conditional_upper'])} | "
            f"{_ratio_text(row['observed_best_maximum_to_conditional_upper'])} | "
            f"{_rate_text(row['empirical_ideal_min_per_second'], unit)}–"
            f"{_rate_text(row['empirical_ideal_max_per_second'], unit)} | "
            f"{_ratio_text(row['observed_to_empirical_ideal_min'])}–"
            f"{_ratio_text(row['observed_to_empirical_ideal_max'])} | "
            f"{_ratio_text(row['observed_best_to_empirical_ideal_min'])}–"
            f"{_ratio_text(row['observed_best_to_empirical_ideal_max'])} |")

    lines.extend(["", "## Findings", ""])
    findings = analysis.get("findings", [])
    if findings:
        for finding in findings:
            lines.append(
                f"- **{finding['severity']} `{finding['code']}`**："
                f"{finding['message']}")
    else:
        lines.append("- 没有条件上界违规或经验包络重校准 warning。")
    return "\n".join(lines) + "\n"


def write_closure_report(
    analysis: dict[str, Any], *, json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_closure_markdown(analysis))
