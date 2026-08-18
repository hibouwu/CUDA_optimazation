from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .model import (
    Capacity, Hardware, PipelineProfile, Schedule, Workload,
    aggregate_compute_resource, capacity_applies_to_hardware,
    evaluate_manifest, precision_specs,
)
from .observations import ObservedBest


@dataclass(frozen=True)
class PrecisionCoverage:
    precision_id: str
    implementation_domain: str
    domain_compute_upper: bool
    empirical_compute_rate: bool
    closure_qualified_compute_rate: bool
    full_gemm_observed: bool
    closure_qualified_full_gemm: bool
    same_precision_performance_denominator: bool
    numeric_closure: bool
    calibration_scenario_closure: bool
    holdout_scenario_closure: bool
    absolute_three_layer_closure: bool
    same_precision_ratio_closure: bool
    evidence_missing: tuple[str, ...]
    missing: tuple[str, ...]
    comparison_missing: tuple[str, ...]
    strict_compute_upper: bool
    required_compute_shapes: tuple[str, ...]
    closure_qualified_compute_shapes: tuple[str, ...]
    compute_shape_matrix_complete: bool
    required_full_gemm_shapes: tuple[int, ...]
    observed_full_gemm_shapes: tuple[int, ...]
    closure_qualified_full_gemm_shapes: tuple[int, ...]
    full_gemm_shape_matrix_complete: bool
    full_gemm_numerical_validation_complete: bool
    missing_compute_shapes: tuple[str, ...]
    missing_full_gemm_shapes: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioCoverage:
    workload_id: str
    precision_id: str
    validation_split: str
    implementation_domain: str
    conditional_upper_numeric: bool
    conditional_upper_complete: bool
    closure_qualified_empirical_envelope: bool
    contract_matched_full_gemm: bool
    scenario_aligned_full_gemm: bool
    numeric_closure: bool
    upper_consistent: bool
    empirical_envelope_consistent: bool
    absolute_three_layer_closure: bool
    same_precision_ratio_closure: bool
    empirical_schedule_id: str | None
    conditional_upper_per_second: float | None
    empirical_envelope_per_second: float | None
    observed_median_per_second: float | None
    performance_unit: str
    empirical_to_observed_ratio: float | None
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadManifestCoverage:
    precision_id: str
    implementation_domain: str
    calibration_workload_ids: tuple[str, ...]
    holdout_workload_ids: tuple[str, ...]
    complete: bool
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


COMMON_REQUIRED_EMPIRICAL_RESOURCES = (
    "hbm.duplex",
    "l2.duplex",
    "hbm.read",
    "hbm.write",
    "l2.read",
    "l2.write",
    "tma.hbm",
    "tma.hbm.inflight4",
    "tma.smem_ingress.per_sm",
    "tma.smem_ingress.per_sm.inflight4",
    "tmem.scale_ingress",
    "tmem.readback",
)

CAMPAIGN_FULL_PRECISIONS = (
    "fp16_f32",
    "bf16_f32",
    "tf32_f32",
    "e4m3_f32",
    "s8_s32",
)

CAMPAIGN_COMPONENT_CASE_RESOURCES = {
    "tma_l2_hit_32k": "tma.smem_ingress.diagnostic.serial32k.per_sm",
    "tma_dram_stream_32k": "tma.hbm.diagnostic.serial32k",
    "tma_l2_hit_32k_inflight4":
        "tma.smem_ingress.per_sm.inflight4",
    "tma_dram_stream_32k_inflight4":
        "tma.hbm.inflight4",
    "tma_l2_hit_tc5a_ab_inflight8": "tma.smem_ingress.per_sm",
    "tma_dram_stream_tc5a_ab_inflight8": "tma.hbm",
    "tmem_scale_ingress_32x128b_warpx4": "tmem.scale_ingress",
    "hbm_read_aggregate": "hbm.read",
    "hbm_write_aggregate": "hbm.write",
    "l2_read_aggregate": "l2.read",
    "l2_write_aggregate": "l2.write",
    "tmem_ld_32x32b_x8_warps1": "tmem.readback.x8.warps1",
    "tmem_ld_32x32b_x8_warps4": "tmem.readback.x8.warps4",
    "tmem_ld_32x32b_x16_warps1": "tmem.readback.x16.warps1",
    "tmem_ld_32x32b_x16_warps4": "tmem.readback",
    "nvfp4_requant_4096x1024_normal": "epilogue.nvfp4_requant",
    "nvfp4_requant_4096x1024_outlier": "epilogue.nvfp4_requant",
    "nvfp4_requant_4096x1024_constant": "epilogue.nvfp4_requant",
}
CAMPAIGN_COMPONENT_RESOURCE_COUNTS = dict(Counter(
    CAMPAIGN_COMPONENT_CASE_RESOURCES.values()))

CAMPAIGN_FULL_SHAPES = (1024, 2048, 4096)

CAMPAIGN_COMPUTE_SELECTION = {
    "launch": "full_sm_4warp_block",
    "m": 128,
    "n_values": [64, 128, 256],
}


def precision_coverage(
    capacities: Iterable[Capacity],
    observed: Iterable[ObservedBest],
    hardware: Hardware,
    scenarios: Iterable[ScenarioCoverage] = (),
) -> list[PrecisionCoverage]:
    capacities = list(capacities)
    observed = list(observed)
    scenarios = list(scenarios)
    hardware.validate()
    for capacity in capacities:
        capacity.validate()
    for observation in observed:
        observation.validate()
    capacities = [
        capacity for capacity in capacities
        if capacity_applies_to_hardware(capacity, hardware)
    ]
    rows: list[PrecisionCoverage] = []
    for precision_id, spec in precision_specs().items():
        domains = sorted({
            row.implementation_domain for row in scenarios
            if row.precision_id == precision_id
        }) or ["tensor_core_classical"]
        for implementation_domain in domains:
            required_compute_shapes = tuple(
                f"m{CAMPAIGN_COMPUTE_SELECTION['m']}n{n}"
                for n in CAMPAIGN_COMPUTE_SELECTION["n_values"]
            )
            domain_resource = (
                spec.compute_resource
                if implementation_domain == "tensor_core_classical"
                else aggregate_compute_resource(precision_id)
            )
            allowed_upper_scopes = (
                {"tensor_core_classical", "all_classical"}
                if implementation_domain == "tensor_core_classical"
                else {"all_classical"}
            )
            domain_upper = any(
                cap.resource == domain_resource
                and cap.evidence_kind.is_rate_upper_bound
                and cap.upper_scope in allowed_upper_scopes
                for cap in capacities
            )
            empirical = any(
                cap.resource.startswith(f"{spec.compute_resource}.m")
                and cap.evidence_kind.is_empirical_rate
                for cap in capacities
            )
            qualified_empirical = any(
                cap.resource.startswith(f"{spec.compute_resource}.m")
                and cap.evidence_kind.is_empirical_rate
                and cap.is_closure_qualified
                for cap in capacities
            )
            qualified_compute_shapes = tuple(
                shape for shape in required_compute_shapes
                if any(
                    cap.resource == f"{spec.compute_resource}.{shape}"
                    and cap.evidence_kind.is_empirical_rate
                    and cap.is_closure_qualified
                    for cap in capacities
                )
            )
            missing_compute_shapes = tuple(
                shape for shape in required_compute_shapes
                if shape not in qualified_compute_shapes
            )
            observed_rows = [
                row for row in observed
                if row.precision_id == precision_id
                and row.hardware_id == hardware.hardware_id
                and row.sm_count == hardware.sm_count
                and row.operating_mode == hardware.operating_mode
                and (
                    implementation_domain == "all_classical"
                    or row.arithmetic_path == "tensor_core"
                )
            ]
            full_gemm = bool(observed_rows)
            required_full_gemm_shapes = CAMPAIGN_FULL_SHAPES
            observed_full_gemm_shapes = tuple(sorted({
                row.n for row in observed_rows if row.m == row.n == row.k
            }))
            qualified_rows = [
                row for row in observed_rows
                if row.qualification == "closure_qualified"
            ]
            qualified_full_gemm_shapes = tuple(sorted({
                row.n for row in qualified_rows if row.m == row.n == row.k
            }))
            missing_full_gemm_shapes = tuple(
                n for n in required_full_gemm_shapes
                if n not in qualified_full_gemm_shapes
            )
            numerical_shapes = {
                row.n for row in qualified_rows
                if row.m == row.n == row.k
                and row.correctness_reference_relation
                == "independent_same_contract"
            }
            qualified_full_gemm = any(
                row.qualification == "closure_qualified"
                for row in observed_rows
            )
            numeric_closure = any(
                row.qualification == "closure_qualified"
                and row.correctness_reference_relation
                == "independent_same_contract"
                for row in observed_rows
            )
            same_denominator = any(
                row.qualification == "closure_qualified"
                and row.performance_reference_relation == "same_precision"
                for row in observed_rows
            )
            scenario_rows = [
                row for row in scenarios
                if row.precision_id == precision_id
                and row.implementation_domain == implementation_domain
            ]
            calibration_closed = any(
                row.validation_split == "calibration"
                and row.absolute_three_layer_closure
                for row in scenario_rows
            )
            holdout_closed = any(
                row.validation_split == "holdout"
                and row.absolute_three_layer_closure
                for row in scenario_rows
            )
            ratio_calibration_closed = any(
                row.validation_split == "calibration"
                and row.same_precision_ratio_closure
                for row in scenario_rows
            )
            ratio_holdout_closed = any(
                row.validation_split == "holdout"
                and row.same_precision_ratio_closure
                for row in scenario_rows
            )
            evidence_missing: list[str] = []
            if not domain_upper:
                evidence_missing.append("domain_compute_upper")
            if not empirical:
                evidence_missing.append("empirical_compute_rate")
            elif not qualified_empirical:
                evidence_missing.append("closure_qualified_compute_rate")
            if not full_gemm:
                evidence_missing.append("full_gemm_observed")
            elif not qualified_full_gemm:
                evidence_missing.append("closure_qualified_full_gemm")
            elif not numeric_closure:
                evidence_missing.append(
                    "independent_same_contract_correctness_reference"
                )
            missing: list[str] = []
            if not calibration_closed:
                missing.append("calibration_scenario_three_layer_closure")
            if not holdout_closed:
                missing.append("holdout_scenario_three_layer_closure")
            comparison_missing = (
                () if same_denominator
                else ("same_precision_performance_denominator",)
            )
            absolute_closed = not missing
            rows.append(PrecisionCoverage(
                precision_id=precision_id,
                implementation_domain=implementation_domain,
                domain_compute_upper=domain_upper,
                empirical_compute_rate=empirical,
                closure_qualified_compute_rate=qualified_empirical,
                full_gemm_observed=full_gemm,
                closure_qualified_full_gemm=qualified_full_gemm,
                same_precision_performance_denominator=same_denominator,
                numeric_closure=numeric_closure,
                calibration_scenario_closure=calibration_closed,
                holdout_scenario_closure=holdout_closed,
                absolute_three_layer_closure=absolute_closed,
                same_precision_ratio_closure=(
                    absolute_closed
                    and ratio_calibration_closed
                    and ratio_holdout_closed
                ),
                evidence_missing=tuple(evidence_missing),
                missing=tuple(missing),
                comparison_missing=comparison_missing,
                strict_compute_upper=any(
                    cap.resource == spec.compute_resource
                    and cap.evidence_kind.is_rate_upper_bound
                    for cap in capacities
                ),
                required_compute_shapes=required_compute_shapes,
                closure_qualified_compute_shapes=qualified_compute_shapes,
                compute_shape_matrix_complete=not missing_compute_shapes,
                required_full_gemm_shapes=required_full_gemm_shapes,
                observed_full_gemm_shapes=observed_full_gemm_shapes,
                closure_qualified_full_gemm_shapes=
                    qualified_full_gemm_shapes,
                full_gemm_shape_matrix_complete=not missing_full_gemm_shapes,
                full_gemm_numerical_validation_complete=all(
                    n in numerical_shapes for n in required_full_gemm_shapes
                ),
                missing_compute_shapes=missing_compute_shapes,
                missing_full_gemm_shapes=missing_full_gemm_shapes,
            ))
    return rows


def workload_manifest_coverage(
    workloads: Iterable[Workload],
) -> list[WorkloadManifestCoverage]:
    workloads = list(workloads)
    seen: set[str] = set()
    for workload in workloads:
        workload.validate(precision_specs())
        if workload.workload_id in seen:
            raise ValueError(f"duplicate workload_id: {workload.workload_id}")
        seen.add(workload.workload_id)
    rows: list[WorkloadManifestCoverage] = []
    for precision_id in precision_specs():
        domains = sorted({
            workload.implementation_domain for workload in workloads
            if workload.precision_id == precision_id
        }) or ["tensor_core_classical"]
        for domain in domains:
            selected = [
                workload for workload in workloads
                if workload.precision_id == precision_id
                and workload.implementation_domain == domain
            ]
            calibration = tuple(sorted(
                row.workload_id for row in selected
                if row.validation_split == "calibration"))
            holdout = tuple(sorted(
                row.workload_id for row in selected
                if row.validation_split == "holdout"))
            missing = tuple(
                name for name, present in (
                    ("calibration_workload", bool(calibration)),
                    ("holdout_workload", bool(holdout)),
                ) if not present
            )
            rows.append(WorkloadManifestCoverage(
                precision_id, domain, calibration, holdout, not missing, missing))
    return rows


def _scenario_observation_matches(
    workload: Workload, observed: ObservedBest, hardware: Hardware,
) -> bool:
    precision = precision_specs()[workload.precision_id]
    arithmetic_path_matches = (
        observed.arithmetic_path == "tensor_core"
        if workload.implementation_domain == "tensor_core_classical"
        else observed.arithmetic_path in {
            "tensor_core", "classical_non_tensor_or_mixed"
        }
    )
    return bool(
        observed.precision_id == workload.precision_id
        and (observed.m, observed.n, observed.k)
        == (workload.m, workload.n, workload.k)
        and observed.transpose_a == workload.transpose_a
        and observed.transpose_b == workload.transpose_b
        and math.isclose(observed.alpha, workload.alpha)
        and math.isclose(observed.beta, workload.beta)
        and observed.epilogue == workload.epilogue
        and observed.output_mode == workload.output_mode
        and observed.calibration_split == workload.validation_split
        and observed.numerical_contract == precision.numerical_contract
        and arithmetic_path_matches
        and observed.hardware_id == hardware.hardware_id
        and observed.sm_count == hardware.sm_count
        and observed.operating_mode == hardware.operating_mode
    )


def scenario_coverage(
    hardware: Hardware,
    capacities: Iterable[Capacity],
    workloads: Iterable[Workload],
    schedules: Iterable[Schedule],
    observed: Iterable[ObservedBest],
    *,
    pipeline_profiles: Iterable[PipelineProfile] = (),
    upper_tolerance: float = 0.02,
) -> list[ScenarioCoverage]:
    capacities = list(capacities)
    schedules = list(schedules)
    observed = list(observed)
    pipeline_profiles = list(pipeline_profiles)
    hardware.validate()
    for capacity in capacities:
        capacity.validate()
    for observation in observed:
        observation.validate()
    closure_capacities = [
        capacity for capacity in capacities
        if capacity.evidence_kind.is_rate_upper_bound
        or (
            capacity.evidence_kind.is_empirical_rate
            and capacity.is_closure_qualified
        )
    ]
    rows: list[ScenarioCoverage] = []
    for workload in workloads:
        envelope = evaluate_manifest(
            workload,
            schedules,
            hardware,
            closure_capacities,
            pipeline_profiles,
        )
        conditional = envelope.domain_conditional_upper
        empirical = envelope.empirical_ideal_envelope
        contract_rows = [
            row for row in observed
            if _scenario_observation_matches(workload, row, hardware)
        ]
        aligned = [
            row for row in contract_rows
            if row.residency == workload.residency
            and row.timed_scope == workload.timed_scope
        ]
        numeric = [
            row for row in aligned
            if row.qualification == "closure_qualified"
            and row.correctness_reference_relation
            == "independent_same_contract"
        ]
        best = max(numeric, key=lambda row: row.median_per_second, default=None)
        conditional_numeric = conditional.performance_per_second is not None
        conditional_complete = conditional.status == "ok" and conditional_numeric
        empirical_available = (
            empirical.status == "ok"
            and empirical.performance_per_second is not None)
        upper_consistent = bool(
            best is not None and conditional_numeric
            and best.performance_unit == conditional.performance_unit
            and best.maximum_per_second
            <= float(conditional.performance_per_second) * (1 + upper_tolerance))
        empirical_consistent = bool(
            best is not None and empirical_available
            and best.performance_unit == empirical.performance_unit
            and best.maximum_per_second
            <= float(empirical.performance_per_second) * (1 + upper_tolerance))
        closed = bool(
            conditional_complete and empirical_available and best is not None
            and upper_consistent and empirical_consistent)
        missing: list[str] = []
        if not conditional_complete:
            missing.append("conditional_upper_complete")
        if not empirical_available:
            missing.append("closure_qualified_empirical_envelope")
        if not contract_rows:
            missing.append("contract_matched_full_gemm")
        elif not aligned:
            missing.append("residency_or_timed_scope_aligned_full_gemm")
        elif not numeric:
            missing.append("closure_qualified_independent_correctness")
        if best is not None and conditional_numeric and not upper_consistent:
            missing.append("observed_within_conditional_upper")
        if best is not None and empirical_available and not empirical_consistent:
            missing.append("observed_not_above_empirical_envelope")
        rows.append(ScenarioCoverage(
            workload_id=workload.workload_id,
            precision_id=workload.precision_id,
            validation_split=workload.validation_split,
            implementation_domain=workload.implementation_domain,
            conditional_upper_numeric=conditional_numeric,
            conditional_upper_complete=conditional_complete,
            closure_qualified_empirical_envelope=empirical_available,
            contract_matched_full_gemm=bool(contract_rows),
            scenario_aligned_full_gemm=bool(aligned),
            numeric_closure=bool(numeric),
            upper_consistent=upper_consistent,
            empirical_envelope_consistent=empirical_consistent,
            absolute_three_layer_closure=closed,
            same_precision_ratio_closure=bool(
                closed and best is not None
                and best.performance_reference_relation == "same_precision"),
            empirical_schedule_id=envelope.empirical_schedule_id,
            conditional_upper_per_second=conditional.performance_per_second,
            empirical_envelope_per_second=empirical.performance_per_second,
            observed_median_per_second=(best.median_per_second if best else None),
            performance_unit=(conditional.performance_unit if conditional_numeric
                              else empirical.performance_unit),
            empirical_to_observed_ratio=(
                float(empirical.performance_per_second) / best.median_per_second
                if best is not None and empirical_available else None),
            missing=tuple(missing),
        ))
    return rows


def common_resource_coverage(
    capacities: Iterable[Capacity], hardware: Hardware,
) -> dict[str, bool]:
    capacities = list(capacities)
    hardware.validate()
    for capacity in capacities:
        capacity.validate()
    return {
        resource: any(
            cap.resource == resource
            and cap.evidence_kind.is_empirical_rate
            and cap.is_closure_qualified
            and capacity_applies_to_hardware(cap, hardware)
            for cap in capacities
        )
        for resource in COMMON_REQUIRED_EMPIRICAL_RESOURCES
    }


def campaign_measurement_coverage(
    capacities: Iterable[Capacity], observed: Iterable[ObservedBest]
) -> dict[str, object]:
    capacities = list(capacities)
    observed = list(observed)
    precision_closed: dict[str, bool] = {}
    compute_shape_closed: dict[str, dict[str, bool]] = {}
    for precision_id in CAMPAIGN_FULL_PRECISIONS:
        spec = precision_specs()[precision_id]
        compute_shape_closed[precision_id] = {
            f"m{CAMPAIGN_COMPUTE_SELECTION['m']}n{n}": sum(
                1
                for capacity in capacities
                if capacity.resource == (
                    f"{spec.compute_resource}."
                    f"m{CAMPAIGN_COMPUTE_SELECTION['m']}n{n}"
                )
                and capacity.evidence_kind.is_empirical_rate
                and capacity.is_closure_qualified
            ) == 1
            for n in CAMPAIGN_COMPUTE_SELECTION["n_values"]
        }
        rows = [
            row for row in observed
            if row.precision_id == precision_id
            and row.qualification == "closure_qualified"
            and row.performance_reference_relation == "same_precision"
            and row.m == row.n == row.k
        ]
        shape_counts = {
            n: sum(1 for row in rows if row.n == n)
            for n in CAMPAIGN_FULL_SHAPES
        }
        precision_closed[precision_id] = (
            all(compute_shape_closed[precision_id].values())
            and shape_counts == {n: 1 for n in CAMPAIGN_FULL_SHAPES}
            and len(rows) == len(CAMPAIGN_FULL_SHAPES)
        )
    component_case_closed = {
        case_id: sum(
            1
            for capacity in capacities
            if capacity.resource == resource
            and capacity.capacity_id.endswith(f".component.{case_id}")
            and capacity.evidence_kind.is_empirical_rate
            and capacity.is_closure_qualified
        ) == 1
        for case_id, resource in CAMPAIGN_COMPONENT_CASE_RESOURCES.items()
    }
    component_closed = {
        resource: all(
            component_case_closed[case_id]
            for case_id, case_resource
            in CAMPAIGN_COMPONENT_CASE_RESOURCES.items()
            if case_resource == resource
        )
        for resource in CAMPAIGN_COMPONENT_RESOURCE_COUNTS
    }
    return {
        "precision_ids": list(CAMPAIGN_FULL_PRECISIONS),
        "precision_closed": precision_closed,
        "compute_shape_closed": compute_shape_closed,
        "compute_selection": dict(CAMPAIGN_COMPUTE_SELECTION),
        "full_shapes": list(CAMPAIGN_FULL_SHAPES),
        "component_resource_counts": dict(CAMPAIGN_COMPONENT_RESOURCE_COUNTS),
        "component_case_closed": component_case_closed,
        "component_closed": component_closed,
        "all_campaign_precisions_closed": all(precision_closed.values()),
        "all_campaign_components_closed": all(component_closed.values()),
        "all_campaign_measurements_closed": (
            all(precision_closed.values()) and all(component_closed.values())
        ),
    }
