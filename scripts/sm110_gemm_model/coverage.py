from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from .model import Capacity, precision_specs
from .observations import ObservedBest


@dataclass(frozen=True)
class PrecisionCoverage:
    precision_id: str
    strict_compute_upper: bool
    empirical_compute_rate: bool
    closure_qualified_compute_rate: bool
    full_gemm_observed: bool
    closure_qualified_full_gemm: bool
    same_precision_performance_denominator: bool
    numeric_closure: bool
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


COMMON_REQUIRED_EMPIRICAL_RESOURCES = (
    "hbm.read",
    "hbm.write",
    "l2.read",
    "l2.write",
    "tma.hbm",
    "tma.smem_ingress.per_sm",
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
    "tma_l2_hit_32k": "tma.smem_ingress.per_sm",
    "tma_dram_stream_32k": "tma.hbm",
    "tma_l2_hit_32k_inflight4": "tma.smem_ingress.per_sm",
    "tma_dram_stream_32k_inflight4": "tma.hbm",
    "tma_l2_hit_16k_inflight8": "tma.smem_ingress.per_sm",
    "tma_dram_stream_16k_inflight8": "tma.hbm",
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
) -> list[PrecisionCoverage]:
    capacities = list(capacities)
    observed = list(observed)
    rows: list[PrecisionCoverage] = []
    for precision_id, spec in precision_specs().items():
        strict = any(
            cap.resource == spec.compute_resource and cap.evidence_kind.is_rate_upper_bound
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
        observed_rows = [row for row in observed if row.precision_id == precision_id]
        full_gemm = bool(observed_rows)
        qualified_full_gemm = any(
            row.qualification == "closure_qualified" for row in observed_rows
        )
        same_denominator = any(
            row.performance_reference_relation == "same_precision"
            for row in observed_rows
        )
        missing: list[str] = []
        if not strict:
            missing.append("strict_compute_upper")
        if not empirical:
            missing.append("empirical_compute_rate")
        elif not qualified_empirical:
            missing.append("closure_qualified_compute_rate")
        if not full_gemm:
            missing.append("full_gemm_observed")
        elif not qualified_full_gemm:
            missing.append("closure_qualified_full_gemm")
        if full_gemm and not same_denominator:
            missing.append("same_precision_performance_denominator")
        rows.append(
            PrecisionCoverage(
                precision_id=precision_id,
                strict_compute_upper=strict,
                empirical_compute_rate=empirical,
                closure_qualified_compute_rate=qualified_empirical,
                full_gemm_observed=full_gemm,
                closure_qualified_full_gemm=qualified_full_gemm,
                same_precision_performance_denominator=same_denominator,
                numeric_closure=not missing,
                missing=tuple(missing),
            )
        )
    return rows


def common_resource_coverage(capacities: Iterable[Capacity]) -> dict[str, bool]:
    capacities = list(capacities)
    return {
        resource: any(
            cap.resource == resource
            and cap.evidence_kind.is_empirical_rate
            and cap.is_closure_qualified
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
