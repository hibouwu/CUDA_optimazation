from __future__ import annotations

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
    "tma.l2",
    "tmem.readback",
)


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
            cap.resource == spec.compute_resource and cap.evidence_kind.is_empirical_rate
            for cap in capacities
        )
        qualified_empirical = any(
            cap.resource == spec.compute_resource
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
