from __future__ import annotations

import math
import csv
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class ModelError(ValueError):
    """Raised when an input would make the model semantically ambiguous."""


class EvidenceKind(str, Enum):
    SPECIFIED_UPPER = "specified_upper"
    DERIVED_UPPER = "derived_upper"
    PROFILER_MODEL_PEAK = "profiler_model_peak"
    MEASURED_SUSTAINED = "measured_sustained"
    MEASURED_JOINT = "measured_joint"
    OBSERVED_GEMM = "observed_gemm"
    DERIVED_WORK = "derived_work"
    UNKNOWN = "unknown"

    @property
    def is_rate_upper_bound(self) -> bool:
        return self in {
            EvidenceKind.SPECIFIED_UPPER,
            EvidenceKind.DERIVED_UPPER,
            EvidenceKind.PROFILER_MODEL_PEAK,
        }

    @property
    def is_empirical_rate(self) -> bool:
        return self in {
            EvidenceKind.MEASURED_SUSTAINED,
            EvidenceKind.MEASURED_JOINT,
        }


@dataclass(frozen=True)
class PrecisionSpec:
    precision_id: str
    input_bytes: float
    accumulator_bytes: int
    output_bytes: float
    mma_k: int
    compute_resource: str
    compute_work_unit: str = "flop"
    input_scale_block: int | None = None
    input_scale_bytes: int = 0
    output_scale_block: int | None = None
    output_scale_bytes: int = 0

    def validate(self) -> None:
        if self.input_bytes <= 0 or self.output_bytes <= 0:
            raise ModelError(f"{self.precision_id}: element byte sizes must be positive")
        if self.accumulator_bytes <= 0 or self.mma_k <= 0:
            raise ModelError(f"{self.precision_id}: accumulator bytes and mma_k must be positive")
        if self.compute_work_unit not in {"flop", "operation"}:
            raise ModelError(
                f"{self.precision_id}: compute_work_unit must be flop or operation"
            )
        for block, size, label in (
            (self.input_scale_block, self.input_scale_bytes, "input"),
            (self.output_scale_block, self.output_scale_bytes, "output"),
        ):
            if (block is None) != (size == 0):
                raise ModelError(
                    f"{self.precision_id}: {label} scale block and byte size must be defined together"
                )


def precision_specs() -> dict[str, PrecisionSpec]:
    specs = [
        PrecisionSpec("fp16_f32", 2.0, 4, 4.0, 16, "tensor.fp16"),
        PrecisionSpec("bf16_f32", 2.0, 4, 4.0, 16, "tensor.bf16"),
        PrecisionSpec("tf32_f32", 4.0, 4, 4.0, 8, "tensor.tf32"),
        PrecisionSpec("e4m3_f32", 1.0, 4, 4.0, 32, "tensor.e4m3"),
        PrecisionSpec("e5m2_f32", 1.0, 4, 4.0, 32, "tensor.e5m2"),
        PrecisionSpec("e3m2_f32", 0.75, 4, 4.0, 32, "tensor.e3m2"),
        PrecisionSpec("e2m3_f32", 0.75, 4, 4.0, 32, "tensor.e2m3"),
        PrecisionSpec("e2m1_f32", 0.5, 4, 4.0, 32, "tensor.e2m1"),
        PrecisionSpec(
            "mxfp4_f32",
            0.5,
            4,
            4.0,
            64,
            "tensor.mxfp4",
            input_scale_block=32,
            input_scale_bytes=1,
        ),
        PrecisionSpec(
            "nvfp4_f32",
            0.5,
            4,
            4.0,
            64,
            "tensor.nvfp4",
            input_scale_block=16,
            input_scale_bytes=1,
        ),
        PrecisionSpec(
            "s8_s32", 1.0, 4, 4.0, 32, "tensor.s8", compute_work_unit="operation"
        ),
        PrecisionSpec(
            "u8_s32", 1.0, 4, 4.0, 32, "tensor.u8", compute_work_unit="operation"
        ),
    ]
    for spec in specs:
        spec.validate()
    return {spec.precision_id: spec for spec in specs}


@dataclass(frozen=True)
class Workload:
    workload_id: str
    m: int
    n: int
    k: int
    precision_id: str
    transpose_a: bool = False
    transpose_b: bool = False
    alpha: float = 1.0
    beta: float = 0.0
    epilogue: str = "none"
    residency: str = "cold_hbm"
    output_mode: str = "accumulator"
    include_launch: bool = True

    def validate(self, precisions: dict[str, PrecisionSpec]) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ModelError(f"{self.workload_id}: M, N, and K must be positive")
        if self.precision_id not in precisions:
            raise ModelError(f"{self.workload_id}: unknown precision {self.precision_id}")
        if self.residency not in {"cold_hbm", "hot_l2", "compute_oracle"}:
            raise ModelError(f"{self.workload_id}: unsupported residency {self.residency}")
        if self.output_mode not in {"accumulator", "packed_quantized"}:
            raise ModelError(f"{self.workload_id}: unsupported output_mode {self.output_mode}")
        if self.output_mode != "accumulator":
            raise ModelError(
                f"{self.workload_id}: packed_quantized output requires an explicit "
                "epilogue work and I/O contract not implemented in model v1"
            )
        if self.transpose_a or self.transpose_b:
            raise ModelError(
                f"{self.workload_id}: transposed data-movement schedule legality is "
                "not implemented in model v1"
            )
        if self.epilogue not in {
            "none",
            "bias",
            "relu",
            "gelu",
            "residual",
            "requant",
        }:
            raise ModelError(f"{self.workload_id}: unsupported epilogue {self.epilogue}")
        if self.epilogue != "none":
            raise ModelError(
                f"{self.workload_id}: epilogue {self.epilogue} is declared but its "
                "work and I/O contract is not implemented in model v1"
            )
        if self.alpha == 0.0:
            raise ModelError(
                f"{self.workload_id}: alpha=0 is not a GEMM workload in the v1 model"
            )


@dataclass(frozen=True)
class Schedule:
    schedule_id: str
    bm: int
    bn: int
    bk: int
    stages: int
    mma_m: int = 128
    mma_n: int = 128
    cta_group: int = 1
    split_k: int = 1
    tail_policy: str = "exact"
    supported_precisions: tuple[str, ...] = ()
    smem_limit_bytes: int = 228 * 1024
    tmem_columns: int = 128
    threads: int = 128
    registers_per_thread: int | None = None
    uses_tma: bool = True
    input_transport_layout: str = "logical_packed"
    persistent: bool = False
    fixed_seconds: float = 0.0

    def validate(self, precision: PrecisionSpec) -> None:
        if min(
            self.bm,
            self.bn,
            self.bk,
            self.mma_m,
            self.mma_n,
            self.stages,
            self.split_k,
            self.tmem_columns,
            self.threads,
        ) <= 0:
            raise ModelError(f"{self.schedule_id}: tile, stage, and split values must be positive")
        if self.cta_group not in {1, 2}:
            raise ModelError(f"{self.schedule_id}: cta_group must be 1 or 2")
        if self.cta_group != 1:
            raise ModelError(
                f"{self.schedule_id}: CTA-group-2 cluster and data-movement contract "
                "is not implemented in model v1"
            )
        if self.split_k != 1:
            raise ModelError(
                f"{self.schedule_id}: split-K partial-storage and reduction contract "
                "is not implemented in model v1"
            )
        if self.persistent:
            raise ModelError(
                f"{self.schedule_id}: persistent scheduling contract is not implemented "
                "in model v1"
            )
        if self.tail_policy not in {"exact", "pad"}:
            raise ModelError(f"{self.schedule_id}: tail_policy must be exact or pad")
        if self.input_transport_layout not in {
            "logical_packed",
            "byte_padded",
            "b6x16_p32",
            "b4x16_p64",
        }:
            raise ModelError(
                f"{self.schedule_id}: unsupported input_transport_layout "
                f"{self.input_transport_layout}"
            )
        if self.input_transport_layout == "b6x16_p32" and precision.input_bytes != 0.75:
            raise ModelError(
                f"{self.schedule_id}: b6x16_p32 is only valid for six-bit inputs"
            )
        if self.input_transport_layout == "b4x16_p64" and precision.input_bytes != 0.5:
            raise ModelError(
                f"{self.schedule_id}: b4x16_p64 is only valid for four-bit inputs"
            )
        if self.bk % precision.mma_k != 0:
            raise ModelError(
                f"{self.schedule_id}: BK={self.bk} is not divisible by "
                f"{precision.precision_id} MMA K={precision.mma_k}"
            )
        if self.bm % self.mma_m != 0 or self.bn % self.mma_n != 0:
            raise ModelError(
                f"{self.schedule_id}: CTA tile must be divisible by its MMA M/N atom"
            )
        if self.mma_m not in ({64, 128} if self.cta_group == 1 else {128, 256}):
            raise ModelError(
                f"{self.schedule_id}: MMA M={self.mma_m} is invalid for CTA group "
                f"{self.cta_group} in the v1 dense manifest"
            )
        if precision.precision_id in {"s8_s32", "u8_s32"}:
            valid_n = self.mma_n in {8, 16, 24, 32} or (
                48 <= self.mma_n <= 256 and self.mma_n % 16 == 0
            )
        else:
            valid_n = 8 <= self.mma_n <= 256 and self.mma_n % 8 == 0
        if not valid_n:
            raise ModelError(
                f"{self.schedule_id}: MMA N={self.mma_n} is invalid for CTA group "
                f"{self.cta_group} and precision {precision.precision_id}"
            )
        if precision.precision_id in {"mxfp4_f32", "nvfp4_f32"} and self.mma_m != 128:
            raise ModelError(
                f"{self.schedule_id}: block-scaled CTA group 1 requires MMA M=128"
            )
        if self.threads > 1024 or self.threads % 32 != 0:
            raise ModelError(
                f"{self.schedule_id}: threads must be a multiple of 32 and no more than 1024"
            )
        if (
            self.tmem_columns > 512
            or self.tmem_columns % 32 != 0
            or self.tmem_columns & (self.tmem_columns - 1)
        ):
            raise ModelError(
                f"{self.schedule_id}: TMEM columns must be a power of two in [32, 512]"
            )
        if self.registers_per_thread is not None and self.registers_per_thread <= 0:
            raise ModelError(f"{self.schedule_id}: registers_per_thread must be positive")
        if self.registers_per_thread is not None and self.registers_per_thread > 255:
            raise ModelError(f"{self.schedule_id}: registers_per_thread exceeds 255")
        if self.tmem_columns < self.mma_n:
            raise ModelError(
                f"{self.schedule_id}: TMEM allocation has {self.tmem_columns} columns "
                f"but the MMA atom requires at least {self.mma_n}"
            )
        if self.supported_precisions and precision.precision_id not in self.supported_precisions:
            raise ModelError(
                f"{self.schedule_id}: precision {precision.precision_id} is not supported"
            )
        smem_bytes = self.stages * (
            _transport_value_bytes(
                self.bm * self.bk, precision, self.input_transport_layout
            )
            + _transport_value_bytes(
                self.bk * self.bn, precision, self.input_transport_layout
            )
            + _scale_bytes(
                self.bm * self.bk,
                precision.input_scale_block,
                precision.input_scale_bytes,
            )
            + _scale_bytes(
                self.bk * self.bn,
                precision.input_scale_block,
                precision.input_scale_bytes,
            )
        )
        if smem_bytes > self.smem_limit_bytes:
            raise ModelError(
                f"{self.schedule_id}: modeled SMEM footprint {smem_bytes:g} B exceeds "
                f"{self.smem_limit_bytes} B"
            )
        if self.fixed_seconds < 0:
            raise ModelError(f"{self.schedule_id}: fixed_seconds cannot be negative")


@dataclass(frozen=True)
class Hardware:
    hardware_id: str
    sm_count: int
    clock_hz: float

    def validate(self) -> None:
        if self.sm_count <= 0 or self.clock_hz <= 0:
            raise ModelError("hardware sm_count and clock_hz must be positive")


@dataclass(frozen=True)
class Capacity:
    capacity_id: str
    resource: str
    rate_per_second: float
    work_unit: str
    evidence_kind: EvidenceKind
    source_id: str
    source_path: str
    source_locator: str
    original_value: float | None = None
    original_unit: str | None = None
    condition: str = ""
    uncertainty_fraction: float = 0.0
    qualification: str = "snapshot_only"
    trial_count: int = 1
    source_url: str = ""
    artifact_paths: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.rate_per_second <= 0 or not math.isfinite(self.rate_per_second):
            raise ModelError(f"{self.capacity_id}: rate_per_second must be finite and positive")
        if self.work_unit not in {"flop", "byte", "operation"}:
            raise ModelError(f"{self.capacity_id}: unsupported work unit {self.work_unit}")
        if not self.source_id or not self.source_path or not self.source_locator:
            raise ModelError(f"{self.capacity_id}: provenance fields cannot be empty")
        if not 0.0 <= self.uncertainty_fraction < 1.0:
            raise ModelError(f"{self.capacity_id}: invalid uncertainty_fraction")
        if self.qualification not in {
            "snapshot_only",
            "closure_qualified",
            "quarantined",
        }:
            raise ModelError(f"{self.capacity_id}: unsupported qualification")
        if self.trial_count <= 0:
            raise ModelError(f"{self.capacity_id}: trial_count must be positive")
        if self.qualification == "closure_qualified" and self.trial_count < 10:
            raise ModelError(
                f"{self.capacity_id}: closure qualification requires at least 10 trials"
            )
        if self.qualification == "closure_qualified" and not self.artifact_paths:
            raise ModelError(
                f"{self.capacity_id}: closure qualification requires artifact_paths"
            )
        if self.qualification == "quarantined" and self.evidence_kind != EvidenceKind.UNKNOWN:
            raise ModelError(
                f"{self.capacity_id}: quarantined capacity must use unknown evidence kind"
            )
        if self.evidence_kind == EvidenceKind.UNKNOWN and self.qualification != "quarantined":
            raise ModelError(
                f"{self.capacity_id}: unknown evidence must be explicitly quarantined"
            )

    @property
    def is_closure_qualified(self) -> bool:
        return self.qualification == "closure_qualified"


@dataclass(frozen=True)
class WorkAccounting:
    useful_compute_work: float
    issued_compute_work: float
    compute_work_unit: str
    input_value_bytes_min: float
    input_scale_bytes_min: float
    c_read_bytes_min: float
    output_value_bytes_min: float
    output_scale_bytes_min: float
    tma_input_bytes: float
    accumulator_readback_bytes: float
    reduction_bytes: float
    task_count: int
    output_tiles: int
    k_tiles: int

    @property
    def shape_efficiency(self) -> float:
        return self.useful_compute_work / self.issued_compute_work


@dataclass
class LayerResult:
    status: str
    seconds: float | None
    performance_per_second: float | None
    performance_unit: str
    bottlenecks: list[str] = field(default_factory=list)
    resource_seconds: dict[str, float] = field(default_factory=dict)
    missing_resources: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)


@dataclass
class ModelResult:
    workload_id: str
    schedule_id: str
    work: WorkAccounting
    conditional_upper: LayerResult
    empirical_envelope: LayerResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "schedule_id": self.schedule_id,
            "work": asdict(self.work),
            "conditional_upper": asdict(self.conditional_upper),
            "empirical_envelope": asdict(self.empirical_envelope),
        }


@dataclass
class WorkloadEnvelope:
    workload_id: str
    valid_schedule_count: int
    rejected_schedule_count: int
    manifest_conditional_upper: LayerResult
    empirical_ideal_envelope: LayerResult
    conditional_schedule_id: str | None
    empirical_schedule_id: str | None
    rejected: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _scale_bytes(elements: int, block: int | None, bytes_per_block: int) -> int:
    if block is None:
        return 0
    return ceil_div(elements, block) * bytes_per_block


def _transport_value_bytes(
    elements: int, precision: PrecisionSpec, layout: str
) -> float:
    """Return physical bytes moved by the selected input transport layout.

    `PrecisionSpec.input_bytes` is the logical packed representation.  PTX
    decompression layouts have a different physical transaction size and must
    be selected explicitly instead of silently charging the logical minimum.
    """
    if layout == "logical_packed":
        return elements * precision.input_bytes
    if layout == "byte_padded":
        return float(elements)
    if layout in {"b6x16_p32", "b4x16_p64"}:
        # Both layouts consume 16 physical bytes for each group of 16 values:
        # 12 B payload + 4 B padding for b6, or 8 B payload + 8 B padding for b4.
        return float(ceil_div(elements, 16) * 16)
    raise ModelError(f"unsupported input transport layout {layout}")


def account_work(workload: Workload, schedule: Schedule, precision: PrecisionSpec) -> WorkAccounting:
    schedule.validate(precision)
    nm = ceil_div(workload.m, schedule.bm)
    nn = ceil_div(workload.n, schedule.bn)
    nk = ceil_div(workload.k, schedule.bk)
    if schedule.tail_policy == "exact" and (
        workload.m % schedule.bm
        or workload.n % schedule.bn
        or workload.k % schedule.bk
    ):
        raise ModelError(
            f"{schedule.schedule_id}: exact tail requires an explicit legal tail-kernel "
            "manifest, which model v1 does not yet implement"
        )
    if schedule.tail_policy == "pad":
        issued_m, issued_n, issued_k = nm * schedule.bm, nn * schedule.bn, nk * schedule.bk
    else:
        issued_m, issued_n, issued_k = workload.m, workload.n, workload.k

    useful_compute_work = float(2 * workload.m * workload.n * workload.k)
    reduction_compute_work = float((schedule.split_k - 1) * workload.m * workload.n)
    issued_compute_work = (
        float(2 * issued_m * issued_n * issued_k) + reduction_compute_work
    )

    a_elements = workload.m * workload.k
    b_elements = workload.k * workload.n
    input_value_min = (a_elements + b_elements) * precision.input_bytes
    input_scale_min = _scale_bytes(
        a_elements, precision.input_scale_block, precision.input_scale_bytes
    ) + _scale_bytes(
        b_elements, precision.input_scale_block, precision.input_scale_bytes
    )
    c_read_min = workload.m * workload.n * precision.accumulator_bytes if workload.beta != 0 else 0

    if workload.output_mode == "packed_quantized":
        output_value = workload.m * workload.n * precision.input_bytes
        output_scale = _scale_bytes(
            workload.m * workload.n,
            precision.output_scale_block or precision.input_scale_block,
            precision.output_scale_bytes or precision.input_scale_bytes,
        )
    else:
        output_value = workload.m * workload.n * precision.output_bytes
        output_scale = 0

    per_full_tile_values = _transport_value_bytes(
        schedule.bm * schedule.bk,
        precision,
        schedule.input_transport_layout,
    ) + _transport_value_bytes(
        schedule.bk * schedule.bn,
        precision,
        schedule.input_transport_layout,
    )
    per_full_tile_scales = _scale_bytes(
        schedule.bm * schedule.bk,
        precision.input_scale_block,
        precision.input_scale_bytes,
    ) + _scale_bytes(
        schedule.bk * schedule.bn,
        precision.input_scale_block,
        precision.input_scale_bytes,
    )
    tma_input = nm * nn * nk * (per_full_tile_values + per_full_tile_scales)
    accumulator_readback = workload.m * workload.n * precision.accumulator_bytes
    reduction_bytes = (
        2.0
        * (schedule.split_k - 1)
        * workload.m
        * workload.n
        * precision.accumulator_bytes
    )
    return WorkAccounting(
        useful_compute_work=useful_compute_work,
        issued_compute_work=issued_compute_work,
        compute_work_unit=precision.compute_work_unit,
        input_value_bytes_min=float(input_value_min),
        input_scale_bytes_min=float(input_scale_min),
        c_read_bytes_min=float(c_read_min),
        output_value_bytes_min=float(output_value),
        output_scale_bytes_min=float(output_scale),
        tma_input_bytes=float(tma_input),
        accumulator_readback_bytes=float(accumulator_readback),
        reduction_bytes=float(reduction_bytes),
        task_count=nm * nn * schedule.split_k,
        output_tiles=nm * nn,
        k_tiles=nk,
    )


def _select_capacity(
    capacities: Iterable[Capacity], resource: str, *, strict: bool
) -> Capacity | None:
    candidates = [
        cap
        for cap in capacities
        if cap.resource == resource
        and (
            cap.evidence_kind.is_rate_upper_bound
            if strict
            else cap.evidence_kind.is_empirical_rate
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda cap: cap.rate_per_second)


def _resource_demands(
    workload: Workload,
    schedule: Schedule,
    work: WorkAccounting,
    precision: PrecisionSpec,
    *,
    empirical: bool,
) -> dict[str, tuple[float, str]]:
    read_min = work.input_value_bytes_min + work.input_scale_bytes_min + work.c_read_bytes_min
    write_min = work.output_value_bytes_min + work.output_scale_bytes_min
    demands: dict[str, tuple[float, str]] = {
        precision.compute_resource: (
            work.issued_compute_work if empirical else work.useful_compute_work,
            precision.compute_work_unit,
        )
    }
    if workload.residency == "cold_hbm":
        if empirical:
            demands["hbm.read"] = (read_min, "byte")
            demands["hbm.write"] = (write_min, "byte")
        else:
            demands["hbm.total"] = (read_min + write_min, "byte")
    elif workload.residency == "hot_l2":
        demands["l2.read"] = (read_min, "byte")
        demands["l2.write"] = (write_min, "byte")

    if empirical:
        tma_resource = "tma.hbm" if workload.residency == "cold_hbm" else "tma.l2"
        if workload.residency != "compute_oracle" and schedule.uses_tma:
            demands[tma_resource] = (work.tma_input_bytes, "byte")
        demands["tmem.readback"] = (work.accumulator_readback_bytes, "byte")
        if work.reduction_bytes:
            demands["reduction.io"] = (work.reduction_bytes, "byte")
    return demands


def _evaluate_layer(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
    precision: PrecisionSpec,
    work: WorkAccounting,
    capacities: list[Capacity],
    *,
    strict: bool,
) -> LayerResult:
    demands = _resource_demands(
        workload, schedule, work, precision, empirical=not strict
    )
    seconds: dict[str, float] = {}
    missing: list[str] = []
    conditions: list[str] = []
    selected: dict[str, Capacity] = {}
    for resource, (quantity, unit) in demands.items():
        cap = _select_capacity(capacities, resource, strict=strict)
        if cap is None:
            missing.append(resource)
            continue
        if cap.work_unit != unit:
            raise ModelError(
                f"{cap.capacity_id}: capacity unit {cap.work_unit} does not match "
                f"{resource} demand unit {unit}"
            )
        selected[resource] = cap
        seconds[resource] = quantity / cap.rate_per_second
        if cap.condition:
            conditions.append(f"{cap.capacity_id}: {cap.condition}")

    compute_cap = selected.get(precision.compute_resource)
    if compute_cap is not None and not strict:
        service_units = max(1, hardware.sm_count // schedule.cta_group)
        per_group_rate = (
            compute_cap.rate_per_second * schedule.cta_group / hardware.sm_count
        )
        task_work = work.issued_compute_work / work.task_count
        seconds["parallel_span"] = task_work / per_group_rate
        seconds["parallel_makespan"] = (
            ceil_div(work.task_count, service_units) * task_work / per_group_rate
        )

    if workload.include_launch and schedule.fixed_seconds and not strict:
        seconds["fixed"] = schedule.fixed_seconds

    if not seconds or (not strict and missing):
        return LayerResult(
            status="insufficient_evidence",
            seconds=None,
            performance_per_second=None,
            performance_unit=f"{precision.compute_work_unit}/s",
            resource_seconds=dict(sorted(seconds.items())),
            missing_resources=sorted(set(missing)),
            conditions=sorted(set(conditions)),
        )

    total = max(seconds.values())
    bottlenecks = sorted(
        resource
        for resource, value in seconds.items()
        if math.isclose(value, total, rel_tol=1e-9, abs_tol=0.0)
    )
    status = "ok" if not missing else "partial"
    return LayerResult(
        status=status,
        seconds=total,
        performance_per_second=work.useful_compute_work / total,
        performance_unit=f"{precision.compute_work_unit}/s",
        bottlenecks=bottlenecks,
        resource_seconds=dict(sorted(seconds.items())),
        missing_resources=sorted(set(missing)),
        conditions=sorted(set(conditions)),
    )


def evaluate(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
    capacities: list[Capacity],
) -> ModelResult:
    precisions = precision_specs()
    workload.validate(precisions)
    hardware.validate()
    precision = precisions[workload.precision_id]
    for cap in capacities:
        cap.validate()
    work = account_work(workload, schedule, precision)
    return ModelResult(
        workload_id=workload.workload_id,
        schedule_id=schedule.schedule_id,
        work=work,
        conditional_upper=_evaluate_layer(
            workload, schedule, hardware, precision, work, capacities, strict=True
        ),
        empirical_envelope=_evaluate_layer(
            workload, schedule, hardware, precision, work, capacities, strict=False
        ),
    )


def evaluate_manifest(
    workload: Workload,
    schedules: Iterable[Schedule],
    hardware: Hardware,
    capacities: list[Capacity],
) -> WorkloadEnvelope:
    results: list[ModelResult] = []
    rejected: list[dict[str, str]] = []
    for schedule in schedules:
        try:
            results.append(evaluate(workload, schedule, hardware, capacities))
        except ModelError as exc:
            rejected.append({"schedule_id": schedule.schedule_id, "reason": str(exc)})

    def best_layer(name: str) -> tuple[LayerResult, str | None]:
        candidates = [
            (getattr(row, name), row.schedule_id)
            for row in results
            if getattr(row, name).performance_per_second is not None
        ]
        if not candidates:
            missing = sorted(
                {
                    resource
                    for row in results
                    for resource in getattr(row, name).missing_resources
                }
            )
            return (
                LayerResult(
                    status="insufficient_evidence",
                    seconds=None,
                    performance_per_second=None,
                    performance_unit="unknown/s",
                    missing_resources=missing,
                ),
                None,
            )
        # The best implementation inside the manifest is bounded by the
        # largest per-schedule performance upper. The empirical ideal envelope
        # likewise chooses the schedule with the smallest predicted time.
        return max(
            candidates,
            key=lambda pair: float(pair[0].performance_per_second),
        )

    strict, strict_id = best_layer("conditional_upper")
    empirical, empirical_id = best_layer("empirical_envelope")
    return WorkloadEnvelope(
        workload_id=workload.workload_id,
        valid_schedule_count=len(results),
        rejected_schedule_count=len(rejected),
        manifest_conditional_upper=strict,
        empirical_ideal_envelope=empirical,
        conditional_schedule_id=strict_id,
        empirical_schedule_id=empirical_id,
        rejected=rejected,
    )


def audit_inputs(
    capacities: Iterable[Capacity],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    seen: set[str] = set()
    for cap in capacities:
        try:
            cap.validate()
        except ModelError as exc:
            findings.append({"severity": "error", "code": "invalid_capacity", "message": str(exc)})
            continue
        if cap.capacity_id in seen:
            findings.append(
                {"severity": "error", "code": "duplicate_capacity_id", "message": cap.capacity_id}
            )
        seen.add(cap.capacity_id)
        if cap.evidence_kind.is_empirical_rate and "upper" in cap.capacity_id:
            findings.append(
                {
                    "severity": "error",
                    "code": "measured_mislabeled_as_upper",
                    "message": cap.capacity_id,
                }
            )
        if cap.evidence_kind == EvidenceKind.PROFILER_MODEL_PEAK and not cap.condition:
            findings.append(
                {
                    "severity": "error",
                    "code": "conditional_peak_without_condition",
                    "message": cap.capacity_id,
                }
            )
        if repo_root is not None and not (repo_root / cap.source_path).is_file():
            findings.append(
                {
                    "severity": "error",
                    "code": "missing_source_path",
                    "message": f"{cap.capacity_id}: {cap.source_path}",
                }
            )
        elif repo_root is not None and cap.source_path.lower().endswith(".csv"):
            source_path = repo_root / cap.source_path
            tokens = [token.strip() for token in cap.source_locator.split(",")]
            predicates: dict[str, str] = {}
            value_field: str | None = None
            malformed = False
            for token in tokens:
                if "=" in token:
                    key, value = token.split("=", 1)
                    predicates[key] = value
                elif token:
                    if value_field is not None:
                        malformed = True
                    value_field = token
            if malformed or value_field is None:
                findings.append(
                    {
                        "severity": "error",
                        "code": "invalid_csv_locator",
                        "message": f"{cap.capacity_id}: {cap.source_locator}",
                    }
                )
                continue
            with source_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            matches = [
                row
                for row in rows
                if all(row.get(key) == value for key, value in predicates.items())
            ]
            if not matches:
                findings.append(
                    {
                        "severity": "error",
                        "code": "csv_locator_no_match",
                        "message": f"{cap.capacity_id}: {cap.source_locator}",
                    }
                )
                continue
            if value_field not in matches[0]:
                findings.append(
                    {
                        "severity": "error",
                        "code": "csv_value_field_missing",
                        "message": f"{cap.capacity_id}: {value_field}",
                    }
                )
                continue
            if cap.original_value is not None:
                try:
                    located_value = float(matches[0][value_field])
                except (TypeError, ValueError):
                    findings.append(
                        {
                            "severity": "error",
                            "code": "csv_value_not_numeric",
                            "message": f"{cap.capacity_id}: {matches[0][value_field]}",
                        }
                    )
                else:
                    if not math.isclose(
                        located_value, cap.original_value, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        findings.append(
                            {
                                "severity": "error",
                                "code": "csv_original_value_mismatch",
                                "message": (
                                    f"{cap.capacity_id}: located {located_value:g}, "
                                    f"declared {cap.original_value:g}"
                                ),
                            }
                        )
    return findings
