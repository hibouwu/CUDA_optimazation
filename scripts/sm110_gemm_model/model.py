from __future__ import annotations

import math
import csv
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class ModelError(ValueError):
    """Raised when an input would make the model semantically ambiguous."""


# The executable has a latency-weighted persistent-worker DAG solver.  This
# flag means the algorithm exists; it does not mean that every schedule has a
# closure-qualified timing profile.  Per-schedule evidence remains fail-closed.
CAUSAL_PIPELINE_DAG_IMPLEMENTED = True


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
        if (not math.isfinite(self.input_bytes)
                or not math.isfinite(self.output_bytes)
                or self.input_bytes <= 0 or self.output_bytes <= 0):
            raise ModelError(f"{self.precision_id}: element byte sizes must be positive")
        if (not isinstance(self.accumulator_bytes, int)
                or isinstance(self.accumulator_bytes, bool)
                or not isinstance(self.mma_k, int)
                or isinstance(self.mma_k, bool)
                or self.accumulator_bytes <= 0 or self.mma_k <= 0):
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
            if block is not None and (
                    not isinstance(block, int) or isinstance(block, bool)
                    or not isinstance(size, int) or isinstance(size, bool)
                    or block <= 0 or size <= 0):
                raise ModelError(
                    f"{self.precision_id}: {label} scale block and byte size "
                    "must be positive integers"
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
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in (self.m, self.n, self.k)):
            raise ModelError(f"{self.workload_id}: M, N, and K must be integers")
        if min(self.m, self.n, self.k) <= 0:
            raise ModelError(f"{self.workload_id}: M, N, and K must be positive")
        if self.precision_id not in precisions:
            raise ModelError(f"{self.workload_id}: unknown precision {self.precision_id}")
        if any(not isinstance(value, bool) for value in (
                self.transpose_a, self.transpose_b, self.include_launch)):
            raise ModelError(
                f"{self.workload_id}: transpose and include_launch fields must be boolean")
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
        if not math.isfinite(self.alpha) or not math.isfinite(self.beta):
            raise ModelError(
                f"{self.workload_id}: alpha and beta must be finite")
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
    tmem_load_registers: int = 16
    tmem_consumer_warps: int | None = None
    registers_per_thread: int | None = None
    uses_tma: bool = True
    tma_ingress_capacity_resource: str | None = None
    tma_hbm_capacity_resource: str | None = None
    tma_contract_family_by_precision: dict[str, str] = field(
        default_factory=dict
    )
    tma_contract_row_stride_elements: tuple[int, ...] = ()
    causal_pipeline_resource: str | None = None
    input_transport_layout: str = "logical_packed"
    persistent: bool = False
    fixed_seconds: float = 0.0

    def validate(self, precision: PrecisionSpec) -> None:
        integer_fields = (
            self.bm, self.bn, self.bk, self.stages, self.mma_m, self.mma_n,
            self.cta_group, self.split_k, self.smem_limit_bytes,
            self.tmem_columns, self.threads, self.tmem_load_registers,
        )
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in integer_fields):
            raise ModelError(
                f"{self.schedule_id}: tile and resource sizes must be integers")
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
        if self.smem_limit_bytes <= 0:
            raise ModelError(f"{self.schedule_id}: smem_limit_bytes must be positive")
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
        if self.persistent and self.causal_pipeline_resource is None:
            raise ModelError(
                f"{self.schedule_id}: persistent scheduling requires an exact "
                "causal_pipeline_resource contract"
            )
        if not isinstance(self.uses_tma, bool):
            raise ModelError(f"{self.schedule_id}: uses_tma must be boolean")
        if not self.uses_tma:
            raise ModelError(
                f"{self.schedule_id}: non-TMA input paths require a separate "
                "issued-traffic and ingress-capacity contract not implemented in model v1"
            )
        if self.causal_pipeline_resource is not None and not self.persistent:
            raise ModelError(
                f"{self.schedule_id}: causal pipeline binding requires the "
                "persistent worker contract"
            )
        for resource, label in (
            (self.tma_ingress_capacity_resource, "TMA ingress"),
            (self.tma_hbm_capacity_resource, "TMA HBM"),
            (self.causal_pipeline_resource, "causal pipeline"),
        ):
            if resource is not None and (
                not isinstance(resource, str) or not resource.strip()
            ):
                raise ModelError(
                    f"{self.schedule_id}: {label} capacity resource must be "
                    "a nonempty string when declared"
                )
        if not isinstance(self.tma_contract_family_by_precision, dict):
            raise ModelError(
                f"{self.schedule_id}: tma_contract_family_by_precision must "
                "be an object"
            )
        if bool(self.tma_contract_family_by_precision) != bool(
            self.tma_contract_row_stride_elements
        ):
            raise ModelError(
                f"{self.schedule_id}: exact TMA family mapping and row-stride "
                "set must be declared together"
            )
        if self.tma_contract_family_by_precision and (
            self.tma_ingress_capacity_resource is not None
            or self.tma_hbm_capacity_resource is not None
        ):
            raise ModelError(
                f"{self.schedule_id}: exact TMA family mapping cannot be mixed "
                "with legacy fixed TMA resource IDs"
            )
        for precision_id, family_id in (
            self.tma_contract_family_by_precision.items()
        ):
            if (
                not isinstance(precision_id, str)
                or not precision_id
                or not isinstance(family_id, str)
                or re.fullmatch(r"[a-z0-9_]+", family_id) is None
            ):
                raise ModelError(
                    f"{self.schedule_id}: invalid exact TMA family mapping"
                )
            if (
                self.supported_precisions
                and precision_id not in self.supported_precisions
            ):
                raise ModelError(
                    f"{self.schedule_id}: TMA family is mapped for unsupported "
                    f"precision {precision_id}"
                )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in self.tma_contract_row_stride_elements
        ):
            raise ModelError(
                f"{self.schedule_id}: exact TMA row strides must be positive "
                "integers"
            )
        if len(set(self.tma_contract_row_stride_elements)) != len(
            self.tma_contract_row_stride_elements
        ):
            raise ModelError(
                f"{self.schedule_id}: exact TMA row strides must be unique"
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
        if (precision.precision_id in {"e3m2_f32", "e2m3_f32", "e2m1_f32"}
                and self.input_transport_layout == "logical_packed"):
            raise ModelError(
                f"{self.schedule_id}: {precision.precision_id} direct-SMEM path "
                "uses byte containers; select byte_padded or an explicit tcgen05.cp layout"
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
        if self.tmem_load_registers not in {8, 16}:
            raise ModelError(
                f"{self.schedule_id}: v1 TMEM readback must use LDTM.x8 or LDTM.x16")
        if self.tmem_consumer_warps is not None and (
                not isinstance(self.tmem_consumer_warps, int)
                or isinstance(self.tmem_consumer_warps, bool)
                or self.tmem_consumer_warps <= 0
                or self.tmem_consumer_warps > self.threads // 32):
            raise ModelError(
                f"{self.schedule_id}: tmem_consumer_warps must be a positive "
                "integer no greater than the CTA warp count")
        if (
            self.tmem_columns > 512
            or self.tmem_columns % 32 != 0
            or self.tmem_columns & (self.tmem_columns - 1)
        ):
            raise ModelError(
                f"{self.schedule_id}: TMEM columns must be a power of two in [32, 512]"
            )
        if self.registers_per_thread is not None and (
                not isinstance(self.registers_per_thread, int)
                or isinstance(self.registers_per_thread, bool)
                or self.registers_per_thread <= 0):
            raise ModelError(
                f"{self.schedule_id}: registers_per_thread must be a positive integer")
        if self.registers_per_thread is not None and self.registers_per_thread > 255:
            raise ModelError(f"{self.schedule_id}: registers_per_thread exceeds 255")
        if self.tmem_columns < self.mma_n:
            raise ModelError(
                f"{self.schedule_id}: TMEM allocation has {self.tmem_columns} columns "
                f"but the MMA atom requires at least {self.mma_n}"
            )
        if (precision.precision_id in {"mxfp4_f32", "nvfp4_f32"}
                and self.tmem_columns < 512):
            raise ModelError(
                f"{self.schedule_id}: block-scaled accumulator plus SFA/SFB "
                "requires the v1 512-column TMEM allocation contract"
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
            + _block_scale_transport_bytes(
                self.bm,
                self.bk,
                precision.input_scale_block,
                precision.input_scale_bytes,
            )
            + _block_scale_transport_bytes(
                self.bn,
                self.bk,
                precision.input_scale_block,
                precision.input_scale_bytes,
            )
        )
        if smem_bytes > self.smem_limit_bytes:
            raise ModelError(
                f"{self.schedule_id}: modeled SMEM footprint {smem_bytes:g} B exceeds "
                f"{self.smem_limit_bytes} B"
            )
        if not math.isfinite(self.fixed_seconds) or self.fixed_seconds < 0:
            raise ModelError(
                f"{self.schedule_id}: fixed_seconds must be finite and nonnegative")


@dataclass(frozen=True)
class Hardware:
    hardware_id: str
    sm_count: int
    clock_hz: float

    def validate(self) -> None:
        if (not isinstance(self.sm_count, int) or isinstance(self.sm_count, bool)
                or self.sm_count <= 0 or not math.isfinite(self.clock_hz)
                or self.clock_hz <= 0):
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
        if self.work_unit not in {"flop", "byte", "operation", "element"}:
            raise ModelError(f"{self.capacity_id}: unsupported work unit {self.work_unit}")
        if not self.source_id or not self.source_path or not self.source_locator:
            raise ModelError(f"{self.capacity_id}: provenance fields cannot be empty")
        if (self.evidence_kind == EvidenceKind.SPECIFIED_UPPER
                and not self.source_url.startswith("https://")):
            raise ModelError(
                f"{self.capacity_id}: specified upper requires an HTTPS primary-source URL"
            )
        if not 0.0 <= self.uncertainty_fraction < 1.0:
            raise ModelError(f"{self.capacity_id}: invalid uncertainty_fraction")
        if self.qualification not in {
            "snapshot_only",
            "closure_qualified",
            "quarantined",
        }:
            raise ModelError(f"{self.capacity_id}: unsupported qualification")
        if (not isinstance(self.trial_count, int)
                or isinstance(self.trial_count, bool)
                or self.trial_count <= 0):
            raise ModelError(
                f"{self.capacity_id}: trial_count must be a positive integer")
        if (self.original_value is None) != (self.original_unit is None):
            raise ModelError(
                f"{self.capacity_id}: original_value and original_unit must be paired")
        if (self.original_value is not None
                and not math.isfinite(self.original_value)):
            raise ModelError(
                f"{self.capacity_id}: original_value must be finite")
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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_kind"] = self.evidence_kind.value
        return data


@dataclass(frozen=True)
class PipelineProfile:
    profile_id: str
    resource: str
    schedule_id: str
    precision_ids: tuple[str, ...]
    evidence_kind: EvidenceKind
    qualification: str
    trial_count_per_case: int
    source_id: str
    expected_commit: str
    source_path: str
    source_locator: str
    input_residency: str
    stages: int
    accumulator_buffers: int
    resident_ctas_per_sm: int
    maximum_k_tiles: int
    maximum_output_tasks_per_worker: int
    tma_first_completion_seconds: float
    tma_completion_interval_seconds: float
    mma_first_completion_seconds: float
    mma_completion_interval_seconds: float
    joint_first_mma_completion_seconds: float
    joint_completion_interval_seconds: float
    epilogue_latency_seconds: float
    component_r_squared: dict[str, float]
    max_calibration_relative_error: float
    max_holdout_relative_error: float
    fit_contract: dict[str, Any]
    validation: tuple[dict[str, Any], ...]
    closure_qualified: bool
    artifact_paths: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.profile_id or not self.resource or not self.schedule_id:
            raise ModelError("pipeline profile identity fields cannot be empty")
        if (
            not isinstance(self.precision_ids, tuple)
            or not self.precision_ids
            or any(not isinstance(value, str) or not value
                   for value in self.precision_ids)
            or len(set(self.precision_ids)) != len(self.precision_ids)
        ):
            raise ModelError(
                f"{self.profile_id}: precision_ids must be a nonempty tuple "
                "of unique precision IDs"
            )
        unknown_precisions = sorted(
            set(self.precision_ids) - set(precision_specs())
        )
        if unknown_precisions:
            raise ModelError(
                f"{self.profile_id}: unknown causal profile precisions "
                f"{unknown_precisions}"
            )
        if self.evidence_kind != EvidenceKind.MEASURED_JOINT:
            raise ModelError(
                f"{self.profile_id}: causal profile must use measured_joint evidence"
            )
        if self.qualification not in {"closure_qualified", "quarantined"}:
            raise ModelError(
                f"{self.profile_id}: unsupported profile qualification"
            )
        if self.closure_qualified != (
            self.qualification == "closure_qualified"
        ):
            raise ModelError(
                f"{self.profile_id}: qualification boolean is inconsistent"
            )
        integer_fields = (
            self.trial_count_per_case, self.stages, self.accumulator_buffers,
            self.resident_ctas_per_sm, self.maximum_k_tiles,
            self.maximum_output_tasks_per_worker,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in integer_fields
        ):
            raise ModelError(
                f"{self.profile_id}: count and topology fields must be positive integers"
            )
        if self.closure_qualified and self.trial_count_per_case < 10:
            raise ModelError(
                f"{self.profile_id}: closure qualification requires ten trials per case"
            )
        if not re.fullmatch(r"[0-9a-f]{40}", self.expected_commit):
            raise ModelError(
                f"{self.profile_id}: expected_commit is not a 40-hex Git commit"
            )
        if not self.source_id or not self.source_path or not self.source_locator:
            raise ModelError(
                f"{self.profile_id}: provenance fields cannot be empty"
            )
        if self.input_residency != "hot_l2":
            raise ModelError(
                f"{self.profile_id}: v1 causal calibration must use hot_l2 inputs"
            )
        timing_fields = (
            self.tma_first_completion_seconds,
            self.tma_completion_interval_seconds,
            self.mma_first_completion_seconds,
            self.mma_completion_interval_seconds,
            self.joint_first_mma_completion_seconds,
            self.joint_completion_interval_seconds,
            self.epilogue_latency_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in timing_fields):
            raise ModelError(
                f"{self.profile_id}: timing fields must be finite and positive"
            )
        if set(self.component_r_squared) != {"tma", "mma", "joint"}:
            raise ModelError(
                f"{self.profile_id}: component R-squared set is incomplete"
            )
        # R-squared may legitimately be negative when the frozen linear model
        # is worse than a constant predictor.  Such a profile must remain
        # importable as quarantined evidence, so only non-finite values and
        # values above the mathematical maximum are structurally invalid.
        if any(
            not math.isfinite(value) or value > 1.0
            for value in self.component_r_squared.values()
        ):
            raise ModelError(
                f"{self.profile_id}: component R-squared values are invalid"
            )
        for value, label in (
            (self.max_calibration_relative_error, "calibration"),
            (self.max_holdout_relative_error, "holdout"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ModelError(
                    f"{self.profile_id}: {label} error must be finite and nonnegative"
                )
        required_fit_fields = {
            "calibration_points",
            "holdout_points",
            "calibration_output_tasks",
            "epilogue_calibration_output_tasks",
            "holdout_output_tasks",
            "minimum_r_squared",
            "maximum_holdout_relative_error",
            "profile_stages",
            "profile_accumulator_buffers",
            "profile_resident_ctas_per_sm",
        }
        if not required_fit_fields.issubset(self.fit_contract):
            raise ModelError(
                f"{self.profile_id}: fit contract is missing "
                f"{sorted(required_fit_fields - set(self.fit_contract))}"
            )
        minimum_r2 = self.fit_contract["minimum_r_squared"]
        maximum_error = self.fit_contract["maximum_holdout_relative_error"]
        if (
            not isinstance(minimum_r2, (int, float))
            or isinstance(minimum_r2, bool)
            or not math.isfinite(float(minimum_r2))
            or not 0.0 <= float(minimum_r2) <= 1.0
            or not isinstance(maximum_error, (int, float))
            or isinstance(maximum_error, bool)
            or not math.isfinite(float(maximum_error))
            or not 0.0 <= float(maximum_error) < 1.0
        ):
            raise ModelError(f"{self.profile_id}: fit thresholds are invalid")
        topology = (
            ("profile_stages", self.stages),
            ("profile_accumulator_buffers", self.accumulator_buffers),
            ("profile_resident_ctas_per_sm", self.resident_ctas_per_sm),
        )
        for field_name, expected in topology:
            if self.fit_contract[field_name] != expected:
                raise ModelError(
                    f"{self.profile_id}: {field_name} disagrees with profile topology"
                )
        sweep_fields = (
            "calibration_points",
            "holdout_points",
            "calibration_output_tasks",
            "epilogue_calibration_output_tasks",
            "holdout_output_tasks",
        )
        sweeps: dict[str, tuple[int, ...]] = {}
        for field_name in sweep_fields:
            raw = self.fit_contract[field_name]
            if (
                not isinstance(raw, (list, tuple))
                or not raw
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value <= 0
                    for value in raw
                )
                or len(set(raw)) != len(raw)
            ):
                raise ModelError(
                    f"{self.profile_id}: {field_name} must be a nonempty "
                    "unique positive-integer sequence"
                )
            sweeps[field_name] = tuple(raw)
        calibration_k = set(sweeps["calibration_points"])
        holdout_k = set(sweeps["holdout_points"])
        calibration_output = set(sweeps["calibration_output_tasks"])
        epilogue_output = set(sweeps["epilogue_calibration_output_tasks"])
        holdout_output = set(sweeps["holdout_output_tasks"])
        if calibration_k & holdout_k:
            raise ModelError(
                f"{self.profile_id}: calibration and holdout K points overlap"
            )
        if calibration_output & holdout_output:
            raise ModelError(
                f"{self.profile_id}: calibration and holdout output tasks overlap"
            )
        if not epilogue_output.issubset(calibration_output):
            raise ModelError(
                f"{self.profile_id}: epilogue calibration tasks must be a "
                "subset of calibration output tasks"
            )
        all_k = calibration_k | holdout_k
        all_output = calibration_output | holdout_output
        if max(all_k) != self.maximum_k_tiles:
            raise ModelError(
                f"{self.profile_id}: maximum_k_tiles disagrees with fit sweep"
            )
        if max(all_output) != self.maximum_output_tasks_per_worker:
            raise ModelError(
                f"{self.profile_id}: maximum output-task range disagrees with fit sweep"
            )
        if self.joint_completion_interval_seconds < max(
            self.tma_completion_interval_seconds,
            self.mma_completion_interval_seconds,
        ) and not math.isclose(
            self.joint_completion_interval_seconds,
            max(
                self.tma_completion_interval_seconds,
                self.mma_completion_interval_seconds,
            ),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ModelError(
                f"{self.profile_id}: joint interval is faster than an isolated component"
            )
        if not self.validation:
            raise ModelError(f"{self.profile_id}: validation rows cannot be empty")
        recomputed_errors: dict[str, list[float]] = {
            "calibration": [],
            "holdout": [],
        }
        expected_coordinates = {
            (k_tiles, output_tasks)
            for k_tiles in all_k
            for output_tasks in all_output
        }
        observed_coordinates: set[tuple[int, int]] = set()
        case_ids: set[str] = set()
        for index, row in enumerate(self.validation):
            if not isinstance(row, dict):
                raise ModelError(
                    f"{self.profile_id}: validation row {index} is not an object"
                )
            split = row.get("split")
            if split not in recomputed_errors:
                raise ModelError(
                    f"{self.profile_id}: validation row {index} has invalid split"
                )
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                raise ModelError(
                    f"{self.profile_id}: validation row {index} has a missing "
                    "or duplicate case_id"
                )
            case_ids.add(case_id)
            k_tiles = row.get("k_tiles")
            output_tasks = row.get("output_tasks")
            if (
                not isinstance(k_tiles, int)
                or isinstance(k_tiles, bool)
                or not isinstance(output_tasks, int)
                or isinstance(output_tasks, bool)
                or (k_tiles, output_tasks) not in expected_coordinates
                or (k_tiles, output_tasks) in observed_coordinates
            ):
                raise ModelError(
                    f"{self.profile_id}: validation row {index} has an invalid "
                    "or duplicate K/output coordinate"
                )
            observed_coordinates.add((k_tiles, output_tasks))
            expected_split = (
                "calibration"
                if k_tiles in calibration_k and output_tasks in calibration_output
                else "holdout"
            )
            if split != expected_split:
                raise ModelError(
                    f"{self.profile_id}: validation row {index} split disagrees "
                    "with the frozen sweep"
                )
            try:
                actual = float(row["actual_median_ns"])
                predicted = float(row["predicted_ns"])
                recorded_error = float(row["relative_error"])
            except (KeyError, TypeError, ValueError) as error:
                raise ModelError(
                    f"{self.profile_id}: validation row {index} is incomplete"
                ) from error
            if (
                not math.isfinite(actual)
                or not math.isfinite(predicted)
                or not math.isfinite(recorded_error)
                or actual <= 0
                or predicted <= 0
                or recorded_error < 0
            ):
                raise ModelError(
                    f"{self.profile_id}: validation row {index} is invalid"
                )
            previous_mma_done = 0.0
            epilogue_done: list[float] = []
            for task in range(output_tasks):
                first_mma_done = (
                    self.joint_first_mma_completion_seconds
                    if task == 0
                    else previous_mma_done
                    + self.joint_completion_interval_seconds
                )
                if task >= self.accumulator_buffers:
                    first_mma_done = max(
                        first_mma_done,
                        epilogue_done[task - self.accumulator_buffers]
                        + self.joint_completion_interval_seconds,
                    )
                last_mma_done = (
                    first_mma_done
                    + (k_tiles - 1)
                    * self.joint_completion_interval_seconds
                )
                epilogue_start = max(
                    last_mma_done,
                    epilogue_done[-1] if epilogue_done else 0.0,
                )
                epilogue_done.append(
                    epilogue_start + self.epilogue_latency_seconds
                )
                previous_mma_done = last_mma_done
            parameter_prediction_ns = epilogue_done[-1] * 1e9
            if not math.isclose(
                parameter_prediction_ns,
                predicted,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ModelError(
                    f"{self.profile_id}: validation row {index} prediction "
                    "does not match profile parameters"
                )
            recomputed = abs(predicted - actual) / actual
            if not math.isclose(
                recomputed, recorded_error, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ModelError(
                    f"{self.profile_id}: validation row {index} error arithmetic mismatch"
                )
            recomputed_errors[split].append(recomputed)
        if observed_coordinates != expected_coordinates:
            raise ModelError(
                f"{self.profile_id}: validation coordinate grid is incomplete"
            )
        if any(not values for values in recomputed_errors.values()):
            raise ModelError(
                f"{self.profile_id}: calibration and holdout validation are both required"
            )
        if not math.isclose(
            max(recomputed_errors["calibration"]),
            self.max_calibration_relative_error,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ) or not math.isclose(
            max(recomputed_errors["holdout"]),
            self.max_holdout_relative_error,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ModelError(
                f"{self.profile_id}: summarized validation maxima do not match rows"
            )
        gates_pass = bool(
            all(
                value >= float(minimum_r2)
                for value in self.component_r_squared.values()
            )
            and self.max_calibration_relative_error <= float(maximum_error)
            and self.max_holdout_relative_error <= float(maximum_error)
        )
        if self.closure_qualified != gates_pass:
            raise ModelError(
                f"{self.profile_id}: qualification disagrees with predeclared fit gates"
            )
        if len(set(self.artifact_paths)) != len(self.artifact_paths) or any(
            not path or Path(path).is_absolute() or ".." in Path(path).parts
            for path in self.artifact_paths
        ):
            raise ModelError(
                f"{self.profile_id}: artifact paths must be unique repository-relative files"
            )
        if self.closure_qualified and not self.artifact_paths:
            raise ModelError(
                f"{self.profile_id}: qualified profile requires artifact paths"
            )

    @property
    def is_closure_qualified(self) -> bool:
        return self.closure_qualified

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence_kind"] = self.evidence_kind.value
        return result


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
    tma_unique_input_bytes: float
    tma_value_input_bytes: float
    tma_scale_input_bytes: float
    tma_input_bytes: float
    tmem_scale_ingress_bytes: float
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
    selected_capacity_ids: dict[str, str] = field(default_factory=dict)
    selected_capacity_evidence_kinds: dict[str, str] = field(
        default_factory=dict
    )
    selected_capacity_qualifications: dict[str, str] = field(
        default_factory=dict
    )


@dataclass
class ModelResult:
    workload_id: str
    schedule_id: str
    work: WorkAccounting
    conditional_upper: LayerResult
    empirical_envelope: LayerResult
    causal_pipeline: LayerResult
    empirical_ideal_envelope: LayerResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "schedule_id": self.schedule_id,
            "work": asdict(self.work),
            "conditional_upper": asdict(self.conditional_upper),
            "empirical_envelope": asdict(self.empirical_envelope),
            "causal_pipeline": asdict(self.causal_pipeline),
            "empirical_ideal_envelope": asdict(self.empirical_ideal_envelope),
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
    manifest_empirical_resource_envelope: LayerResult = field(
        default_factory=lambda: LayerResult(
            status="insufficient_evidence",
            seconds=None,
            performance_per_second=None,
            performance_unit="unknown/s",
        )
    )
    empirical_resource_schedule_id: str | None = None
    causal_pipeline_envelope: LayerResult = field(
        default_factory=lambda: LayerResult(
            status="insufficient_evidence",
            seconds=None,
            performance_per_second=None,
            performance_unit="unknown/s",
        )
    )
    causal_pipeline_schedule_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def resolve_repo_artifact(repo_root: Path, relative_path: str) -> Path:
    """Resolve one evidence path without allowing escape through '..' or symlinks."""
    root = repo_root.resolve()
    path = Path(relative_path)
    if path.is_absolute():
        raise ModelError(f"evidence path must be repository-relative: {relative_path}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ModelError(
            f"evidence path escapes repository root: {relative_path}") from error
    if not resolved.is_file():
        raise ModelError(f"evidence file is missing: {relative_path}")
    return resolved


def _linear_scale_bytes(
    elements: int, block: int | None, bytes_per_block: int
) -> int:
    if block is None:
        return 0
    return ceil_div(elements, block) * bytes_per_block


def _matrix_scale_bytes(
    vector_count: int,
    reduction_elements: int,
    block: int | None,
    bytes_per_block: int,
) -> int:
    """Count block scales without allowing a block to cross K vectors."""
    if block is None:
        return 0
    return vector_count * ceil_div(reduction_elements, block) * bytes_per_block


def _block_scale_transport_bytes(
    vector_count: int,
    reduction_elements: int,
    block: int | None,
    bytes_per_block: int,
) -> int:
    """Count the physical SFA/SFB scale tensor consumed by TMA/S2T.

    Blackwell block-scaled MMA stores scale factors in 128-vector by four-
    scale-group atoms (CUTLASS ``Swizzle32x4x4``).  Logical scale counts are
    still reported separately by :func:`_matrix_scale_bytes`; transport and
    SMEM/TMEM ingress must charge both layout paddings instead of silently
    treating the scale tensor as a compact matrix.
    """
    if block is None:
        return 0
    vectors_padded = ceil_div(vector_count, 128) * 128
    scale_groups = ceil_div(reduction_elements, block)
    scale_groups_padded = ceil_div(scale_groups, 4) * 4
    return vectors_padded * scale_groups_padded * bytes_per_block


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
    input_scale_min = _matrix_scale_bytes(
        workload.m,
        workload.k,
        precision.input_scale_block,
        precision.input_scale_bytes,
    ) + _matrix_scale_bytes(
        workload.n,
        workload.k,
        precision.input_scale_block,
        precision.input_scale_bytes,
    )
    c_read_min = workload.m * workload.n * precision.accumulator_bytes if workload.beta != 0 else 0

    if workload.output_mode == "packed_quantized":
        output_value = workload.m * workload.n * precision.input_bytes
        output_scale = _linear_scale_bytes(
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
    per_full_tile_scales = _block_scale_transport_bytes(
        schedule.bm,
        schedule.bk,
        precision.input_scale_block,
        precision.input_scale_bytes,
    ) + _block_scale_transport_bytes(
        schedule.bn,
        schedule.bk,
        precision.input_scale_block,
        precision.input_scale_bytes,
    )
    unique_tma_input = (
        _transport_value_bytes(
            issued_m * issued_k,
            precision,
            schedule.input_transport_layout,
        )
        + _transport_value_bytes(
            issued_n * issued_k,
            precision,
            schedule.input_transport_layout,
        )
        + _block_scale_transport_bytes(
            issued_m,
            issued_k,
            precision.input_scale_block,
            precision.input_scale_bytes,
        )
        + _block_scale_transport_bytes(
            issued_n,
            issued_k,
            precision.input_scale_block,
            precision.input_scale_bytes,
        )
    )
    tma_value_input = nm * nn * nk * per_full_tile_values
    tma_scale_input = nm * nn * nk * per_full_tile_scales
    tma_input = tma_value_input + tma_scale_input
    accumulator_readback = issued_m * issued_n * precision.accumulator_bytes
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
        tma_unique_input_bytes=float(unique_tma_input),
        tma_value_input_bytes=float(tma_value_input),
        tma_scale_input_bytes=float(tma_scale_input),
        tma_input_bytes=float(tma_input),
        tmem_scale_ingress_bytes=float(tma_scale_input),
        accumulator_readback_bytes=float(accumulator_readback),
        reduction_bytes=float(reduction_bytes),
        task_count=nm * nn * schedule.split_k,
        output_tiles=nm * nn,
        k_tiles=nk,
    )


def _select_capacity(
    capacities: Iterable[Capacity], resource: str, *, strict: bool
) -> Capacity | None:
    # The historical closure campaign already measured the exact tc5a stage-4
    # contract at packed row stride 2048, before resource IDs encoded the row
    # stride.  These one-way aliases preserve that valid point for N=K=2048;
    # they never let the legacy result satisfy 1024/4096 or another family.
    legacy_empirical_aliases = {
        (
            "tma.smem_ingress.contract."
            "tc5a_m128n256k64_s4_v16.stride2048.per_sm"
        ): ("tma.smem_ingress.per_sm",),
        (
            "tma.hbm.contract."
            "tc5a_m128n256k64_s4_v16.stride2048"
        ): ("tma.hbm",),
    }
    candidate_resources = {resource}
    if not strict:
        candidate_resources.update(legacy_empirical_aliases.get(resource, ()))
    candidates = [
        cap
        for cap in capacities
        if cap.resource in candidate_resources
        and (
            cap.evidence_kind.is_rate_upper_bound
            if strict
            else cap.evidence_kind.is_empirical_rate
        )
    ]
    if not candidates:
        return None
    if not strict:
        qualified = [cap for cap in candidates if cap.is_closure_qualified]
        if qualified:
            candidates = qualified
    # Every proven upper applies simultaneously, so their intersection is the
    # smallest rate upper.  Empirical capacities mean "hardware has sustained
    # at least this rate" under the recorded condition, so the best comparable
    # point is the largest rate.  Once a resource has closure-qualified points,
    # weaker snapshots with legacy timing/provenance contracts are not mixed
    # back into that resource's calibration set.
    selector = min if strict else max
    return selector(candidates, key=lambda cap: cap.rate_per_second)


def _resolve_tma_capacity_resources(
    workload: Workload,
    schedule: Schedule,
    precision: PrecisionSpec,
) -> tuple[str | None, str | None, str | None]:
    """Resolve an exact schedule/precision/leading-dimension TMA contract.

    Model v1 accepts only non-transposed packed matrices, so A's physical row
    stride is K elements and B's is N elements.  The frozen resource campaign
    uses one common row stride for both tensor maps.  It is therefore an exact
    match only when ``K == N`` and that common stride was predeclared.  Any
    other shape fails closed instead of borrowing a nearby measurement.
    """

    families = schedule.tma_contract_family_by_precision
    if not families:
        return (
            schedule.tma_ingress_capacity_resource,
            schedule.tma_hbm_capacity_resource,
            None,
        )
    family = families.get(precision.precision_id)
    if family is None:
        return (
            None,
            None,
            (
                f"tma_transport_family_contract:{schedule.schedule_id}:"
                f"{precision.precision_id}"
            ),
        )
    if workload.k != workload.n:
        return (
            None,
            None,
            (
                f"tma_equal_ab_row_stride_contract:{schedule.schedule_id}:"
                f"a_ld={workload.k}:b_ld={workload.n}"
            ),
        )
    stride = workload.k
    if stride not in schedule.tma_contract_row_stride_elements:
        return (
            None,
            None,
            f"tma_row_stride_contract:{schedule.schedule_id}:{stride}",
        )
    stem = f"{family}.stride{stride}"
    return (
        f"tma.smem_ingress.contract.{stem}.per_sm",
        f"tma.hbm.contract.{stem}",
        None,
    )


def _resource_demands(
    workload: Workload,
    schedule: Schedule,
    work: WorkAccounting,
    precision: PrecisionSpec,
    *,
    empirical: bool,
) -> dict[str, tuple[float, str]]:
    _, resolved_tma_hbm, _ = _resolve_tma_capacity_resources(
        workload, schedule, precision
    )
    read_min = work.input_value_bytes_min + work.input_scale_bytes_min + work.c_read_bytes_min
    write_min = work.output_value_bytes_min + work.output_scale_bytes_min
    compute_resource = (
        precision.compute_resource
        if not empirical
        else (
            f"{precision.compute_resource}."
            f"m{schedule.mma_m}n{schedule.mma_n}"
        )
    )
    demands: dict[str, tuple[float, str]] = {
        compute_resource: (
            work.issued_compute_work if empirical else work.useful_compute_work,
            precision.compute_work_unit,
        )
    }
    if workload.residency == "cold_hbm":
        if empirical:
            demands["hbm.read"] = (
                work.tma_unique_input_bytes + work.c_read_bytes_min,
                "byte",
            )
            demands["hbm.write"] = (write_min, "byte")
            demands["l2.read"] = (
                work.tma_input_bytes + work.c_read_bytes_min,
                "byte",
            )
            demands["l2.write"] = (write_min, "byte")
        else:
            demands["hbm.total"] = (read_min + write_min, "byte")
            # Device-memory traffic still crosses the GPU-wide shared L2
            # fabric.  These are aggregate GPU capacities, not per-SM rates;
            # keep the minimum read and write work as independent strict
            # constraints alongside the shared HBM total constraint.
            demands["l2.read"] = (read_min, "byte")
            demands["l2.write"] = (write_min, "byte")
    elif workload.residency == "hot_l2":
        demands["l2.read"] = (
            (work.tma_input_bytes + work.c_read_bytes_min)
            if empirical else read_min,
            "byte",
        )
        demands["l2.write"] = (write_min, "byte")

    if empirical:
        if workload.residency != "compute_oracle" and schedule.uses_tma:
            if (
                workload.residency == "cold_hbm"
                and resolved_tma_hbm is not None
            ):
                demands[resolved_tma_hbm] = (
                    work.tma_unique_input_bytes, "byte")
        tmem_warps = (
            schedule.tmem_consumer_warps
            if schedule.tmem_consumer_warps is not None
            else schedule.threads // 32
        )
        tmem_resource = (
            "tmem.readback"
            if schedule.tmem_load_registers == 16 and tmem_warps == 4
            else (
                f"tmem.readback.x{schedule.tmem_load_registers}."
                f"warps{tmem_warps}"
            )
        )
        demands[tmem_resource] = (work.accumulator_readback_bytes, "byte")
        if work.tmem_scale_ingress_bytes:
            demands["tmem.scale_ingress"] = (
                work.tmem_scale_ingress_bytes,
                "byte",
            )
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
    resolved_tma_ingress, resolved_tma_hbm, tma_contract_gap = (
        _resolve_tma_capacity_resources(workload, schedule, precision)
    )
    demands = _resource_demands(
        workload, schedule, work, precision, empirical=not strict
    )
    seconds: dict[str, float] = {}
    missing: list[str] = []
    conditions: list[str] = []
    selected: dict[str, Capacity] = {}
    if not strict and workload.residency != "compute_oracle" and schedule.uses_tma:
        if tma_contract_gap is not None:
            missing.append(tma_contract_gap)
        elif resolved_tma_ingress is None:
            missing.append(
                f"tma_ingress_capacity_contract:{schedule.schedule_id}"
            )
        if (
            workload.residency == "cold_hbm"
            and tma_contract_gap is None
            and resolved_tma_hbm is None
        ):
            missing.append(
                f"tma_hbm_capacity_contract:{schedule.schedule_id}"
            )
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

    if not strict:
        # An empirical resource point proves that the recorded kernel sustained
        # a rate; it does not remove any independently established physical
        # ceiling.  Intersect every empirical schedule with all applicable
        # conditional uppers.  This is especially important for cold-HBM GEMM:
        # separate read and write probes may overlap, but their combined traffic
        # must still respect the shared LPDDR bandwidth ceiling.
        ceiling_demands = _resource_demands(
            workload, schedule, work, precision, empirical=False
        )
        for resource, (quantity, unit) in ceiling_demands.items():
            cap = _select_capacity(capacities, resource, strict=True)
            if cap is None:
                continue
            if cap.work_unit != unit:
                raise ModelError(
                    f"{cap.capacity_id}: capacity unit {cap.work_unit} does not "
                    f"match {resource} ceiling demand unit {unit}"
                )
            hard_key = f"hard_upper:{resource}"
            seconds[hard_key] = (
                quantity / cap.rate_per_second
            )
            selected[hard_key] = cap
            if cap.condition:
                conditions.append(f"{cap.capacity_id}: {cap.condition}")

        if workload.residency != "compute_oracle" and schedule.uses_tma:
            ingress_resource = resolved_tma_ingress
            ingress_cap = (
                _select_capacity(capacities, ingress_resource, strict=False)
                if ingress_resource is not None
                else None
            )
            if ingress_resource is not None and ingress_cap is None:
                missing.append(ingress_resource)
            elif ingress_cap is not None:
                if ingress_cap.work_unit != "byte":
                    raise ModelError(
                        f"{ingress_cap.capacity_id}: per-SM TMA ingress "
                        f"capacity unit {ingress_cap.work_unit} is not byte"
                    )
                service_sms = max(1, hardware.sm_count // schedule.cta_group)
                task_bytes = work.tma_input_bytes / work.task_count
                per_group_rate = (
                    ingress_cap.rate_per_second * schedule.cta_group
                )
                seconds["tma.per_sm_parallel_span"] = (
                    task_bytes / per_group_rate
                )
                seconds["tma.per_sm_parallel_makespan"] = (
                    ceil_div(work.task_count, service_sms)
                    * task_bytes / per_group_rate
                )
                if ingress_cap.condition:
                    conditions.append(
                        f"{ingress_cap.capacity_id}: "
                        f"{ingress_cap.condition}"
                    )
                selected[ingress_resource] = ingress_cap

    selected_ids = {
        resource: cap.capacity_id
        for resource, cap in sorted(selected.items())
    }
    selected_kinds = {
        resource: cap.evidence_kind.value
        for resource, cap in sorted(selected.items())
    }
    selected_qualifications = {
        resource: cap.qualification
        for resource, cap in sorted(selected.items())
    }

    empirical_compute_resource = (
        f"{precision.compute_resource}.m{schedule.mma_m}n{schedule.mma_n}")
    compute_cap = selected.get(empirical_compute_resource)
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
            selected_capacity_ids=selected_ids,
            selected_capacity_evidence_kinds=selected_kinds,
            selected_capacity_qualifications=selected_qualifications,
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
        selected_capacity_ids=selected_ids,
        selected_capacity_evidence_kinds=selected_kinds,
        selected_capacity_qualifications=selected_qualifications,
    )


def predict_pipeline_worker_seconds(
    profile: PipelineProfile,
    *,
    k_tiles: int,
    output_tasks: int,
) -> float:
    """Predict the slowest persistent worker's causal completion time.

    ``k_tiles`` is the number of BK-sized reduction tiles consumed by one
    output task. ``output_tasks`` is the number of output tiles assigned to
    the slowest resident worker.  The recurrence preserves the measured
    steady-state TMA/MMA interval, the two-accumulator reuse dependency, and
    serialized accumulator readback/store completion.
    """

    profile.validate()
    for value, label in ((k_tiles, "k_tiles"), (output_tasks, "output_tasks")):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ModelError(
                f"{profile.profile_id}: {label} must be a positive integer"
            )
    if k_tiles > profile.maximum_k_tiles:
        raise ModelError(
            f"{profile.profile_id}: k_tiles={k_tiles} exceeds calibrated/holdout "
            f"range {profile.maximum_k_tiles}"
        )
    if output_tasks > profile.maximum_output_tasks_per_worker:
        raise ModelError(
            f"{profile.profile_id}: output_tasks={output_tasks} exceeds "
            f"calibrated/holdout range "
            f"{profile.maximum_output_tasks_per_worker}"
        )

    previous_mma_done = 0.0
    epilogue_done: list[float] = []
    for task in range(output_tasks):
        first_mma_done = (
            profile.joint_first_mma_completion_seconds
            if task == 0
            else previous_mma_done + profile.joint_completion_interval_seconds
        )
        if task >= profile.accumulator_buffers:
            first_mma_done = max(
                first_mma_done,
                epilogue_done[task - profile.accumulator_buffers]
                + profile.joint_completion_interval_seconds,
            )
        last_mma_done = (
            first_mma_done
            + (k_tiles - 1) * profile.joint_completion_interval_seconds
        )
        epilogue_start = max(
            last_mma_done,
            epilogue_done[-1] if epilogue_done else 0.0,
        )
        epilogue_done.append(
            epilogue_start + profile.epilogue_latency_seconds
        )
        previous_mma_done = last_mma_done
    return epilogue_done[-1]


def _evaluate_causal_pipeline(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
    precision: PrecisionSpec,
    work: WorkAccounting,
    profiles: list[PipelineProfile],
) -> LayerResult:
    resource = schedule.causal_pipeline_resource
    if resource is None:
        return LayerResult(
            status="insufficient_evidence",
            seconds=None,
            performance_per_second=None,
            performance_unit=f"{precision.compute_work_unit}/s",
            missing_resources=[
                f"causal_pipeline_contract:{schedule.schedule_id}"
            ],
        )
    if workload.residency == "compute_oracle":
        return LayerResult(
            status="insufficient_evidence",
            seconds=None,
            performance_per_second=None,
            performance_unit=f"{precision.compute_work_unit}/s",
            missing_resources=["causal_pipeline_residency:compute_oracle"],
        )

    same_pipeline = [
        profile
        for profile in profiles
        if profile.resource == resource
        and profile.schedule_id == schedule.schedule_id
        and profile.stages == schedule.stages
    ]
    exact = [
        profile for profile in same_pipeline
        if precision.precision_id in profile.precision_ids
    ]
    qualified = [
        profile for profile in exact if profile.is_closure_qualified
    ]
    if not qualified:
        conditions = [
            f"{profile.profile_id}: {profile.qualification}"
            for profile in exact
        ]
        conditions.extend(
            f"{profile.profile_id}: precision_ids={','.join(profile.precision_ids)}"
            for profile in same_pipeline
            if precision.precision_id not in profile.precision_ids
        )
        return LayerResult(
            status="insufficient_evidence",
            seconds=None,
            performance_per_second=None,
            performance_unit=f"{precision.compute_work_unit}/s",
            missing_resources=[
                f"{resource}:precision={precision.precision_id}"
            ],
            conditions=sorted(conditions),
        )

    predictions: list[tuple[float, int, PipelineProfile]] = []
    range_errors: list[str] = []
    for profile in qualified:
        service_units = (
            max(1, hardware.sm_count // schedule.cta_group)
            * profile.resident_ctas_per_sm
        )
        output_tasks = ceil_div(work.task_count, service_units)
        try:
            predicted = predict_pipeline_worker_seconds(
                profile,
                k_tiles=work.k_tiles,
                output_tasks=output_tasks,
            )
        except ModelError as error:
            range_errors.append(str(error))
            continue
        predictions.append((predicted, output_tasks, profile))
    if not predictions:
        return LayerResult(
            status="insufficient_evidence",
            seconds=None,
            performance_per_second=None,
            performance_unit=f"{precision.compute_work_unit}/s",
            missing_resources=[f"{resource}:profile_range"],
            conditions=sorted(range_errors),
        )

    # A profile is an indivisible joint measurement contract.  If repeated
    # closure-qualified campaigns exist, select the fastest complete profile;
    # never mix the best component from different campaigns.
    total, output_tasks, selected = min(predictions, key=lambda row: row[0])
    component_seconds = {
        "causal.pipeline.joint_first_mma":
            selected.joint_first_mma_completion_seconds,
        "causal.pipeline.one_k_tile_interval":
            selected.joint_completion_interval_seconds,
        "causal.pipeline.one_epilogue": selected.epilogue_latency_seconds,
        "causal.pipeline.persistent_worker": total,
    }
    condition = (
        f"{selected.profile_id}: hot-L2 joint profile; "
        f"{hardware.sm_count * selected.resident_ctas_per_sm} resident workers; "
        f"slowest worker has {output_tasks} output task(s), "
        f"{work.k_tiles} K tile(s) per task"
    )
    return LayerResult(
        status="ok",
        seconds=total,
        performance_per_second=work.useful_compute_work / total,
        performance_unit=f"{precision.compute_work_unit}/s",
        bottlenecks=["causal.pipeline.persistent_worker"],
        resource_seconds=component_seconds,
        conditions=[condition],
        selected_capacity_ids={resource: selected.profile_id},
        selected_capacity_evidence_kinds={
            resource: selected.evidence_kind.value
        },
        selected_capacity_qualifications={
            resource: selected.qualification
        },
    )


def _combine_empirical_layers(
    resource_layer: LayerResult,
    causal_layer: LayerResult,
    work: WorkAccounting,
) -> LayerResult:
    resource_seconds = dict(resource_layer.resource_seconds)
    resource_seconds.update(causal_layer.resource_seconds)
    missing = sorted(set(
        resource_layer.missing_resources + causal_layer.missing_resources
    ))
    conditions = sorted(set(resource_layer.conditions + causal_layer.conditions))
    selected_ids = {
        **resource_layer.selected_capacity_ids,
        **causal_layer.selected_capacity_ids,
    }
    selected_kinds = {
        **resource_layer.selected_capacity_evidence_kinds,
        **causal_layer.selected_capacity_evidence_kinds,
    }
    selected_qualifications = {
        **resource_layer.selected_capacity_qualifications,
        **causal_layer.selected_capacity_qualifications,
    }
    if resource_layer.seconds is None or causal_layer.seconds is None or missing:
        return LayerResult(
            status="insufficient_evidence",
            seconds=None,
            performance_per_second=None,
            performance_unit=f"{work.compute_work_unit}/s",
            resource_seconds=dict(sorted(resource_seconds.items())),
            missing_resources=missing,
            conditions=conditions,
            selected_capacity_ids=selected_ids,
            selected_capacity_evidence_kinds=selected_kinds,
            selected_capacity_qualifications=selected_qualifications,
        )

    total = max(resource_layer.seconds, causal_layer.seconds)
    bottlenecks: list[str] = []
    if math.isclose(resource_layer.seconds, total, rel_tol=1e-9):
        bottlenecks.extend(
            f"resource:{name}" for name in resource_layer.bottlenecks
        )
    if math.isclose(causal_layer.seconds, total, rel_tol=1e-9):
        bottlenecks.extend(
            f"causal:{name}" for name in causal_layer.bottlenecks
        )
    return LayerResult(
        status="ok",
        seconds=total,
        performance_per_second=work.useful_compute_work / total,
        performance_unit=f"{work.compute_work_unit}/s",
        bottlenecks=sorted(bottlenecks),
        resource_seconds=dict(sorted(resource_seconds.items())),
        conditions=conditions,
        selected_capacity_ids=selected_ids,
        selected_capacity_evidence_kinds=selected_kinds,
        selected_capacity_qualifications=selected_qualifications,
    )


def evaluate(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
    capacities: list[Capacity],
    pipeline_profiles: Iterable[PipelineProfile] = (),
) -> ModelResult:
    precisions = precision_specs()
    workload.validate(precisions)
    hardware.validate()
    precision = precisions[workload.precision_id]
    for cap in capacities:
        cap.validate()
    profiles = list(pipeline_profiles)
    for profile in profiles:
        profile.validate()
    work = account_work(workload, schedule, precision)
    empirical_resource = _evaluate_layer(
        workload, schedule, hardware, precision, work, capacities, strict=False
    )
    causal = _evaluate_causal_pipeline(
        workload, schedule, hardware, precision, work, profiles
    )
    return ModelResult(
        workload_id=workload.workload_id,
        schedule_id=schedule.schedule_id,
        work=work,
        conditional_upper=_evaluate_layer(
            workload, schedule, hardware, precision, work, capacities, strict=True
        ),
        empirical_envelope=empirical_resource,
        causal_pipeline=causal,
        empirical_ideal_envelope=_combine_empirical_layers(
            empirical_resource, causal, work
        ),
    )


def evaluate_manifest(
    workload: Workload,
    schedules: Iterable[Schedule],
    hardware: Hardware,
    capacities: list[Capacity],
    pipeline_profiles: Iterable[PipelineProfile] = (),
) -> WorkloadEnvelope:
    profiles = list(pipeline_profiles)
    results: list[ModelResult] = []
    rejected: list[dict[str, str]] = []
    for schedule in schedules:
        try:
            results.append(evaluate(
                workload, schedule, hardware, capacities, profiles
            ))
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
    empirical_resource, empirical_resource_id = best_layer(
        "empirical_envelope"
    )
    causal, causal_id = best_layer("causal_pipeline")
    empirical, empirical_id = best_layer("empirical_ideal_envelope")
    return WorkloadEnvelope(
        workload_id=workload.workload_id,
        valid_schedule_count=len(results),
        rejected_schedule_count=len(rejected),
        manifest_conditional_upper=strict,
        empirical_ideal_envelope=empirical,
        conditional_schedule_id=strict_id,
        empirical_schedule_id=empirical_id,
        rejected=rejected,
        manifest_empirical_resource_envelope=empirical_resource,
        empirical_resource_schedule_id=empirical_resource_id,
        causal_pipeline_envelope=causal,
        causal_pipeline_schedule_id=causal_id,
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
        source_path: Path | None = None
        if repo_root is not None:
            try:
                source_path = resolve_repo_artifact(repo_root, cap.source_path)
            except ModelError as error:
                findings.append({
                    "severity": "error",
                    "code": "invalid_source_path",
                    "message": f"{cap.capacity_id}: {error}",
                })
        if source_path is not None and cap.source_path.lower().endswith(".csv"):
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
        elif source_path is not None:
            source_text = source_path.read_text(
                encoding="utf-8", errors="replace")
            if cap.source_locator not in source_text:
                findings.append(
                    {
                        "severity": "error",
                        "code": "text_locator_no_match",
                        "message": (
                            f"{cap.capacity_id}: {cap.source_locator}"
                        ),
                    }
                )
        if repo_root is not None and cap.is_closure_qualified:
            for artifact_path in cap.artifact_paths:
                try:
                    resolve_repo_artifact(repo_root, artifact_path)
                except ModelError as error:
                    findings.append(
                        {
                            "severity": "error",
                            "code": "invalid_closure_artifact_path",
                            "message": f"{cap.capacity_id}: {error}",
                        }
                    )
    return findings


def audit_pipeline_profiles(
    profiles: Iterable[PipelineProfile],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, str]]:
    """Audit profile semantics and repository-relative provenance files."""

    findings: list[dict[str, str]] = []
    seen: set[str] = set()
    for profile in profiles:
        try:
            profile.validate()
        except ModelError as error:
            findings.append({
                "severity": "error",
                "code": "invalid_pipeline_profile",
                "message": str(error),
            })
            continue
        if profile.profile_id in seen:
            findings.append({
                "severity": "error",
                "code": "duplicate_pipeline_profile_id",
                "message": profile.profile_id,
            })
        seen.add(profile.profile_id)
        if repo_root is None:
            continue
        try:
            resolve_repo_artifact(repo_root, profile.source_path)
        except ModelError as error:
            findings.append({
                "severity": "error",
                "code": "invalid_pipeline_source_path",
                "message": f"{profile.profile_id}: {error}",
            })
        for artifact_path in profile.artifact_paths:
            try:
                resolve_repo_artifact(repo_root, artifact_path)
            except ModelError as error:
                findings.append({
                    "severity": "error",
                    "code": "invalid_pipeline_artifact_path",
                    "message": f"{profile.profile_id}: {error}",
                })
    return findings
