from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import (
    Capacity,
    EvidenceKind,
    Hardware,
    ModelError,
    PipelineProfile,
    Schedule,
    Workload,
)
from .observations import ObservedBest


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_hardware(path: Path) -> Hardware:
    return Hardware(**read_json(path))


def capacities_from_rows(rows: list[dict[str, Any]]) -> list[Capacity]:
    capacities: list[Capacity] = []
    for row in rows:
        data = dict(row)
        data["evidence_kind"] = EvidenceKind(data["evidence_kind"])
        data["artifact_paths"] = tuple(data.get("artifact_paths", ()))
        for key in (
            "applicable_precision_ids", "applicable_mma_shapes",
            "applicable_cta_groups", "applicable_sm_counts",
            "applicable_hardware_ids", "applicable_operating_modes",
            "applicable_clock_hz", "applicable_residencies",
            "applicable_tma_tile_bytes", "applicable_tmem_load_registers",
            "applicable_readback_warps", "applicable_tma_destination_slots",
            "applicable_threads_per_cta", "applicable_resident_ctas_per_sm",
            "applicable_read_write_ratios", "applicable_access_patterns",
            "applicable_schedule_ids", "applicable_workload_ids",
        ):
            data[key] = tuple(data.get(key, ()))
        capacities.append(Capacity(**data))
    return capacities


def load_capacities(path: Path) -> list[Capacity]:
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("capacities", [])
    if not isinstance(payload, list):
        raise ModelError(f"{path}: capacity input must be a list or object")
    return capacities_from_rows(payload)


def load_capacity_files(paths: list[Path]) -> list[Capacity]:
    capacities: list[Capacity] = []
    for path in paths:
        capacities.extend(load_capacities(path))
    return capacities


def pipeline_profiles_from_rows(
    rows: list[dict[str, Any]],
) -> list[PipelineProfile]:
    profiles: list[PipelineProfile] = []
    for row in rows:
        data = dict(row)
        data.pop("schema_version", None)
        data["evidence_kind"] = EvidenceKind(data["evidence_kind"])
        data["precision_ids"] = tuple(data["precision_ids"])
        data["validation"] = tuple(dict(item) for item in data["validation"])
        data["artifact_paths"] = tuple(data.get("artifact_paths", ()))
        for key in (
            "applicable_sm_counts", "applicable_hardware_ids",
            "applicable_operating_modes", "applicable_clock_hz",
        ):
            data[key] = tuple(data.get(key, ()))
        profiles.append(PipelineProfile(**data))
    return profiles


def load_pipeline_profiles(path: Path) -> list[PipelineProfile]:
    raw = read_json(path)
    if isinstance(raw, dict):
        raw = raw.get("pipeline_profiles", [raw])
    if not isinstance(raw, list):
        raise ModelError(f"{path}: pipeline profile input must be a list or object")
    return pipeline_profiles_from_rows(raw)


def observations_from_rows(rows: list[dict[str, Any]]) -> list[ObservedBest]:
    observations: list[ObservedBest] = []
    for row in rows:
        data = dict(row)
        data["artifact_paths"] = tuple(data.get("artifact_paths", ()))
        observations.append(ObservedBest(**data))
    return observations


def load_closure_inputs(path: Path) -> tuple[list[Capacity], list[ObservedBest]]:
    data = read_json(path)
    if data.get("schema_version") != 1:
        raise ModelError(f"{path}: unsupported closure input schema")
    return (
        capacities_from_rows(data.get("capacities", [])),
        observations_from_rows(data.get("observed_best", [])),
    )


def load_workloads(path: Path) -> list[Workload]:
    return [Workload(**row) for row in read_json(path)]


def load_schedules(path: Path) -> list[Schedule]:
    schedules: list[Schedule] = []
    for row in read_json(path):
        data = dict(row)
        data["supported_precisions"] = tuple(data.get("supported_precisions", ()))
        data["tma_contract_family_by_precision"] = dict(
            data.get("tma_contract_family_by_precision", {})
        )
        data["tma_contract_row_stride_elements"] = tuple(
            data.get("tma_contract_row_stride_elements", ())
        )
        schedules.append(Schedule(**data))
    return schedules
