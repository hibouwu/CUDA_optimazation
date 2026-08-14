from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import Capacity, EvidenceKind, Hardware, ModelError, Schedule, Workload
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
        capacities.append(Capacity(**data))
    return capacities


def load_capacities(path: Path) -> list[Capacity]:
    return capacities_from_rows(read_json(path))


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
        schedules.append(Schedule(**data))
    return schedules
