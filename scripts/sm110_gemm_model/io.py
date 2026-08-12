from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import Capacity, EvidenceKind, Hardware, Schedule, Workload


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_hardware(path: Path) -> Hardware:
    return Hardware(**read_json(path))


def load_capacities(path: Path) -> list[Capacity]:
    capacities: list[Capacity] = []
    for row in read_json(path):
        data = dict(row)
        data["evidence_kind"] = EvidenceKind(data["evidence_kind"])
        data["artifact_paths"] = tuple(data.get("artifact_paths", ()))
        capacities.append(Capacity(**data))
    return capacities


def load_workloads(path: Path) -> list[Workload]:
    return [Workload(**row) for row in read_json(path)]


def load_schedules(path: Path) -> list[Schedule]:
    schedules: list[Schedule] = []
    for row in read_json(path):
        data = dict(row)
        data["supported_precisions"] = tuple(data.get("supported_precisions", ()))
        schedules.append(Schedule(**data))
    return schedules
