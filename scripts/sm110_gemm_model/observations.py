from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .model import ModelError


PRECISION_MAP = {
    "fp16->fp32": "fp16_f32",
    "FP8": "e4m3_f32",
    "INT8": "s8_s32",
    "MXFP4": "mxfp4_f32",
    "NVFP4": "nvfp4_f32",
}


@dataclass(frozen=True)
class ObservedBest:
    observation_id: str
    precision_id: str
    m: int
    n: int
    k: int
    backend_id: str
    reference: str
    performance_reference_relation: str
    trial_count: int
    matched_count: int
    median_per_second: float
    maximum_per_second: float
    minimum_per_second: float
    performance_unit: str
    source_path: str
    residency: str = "warm_repeated_unspecified"
    timed_scope: str = "device_kernel"
    qualification: str = "snapshot_only"
    selection_rule: str = "largest median among fully matched backends"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _performance_reference_relation(precision_id: str, reference: str) -> str:
    lower = reference.lower()
    if precision_id in {"nvfp4_f32", "mxfp4_f32"} and "fp16" in lower:
        return "cross_precision_denominator"
    if precision_id == "e4m3_f32" and "fp8" in lower:
        return "same_precision"
    if precision_id == "s8_s32" and "int8" in lower:
        return "same_precision"
    if precision_id == "fp16_f32" and "tensor core" in lower:
        return "same_precision"
    return "unspecified"


def _performance_unit(precision_id: str) -> str:
    return "operation/s" if precision_id in {"s8_s32", "u8_s32"} else "flop/s"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _normalize(path: Path) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in _read_rows(path):
        precision_raw = row.get("Precision", "")
        precision_id = PRECISION_MAP.get(precision_raw)
        if precision_id is None:
            continue
        status = row.get("Status", "ok")
        gflops = row.get("GFLOPS", "")
        if status != "ok" or not gflops:
            continue
        n = int(row["N"])
        normalized.append(
            {
                "precision_id": precision_id,
                "m": n,
                "n": n,
                "k": n,
                "backend_id": row["BackendId"],
                "reference": row["Reference"],
                "matched": row.get("Matched") == "1",
                "flop_per_second": float(gflops) * 1e9,
            }
        )
    return normalized


def summarize_observed_csvs(
    paths: Iterable[Path],
    *,
    repo_root: Path,
    minimum_trials: int = 10,
) -> list[ObservedBest]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    path_by_group: dict[tuple[object, ...], Path] = {}
    for path in paths:
        for row in _normalize(path):
            key = (
                row["precision_id"],
                row["m"],
                row["n"],
                row["k"],
                row["backend_id"],
                row["reference"],
            )
            groups.setdefault(key, []).append(row)
            path_by_group[key] = path

    eligible: list[ObservedBest] = []
    for key, rows in groups.items():
        precision_id, m, n, k, backend_id, reference = key
        values = [float(row["flop_per_second"]) for row in rows]
        matched_count = sum(bool(row["matched"]) for row in rows)
        if len(rows) < minimum_trials or matched_count != len(rows):
            continue
        source = path_by_group[key]
        try:
            source_path = str(source.resolve().relative_to(repo_root.resolve()))
        except ValueError as exc:
            raise ModelError(f"observation source is outside repo: {source}") from exc
        eligible.append(
            ObservedBest(
                observation_id=f"{precision_id}_m{m}_n{n}_k{k}",
                precision_id=str(precision_id),
                m=int(m),
                n=int(n),
                k=int(k),
                backend_id=str(backend_id),
                reference=str(reference),
                performance_reference_relation=_performance_reference_relation(
                    str(precision_id), str(reference)
                ),
                trial_count=len(rows),
                matched_count=matched_count,
                median_per_second=statistics.median(values),
                maximum_per_second=max(values),
                minimum_per_second=min(values),
                performance_unit=_performance_unit(str(precision_id)),
                source_path=source_path,
            )
        )

    best: dict[tuple[str, int, int, int], ObservedBest] = {}
    for row in eligible:
        key = (row.precision_id, row.m, row.n, row.k)
        current = best.get(key)
        if current is None or row.median_per_second > current.median_per_second:
            best[key] = row
    return sorted(best.values(), key=lambda row: (row.precision_id, row.m, row.n, row.k))


def audit_observed_against_upper(
    observed: ObservedBest,
    upper_per_second: float | None,
    *,
    upper_performance_unit: str | None = None,
    upper_residency: str | None = None,
    relative_tolerance: float = 0.02,
) -> list[dict[str, str]]:
    if upper_performance_unit is not None and upper_performance_unit != observed.performance_unit:
        return [
            {
                "severity": "error",
                "code": "performance_unit_mismatch",
                "message": (
                    f"{observed.observation_id}: {observed.performance_unit} versus "
                    f"{upper_performance_unit}"
                ),
            }
        ]
    if upper_residency is not None and upper_residency != observed.residency:
        return [
            {
                "severity": "warning",
                "code": "residency_mismatch",
                "message": (
                    f"{observed.observation_id}: observed {observed.residency} versus "
                    f"upper {upper_residency}; comparison suppressed"
                ),
            }
        ]
    if upper_per_second is None:
        return [
            {
                "severity": "warning",
                "code": "upper_unavailable",
                "message": observed.observation_id,
            }
        ]
    if observed.maximum_per_second > upper_per_second * (1.0 + relative_tolerance):
        return [
            {
                "severity": "error",
                "code": "observed_exceeds_conditional_upper",
                "message": (
                    f"{observed.observation_id}: observed max "
                    f"{observed.maximum_per_second:g} > upper "
                    f"{upper_per_second:g}"
                ),
            }
        ]
    return []
