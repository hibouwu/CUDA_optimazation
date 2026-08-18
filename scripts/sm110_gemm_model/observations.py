from __future__ import annotations

import csv
import dataclasses
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .model import (
    Hardware, ModelError, precision_specs, resolve_repo_artifact,
)
from .suite import SuiteLinkage


PRECISION_MAP = {
    "fp16->fp32": "fp16_f32",
    "FP8": "e4m3_f32",
    "INT8": "s8_s32",
    "MXFP4": "mxfp4_f32",
    "NVFP4": "nvfp4_f32",
}

LEGACY_NUMERICAL_CONTRACTS = {
    ("fp16_f32", "fp_accumulator"): "fp16_f32",
    ("e4m3_f32", "fp8_e4m3_f32"): "e4m3_f32",
    ("s8_s32", "s8_s32_exact"): "s8_s32",
}


def _canonical_numerical_contract(
    precision_id: str, contract: str,
) -> str:
    canonical = LEGACY_NUMERICAL_CONTRACTS.get(
        (precision_id, contract), contract
    )
    if canonical != precision_id:
        raise ModelError(
            f"{precision_id}: numerical contract {contract!r} does not match "
            "the executable precision contract"
        )
    return canonical


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
    source_locator: str = ""
    artifact_paths: tuple[str, ...] = ()
    run_id: str = ""
    reference_median_per_second: float | None = None
    reference_maximum_per_second: float | None = None
    reference_minimum_per_second: float | None = None
    ratio_of_paired_medians: float | None = None
    residency: str = "warm_repeated_unspecified"
    timed_scope: str = "device_kernel"
    qualification: str = "snapshot_only"
    selection_rule: str = "largest median among fully matched backends"
    correctness_reference: str = "unspecified"
    correctness_reference_relation: str = "unspecified"
    numerical_contract: str = "unspecified"
    calibration_split: str = "unspecified"
    transpose_a: bool = False
    transpose_b: bool = False
    alpha: float = 1.0
    beta: float = 0.0
    epilogue: str = "none"
    output_mode: str = "accumulator"
    arithmetic_path: str = "unspecified"
    hardware_id: str = "unspecified"
    sm_count: int = 0
    operating_mode: str = "unspecified"

    def validate(self, *, repo_root: Path | None = None) -> None:
        if (any(not isinstance(value, int) or isinstance(value, bool)
                for value in (self.m, self.n, self.k))
                or min(self.m, self.n, self.k) <= 0):
            raise ModelError(
                f"{self.observation_id}: dimensions must be positive integers")
        if (any(not isinstance(value, int) or isinstance(value, bool)
                for value in (self.trial_count, self.matched_count))
                or self.trial_count <= 0
                or self.matched_count != self.trial_count):
            raise ModelError(
                f"{self.observation_id}: all declared trials must be matched")
        values = (
            self.minimum_per_second,
            self.median_per_second,
            self.maximum_per_second,
        )
        if (not all(math.isfinite(value) and value > 0 for value in values)
                or not self.minimum_per_second <= self.median_per_second
                <= self.maximum_per_second):
            raise ModelError(
                f"{self.observation_id}: invalid min/median/max performance")
        if self.performance_unit not in {"flop/s", "operation/s"}:
            raise ModelError(
                f"{self.observation_id}: invalid performance unit")
        if self.qualification not in {"snapshot_only", "closure_qualified"}:
            raise ModelError(
                f"{self.observation_id}: invalid qualification")
        if self.precision_id not in precision_specs():
            raise ModelError(
                f"{self.observation_id}: unknown precision {self.precision_id}"
            )
        if self.residency not in {
            "cold_hbm", "hot_l2", "compute_oracle",
            "warm_repeated_unspecified", "warm_repeated_device_gemm",
        }:
            raise ModelError(
                f"{self.observation_id}: unsupported residency")
        if self.timed_scope not in {
            "device_kernel", "device_kernel_plus_launch"
        }:
            raise ModelError(
                f"{self.observation_id}: unsupported timed scope")
        if not isinstance(self.transpose_a, bool) or not isinstance(
            self.transpose_b, bool
        ):
            raise ModelError(
                f"{self.observation_id}: transpose flags must be boolean")
        if (
            not math.isfinite(self.alpha)
            or not math.isfinite(self.beta)
            or self.alpha == 0.0
        ):
            raise ModelError(
                f"{self.observation_id}: alpha/beta contract is invalid")
        if self.epilogue not in {
            "none", "bias", "relu", "gelu", "residual", "requant"
        }:
            raise ModelError(
                f"{self.observation_id}: unsupported epilogue")
        if self.output_mode not in {"accumulator", "packed_quantized"}:
            raise ModelError(
                f"{self.observation_id}: unsupported output mode")
        if self.qualification == "closure_qualified":
            if self.trial_count < 10:
                raise ModelError(
                    f"{self.observation_id}: closure requires at least 10 trials")
            if (not self.run_id or not self.source_locator
                    or not self.artifact_paths):
                raise ModelError(
                    f"{self.observation_id}: closure provenance is incomplete")
            if self.performance_reference_relation != "same_precision":
                raise ModelError(
                    f"{self.observation_id}: closure requires same-precision denominator")
            if (self.reference_median_per_second is None
                    or not math.isfinite(self.reference_median_per_second)
                    or self.reference_median_per_second <= 0):
                raise ModelError(
                    f"{self.observation_id}: closure reference median is invalid")
            reference_values = (
                self.reference_minimum_per_second,
                self.reference_median_per_second,
                self.reference_maximum_per_second,
            )
            if (not all(value is not None and math.isfinite(value) and value > 0
                        for value in reference_values)
                    or not self.reference_minimum_per_second
                    <= self.reference_median_per_second
                    <= self.reference_maximum_per_second):
                raise ModelError(
                    f"{self.observation_id}: closure reference min/median/max "
                    "performance is invalid")
            expected_ratio = (
                self.median_per_second / self.reference_median_per_second)
            if (self.ratio_of_paired_medians is None
                    or not math.isclose(
                        self.ratio_of_paired_medians, expected_ratio,
                        rel_tol=1e-12)):
                raise ModelError(
                    f"{self.observation_id}: paired-median ratio is invalid")
            if self.correctness_reference_relation != "independent_same_contract":
                raise ModelError(
                    f"{self.observation_id}: closure requires an independent "
                    "same-contract correctness reference"
                )
            if not self.correctness_reference or self.correctness_reference == "unspecified":
                raise ModelError(
                    f"{self.observation_id}: closure correctness reference is "
                    "missing"
                )
            _canonical_numerical_contract(
                self.precision_id, self.numerical_contract
            )
            if self.calibration_split not in {"calibration", "holdout"}:
                raise ModelError(
                    f"{self.observation_id}: closure requires a calibration "
                    "or holdout split"
                )
            if self.arithmetic_path not in {
                "tensor_core", "classical_non_tensor_or_mixed"
            }:
                raise ModelError(
                    f"{self.observation_id}: closure requires an audited "
                    "arithmetic path"
                )
            if (
                self.hardware_id == "unspecified"
                or self.sm_count <= 0
                or self.operating_mode == "unspecified"
            ):
                raise ModelError(
                    f"{self.observation_id}: closure requires an exact hardware "
                    "ID, SM count, and operating mode"
                )
        if repo_root is not None:
            try:
                source = resolve_repo_artifact(repo_root, self.source_path)
            except ModelError as error:
                raise ModelError(
                    f"{self.observation_id}: invalid source path: {error}") from error
            if (self.source_locator
                    and self.source_locator not in source.read_text(
                        encoding="utf-8", errors="replace")):
                raise ModelError(
                    f"{self.observation_id}: source locator does not match")
            if self.qualification == "closure_qualified":
                for artifact_path in self.artifact_paths:
                    try:
                        resolve_repo_artifact(repo_root, artifact_path)
                    except ModelError as error:
                        raise ModelError(
                            f"{self.observation_id}: invalid closure artifact: "
                            f"{error}") from error

    @property
    def is_closure_qualified(self) -> bool:
        return self.qualification == "closure_qualified"

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


def audit_observations(
    observations: Iterable[ObservedBest], *, repo_root: Path | None = None
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    seen: set[str] = set()
    for observation in observations:
        if observation.observation_id in seen:
            findings.append({
                "severity": "error", "code": "duplicate_observation_id",
                "message": observation.observation_id,
            })
        seen.add(observation.observation_id)
        try:
            observation.validate(repo_root=repo_root)
        except ModelError as error:
            findings.append({
                "severity": "error", "code": "invalid_observation",
                "message": str(error),
            })
    return findings


def summarize_closure_campaign(
    run_dir: Path, *, repo_root: Path, hardware: Hardware,
) -> list[ObservedBest]:
    repo_root = repo_root.resolve()
    run_dir = run_dir.resolve()
    try:
        relative_run = run_dir.relative_to(repo_root)
    except ValueError as error:
        raise ModelError(f"campaign directory is outside repo: {run_dir}") from error
    auditor = repo_root / "microbench/sm110_full_gemm_campaign/audit_campaign.py"
    audit = subprocess.run(
        [sys.executable, str(auditor), str(run_dir)], cwd=repo_root,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False)
    if audit.returncode:
        raise ModelError(f"full-GEMM campaign audit failed:\n{audit.stdout}")
    spec = json.loads((run_dir / "run_spec.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    support_path = repo_root / str(spec["support_manifest"])
    support = {
        row["precision_id"]: row
        for row in json.loads(support_path.read_text())["precisions"]
    }
    hardware.validate()
    if (
        hardware.hardware_id != "thor_t5000_sm110_20sm"
        or hardware.sm_count != 20
        or hardware.operating_mode != "MAXN"
    ):
        raise ModelError(
            "the SM110 closure campaign can only be imported into the "
            "thor_t5000_sm110_20sm / 20 SM / MAXN hardware contract"
        )
    cases = {case["id"]: case for case in spec["cases"]}
    observations: list[ObservedBest] = []
    for result in summary["results"]:
        case_id = str(result["case_id"])
        case = cases[case_id]
        precision_id = str(result["precision_id"])
        contract = support[precision_id]
        correctness = contract.get("numerical_reference") or {}
        denominator = contract.get("performance_denominator") or {}
        if not (
            correctness.get("same_input_precision") is True
            and correctness.get("same_output_type") is True
            and correctness.get("backend_id")
        ):
            raise ModelError(
                f"{case_id}: support manifest does not prove a same-contract "
                "correctness reference"
            )
        same_denominator = bool(
            denominator.get("same_precision") is True
            and denominator.get("status") == "ready")
        reference = str(denominator.get("backend_id", "unavailable"))
        n = int(result["n"])
        trial_path = run_dir / "cases" / case_id / "trials.jsonl"
        trial_rows = [
            json.loads(line)
            for line in trial_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        try:
            reference_rates = [
                float(row["reference_rate_per_second"])
                for row in trial_rows
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ModelError(
                f"{case_id}: invalid reference rate in full-GEMM trials"
            ) from error
        if len(reference_rates) != int(result["trial_count"]) or not all(
            math.isfinite(rate) and rate > 0 for rate in reference_rates
        ):
            raise ModelError(
                f"{case_id}: reference trial evidence is incomplete"
            )
        reference_median = statistics.median(reference_rates)
        if (
            "reference_rate_per_second_median" in result
            and not math.isclose(
                float(result["reference_rate_per_second_median"]),
                reference_median,
                rel_tol=1e-12,
            )
        ):
            raise ModelError(
                f"{case_id}: reference median differs from raw trials"
            )
        artifact_paths = (
            str(relative_run / "run_spec.json"),
            str(relative_run / "environment.json"),
            str(relative_run / "environment_snapshots.jsonl"),
            str(relative_run / "summary.json"),
            str(relative_run / "COMPLETE"),
            str(relative_run / "cases" / case_id / "trials.jsonl"),
            str(relative_run / "cases" / case_id / "result.json"),
            str(relative_run / str(result["sass_path"])),
            str(relative_run / str(result["binary_hash_path"])),
        )
        row = ObservedBest(
            observation_id=f"{precision_id}_m{n}_n{n}_k{n}_{case_id}",
            precision_id=precision_id, m=n, n=n, k=n,
            backend_id=str(result["backend_id"]), reference=reference,
            performance_reference_relation=(
                "same_precision" if same_denominator
                else "cross_precision_denominator"),
            trial_count=int(result["trial_count"]),
            matched_count=int(result["trial_count"]),
            median_per_second=float(result["custom_rate_per_second_median"]),
            maximum_per_second=float(result["custom_rate_per_second_max"]),
            minimum_per_second=float(result["custom_rate_per_second_min"]),
            performance_unit=(
                "operation/s" if result["work_unit"] == "operation" else "flop/s"),
            source_path=str(relative_run / "summary.json"),
            source_locator=f'"case_id": "{case_id}"',
            artifact_paths=artifact_paths,
            run_id=str(spec["run_id"]),
            reference_median_per_second=reference_median,
            reference_minimum_per_second=min(reference_rates),
            reference_maximum_per_second=max(reference_rates),
            ratio_of_paired_medians=float(result["ratio_of_paired_medians"]),
            residency="warm_repeated_unspecified",
            timed_scope="device_kernel",
            qualification="snapshot_only",
            correctness_reference=str(correctness.get("backend_id", "unspecified")),
            correctness_reference_relation="independent_same_contract",
            numerical_contract=_canonical_numerical_contract(
                precision_id, str(case["numerical_contract"])
            ),
            calibration_split=str(result["split"]),
            arithmetic_path="tensor_core",
            hardware_id="unspecified", sm_count=0,
            operating_mode=hardware.operating_mode,
        )
        row.validate(repo_root=repo_root)
        observations.append(row)
    return sorted(observations, key=lambda row: (row.precision_id, row.n))


def qualify_observations_for_suite(
    observations: Iterable[ObservedBest], *, linkage: SuiteLinkage,
    full_gemm_run_dir: Path,
    repo_root: Path, hardware: Hardware,
) -> list[ObservedBest]:
    run_dir = full_gemm_run_dir.resolve()
    try:
        relative_run = run_dir.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ModelError(f"campaign directory is outside repo: {run_dir}") from error
    if run_dir.name != linkage.full_gemm_run_id:
        raise ModelError("suite/full-GEMM run ID mismatch")
    hardware.validate()
    if (
        hardware.hardware_id != "thor_t5000_sm110_20sm"
        or hardware.sm_count != 20
        or hardware.operating_mode != "MAXN"
    ):
        raise ModelError(
            "suite qualification requires thor_t5000_sm110_20sm / 20 SM / MAXN"
        )
    prefix = f"{relative_run}/"
    qualified: list[ObservedBest] = []
    for observation in observations:
        observation.validate(repo_root=repo_root)
        if observation.qualification != "snapshot_only":
            raise ModelError("suite qualification expects snapshot observations")
        if not observation.source_path.startswith(prefix):
            raise ModelError(
                f"{observation.observation_id}: source does not belong to "
                "linked full-GEMM run"
            )
        row = dataclasses.replace(
            observation, qualification="closure_qualified",
            hardware_id=hardware.hardware_id, sm_count=hardware.sm_count,
            operating_mode=hardware.operating_mode)
        row.validate(repo_root=repo_root)
        qualified.append(row)
    return qualified
