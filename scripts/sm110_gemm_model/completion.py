from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from .model import (
    Capacity,
    Hardware,
    ModelError,
    PipelineProfile,
    Schedule,
    Workload,
    account_work,
    capacity_applies_to_hardware,
    ceil_div,
    precision_specs,
)
from .observations import ObservedBest


@dataclass(frozen=True)
class TargetPrecisionAudit:
    precision_id: str
    compute_campaign_case_ids: tuple[str, ...]
    compute_campaign_full_gpu_case_ids: tuple[str, ...]
    legal_schedule_ids: tuple[str, ...]
    complete_data_path_schedule_ids: tuple[str, ...]
    candidate_tma_payload_bytes: tuple[int, ...]
    required_tma_payload_bytes: tuple[int, ...]
    closure_qualified_tma_payload_bytes: tuple[int, ...]
    required_tmem_readback_contracts: tuple[str, ...]
    required_hbm_duplex_read_write_ratios: tuple[str, ...]
    closure_qualified_hbm_duplex_proxy_ratios: tuple[str, ...]
    closure_qualified_hbm_duplex_ratios: tuple[str, ...]
    candidate_l2_duplex_read_write_ratios: tuple[str, ...]
    required_l2_duplex_read_write_ratios: tuple[str, ...]
    closure_qualified_l2_duplex_ratios: tuple[str, ...]
    required_joint_pipeline_contracts: tuple[str, ...]
    closure_qualified_joint_pipeline_contracts: tuple[str, ...]
    full_gemm_support_status: str
    full_gemm_campaign_case_ids: tuple[str, ...]
    full_gemm_calibration_case_ids: tuple[str, ...]
    full_gemm_holdout_case_ids: tuple[str, ...]
    full_gemm_scenario_qualified_case_ids: tuple[str, ...]
    conditional_upper_numeric: bool
    conditional_upper_complete: bool
    closure_qualified_compute_rate: bool
    closure_qualified_full_gemm: bool
    absolute_three_layer_closure: bool
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TargetCompletionAudit:
    precision_audits: tuple[TargetPrecisionAudit, ...]
    all_precision_contracts_present: bool
    all_compute_campaigns_planned: bool
    all_complete_data_paths_modeled: bool
    all_required_tma_payloads_planned: bool
    all_required_tma_payloads_measured: bool
    all_required_hbm_duplex_proxies_measured: bool
    all_required_hbm_duplex_ratios_measured: bool
    all_required_l2_duplex_ratios_measured: bool
    all_full_gemm_campaigns_planned: bool
    all_full_gemm_scenarios_planned: bool
    duplex_campaign_frozen: bool
    epilogue_campaign_frozen: bool
    joint_pipeline_campaign_frozen: bool
    dependency_span_model_complete: bool
    hardware_capacity_source_present: bool
    cache_residency_model_complete: bool
    joint_overlap_model_complete: bool
    all_precisions_absolute_three_layer_closed: bool
    final_source_appendix_generated: bool
    complete: bool
    global_missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["precision_audits"] = [
            row.to_dict() for row in self.precision_audits
        ]
        return payload


def _ratio(read_bytes: float, write_bytes: float) -> str:
    read = Fraction(str(read_bytes))
    write = Fraction(str(write_bytes))
    quotient = read / write
    return f"{quotient.numerator}:{quotient.denominator}"


def _repo_file(repo_root: Path, relative_path: str) -> Path | None:
    path = repo_root / relative_path
    try:
        path.resolve().relative_to(repo_root.resolve())
    except (OSError, ValueError):
        return None
    return path if path.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def campaign_freeze_manifest_valid(
    repo_root: Path,
    *,
    campaign_id: str,
    manifest_path: str,
    required_artifacts: tuple[str, ...],
) -> bool:
    """Accept a frozen campaign only when its design basis and files are bound.

    File existence alone is not evidence that a campaign's cases or semantics
    were frozen.  The manifest must bind a non-empty case set, the audited
    first-round design basis, and the exact digest of every required artifact.
    """
    manifest_file = _repo_file(repo_root, manifest_path)
    if manifest_file is None:
        return False
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or any(
        (
            payload.get("schema_version") != 1,
            payload.get("campaign_id") != campaign_id,
            payload.get("status") != "frozen",
        )
    ):
        return False
    case_ids = payload.get("frozen_case_ids")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
    ):
        return False
    case_manifest_path = payload.get("case_manifest_path")
    case_manifest_digest = payload.get("case_manifest_sha256")
    digest_pattern = re.compile(r"^[0-9a-f]{64}$")
    basis_path = payload.get("basis_artifact_path")
    basis_digest = payload.get("basis_artifact_sha256")
    basis_expected_commit = payload.get("basis_expected_commit")
    if (
        not isinstance(basis_path, str)
        or not isinstance(basis_digest, str)
        or digest_pattern.fullmatch(basis_digest) is None
        or not isinstance(basis_expected_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", basis_expected_commit) is None
        or not isinstance(case_manifest_path, str)
        or not isinstance(case_manifest_digest, str)
        or digest_pattern.fullmatch(case_manifest_digest) is None
    ):
        return False
    basis_file = _repo_file(repo_root, basis_path)
    if basis_file is None or _sha256(basis_file) != basis_digest:
        return False
    case_manifest_file = _repo_file(repo_root, case_manifest_path)
    if (
        case_manifest_file is None
        or _sha256(case_manifest_file) != case_manifest_digest
    ):
        return False
    try:
        case_manifest = json.loads(
            case_manifest_file.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    cases = (
        case_manifest.get("cases")
        if isinstance(case_manifest, dict)
        else None
    )
    if (
        not isinstance(cases, list)
        or any(not isinstance(case, dict) for case in cases)
        or [case.get("case_id") for case in cases] != case_ids
    ):
        return False
    try:
        basis = json.loads(basis_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    linkage = basis.get("suite_linkage") if isinstance(basis, dict) else None
    findings = basis.get("capacity_findings") if isinstance(basis, dict) else None
    coverage = basis.get("coverage") if isinstance(basis, dict) else None
    if (
        not isinstance(linkage, dict)
        or re.fullmatch(r"[0-9a-f]{40}", str(linkage.get("expected_commit", "")))
        is None
        or linkage.get("ncu_required") is not True
        or linkage.get("expected_commit") != basis_expected_commit
        or not all(
            isinstance(linkage.get(key), str) and linkage.get(key)
            for key in (
                "suite_id",
                "hostname",
                "gpu_identity",
                "compute_run_id",
                "component_run_id",
                "full_gemm_run_id",
            )
        )
        or not isinstance(findings, list)
        or any(
            isinstance(finding, dict) and finding.get("severity") == "error"
            for finding in findings
        )
        or not isinstance(coverage, dict)
        or not isinstance(coverage.get("target_completion"), dict)
    ):
        return False
    artifact_digests = payload.get("artifact_sha256")
    if (
        not isinstance(artifact_digests, dict)
        or set(artifact_digests) != set(required_artifacts)
    ):
        return False
    for relative_path in required_artifacts:
        expected = artifact_digests.get(relative_path)
        artifact = _repo_file(repo_root, relative_path)
        if (
            artifact is None
            or not isinstance(expected, str)
            or digest_pattern.fullmatch(expected) is None
            or _sha256(artifact) != expected
        ):
            return False
    return True


def _hardware_capacity_source_present(
    repo_root: Path,
    hardware: Hardware,
) -> bool:
    """Verify the declared L2-capacity evidence and its exact locator."""
    if hardware.l2_capacity_bytes is None:
        return False
    source = repo_root / hardware.l2_capacity_source_path
    try:
        source.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    if not source.is_file():
        return False
    locator = hardware.l2_capacity_source_locator
    if source.suffix.lower() != ".csv":
        return locator in source.read_text(encoding="utf-8")
    predicates: dict[str, str] = {}
    for token in locator.split(","):
        if "=" not in token:
            return False
        key, value = token.split("=", 1)
        predicates[key.strip()] = value.strip()
    if not predicates:
        return False
    if predicates.get("l2_cache_bytes") != str(hardware.l2_capacity_bytes):
        return False
    with source.open(newline="", encoding="utf-8-sig") as handle:
        return any(
            all(row.get(key) == value for key, value in predicates.items())
            for row in csv.DictReader(handle)
        )


def _canonical_campaign_manifests(
    repo_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, str], set[int]]:
    # These modules only construct immutable manifests; they do not touch a GPU
    # at import time. Keeping this derivation executable prevents documentation
    # tables from drifting away from the campaign actually sent to Thor.
    from microbench.sm110_full_gemm_campaign.run_full_gemm_campaign import (
        CASES as full_gemm_cases,
    )
    from microbench.sm110_gemm_campaign.run_compute_campaign import make_manifest
    from microbench.sm110_tma_payload_campaign.run_tma_payload_campaign import (
        PAYLOAD_BYTES,
    )

    support_path = (
        repo_root
        / "microbench/sm110_full_gemm_campaign/support_manifest.json"
    )
    support = json.loads(support_path.read_text(encoding="utf-8"))
    statuses = {
        str(row["precision_id"]): str(row["status"])
        for row in support["precisions"]
    }
    return (
        list(make_manifest()),
        list(full_gemm_cases),
        statuses,
        {int(value) for value in PAYLOAD_BYTES},
    )


def _joint_contract_id(workload: Workload, schedule: Schedule) -> str:
    return (
        f"{workload.workload_id}|{schedule.schedule_id}|{workload.residency}|"
        f"{workload.timed_scope}"
    )


def joint_pipeline_capacity_matches(
    capacity: Capacity,
    *,
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> bool:
    """Return whether one joint measurement closes one exact scenario contract."""
    try:
        capacity.validate()
    except ModelError:
        return False
    expected_residency_evidence = (
        "ncu_proven"
        if workload.residency in {"cold_hbm", "hot_l2"}
        else "construction_proven"
    )
    return bool(
        capacity.resource == f"pipeline.joint.{workload.precision_id}"
        and capacity.evidence_kind.value == "measured_joint"
        and capacity.is_closure_qualified
        and workload.precision_id in capacity.applicable_precision_ids
        and workload.workload_id in capacity.applicable_workload_ids
        and schedule.schedule_id in capacity.applicable_schedule_ids
        and workload.residency in capacity.applicable_residencies
        and capacity.timed_scope == workload.timed_scope
        and hardware.sm_count in capacity.applicable_sm_counts
        and hardware.hardware_id in capacity.applicable_hardware_ids
        and hardware.operating_mode in capacity.applicable_operating_modes
        and schedule.threads in capacity.applicable_threads_per_cta
        and schedule.resident_ctas_per_sm
        in capacity.applicable_resident_ctas_per_sm
        and capacity.residency_evidence_qualification
        == expected_residency_evidence
    )


def joint_pipeline_profile_matches(
    profile: PipelineProfile,
    *,
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> bool:
    """Accept a causal profile only inside its exact topology and fit range."""
    try:
        profile.validate()
        work = account_work(
            workload, schedule, precision_specs()[workload.precision_id]
        )
    except ModelError:
        return False
    service_units = (
        max(1, hardware.sm_count // schedule.cta_group)
        * profile.resident_ctas_per_sm
    )
    output_tasks = ceil_div(work.task_count, service_units)
    return bool(
        profile.is_closure_qualified
        and schedule.causal_pipeline_resource is not None
        and profile.resource == schedule.causal_pipeline_resource
        and profile.schedule_id == schedule.schedule_id
        and workload.precision_id in profile.precision_ids
        and profile.input_residency == workload.residency
        and profile.timed_scope == workload.timed_scope
        and profile.stages == schedule.stages
        and profile.resident_ctas_per_sm == schedule.resident_ctas_per_sm
        and (not profile.applicable_sm_counts
             or hardware.sm_count in profile.applicable_sm_counts)
        and (not profile.applicable_hardware_ids
             or hardware.hardware_id in profile.applicable_hardware_ids)
        and (not profile.applicable_operating_modes
             or hardware.operating_mode in profile.applicable_operating_modes)
        and (not profile.applicable_clock_hz
             or hardware.clock_hz in profile.applicable_clock_hz)
        and work.k_tiles <= profile.maximum_k_tiles
        and output_tasks <= profile.maximum_output_tasks_per_worker
    )


def audit_target_completeness(
    *,
    repo_root: Path,
    hardware: Hardware,
    capacities: Iterable[Capacity],
    workloads: Iterable[Workload],
    schedules: Iterable[Schedule],
    observed: Iterable[ObservedBest],
    coverage: dict[str, object],
    pipeline_profiles: Iterable[PipelineProfile] = (),
) -> TargetCompletionAudit:
    """Audit the full user objective, not merely currently available data."""
    repo_root = repo_root.resolve()
    hardware.validate()
    capacities = list(capacities)
    workloads = list(workloads)
    schedules = list(schedules)
    observed = list(observed)
    pipeline_profiles = list(pipeline_profiles)
    for capacity in capacities:
        capacity.validate()
    specs = precision_specs()
    hardware_capacities = [
        capacity for capacity in capacities
        if capacity.is_closure_qualified
        and capacity_applies_to_hardware(capacity, hardware)
    ]
    hot_payloads = {
        payload
        for capacity in hardware_capacities
        if capacity.resource.startswith("tma.smem_ingress.per_sm.payload_")
        for payload in capacity.applicable_tma_tile_bytes
    }
    cold_payloads = {
        payload
        for capacity in hardware_capacities
        if capacity.resource.startswith("tma.hbm.payload_")
        for payload in capacity.applicable_tma_tile_bytes
    }
    measured_payloads = hot_payloads & cold_payloads
    hbm_proxy_ratios = {
        ratio
        for capacity in hardware_capacities
        if capacity.resource == "hbm.duplex.proxy"
        for ratio in capacity.applicable_read_write_ratios
    }
    hbm_duplex_ratios = {
        ratio
        for capacity in hardware_capacities
        if capacity.resource == "hbm.duplex"
        for ratio in capacity.applicable_read_write_ratios
    }
    l2_duplex_ratios = {
        ratio
        for capacity in hardware_capacities
        if capacity.resource == "l2.duplex"
        for ratio in capacity.applicable_read_write_ratios
    }
    compute_manifest, full_cases, support_statuses, planned_tma_payloads = (
        _canonical_campaign_manifests(repo_root)
    )

    precision_coverage = {
        str(row["precision_id"]): row
        for row in coverage["precision_coverage"]
        if row["implementation_domain"] == "tensor_core_classical"
    }
    scenario_coverage = {
        str(row["workload_id"]): row for row in coverage["scenario_coverage"]
    }
    rows: list[TargetPrecisionAudit] = []
    for precision_id, spec in specs.items():
        precision_workloads = [
            row for row in workloads if row.precision_id == precision_id
        ]
        representative = next(
            (
                row for row in precision_workloads
                if row.validation_split == "calibration"
            ),
            None,
        )
        legal: list[tuple[Schedule, object]] = []
        if representative is not None:
            for schedule in schedules:
                try:
                    work = account_work(representative, schedule, spec)
                except ModelError:
                    continue
                legal.append((schedule, work))

        complete = [
            (schedule, work)
            for schedule, work in legal
            if schedule.data_path_contract == "complete"
        ]
        candidate_payloads: set[int] = set()
        required_payloads: set[int] = set()
        readback_contracts: set[str] = set()
        candidate_l2_ratios: set[str] = set()
        required_l2_ratios: set[str] = set()
        required_joint_contracts: set[str] = set()
        qualified_joint_contracts: set[str] = set()
        for schedule, work in legal:
            tile_visits = (
                ceil_div(representative.m, schedule.bm)
                * ceil_div(representative.n, schedule.bn)
                * ceil_div(representative.k, schedule.bk)
            )
            payloads = {
                int(total_bytes / tile_visits)
                for total_bytes in (
                    work.tma_a_value_bytes,
                    work.tma_b_value_bytes,
                    work.tma_a_scale_bytes,
                    work.tma_b_scale_bytes,
                )
                if total_bytes > 0
            }
            candidate_payloads.update(payloads)
            l2_ratio = _ratio(
                work.tma_input_bytes + work.c_read_bytes_min,
                work.output_value_bytes_min + work.output_scale_bytes_min,
            )
            candidate_l2_ratios.add(l2_ratio)
            if schedule.data_path_contract == "complete":
                required_payloads.update(payloads)
                required_l2_ratios.add(l2_ratio)
                readback_contracts.add(
                    f"x{schedule.tmem_load_registers}_warps{schedule.readback_warps}_"
                    f"threads{schedule.threads}_ctas{schedule.resident_ctas_per_sm}"
                )
        for precision_workload in precision_workloads:
            for schedule in schedules:
                try:
                    account_work(precision_workload, schedule, spec)
                except ModelError:
                    continue
                if schedule.data_path_contract != "complete":
                    continue
                contract_id = _joint_contract_id(precision_workload, schedule)
                required_joint_contracts.add(contract_id)
                if any(
                    joint_pipeline_capacity_matches(
                        capacity,
                        workload=precision_workload,
                        schedule=schedule,
                        hardware=hardware,
                    )
                    for capacity in capacities
                ) or any(
                    joint_pipeline_profile_matches(
                        profile,
                        workload=precision_workload,
                        schedule=schedule,
                        hardware=hardware,
                    )
                    for profile in pipeline_profiles
                ):
                    qualified_joint_contracts.add(contract_id)

        compute_cases = sorted(
            str(case["case_id"])
            for case in compute_manifest
            if case["precision"]["precision_id"] == precision_id
        )
        full_gpu_compute = sorted(
            str(case["case_id"])
            for case in compute_manifest
            if case["precision"]["precision_id"] == precision_id
            and case["launch"] == "full_sm_4warp_block"
        )
        precision_full = [
            case for case in full_cases if case["precision_id"] == precision_id
        ]
        calibration_cases = sorted(
            str(case["id"])
            for case in precision_full
            if case["split"] == "calibration"
        )
        holdout_cases = sorted(
            str(case["id"])
            for case in precision_full
            if case["split"] == "holdout"
        )
        required_residencies = {row.residency for row in precision_workloads}
        scenario_qualified_cases = sorted(
            str(case["id"])
            for case in precision_full
            if case.get("residency") in required_residencies
            and case.get("residency_evidence_qualification") == "ncu_proven"
        )
        coverage_row = precision_coverage[precision_id]
        scenario_rows = [
            scenario_coverage[row.workload_id]
            for row in precision_workloads
            if row.workload_id in scenario_coverage
        ]
        upper_numeric = bool(scenario_rows) and all(
            bool(row["conditional_upper_numeric"]) for row in scenario_rows
        )
        upper_complete = bool(scenario_rows) and all(
            bool(row["conditional_upper_complete"]) for row in scenario_rows
        )
        read_bytes = (
            (
                representative.m * representative.k
                + representative.k * representative.n
            )
            * spec.input_bytes
            + (
                0
                if spec.input_scale_block is None
                else (
                    ceil_div(
                        representative.m * representative.k,
                        spec.input_scale_block,
                    )
                    + ceil_div(
                        representative.k * representative.n,
                        spec.input_scale_block,
                    )
                )
                * spec.input_scale_bytes
            )
        ) if representative is not None else 0.0
        write_bytes = (
            representative.m * representative.n * spec.output_bytes
            if representative is not None else 1.0
        )
        required_hbm_ratios = (
            {_ratio(read_bytes, write_bytes)}
            if representative is not None else set()
        )
        qualified_payloads = required_payloads & measured_payloads
        qualified_hbm_proxies = required_hbm_ratios & hbm_proxy_ratios
        qualified_hbm_duplex = required_hbm_ratios & hbm_duplex_ratios
        qualified_l2_duplex = required_l2_ratios & l2_duplex_ratios

        missing: list[str] = []
        if not compute_cases or not full_gpu_compute:
            missing.append("compute_campaign_manifest")
        if not legal:
            missing.append("legal_schedule")
        if not complete:
            missing.append("complete_data_path_schedule")
        if required_payloads - planned_tma_payloads:
            missing.append("planned_exact_tma_payload")
        if required_payloads - qualified_payloads:
            missing.append("closure_qualified_tma_payload")
        if required_hbm_ratios - qualified_hbm_duplex:
            missing.append("closure_qualified_physical_hbm_duplex_ratio")
        if required_l2_ratios - qualified_l2_duplex:
            missing.append("closure_qualified_l2_duplex_ratio")
        if required_joint_contracts != qualified_joint_contracts:
            missing.append("closure_qualified_joint_pipeline_contract")
        if not calibration_cases:
            missing.append("full_gemm_calibration_campaign")
        if not holdout_cases:
            missing.append("full_gemm_holdout_campaign")
        if len(scenario_qualified_cases) != len(precision_full):
            missing.append("full_gemm_scenario_contract")
        if support_statuses.get(precision_id) != "ready_for_closure_campaign":
            missing.append("full_gemm_same_contract_reference")
        if not upper_numeric:
            missing.append("conditional_upper_numeric")
        if not upper_complete:
            missing.append("conditional_upper_complete")
        if not coverage_row["closure_qualified_compute_rate"]:
            missing.append("closure_qualified_compute_rate")
        if not coverage_row["closure_qualified_full_gemm"]:
            missing.append("closure_qualified_full_gemm")
        if not coverage_row["absolute_three_layer_closure"]:
            missing.append("absolute_three_layer_closure")

        rows.append(
            TargetPrecisionAudit(
                precision_id=precision_id,
                compute_campaign_case_ids=tuple(compute_cases),
                compute_campaign_full_gpu_case_ids=tuple(full_gpu_compute),
                legal_schedule_ids=tuple(
                    sorted(schedule.schedule_id for schedule, _ in legal)
                ),
                complete_data_path_schedule_ids=tuple(
                    sorted(schedule.schedule_id for schedule, _ in complete)
                ),
                candidate_tma_payload_bytes=tuple(sorted(candidate_payloads)),
                required_tma_payload_bytes=tuple(sorted(required_payloads)),
                closure_qualified_tma_payload_bytes=tuple(
                    sorted(qualified_payloads)
                ),
                required_tmem_readback_contracts=tuple(sorted(readback_contracts)),
                required_hbm_duplex_read_write_ratios=(
                    tuple(sorted(required_hbm_ratios))
                ),
                closure_qualified_hbm_duplex_proxy_ratios=tuple(
                    sorted(qualified_hbm_proxies)
                ),
                closure_qualified_hbm_duplex_ratios=tuple(
                    sorted(qualified_hbm_duplex)
                ),
                candidate_l2_duplex_read_write_ratios=tuple(
                    sorted(candidate_l2_ratios)
                ),
                required_l2_duplex_read_write_ratios=tuple(
                    sorted(required_l2_ratios)
                ),
                closure_qualified_l2_duplex_ratios=tuple(
                    sorted(qualified_l2_duplex)
                ),
                required_joint_pipeline_contracts=tuple(
                    sorted(required_joint_contracts)
                ),
                closure_qualified_joint_pipeline_contracts=tuple(
                    sorted(qualified_joint_contracts)
                ),
                full_gemm_support_status=support_statuses.get(
                    precision_id, "missing"
                ),
                full_gemm_campaign_case_ids=tuple(
                    sorted(str(case["id"]) for case in precision_full)
                ),
                full_gemm_calibration_case_ids=tuple(calibration_cases),
                full_gemm_holdout_case_ids=tuple(holdout_cases),
                full_gemm_scenario_qualified_case_ids=tuple(
                    scenario_qualified_cases
                ),
                conditional_upper_numeric=upper_numeric,
                conditional_upper_complete=upper_complete,
                closure_qualified_compute_rate=bool(
                    coverage_row["closure_qualified_compute_rate"]
                ),
                closure_qualified_full_gemm=bool(
                    coverage_row["closure_qualified_full_gemm"]
                ),
                absolute_three_layer_closure=bool(
                    coverage_row["absolute_three_layer_closure"]
                ),
                missing=tuple(missing),
            )
        )

    global_missing: list[str] = []
    all_joint_contracts = {
        contract
        for row in rows
        for contract in row.required_joint_pipeline_contracts
    }
    all_qualified_joint_contracts = {
        contract
        for row in rows
        for contract in row.closure_qualified_joint_pipeline_contracts
    }
    required_cache_contracts = {
        contract
        for row in rows
        for contract in row.required_joint_pipeline_contracts
        if "|cold_hbm|" in contract or "|hot_l2|" in contract
    }
    qualified_cache_contracts = required_cache_contracts & all_qualified_joint_contracts
    checks = {
        "all_precision_contracts_present": set(specs) == set(support_statuses),
        "all_compute_campaigns_planned": all(
            row.compute_campaign_case_ids
            and row.compute_campaign_full_gpu_case_ids
            for row in rows
        ),
        "all_complete_data_paths_modeled": all(
            row.complete_data_path_schedule_ids for row in rows
        ),
        "all_required_tma_payloads_planned": all(
            set(row.required_tma_payload_bytes) <= planned_tma_payloads
            for row in rows
        ),
        "all_required_tma_payloads_measured": all(
            set(row.required_tma_payload_bytes)
            <= set(row.closure_qualified_tma_payload_bytes)
            for row in rows
        ),
        "all_required_hbm_duplex_proxies_measured": all(
            set(row.required_hbm_duplex_read_write_ratios)
            <= set(row.closure_qualified_hbm_duplex_proxy_ratios)
            for row in rows
        ),
        "all_required_hbm_duplex_ratios_measured": all(
            set(row.required_hbm_duplex_read_write_ratios)
            <= set(row.closure_qualified_hbm_duplex_ratios)
            for row in rows
        ),
        "all_required_l2_duplex_ratios_measured": all(
            set(row.required_l2_duplex_read_write_ratios)
            <= set(row.closure_qualified_l2_duplex_ratios)
            for row in rows
        ),
        "all_full_gemm_campaigns_planned": all(
            row.full_gemm_calibration_case_ids
            and row.full_gemm_holdout_case_ids
            for row in rows
        ),
        "all_full_gemm_scenarios_planned": all(
            row.full_gemm_campaign_case_ids
            and len(row.full_gemm_scenario_qualified_case_ids)
            == len(row.full_gemm_campaign_case_ids)
            for row in rows
        ),
        "duplex_campaign_frozen": campaign_freeze_manifest_valid(
            repo_root,
            campaign_id="sm110_memory_duplex_campaign",
            manifest_path=(
                "microbench/sm110_memory_duplex_campaign/freeze_manifest.json"
            ),
            required_artifacts=(
                "microbench/sm110_memory_duplex_campaign/README.md",
                "microbench/sm110_memory_duplex_campaign/cases.json",
                "microbench/sm110_memory_duplex_campaign/launch_campaign.sh",
                "microbench/sm110_memory_duplex_campaign/run_campaign.py",
                "microbench/sm110_memory_duplex_campaign/memory_duplex.cu",
                "microbench/sm110_memory_duplex_campaign/audit_campaign.py",
                "microbench/sm110_memory_duplex_campaign/test_campaign.py",
            ),
        ),
        "epilogue_campaign_frozen": campaign_freeze_manifest_valid(
            repo_root,
            campaign_id="sm110_accumulator_store_campaign",
            manifest_path=(
                "microbench/sm110_accumulator_store_campaign/freeze_manifest.json"
            ),
            required_artifacts=(
                "microbench/sm110_accumulator_store_campaign/README.md",
                "microbench/sm110_accumulator_store_campaign/cases.json",
                "microbench/sm110_accumulator_store_campaign/launch_campaign.sh",
                "microbench/sm110_accumulator_store_campaign/run_campaign.py",
                "microbench/sm110_accumulator_store_campaign/accumulator_store.cu",
                "microbench/sm110_accumulator_store_campaign/audit_campaign.py",
                "microbench/sm110_accumulator_store_campaign/test_campaign.py",
            ),
        ),
        "joint_pipeline_campaign_frozen": campaign_freeze_manifest_valid(
            repo_root,
            campaign_id="sm110_joint_pipeline_campaign",
            manifest_path=(
                "microbench/sm110_joint_pipeline_campaign/freeze_manifest.json"
            ),
            required_artifacts=(
                "microbench/sm110_joint_pipeline_campaign/README.md",
                "microbench/sm110_joint_pipeline_campaign/cases.json",
                "microbench/sm110_joint_pipeline_campaign/launch_campaign.sh",
                "microbench/sm110_joint_pipeline_campaign/run_campaign.py",
                "microbench/sm110_joint_pipeline_campaign/joint_pipeline.cu",
                "microbench/sm110_joint_pipeline_campaign/audit_campaign.py",
                "microbench/sm110_joint_pipeline_campaign/test_campaign.py",
            ),
        ),
        # The current executable layer is a resource-service roofline. These
        # stay false until the causal startup/drain and measured joint-overlap
        # terms described in the document are represented in code and tested.
        "dependency_span_model_complete": campaign_freeze_manifest_valid(
            repo_root,
            campaign_id="sm110_accumulator_store_campaign",
            manifest_path=(
                "microbench/sm110_accumulator_store_campaign/freeze_manifest.json"
            ),
            required_artifacts=(
                "microbench/sm110_accumulator_store_campaign/README.md",
                "microbench/sm110_accumulator_store_campaign/cases.json",
                "microbench/sm110_accumulator_store_campaign/launch_campaign.sh",
                "microbench/sm110_accumulator_store_campaign/run_campaign.py",
                "microbench/sm110_accumulator_store_campaign/accumulator_store.cu",
                "microbench/sm110_accumulator_store_campaign/audit_campaign.py",
                "microbench/sm110_accumulator_store_campaign/test_campaign.py",
            ),
        ),
        "hardware_capacity_source_present": _hardware_capacity_source_present(
            repo_root, hardware
        ),
        # Subject to its explicit no-compression/no-external-reuse condition,
        # the strict layer may use unique logical HBM bytes as a lower bound.
        # The empirical layer still needs a source-backed cache residency/reuse
        # schedule or measured physical traffic before cold-HBM reuse or
        # hot-L2 request hits can be called attainable for a concrete schedule.
        "cache_residency_model_complete": bool(required_cache_contracts)
        and required_cache_contracts == qualified_cache_contracts,
        "joint_overlap_model_complete": bool(all_joint_contracts)
        and all_joint_contracts == all_qualified_joint_contracts,
        "all_precisions_absolute_three_layer_closed": all(
            row.absolute_three_layer_closure for row in rows
        ),
        "final_source_appendix_generated": (
            repo_root / "Docs/blackwell_tensorcore/microbenchmark_sources.md"
        ).is_file(),
    }
    for name, passed in checks.items():
        if not passed:
            global_missing.append(name)
    complete = not global_missing and all(not row.missing for row in rows)
    return TargetCompletionAudit(
        precision_audits=tuple(rows),
        complete=complete,
        global_missing=tuple(global_missing),
        **checks,
    )
