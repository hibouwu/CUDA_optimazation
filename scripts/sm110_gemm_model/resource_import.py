from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

from .model import Capacity, EvidenceKind, ModelError, audit_inputs


SUITE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
EXPECTED_RESULT_COUNT = 54


def _load_platform_auditor(repo_root: Path) -> Any:
    path = (
        repo_root
        / "microbench/sm110_gemm_resource_campaign/audit_resource_suite.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sm110_resource_platform_independent_auditor", path
    )
    if spec is None or spec.loader is None:
        raise ModelError(f"cannot load resource platform auditor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"cannot read resource evidence JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelError(f"resource evidence JSON is not an object: {path}")
    return value


def _relative_file(repo_root: Path, path: Path) -> str:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ModelError(f"resource artifact is outside repository: {path}") from error
    if not resolved.is_file():
        raise ModelError(f"resource artifact is missing: {relative}")
    return relative.as_posix()


def _capacity_artifacts(
    *,
    repo_root: Path,
    suite_dir: Path,
    run_dir: Path,
    result: dict[str, Any],
) -> tuple[str, ...]:
    case_dir = run_dir / "cases" / str(result["case_id"])
    paths = [
        suite_dir / "run_contract.json",
        suite_dir / "preflight.txt",
        suite_dir / "oc_before.tsv",
        suite_dir / "oc_after.tsv",
        suite_dir / "suite_launcher.log",
        suite_dir / "suite_audit.json",
        run_dir / "run_spec.json",
        run_dir / "environment.json",
        run_dir / "environment_snapshots.jsonl",
        run_dir / "static_contracts.json",
        run_dir / "summary.json",
        run_dir / "COMPLETE",
        run_dir / "artifact_sha256.txt",
        run_dir / "build/compile_command.json",
        run_dir / "build/compile.log",
        run_dir / "build/artifact.json",
        run_dir / "build/binary.sha256",
        run_dir / "build/tma_ab_contract_bandwidth",
        run_dir / "build/tma_ab_contract_bandwidth.sass.txt",
        case_dir / "result.json",
        case_dir / "trials.jsonl",
        repo_root / str(result["source_path"]),
    ]
    ncu = result.get("ncu", {})
    if isinstance(ncu, dict) and ncu.get("selected") is True:
        paths.extend([
            case_dir / str(ncu["report_path"]),
            case_dir / "ncu/profile.log",
        ])
    relative = tuple(_relative_file(repo_root, path) for path in paths)
    if len(relative) != len(set(relative)):
        raise ModelError(f"{result['case_id']}: duplicate resource artifact path")
    return relative


def import_resource_capacities(
    *,
    repo_root: Path,
    suite_id: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Reaudit one Thor resource suite and emit exact model capacities.

    The imported resource identity retains transport family, packed A/B row
    stride, and residency scope.  Hot-L2 rates remain per-SM ingress points;
    cold-DRAM rates remain aggregate GPU path points.  Neither is promoted to
    a physical upper bound.
    """

    repo_root = repo_root.resolve()
    if SUITE_ID_RE.fullmatch(suite_id) is None:
        raise ModelError("resource suite ID must be a stable path-safe identifier")
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise ModelError("resource expected commit must be 40 lowercase hex digits")
    suite_dir = repo_root / "results/sm110_resource_suite" / suite_id
    platform_auditor = _load_platform_auditor(repo_root)
    platform = platform_auditor.audit_suite(
        suite_dir, expected_commit=expected_commit
    )
    if platform.get("pass") is not True:
        raise ModelError(
            "resource suite independent audit failed: "
            f"{platform.get('errors', [])}"
        )

    contract = _read_json(suite_dir / "run_contract.json")
    run_id = contract.get("resource_run_id")
    if run_id != f"{suite_id}-resources":
        raise ModelError("resource run identity differs from suite contract")
    run_dir = repo_root / "results/sm110_gemm_resource_campaign" / str(run_id)
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    results = summary.get("results")
    if not (
        summary.get("status") == "complete"
        and summary.get("case_count") == EXPECTED_RESULT_COUNT
        and isinstance(results, list)
        and len(results) == EXPECTED_RESULT_COUNT
    ):
        raise ModelError("resource summary is not the complete 54-case matrix")

    capacities: list[Capacity] = []
    seen_resources: set[str] = set()
    for result in sorted(results, key=lambda row: str(row.get("case_id"))):
        if not isinstance(result, dict):
            raise ModelError("resource summary contains a non-object result")
        case_id = str(result.get("case_id", ""))
        resource = str(result.get("resource", ""))
        residency = str(result.get("residency", ""))
        if not case_id or not resource or resource in seen_resources:
            raise ModelError(f"resource result identity is invalid: {case_id}")
        seen_resources.add(resource)
        if residency == "hot_l2":
            if not (
                resource.startswith("tma.smem_ingress.contract.")
                and resource.endswith(".per_sm")
            ):
                raise ModelError(f"{case_id}: hot-L2 resource is not per-SM ingress")
            scope = "one CTA and one observed SM; per-SM TMA-to-SMEM ingress"
        elif residency == "cold_dram":
            if not resource.startswith("tma.hbm.contract."):
                raise ModelError(f"{case_id}: cold-DRAM resource is not aggregate TMA/HBM")
            scope = "one CTA on each of 20 SMs; aggregate cold-entry TMA/DRAM path"
        else:
            raise ModelError(f"{case_id}: unknown resource residency {residency}")
        if result.get("rate_unit") != "B/s":
            raise ModelError(f"{case_id}: resource rate unit is not B/s")
        expected = result.get("expected_contract")
        if not isinstance(expected, dict):
            raise ModelError(f"{case_id}: expected transport contract is missing")
        rate = float(result["rate_per_second_median"])
        row_stride = int(result["row_stride_elements"])
        family_id = str(result["family_id"])
        capacities.append(Capacity(
            capacity_id=f"{suite_id}.resource.{case_id}",
            resource=resource,
            rate_per_second=rate,
            work_unit="byte",
            evidence_kind=EvidenceKind.MEASURED_SUSTAINED,
            source_id=suite_id,
            source_path=_relative_file(repo_root, summary_path),
            source_locator=f'"case_id": "{case_id}"',
            original_value=rate,
            original_unit="B/s",
            condition=(
                f"family={family_id}; packed A/B row_stride_elements={row_stride}; "
                f"residency={residency}; {scope}; stage_bytes="
                f"{expected.get('stage_bytes')}; stages={expected.get('stages')}; "
                f"requests_per_stage={expected.get('requests_per_stage')}; "
                "requested TMA payload numerator"
            ),
            qualification="closure_qualified",
            trial_count=int(result["trial_count"]),
            artifact_paths=_capacity_artifacts(
                repo_root=repo_root,
                suite_dir=suite_dir,
                run_dir=run_dir,
                result=result,
            ),
        ))

    findings = audit_inputs(capacities, repo_root=repo_root)
    errors = [row for row in findings if row["severity"] == "error"]
    if errors:
        raise ModelError(f"resource capacity provenance audit failed: {errors}")
    return {
        "schema_version": 1,
        "kind": "sm110_exact_tma_resource_import",
        "suite_id": suite_id,
        "run_id": run_id,
        "expected_commit": expected_commit,
        "qualification": "closure_qualified",
        "platform_evidence": platform,
        "capacities": [capacity.to_dict() for capacity in capacities],
        "observed_best": [],
    }
