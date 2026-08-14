from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .model import Capacity, EvidenceKind, ModelError, audit_inputs, precision_specs
from .observations import ObservedBest, audit_observations


SUITE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
COMPUTE_SELECTION = {"launch": "full_sm_4warp_block", "m": 128, "n": 256}
COMPONENT_RESOURCES = {
    "tma.l2_hit_ingress": "tma.l2",
    "tma.dram_stream_ingress": "tma.hbm",
    "tmem.accumulator_readback": "tmem.readback",
    "epilogue.nvfp4_requant": "epilogue.nvfp4_requant",
}


@dataclass(frozen=True)
class ClosurePaths:
    repo_root: Path
    suite_id: str

    @property
    def suite(self) -> Path:
        return self.repo_root / "results/sm110_closure_suite" / self.suite_id

    @property
    def epilogue(self) -> Path:
        return self.repo_root / "results/sm110_epilogue_probe" / (
            f"{self.suite_id}-epilogue-preflight")

    @property
    def compute(self) -> Path:
        return self.repo_root / "results/sm110_gemm_campaign" / (
            f"{self.suite_id}-compute")

    @property
    def component(self) -> Path:
        return self.repo_root / "results/sm110_gemm_component_campaign" / (
            f"{self.suite_id}-components")

    @property
    def full(self) -> Path:
        return self.repo_root / "results/sm110_full_gemm_campaign" / (
            f"{self.suite_id}-full")

    def relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.repo_root.resolve()))
        except ValueError as error:
            raise ModelError(f"closure artifact is outside repository: {path}") from error


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"cannot read closure JSON {path}: {error}") from error


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ModelError(f"missing closure artifacts: {missing}")


def _run_auditor(command: list[str], *, repo_root: Path) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError as error:
        raise ModelError(
            f"auditor did not emit JSON ({' '.join(command)}): {proc.stdout}"
        ) from error
    if proc.returncode != 0 or output.get("pass") is not True:
        raise ModelError(
            f"closure auditor failed ({' '.join(command)}): "
            f"{json.dumps(output, sort_keys=True)}"
        )
    return output


def _parse_counter_tsv(path: Path) -> dict[str, int]:
    counters: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or fields[0] in counters:
            raise ModelError(f"{path}:{line_number}: invalid counter row")
        try:
            counters[fields[0]] = int(fields[1])
        except ValueError as error:
            raise ModelError(f"{path}:{line_number}: invalid counter value") from error
    if not counters:
        raise ModelError(f"{path}: no overcurrent counters")
    return counters


def _audit_platform(
    paths: ClosurePaths, expected_commit: str
) -> dict[str, int]:
    contract_path = paths.suite / "run_contract.json"
    preflight = paths.suite / "preflight.txt"
    before_path = paths.suite / "oc_before.tsv"
    after_path = paths.suite / "oc_after.tsv"
    log_path = paths.suite / "suite_launcher.log"
    _require_files((contract_path, preflight, before_path, after_path, log_path))
    contract = _read_json(contract_path)
    if contract != {
        **contract,
        "schema_version": 1,
        "suite_id": paths.suite_id,
        "expected_branch": "codex/thor-sm110-gemm-bounds-v2",
        "expected_commit": expected_commit,
        "ncu_required": True,
    }:
        raise ModelError("suite run contract does not match the closure request")
    preflight_text = preflight.read_text(errors="replace")
    if expected_commit not in preflight_text:
        raise ModelError("suite preflight does not prove the expected commit")
    if "NV Power Mode: MAXN" not in preflight_text:
        raise ModelError("suite preflight does not prove MAXN")
    frequency_contracts = (
        "min_freq=1575000000",
        "max_freq=1575000000",
        "CurrentFreq=1575000000",
        "governor=performance",
    )
    missing_frequency = [token for token in frequency_contracts
                         if token not in preflight_text]
    if missing_frequency:
        raise ModelError(
            f"suite preflight does not prove locked GPU clocks: {missing_frequency}")
    if "SUITE_COMPLETE" not in log_path.read_text(errors="replace"):
        raise ModelError("suite launcher log does not contain SUITE_COMPLETE")

    before = _parse_counter_tsv(before_path)
    after = _parse_counter_tsv(after_path)
    if set(before) != set(after):
        raise ModelError("overcurrent counter sets differ before and after campaign")
    deltas = {key: after[key] - value for key, value in before.items()}
    if any(value < 0 for value in deltas.values()):
        raise ModelError(f"overcurrent counters reset during campaign: {deltas}")
    return deltas


def _audit_commit_environment(run_dir: Path, expected_commit: str) -> None:
    snapshots_path = run_dir / "environment_snapshots.jsonl"
    _require_files((run_dir / "environment.json", snapshots_path))
    documents = [_read_json(run_dir / "environment.json")]
    try:
        documents.extend(
            json.loads(line) for line in snapshots_path.read_text().splitlines() if line)
    except json.JSONDecodeError as error:
        raise ModelError(f"invalid environment snapshot in {snapshots_path}") from error
    if not documents:
        raise ModelError(f"{run_dir}: no environment records")
    for index, document in enumerate(documents):
        git_record = document.get("git_head", {})
        if not git_record:
            git_record = document.get("git_head", document.get("git", {}))
        output = str(git_record.get("output", "")) if isinstance(git_record, dict) else ""
        if expected_commit not in output:
            raise ModelError(
                f"{run_dir}: environment record {index} does not prove expected commit")


def _common_artifacts(paths: ClosurePaths, run_dir: Path) -> tuple[str, ...]:
    return tuple(paths.relative(path) for path in (
        paths.suite / "run_contract.json",
        paths.suite / "preflight.txt",
        paths.suite / "oc_before.tsv",
        paths.suite / "oc_after.tsv",
        paths.suite / "suite_launcher.log",
        paths.epilogue / "summary.json",
        run_dir / "run_spec.json",
        run_dir / "environment.json",
        run_dir / "environment_snapshots.jsonl",
        run_dir / "summary.json",
        run_dir / "COMPLETE",
    ))


def capacities_from_compute(
    summary: dict[str, Any], spec: dict[str, Any], *, paths: ClosurePaths,
    qualification: str,
) -> list[Capacity]:
    manifest = {
        str(row["case_id"]): row for row in spec.get("manifest", [])
        if all(row.get(key) == value for key, value in COMPUTE_SELECTION.items())
    }
    results = {str(row.get("case_id")): row for row in summary.get("results", [])}
    capacities: list[Capacity] = []
    for case_id, entry in sorted(manifest.items()):
        result = results.get(case_id)
        if result is None:
            raise ModelError(f"selected compute result is missing: {case_id}")
        precision_id = str(result["precision_id"])
        try:
            resource = precision_specs()[precision_id].compute_resource
        except KeyError as error:
            raise ModelError(f"unknown compute precision: {precision_id}") from error
        case_dir = paths.compute / "cases" / case_id
        artifacts = (*_common_artifacts(paths, paths.compute),
                     paths.relative(case_dir / "result.json"),
                     paths.relative(case_dir / "trials.jsonl"),
                     paths.relative(case_dir / "sass.txt"))
        if result.get("ncu", {}).get("selected"):
            artifacts += (
                paths.relative(case_dir / "ncu/profile.ncu-rep"),
                paths.relative(case_dir / "ncu/profile.log"),
            )
        capacities.append(Capacity(
            capacity_id=f"{paths.suite_id}.compute.{precision_id}",
            resource=resource,
            rate_per_second=float(result["rate_per_second_median"]),
            work_unit=str(result["work_unit"]),
            evidence_kind=EvidenceKind.MEASURED_SUSTAINED,
            source_id=paths.suite_id,
            source_path=paths.relative(paths.compute / "summary.json"),
            source_locator=f'"case_id": "{case_id}"',
            condition=("full-GPU 20-SM compute oracle; M128N256; SMEM operands; "
                       "device globaltimer issue-to-completion span"),
            qualification=qualification,
            trial_count=int(result["trial_count"]),
            artifact_paths=artifacts,
        ))
    expected = len(precision_specs())
    if len(capacities) != expected:
        raise ModelError(
            f"compute selection has {len(capacities)} precisions, expected {expected}")
    return capacities


def capacities_from_component(
    summary: dict[str, Any], spec: dict[str, Any], *, paths: ClosurePaths,
    qualification: str,
) -> list[Capacity]:
    case_specs = {str(row["id"]): row for row in spec.get("cases", [])}
    capacities: list[Capacity] = []
    for result in sorted(summary.get("results", []), key=lambda row: row["case_id"]):
        case_id = str(result["case_id"])
        raw_resource = str(result["resource"])
        if raw_resource not in COMPONENT_RESOURCES:
            raise ModelError(f"unknown component resource: {raw_resource}")
        rate_unit = str(result["rate_unit"])
        work_unit = "element" if rate_unit == "element/s" else "byte"
        if rate_unit not in {"element/s", "B/s"}:
            raise ModelError(f"{case_id}: unsupported component rate unit {rate_unit}")
        if case_id not in case_specs:
            raise ModelError(f"component case is absent from run spec: {case_id}")
        binary = str(case_specs[case_id]["binary"])
        case_dir = paths.component / "cases" / case_id
        artifacts = (*_common_artifacts(paths, paths.component),
                     paths.relative(case_dir / "result.json"),
                     paths.relative(case_dir / "trials.jsonl"),
                     paths.relative(paths.component / f"build/{binary}.sass.txt"),
                     str(result["source_path"]))
        capacities.append(Capacity(
            capacity_id=f"{paths.suite_id}.component.{case_id}",
            resource=COMPONENT_RESOURCES[raw_resource],
            rate_per_second=float(result["rate_per_second_median"]),
            work_unit=work_unit,
            evidence_kind=EvidenceKind.MEASURED_SUSTAINED,
            source_id=paths.suite_id,
            source_path=paths.relative(paths.component / "summary.json"),
            source_locator=f'"case_id": "{case_id}"',
            condition=("20-SM component campaign; one block per SM; aggregate "
                       "globaltimer span, except CUDA-event NVFP4 epilogue"),
            qualification=qualification,
            trial_count=int(result["trial_count"]),
            artifact_paths=artifacts,
        ))
    if len(capacities) != 9:
        raise ModelError(f"component summary has {len(capacities)} cases, expected 9")
    return capacities


def reference_denominators_from_manifest(
    manifest: dict[str, Any],
) -> dict[str, str]:
    references: dict[str, str] = {}
    for row in manifest.get("precisions", []):
        if row.get("status") != "ready_for_closure_campaign":
            continue
        precision_id = str(row.get("precision_id", ""))
        denominator = row.get("performance_denominator")
        if (not isinstance(denominator, dict)
                or denominator.get("same_precision") is not True
                or denominator.get("status") != "ready"
                or not denominator.get("backend_id")):
            raise ModelError(
                f"{precision_id}: invalid closure performance denominator")
        references[precision_id] = str(denominator["backend_id"])
    if len(references) != 5:
        raise ModelError(
            f"support manifest has {len(references)} ready denominators, expected 5")
    return references


def observations_from_full(
    summary: dict[str, Any], *, references: dict[str, str],
    paths: ClosurePaths, qualification: str,
) -> list[ObservedBest]:
    observations: list[ObservedBest] = []
    for result in sorted(summary.get("results", []), key=lambda row: row["case_id"]):
        case_id = str(result["case_id"])
        precision_id = str(result["precision_id"])
        if precision_id not in references:
            raise ModelError(
                f"full-GEMM result has no same-precision denominator: {precision_id}")
        n = int(result["n"])
        case_dir = paths.full / "cases" / case_id
        trial_count = int(result["trial_count"])
        artifacts = [*_common_artifacts(paths, paths.full),
                     paths.relative(case_dir / "result.json"),
                     paths.relative(case_dir / "trials.jsonl"),
                     paths.relative(paths.full / str(result["sass_path"]))]
        artifacts.extend(
            paths.relative(case_dir / f"trial_{trial:02d}/stdout.log")
            for trial in range(1, trial_count + 1)
        )
        observations.append(ObservedBest(
            observation_id=f"{paths.suite_id}.full.{case_id}",
            precision_id=precision_id,
            m=n,
            n=n,
            k=n,
            backend_id=str(result["backend_id"]),
            reference=references[precision_id],
            performance_reference_relation="same_precision",
            trial_count=trial_count,
            matched_count=trial_count,
            median_per_second=float(result["custom_rate_per_second_median"]),
            maximum_per_second=float(result["custom_rate_per_second_max"]),
            minimum_per_second=float(result["custom_rate_per_second_min"]),
            performance_unit=("operation/s" if result["work_unit"] == "operation"
                              else "flop/s"),
            source_path=paths.relative(paths.full / "summary.json"),
            source_locator=f'"case_id": "{case_id}"',
            artifact_paths=tuple(artifacts),
            run_id=paths.suite_id,
            reference_median_per_second=float(
                result["reference_rate_per_second_median"]),
            ratio_of_paired_medians=float(result["ratio_of_paired_medians"]),
            residency="warm_repeated_device_gemm",
            timed_scope="device_kernel",
            qualification=qualification,
            selection_rule="fixed predeclared candidate and shape; paired same-precision reference",
        ))
    if len(observations) != 15:
        raise ModelError(
            f"full-GEMM summary has {len(observations)} cases, expected 15")
    return observations


def import_closure(
    *, repo_root: Path, suite_id: str, expected_commit: str
) -> dict[str, Any]:
    if not SUITE_ID_RE.fullmatch(suite_id):
        raise ModelError(f"invalid suite ID: {suite_id}")
    if not COMMIT_RE.fullmatch(expected_commit):
        raise ModelError(f"invalid expected commit: {expected_commit}")
    paths = ClosurePaths(repo_root.resolve(), suite_id)
    _require_files((paths.epilogue / "summary.json",
                    paths.compute / "summary.json",
                    paths.component / "summary.json",
                    paths.full / "summary.json"))

    epilogue = _read_json(paths.epilogue / "summary.json")
    if (epilogue.get("schema_version") != 3 or epilogue.get("pass") is not True
            or epilogue.get("expected_commit") != expected_commit):
        raise ModelError("bounded epilogue preflight did not pass at expected commit")
    expected_profiles = {
        "single_cta_smoke", "full_gpu_smoke_bps1", "production_shape_bps1"}
    profiles = epilogue.get("profiles", [])
    if {row.get("profile_id") for row in profiles} != expected_profiles:
        raise ModelError("bounded epilogue preflight profile set is incomplete")
    for row in profiles:
        fields = row.get("fields", {})
        if (row.get("returncode") != 0 or row.get("timed_out") is not False
                or row.get("termination_failed") is not False
                or fields.get("value_mismatches") != "0"
                or fields.get("scale_mismatches") != "0"):
            raise ModelError(f"invalid epilogue preflight profile: {row.get('profile_id')}")

    oc_deltas = _audit_platform(paths, expected_commit)
    qualification = "closure_qualified"
    for run_dir in (paths.compute, paths.component, paths.full):
        _audit_commit_environment(run_dir, expected_commit)

    audit_results = {
        "compute": _run_auditor([
            sys.executable,
            "microbench/sm110_gemm_campaign/audit_campaign.py",
            str(paths.compute),
            "--require-ncu",
        ], repo_root=paths.repo_root),
        "component": _run_auditor([
            sys.executable,
            "microbench/sm110_gemm_component_campaign/audit_campaign.py",
            str(paths.component),
        ], repo_root=paths.repo_root),
        "full_gemm": _run_auditor([
            sys.executable,
            "microbench/sm110_full_gemm_campaign/audit_campaign.py",
            str(paths.full),
        ], repo_root=paths.repo_root),
    }

    compute_summary = _read_json(paths.compute / "summary.json")
    compute_spec = _read_json(paths.compute / "run_spec.json")
    component_summary = _read_json(paths.component / "summary.json")
    component_spec = _read_json(paths.component / "run_spec.json")
    full_summary = _read_json(paths.full / "summary.json")
    full_spec = _read_json(paths.full / "run_spec.json")
    if full_summary.get("ncu_requested") is not True:
        raise ModelError("full-GEMM closure did not request NCU evidence")
    capacities = [
        *capacities_from_compute(
            compute_summary, compute_spec, paths=paths,
            qualification=qualification),
        *capacities_from_component(
            component_summary, component_spec, paths=paths,
            qualification=qualification),
    ]
    support_manifest_path = paths.repo_root / str(full_spec.get("support_manifest", ""))
    references = reference_denominators_from_manifest(
        _read_json(support_manifest_path))
    observations = observations_from_full(
        full_summary, references=references, paths=paths,
        qualification=qualification)
    capacity_findings = audit_inputs(capacities, repo_root=paths.repo_root)
    observation_findings = audit_observations(
        observations, repo_root=paths.repo_root)
    findings = [*capacity_findings, *observation_findings]
    if any(oc_deltas.values()):
        findings.append({
            "severity": "warning",
            "code": "overcurrent_events_observed",
            "message": json.dumps(oc_deltas, sort_keys=True),
        })
    if any(row["severity"] == "error" for row in findings):
        raise ModelError(f"imported closure model inputs failed audit: {findings}")

    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "expected_commit": expected_commit,
        "qualification": qualification,
        "closure_qualified": qualification == "closure_qualified",
        "platform_evidence": {
            "maxn": True,
            "gpu_clock_locked_hz": 1_575_000_000,
            "overcurrent_events_observed": any(oc_deltas.values()),
            "overcurrent_deltas": oc_deltas,
        },
        "epilogue_preflight": epilogue,
        "independent_audits": audit_results,
        "capacities": [capacity.to_dict() for capacity in capacities],
        "observed_best": [observation.to_dict() for observation in observations],
        "model_input_audit": {
            "pass": not any(row["severity"] == "error" for row in findings),
            "findings": findings,
        },
    }
