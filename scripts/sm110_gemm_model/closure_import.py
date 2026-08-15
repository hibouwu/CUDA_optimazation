from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .model import Capacity, EvidenceKind, ModelError, audit_inputs, precision_specs
from .observations import ObservedBest, audit_observations
from .coverage import (
    CAMPAIGN_COMPONENT_RESOURCE_COUNTS,
    CAMPAIGN_COMPUTE_SELECTION,
    CAMPAIGN_FULL_PRECISIONS,
    CAMPAIGN_FULL_SHAPES,
)


SUITE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
COMPUTE_SELECTION = CAMPAIGN_COMPUTE_SELECTION
COMPONENT_RESOURCES = {
    "tma.l2_hit_ingress": "tma.smem_ingress.per_sm",
    "tma.dram_stream_ingress": "tma.hbm",
    "tma.l2_hit_ingress.serial32k":
        "tma.smem_ingress.diagnostic.serial32k.per_sm",
    "tma.dram_stream_ingress.serial32k":
        "tma.hbm.diagnostic.serial32k",
    "tma.l2_hit_ingress.inflight4":
        "tma.smem_ingress.per_sm.inflight4",
    "tma.dram_stream_ingress.inflight4": "tma.hbm.inflight4",
    "tmem.scale_ingress": "tmem.scale_ingress",
    "hbm.read": "hbm.read",
    "hbm.write": "hbm.write",
    "l2.read": "l2.read",
    "l2.write": "l2.write",
    "epilogue.nvfp4_requant": "epilogue.nvfp4_requant",
}
TMEM_MODEL_CASE = "tmem_ld_32x32b_x16_warps4"
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _auditor_path(relative_path: str) -> str:
    path = (SOURCE_ROOT / relative_path).resolve()
    if not path.is_file():
        raise ModelError(f"closure auditor is missing from current source: {path}")
    return str(path)


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


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(
            encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"cannot read closure JSONL {path}: {error}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ModelError(f"closure JSONL contains a non-object row: {path}")
    return rows


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


def _audit_preflight(
    preflight: Path, *, expected_branch: str, expected_commit: str
) -> None:
    preflight_text = preflight.read_text(errors="replace")
    try:
        git_section = preflight_text.split("=== git ===", 1)[1].split(
            "=== nvpmodel ===", 1)[0]
    except IndexError as error:
        raise ModelError("platform preflight Git section is malformed") from error
    git_lines = [line.strip() for line in git_section.splitlines() if line.strip()]
    if git_lines != [expected_branch, expected_commit]:
        raise ModelError(
            f"platform preflight does not prove clean expected checkout: {git_lines}")
    if "NV Power Mode: MAXN" not in preflight_text:
        raise ModelError("platform preflight does not prove MAXN")
    frequency_contracts = (
        "min_freq=1575000000",
        "max_freq=1575000000",
        "CurrentFreq=1575000000",
        "governor=performance",
    )
    missing_frequency = [
        token for token in frequency_contracts if token not in preflight_text
    ]
    if missing_frequency:
        raise ModelError(
            "platform preflight does not prove locked GPU clocks: "
            f"{missing_frequency}")


def _counter_deltas(before_path: Path, after_path: Path) -> dict[str, int]:
    before = _parse_counter_tsv(before_path)
    after = _parse_counter_tsv(after_path)
    if set(before) != set(after):
        raise ModelError("overcurrent counter sets differ before and after campaign")
    deltas = {key: after[key] - value for key, value in before.items()}
    if any(value < 0 for value in deltas.values()):
        raise ModelError(f"overcurrent counters reset during campaign: {deltas}")
    return deltas


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
    _audit_preflight(
        preflight,
        expected_branch="codex/thor-sm110-gemm-bounds-v2",
        expected_commit=expected_commit,
    )
    if "SUITE_COMPLETE" not in log_path.read_text(
            errors="replace").splitlines():
        raise ModelError("suite launcher log does not contain SUITE_COMPLETE")

    return _counter_deltas(before_path, after_path)


def _audit_component_supplement_platform(
    paths: ClosurePaths,
    *,
    base_suite_id: str,
    base_expected_commit: str,
    component_expected_commit: str,
) -> dict[str, int]:
    contract_path = paths.suite / "run_contract.json"
    preflight = paths.suite / "preflight.txt"
    before_path = paths.suite / "oc_before.tsv"
    after_path = paths.suite / "oc_after.tsv"
    log_path = paths.suite / "supplement_launcher.log"
    _require_files((contract_path, preflight, before_path, after_path, log_path))
    contract = _read_json(contract_path)
    required_contract = {
        "schema_version": 2,
        "kind": "component_supplement",
        "supplement_id": paths.suite_id,
        "component_run_id": f"{paths.suite_id}-components",
        "expected_branch": "codex/thor-sm110-gemm-bounds-v2",
        "expected_commit": component_expected_commit,
        "base_suite_id": base_suite_id,
        "base_expected_commit": base_expected_commit,
    }
    if any(contract.get(key) != value for key, value in required_contract.items()):
        raise ModelError(
            "component supplement run contract does not match the composite "
            "closure request")
    _audit_preflight(
        preflight,
        expected_branch="codex/thor-sm110-gemm-bounds-v2",
        expected_commit=component_expected_commit,
    )
    if "COMPONENT_SUPPLEMENT_COMPLETE" not in log_path.read_text(
            errors="replace").splitlines():
        raise ModelError(
            "component supplement log does not contain "
            "COMPONENT_SUPPLEMENT_COMPLETE")
    return _counter_deltas(before_path, after_path)


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
        if (not isinstance(git_record, dict)
                or git_record.get("returncode") != 0
                or str(git_record.get("output", "")).strip() != expected_commit):
            raise ModelError(
                f"{run_dir}: environment record {index} does not prove expected commit")
        status_record = document.get("git_status", {})
        if (not isinstance(status_record, dict)
                or status_record.get("returncode") != 0):
            raise ModelError(
                f"{run_dir}: environment record {index} does not prove Git status")
        status_output = str(status_record.get("output", ""))
        tracked_changes = [
            line for line in status_output.splitlines()
            if line.strip() and not line.startswith("?? ")
        ]
        if tracked_changes:
            raise ModelError(
                f"{run_dir}: environment record {index} has tracked changes: "
                f"{tracked_changes}")


def _common_artifacts(
    paths: ClosurePaths,
    run_dir: Path,
    *,
    include_epilogue: bool = True,
    platform_log: str = "suite_launcher.log",
) -> tuple[str, ...]:
    artifacts = [
        paths.suite / "run_contract.json",
        paths.suite / "preflight.txt",
        paths.suite / "oc_before.tsv",
        paths.suite / "oc_after.tsv",
        paths.suite / platform_log,
        run_dir / "run_spec.json",
        run_dir / "environment.json",
        run_dir / "environment_snapshots.jsonl",
        run_dir / "summary.json",
        run_dir / "COMPLETE",
    ]
    if include_epilogue:
        artifacts.insert(5, paths.epilogue / "summary.json")
    return tuple(paths.relative(path) for path in artifacts)


def capacities_from_compute(
    summary: dict[str, Any], spec: dict[str, Any], *, paths: ClosurePaths,
    qualification: str, identity_id: str | None = None,
) -> list[Capacity]:
    identity_id = identity_id or paths.suite_id
    selected_n = set(COMPUTE_SELECTION["n_values"])
    manifest = {
        str(row["case_id"]): row for row in spec.get("manifest", [])
        if row.get("launch") == COMPUTE_SELECTION["launch"]
        and row.get("m") == COMPUTE_SELECTION["m"]
        and row.get("n") in selected_n
    }
    results = {str(row.get("case_id")): row for row in summary.get("results", [])}
    capacities: list[Capacity] = []
    for case_id, entry in sorted(manifest.items()):
        result = results.get(case_id)
        if result is None:
            raise ModelError(f"selected compute result is missing: {case_id}")
        precision_id = str(result["precision_id"])
        try:
            base_resource = precision_specs()[precision_id].compute_resource
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
            capacity_id=(
                f"{identity_id}.compute.{precision_id}."
                f"m{entry['m']}n{entry['n']}"
            ),
            resource=f"{base_resource}.m{entry['m']}n{entry['n']}",
            rate_per_second=float(result["rate_per_second_median"]),
            work_unit=str(result["work_unit"]),
            evidence_kind=EvidenceKind.MEASURED_SUSTAINED,
            source_id=paths.suite_id,
            source_path=paths.relative(paths.compute / "summary.json"),
            source_locator=f'"case_id": "{case_id}"',
            condition=(f"precision={precision_id}; launch=full_sm_4warp_block; "
                       f"M{entry['m']}N{entry['n']}; "
                       "full-GPU 20-SM compute oracle; SMEM operands; "
                       "device globaltimer issue-to-completion span"),
            qualification=qualification,
            trial_count=int(result["trial_count"]),
            artifact_paths=artifacts,
        ))
    expected = len(precision_specs()) * len(selected_n)
    if len(capacities) != expected:
        raise ModelError(
            f"compute selection has {len(capacities)} cases, expected {expected}")
    return capacities


def capacities_from_component(
    summary: dict[str, Any], spec: dict[str, Any], *, paths: ClosurePaths,
    qualification: str, identity_id: str | None = None,
    include_epilogue: bool = True,
) -> list[Capacity]:
    identity_id = identity_id or paths.suite_id
    case_specs = {str(row["id"]): row for row in spec.get("cases", [])}
    capacities: list[Capacity] = []
    for result in sorted(summary.get("results", []), key=lambda row: row["case_id"]):
        case_id = str(result["case_id"])
        raw_resource = str(result["resource"])
        if raw_resource == "tmem.accumulator_readback":
            resource = (
                "tmem.readback"
                if case_id == TMEM_MODEL_CASE
                else case_id.replace(
                    "tmem_ld_32x32b_", "tmem.readback.").replace(
                        "_warps", ".warps")
            )
        elif raw_resource in COMPONENT_RESOURCES:
            resource = COMPONENT_RESOURCES[raw_resource]
        else:
            raise ModelError(f"unknown component resource: {raw_resource}")
        rate_unit = str(result["rate_unit"])
        work_unit = "element" if rate_unit == "element/s" else "byte"
        if rate_unit not in {"element/s", "B/s"}:
            raise ModelError(f"{case_id}: unsupported component rate unit {rate_unit}")
        if case_id not in case_specs:
            raise ModelError(f"component case is absent from run spec: {case_id}")
        binary = str(case_specs[case_id]["binary"])
        case_arguments = " ".join(
            map(str, case_specs[case_id].get("args", [])))
        case_dir = paths.component / "cases" / case_id
        artifacts = (*_common_artifacts(
                         paths,
                         paths.component,
                         include_epilogue=include_epilogue,
                         platform_log=("suite_launcher.log" if include_epilogue
                                       else "supplement_launcher.log")),
                     paths.relative(case_dir / "result.json"),
                     paths.relative(case_dir / "trials.jsonl"),
                     paths.relative(paths.component / f"build/{binary}.sass.txt"),
                     str(result["source_path"]))
        rate_per_second = float(result["rate_per_second_median"])
        original_value = None
        original_unit = None
        condition_scope = (
            "20-SM component campaign; case-declared blocks per SM; "
            "aggregate globaltimer span"
        )
        if raw_resource.startswith("tma.l2_hit_ingress"):
            condition_scope = (
                "20-SM target contract; exactly one CTA launched and one "
                "SM ID observed; directly isolated per-SM L2-hit ingress"
            )
        if resource == "tma.smem_ingress.per_sm":
            expected_sms = int(spec.get("expected_sm_count", 0))
            if expected_sms != 20:
                raise ModelError(
                    f"{case_id}: per-SM TMA isolation requires the "
                    "frozen 20-SM campaign contract"
                )
            arguments = list(case_specs[case_id].get("args", []))
            try:
                requested_blocks = int(
                    arguments[arguments.index("--blocks") + 1])
                mode = str(arguments[arguments.index("--mode") + 1])
                pattern = str(
                    arguments[arguments.index("--pattern") + 1])
                slots = int(arguments[arguments.index("--slots") + 1])
                inflight = int(
                    arguments[arguments.index("--inflight") + 1])
                threads = int(
                    arguments[arguments.index("--threads") + 1])
            except (ValueError, IndexError) as error:
                raise ModelError(
                    f"{case_id}: per-SM TMA case lacks an explicit launch "
                    "and residency contract"
                ) from error
            if (requested_blocks != 1 or mode != "l2-hit"
                    or pattern != "tc5a-ab" or slots != 8
                    or inflight != 8 or threads != 192):
                raise ModelError(
                    f"{case_id}: per-SM TMA capacity must come from an "
                    "isolated one-CTA exact tc5a L2-hit case"
                )
            condition_scope = (
                "20-SM target contract; exactly one CTA launched and one "
                "SM ID observed; rate is a directly isolated per-SM ingress "
                "measurement and is not divided by the device SM count"
            )
        elif resource == "tma.hbm":
            arguments = list(case_specs[case_id].get("args", []))
            try:
                mode = str(arguments[arguments.index("--mode") + 1])
                pattern = str(
                    arguments[arguments.index("--pattern") + 1])
                slots = int(arguments[arguments.index("--slots") + 1])
                inflight = int(
                    arguments[arguments.index("--inflight") + 1])
                threads = int(
                    arguments[arguments.index("--threads") + 1])
            except (ValueError, IndexError) as error:
                raise ModelError(
                    f"{case_id}: exact tc5a DRAM TMA case lacks its "
                    "pipeline contract"
                ) from error
            if (mode != "dram-stream" or pattern != "tc5a-ab"
                    or slots != 8 or inflight != 8 or threads != 192
                    or "--blocks" in arguments):
                raise ModelError(
                    f"{case_id}: tma.hbm must come from the full-grid exact "
                    "tc5a DRAM-stream case"
                )
        capacities.append(Capacity(
            capacity_id=f"{identity_id}.component.{case_id}",
            resource=resource,
            rate_per_second=rate_per_second,
            work_unit=work_unit,
            evidence_kind=EvidenceKind.MEASURED_SUSTAINED,
            source_id=paths.suite_id,
            source_path=paths.relative(paths.component / "summary.json"),
            source_locator=f'"case_id": "{case_id}"',
            original_value=original_value,
            original_unit=original_unit,
            condition=(f"case_id={case_id}; args={case_arguments}; "
                       f"{condition_scope}, except CUDA-event NVFP4 epilogue"),
            qualification=qualification,
            trial_count=int(result["trial_count"]),
            artifact_paths=artifacts,
        ))
    expected_count = sum(CAMPAIGN_COMPONENT_RESOURCE_COUNTS.values())
    if len(capacities) != expected_count:
        raise ModelError(
            f"component summary has {len(capacities)} cases, expected {expected_count}")
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
    if set(references) != set(CAMPAIGN_FULL_PRECISIONS):
        raise ModelError(
            "support manifest ready denominators differ from the frozen campaign "
            f"precision set: {sorted(references)}")
    return references


def observations_from_full(
    summary: dict[str, Any], *, references: dict[str, str],
    paths: ClosurePaths, qualification: str, identity_id: str | None = None,
) -> list[ObservedBest]:
    identity_id = identity_id or paths.suite_id
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
        trial_rows = _read_json_lines(case_dir / "trials.jsonl")
        if len(trial_rows) != trial_count:
            raise ModelError(
                f"{case_id}: reference trial count differs from summary")
        try:
            reference_rates = [
                float(row["reference_rate_per_second"]) for row in trial_rows
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ModelError(
                f"{case_id}: invalid reference rate in full-GEMM trials") from error
        if (not all(math.isfinite(rate) and rate > 0 for rate in reference_rates)
                or not math.isclose(
                    statistics.median(reference_rates),
                    float(result["reference_rate_per_second_median"]),
                    rel_tol=1e-12)):
            raise ModelError(
                f"{case_id}: reference trial rates do not close to summary")
        for key, expected in (
            ("reference_rate_per_second_min", min(reference_rates)),
            ("reference_rate_per_second_max", max(reference_rates)),
        ):
            if key in result and not math.isclose(
                    float(result[key]), expected, rel_tol=1e-12):
                raise ModelError(f"{case_id}: {key} differs from raw trials")
        artifacts = [*_common_artifacts(paths, paths.full),
                     paths.relative(case_dir / "result.json"),
                     paths.relative(case_dir / "trials.jsonl"),
                     paths.relative(paths.full / str(result["sass_path"]))]
        artifacts.extend(
            paths.relative(case_dir / f"trial_{trial:02d}/stdout.log")
            for trial in range(1, trial_count + 1)
        )
        observations.append(ObservedBest(
            observation_id=f"{identity_id}.full.{case_id}",
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
            reference_maximum_per_second=max(reference_rates),
            reference_minimum_per_second=min(reference_rates),
            ratio_of_paired_medians=float(result["ratio_of_paired_medians"]),
            residency="warm_repeated_device_gemm",
            timed_scope="device_kernel",
            qualification=qualification,
            selection_rule="fixed predeclared candidate and shape; paired same-precision reference",
        ))
    expected_pairs = {
        (precision_id, n)
        for precision_id in CAMPAIGN_FULL_PRECISIONS
        for n in CAMPAIGN_FULL_SHAPES
    }
    actual_pairs = {(row.precision_id, row.n) for row in observations}
    if (len(observations) != len(expected_pairs)
            or actual_pairs != expected_pairs):
        raise ModelError(
            "full-GEMM summary differs from the frozen precision/shape matrix: "
            f"count={len(observations)} pairs={sorted(actual_pairs)}")
    return observations


def _validate_epilogue_preflight(
    paths: ClosurePaths, expected_commit: str
) -> dict[str, Any]:
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
            raise ModelError(
                f"invalid epilogue preflight profile: {row.get('profile_id')}")
    return epilogue


def _campaign_contract(references: dict[str, str]) -> dict[str, Any]:
    return {
        "compute_selection": COMPUTE_SELECTION,
        "compute_precision_count": len(precision_specs()),
        "compute_case_count": (
            len(precision_specs()) * len(COMPUTE_SELECTION["n_values"])
        ),
        "component_case_count": sum(
            CAMPAIGN_COMPONENT_RESOURCE_COUNTS.values()),
        "full_gemm_precisions": sorted(references),
        "full_gemm_shapes": list(CAMPAIGN_FULL_SHAPES),
        "full_gemm_observation_count": (
            len(CAMPAIGN_FULL_PRECISIONS) * len(CAMPAIGN_FULL_SHAPES)),
    }


def _model_input_findings(
    capacities: list[Capacity], observations: list[ObservedBest],
    *, repo_root: Path, platform_deltas: dict[str, dict[str, int]],
) -> list[dict[str, str]]:
    findings = [
        *audit_inputs(capacities, repo_root=repo_root),
        *audit_observations(observations, repo_root=repo_root),
    ]
    for interval, deltas in platform_deltas.items():
        if any(deltas.values()):
            findings.append({
                "severity": "warning",
                "code": "overcurrent_events_observed",
                "message": json.dumps(
                    {"interval": interval, "deltas": deltas}, sort_keys=True),
            })
    return findings


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

    epilogue = _validate_epilogue_preflight(paths, expected_commit)

    oc_deltas = _audit_platform(paths, expected_commit)
    qualification = "closure_qualified"
    for run_dir in (paths.compute, paths.component, paths.full):
        _audit_commit_environment(run_dir, expected_commit)

    audit_results = {
        "compute": _run_auditor([
            sys.executable,
            _auditor_path("microbench/sm110_gemm_campaign/audit_campaign.py"),
            str(paths.compute),
            "--require-ncu",
        ], repo_root=paths.repo_root),
        "component": _run_auditor([
            sys.executable,
            _auditor_path(
                "microbench/sm110_gemm_component_campaign/audit_campaign.py"),
            str(paths.component),
        ], repo_root=paths.repo_root),
        "full_gemm": _run_auditor([
            sys.executable,
            _auditor_path(
                "microbench/sm110_full_gemm_campaign/audit_campaign.py"),
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
    findings = _model_input_findings(
        capacities,
        observations,
        repo_root=paths.repo_root,
        platform_deltas={suite_id: oc_deltas},
    )
    if any(row["severity"] == "error" for row in findings):
        raise ModelError(f"imported closure model inputs failed audit: {findings}")

    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "expected_commit": expected_commit,
        "qualification": qualification,
        "closure_qualified": qualification == "closure_qualified",
        "composition": "single_suite",
        "campaign_sources": {
            "base": {
                "suite_id": suite_id,
                "expected_commit": expected_commit,
                "provides": [
                    "epilogue_preflight",
                    "compute",
                    "component",
                    "full_gemm",
                ],
            },
        },
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
        "campaign_contract": _campaign_contract(references),
    }


def import_composite_closure(
    *,
    repo_root: Path,
    composite_id: str,
    base_suite_id: str,
    base_expected_commit: str,
    component_expected_commit: str,
) -> dict[str, Any]:
    """Compose immutable base compute/full evidence with a new component run.

    The two execution intervals remain independently attributable.  This is not
    a relaxed single-commit import: every environment record is checked against
    the commit that produced that campaign, and both platform intervals retain
    their own preflight and overcurrent evidence.
    """
    for label, value in (
        ("composite ID", composite_id),
        ("base suite ID", base_suite_id),
    ):
        if not SUITE_ID_RE.fullmatch(value):
            raise ModelError(f"invalid {label}: {value}")
    for label, value in (
        ("base expected commit", base_expected_commit),
        ("component expected commit", component_expected_commit),
    ):
        if not COMMIT_RE.fullmatch(value):
            raise ModelError(f"invalid {label}: {value}")
    if composite_id == base_suite_id:
        raise ModelError("composite ID must differ from the base suite ID")

    repo_root = repo_root.resolve()
    base_paths = ClosurePaths(repo_root, base_suite_id)
    component_paths = ClosurePaths(repo_root, composite_id)
    _require_files((
        base_paths.epilogue / "summary.json",
        base_paths.compute / "summary.json",
        base_paths.full / "summary.json",
        component_paths.component / "summary.json",
    ))

    epilogue = _validate_epilogue_preflight(
        base_paths, base_expected_commit)
    base_oc_deltas = _audit_platform(base_paths, base_expected_commit)
    component_oc_deltas = _audit_component_supplement_platform(
        component_paths,
        base_suite_id=base_suite_id,
        base_expected_commit=base_expected_commit,
        component_expected_commit=component_expected_commit,
    )
    _audit_commit_environment(base_paths.compute, base_expected_commit)
    _audit_commit_environment(base_paths.full, base_expected_commit)
    _audit_commit_environment(
        component_paths.component, component_expected_commit)

    audit_results = {
        "compute": _run_auditor([
            sys.executable,
            _auditor_path("microbench/sm110_gemm_campaign/audit_campaign.py"),
            str(base_paths.compute),
            "--require-ncu",
        ], repo_root=repo_root),
        "component": _run_auditor([
            sys.executable,
            _auditor_path(
                "microbench/sm110_gemm_component_campaign/audit_campaign.py"),
            str(component_paths.component),
        ], repo_root=repo_root),
        "full_gemm": _run_auditor([
            sys.executable,
            _auditor_path(
                "microbench/sm110_full_gemm_campaign/audit_campaign.py"),
            str(base_paths.full),
        ], repo_root=repo_root),
    }

    compute_summary = _read_json(base_paths.compute / "summary.json")
    compute_spec = _read_json(base_paths.compute / "run_spec.json")
    component_summary = _read_json(component_paths.component / "summary.json")
    component_spec = _read_json(component_paths.component / "run_spec.json")
    full_summary = _read_json(base_paths.full / "summary.json")
    full_spec = _read_json(base_paths.full / "run_spec.json")
    if full_summary.get("ncu_requested") is not True:
        raise ModelError("base full-GEMM closure did not request NCU evidence")

    qualification = "closure_qualified"
    capacities = [
        *capacities_from_compute(
            compute_summary,
            compute_spec,
            paths=base_paths,
            qualification=qualification,
            identity_id=composite_id,
        ),
        *capacities_from_component(
            component_summary,
            component_spec,
            paths=component_paths,
            qualification=qualification,
            identity_id=composite_id,
            include_epilogue=False,
        ),
    ]
    support_manifest_path = repo_root / str(full_spec.get("support_manifest", ""))
    references = reference_denominators_from_manifest(
        _read_json(support_manifest_path))
    observations = observations_from_full(
        full_summary,
        references=references,
        paths=base_paths,
        qualification=qualification,
        identity_id=composite_id,
    )
    interval_deltas = {
        base_suite_id: base_oc_deltas,
        composite_id: component_oc_deltas,
    }
    findings = _model_input_findings(
        capacities,
        observations,
        repo_root=repo_root,
        platform_deltas=interval_deltas,
    )
    if any(row["severity"] == "error" for row in findings):
        raise ModelError(
            f"imported composite closure model inputs failed audit: {findings}")

    return {
        "schema_version": 1,
        "suite_id": composite_id,
        "expected_commit": component_expected_commit,
        "qualification": qualification,
        "closure_qualified": True,
        "composition": "base_compute_full_plus_component_supplement",
        "campaign_sources": {
            "base": {
                "suite_id": base_suite_id,
                "expected_commit": base_expected_commit,
                "provides": ["epilogue_preflight", "compute", "full_gemm"],
            },
            "component_supplement": {
                "suite_id": composite_id,
                "expected_commit": component_expected_commit,
                "provides": ["component"],
            },
        },
        "platform_evidence": {
            "maxn": True,
            "gpu_clock_locked_hz": 1_575_000_000,
            "overcurrent_events_observed": any(
                value for deltas in interval_deltas.values()
                for value in deltas.values()),
            "overcurrent_deltas": interval_deltas,
        },
        "epilogue_preflight": epilogue,
        "independent_audits": audit_results,
        "capacities": [capacity.to_dict() for capacity in capacities],
        "observed_best": [observation.to_dict() for observation in observations],
        "model_input_audit": {
            "pass": not any(row["severity"] == "error" for row in findings),
            "findings": findings,
        },
        "campaign_contract": _campaign_contract(references),
    }
