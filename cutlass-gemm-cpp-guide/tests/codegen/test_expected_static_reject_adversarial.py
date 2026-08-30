#!/usr/bin/env python3
"""Adversarial checks for fail-closed EXPECTED_STATIC_REJECT primitives."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "tools"))

from codegen_v2.model import (  # noqa: E402
    ContractError,
    _reject_diagnostic_value,
    expected_actual_docker_argv,
    load_run_campaign,
    sha256_bytes,
    sha256_file,
    validate_journal_semantics,
)
from run_codegen_v2 import (  # noqa: E402
    CommandRecord,
    RunnerError,
    classify_expected_reject_diagnostic,
)


PRIMARY = 'error: static assertion failed with "Could not build collective"'
CONTEXT = "instantiation context: reject_fixture KernelScheduleAuto no CollectiveOp"
SECONDARY = "error: no type named CollectiveOp"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def instance_contract() -> dict:
    return {
        "instance_id": "reject_fixture",
        "hypothesis": {
            "reject_contract": {
                "step_id": "compile_type_witness",
                "expected_returncode": 2,
                "expected_error_count": 2,
                "required_diagnostics": [
                    {
                        "id": "primary_failure",
                        "regex": r"(?m)^error: static assertion failed with \"Could not build collective\"$",
                        "min_count": 1,
                        "max_count": 1,
                    },
                    {
                        "id": "subject_instantiation",
                        "regex": r"(?m)^instantiation context: reject_fixture KernelScheduleAuto no CollectiveOp$",
                        "min_count": 1,
                        "max_count": 1,
                    },
                ],
                "source_constraints": [],
            }
        },
    }


def diagnostic_root() -> tuple[Path, CommandRecord, dict]:
    root = Path(tempfile.mkdtemp(prefix="expected-reject-adversarial-"))
    contract_path = root / "tests/codegen/static_codegen_contract.json"
    contract_path.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "tests/codegen/static_codegen_contract.json", contract_path)
    logs = root / "logs"
    logs.mkdir()
    stdout = logs / "compile.stdout"
    stderr = logs / "compile.stderr"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text(f"{PRIMARY}\n{CONTEXT}\n{SECONDARY}\n2 errors detected\n", encoding="utf-8")
    return root, CommandRecord("compile_type_witness", stdout, stderr, 2), instance_contract()


def expect_runner_rejected(name: str, expected: str, mutate) -> None:
    root, record, instance = diagnostic_root()
    try:
        mutate(root, record, instance)
        try:
            classify_expected_reject_diagnostic(root, instance, record)
        except RunnerError as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: wrong rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: deliberate rejection corruption was accepted")
    finally:
        shutil.rmtree(root)


def semantic_fixture() -> tuple[Path, list[dict], dict, dict, Path]:
    root = Path(tempfile.mkdtemp(prefix="expected-reject-journal-"))
    contract_path = root / "tests/codegen/static_codegen_contract.json"
    contract_path.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "tests/codegen/static_codegen_contract.json", contract_path)
    attempt_id = "reject-run.reject_fixture.a001"
    attempt_root = root / "artifacts/codegen-sm110a-v2/reject-run/attempts" / attempt_id
    logs = attempt_root / "full/logs"
    logs.mkdir(parents=True)
    stdout = logs / "compile_type_witness.stdout"
    stderr = logs / "compile_type_witness.stderr"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text(f"{PRIMARY}\n{CONTEXT}\n{SECONDARY}\n2 errors detected\n", encoding="utf-8")
    report = root / "evidence/codegen-sm110a-v2/excerpts" / attempt_id / "rejection-report.json"
    write_json(
        report,
        {
            "failure": {
                "step_id": "compile_type_witness",
                "returncode": 2,
            }
        },
    )
    items = [
        {
            "artifact_id": "compile_type_witness_stdout",
            "role": "COMPILE_STDOUT",
            "path": stdout.relative_to(root).as_posix(),
            "sha256": sha256_file(stdout),
            "producer_step_id": "compile_type_witness",
        },
        {
            "artifact_id": "compile_type_witness_stderr",
            "role": "COMPILE_STDERR",
            "path": stderr.relative_to(root).as_posix(),
            "sha256": sha256_file(stderr),
            "producer_step_id": "compile_type_witness",
        },
        {
            "artifact_id": "rejection_report",
            "role": "REJECTION_REPORT",
            "path": report.relative_to(root).as_posix(),
            "sha256": sha256_file(report),
            "producer_step_id": "reject_contract",
        },
    ]
    artifacts = {
        "instance_ref": {"sha256": "a" * 64},
        "archive_bundle": {"path": attempt_root.relative_to(root).as_posix()},
        "items": items,
    }
    step = {
        "step_id": "compile_type_witness",
        "tool": "nvcc",
        "argv": ["-v", "<attempt>/full/type_witness"],
    }
    fingerprint = {
        "overall_sha256": "b" * 64,
        "execution_plan": [step],
    }
    normalized = expected_actual_docker_argv(root, attempt_root, step, artifacts)
    actual = [
        value.replace("<host-user>", f"{os.getuid()}:{os.getgid()}")
        .replace("<guide-root>", str(root))
        .replace("<attempt-root>", str(attempt_root))
        for value in normalized
    ]
    def event(event_type: str, payload: dict) -> dict:
        return {
            "event_type": event_type,
            "attempt_id": attempt_id,
            "instance_id": "reject_fixture",
            "fingerprint_id": "fp-reject",
            "payload": payload,
        }
    events = [
        event("ATTEMPT_CREATED", {"run_id": "reject-run", "origin_guide_root": str(root), "origin_attempt_root": str(attempt_root), "origin_user": f"{os.getuid()}:{os.getgid()}"}),
        event("INSTANCE_VALIDATED", {"instance_sha256": "a" * 64}),
        event("FINGERPRINT_SEALED", {"overall_sha256": "b" * 64}),
        event("STEP_STARTED", {"step_id": "compile_type_witness", "argv": actual}),
        event("COMMAND_FINISHED", {"step_id": "compile_type_witness", "returncode": 2, "stdout_path": items[0]["path"], "stdout_sha256": items[0]["sha256"], "stderr_path": items[1]["path"], "stderr_sha256": items[1]["sha256"]}),
    ]
    for item in items:
        events.append(event("ARTIFACT_SEALED", {"artifact_id": item["artifact_id"], "path": item["path"], "sha256": item["sha256"], "producer_step_id": item["producer_step_id"]}))
    manifest = root / "evidence/codegen-sm110a-v2/artifact-manifests/am-reject.json"
    write_json(manifest, {"fixture": True})
    events.extend(
        [
            event("ARTIFACT_MANIFEST_SEALED", {"path": manifest.relative_to(root).as_posix(), "sha256": sha256_file(manifest)}),
            event("ATTEMPT_SEALED", {"terminal_pipeline_state": "EXPECTED_STATIC_REJECT", "artifact_manifest_path": manifest.relative_to(root).as_posix(), "artifact_manifest_sha256": sha256_file(manifest), "last_completed_step_id": "compile_type_witness"}),
        ]
    )
    return root, events, fingerprint, artifacts, manifest


def expect_semantic_rejected(name: str, expected: str, mutate) -> None:
    root, events, fingerprint, artifacts, manifest = semantic_fixture()
    try:
        mutate(events, fingerprint, artifacts, manifest)
        try:
            validate_journal_semantics(
                root,
                events,
                fingerprint,
                artifacts,
                manifest,
                "EXPECTED_STATIC_REJECT",
            )
        except ContractError as error:
            if expected not in str(error):
                raise AssertionError(f"{name}: wrong rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: corrupted reject journal was accepted")
    finally:
        shutil.rmtree(root)


def main() -> int:
    root, record, instance = diagnostic_root()
    try:
        _, results, error_count, selected = classify_expected_reject_diagnostic(
            root, instance, record
        )
        diagnostic, model_results, model_count = _reject_diagnostic_value(
            root,
            instance,
            record.stderr_path.read_text(encoding="utf-8"),
            sha256_file(record.stderr_path),
            record.stderr_path.stat().st_size,
        )
        if results != model_results or error_count != model_count or not selected or not diagnostic["selected_lines"]:
            raise AssertionError("positive diagnostic classification differs between runner/model")
    finally:
        shutil.rmtree(root)

    expect_runner_rejected("wrong_returncode", "return code differs", lambda r, x, i: setattr(x, "returncode", 1))
    expect_runner_rejected("docker_125", "infrastructure return code", lambda r, x, i: (setattr(x, "returncode", 125), i["hypothesis"]["reject_contract"].__setitem__("expected_returncode", 125)))
    expect_runner_rejected("signal_137", "infrastructure return code", lambda r, x, i: (setattr(x, "returncode", 137), i["hypothesis"]["reject_contract"].__setitem__("expected_returncode", 137)))
    expect_runner_rejected("empty_stderr", "empty or exceeds", lambda r, x, i: x.stderr_path.write_bytes(b""))
    expect_runner_rejected("non_utf8", "not UTF-8", lambda r, x, i: x.stderr_path.write_bytes(b"\xff"))
    expect_runner_rejected("missing_primary", "primary_failure differs", lambda r, x, i: x.stderr_path.write_text(f"{CONTEXT}\n{SECONDARY}\n", encoding="utf-8"))
    expect_runner_rejected("missing_context", "subject_instantiation differs", lambda r, x, i: x.stderr_path.write_text(f"{PRIMARY}\n{SECONDARY}\n", encoding="utf-8"))
    expect_runner_rejected("error_count", "error count differs", lambda r, x, i: x.stderr_path.write_text(f"{PRIMARY}\n{CONTEXT}\n", encoding="utf-8"))
    infra = ["out of memory", "internal compiler error", "No space left on device", "Permission denied", "No such file or directory", "nvcc fatal"]
    for index, text in enumerate(infra):
        expect_runner_rejected(f"infra_{index}", "infrastructure diagnostic", lambda r, x, i, text=text: x.stderr_path.write_text(f"{PRIMARY}\n{CONTEXT}\n{SECONDARY}\n{text}\n", encoding="utf-8"))

    campaign_root = Path(tempfile.mkdtemp(prefix="run-campaign-adversarial-"))
    try:
        campaign_path = campaign_root / "tests/codegen/run-campaigns/good.json"
        write_json(campaign_path, {"schema_version": 1, "objective_sha256": "463fa7015808acd883b28d115fa33708f66064aaceed96f728996a01ca0e4091", "scope": "STATIC_CODEGEN_ONLY", "phase_id": 4, "campaign_id": "good", "run_id": "good", "current_summary": "evidence/codegen-sm110a-v2/summary-good.json", "allowed_terminal_statuses": ["STATIC_PASS", "EXPECTED_STATIC_REJECT"], "ordered_instances": ["a", "b"]})
        load_run_campaign(campaign_root, "good")
        value = json.loads(campaign_path.read_text())
        value["ordered_instances"] = ["a", "a"]
        write_json(campaign_path, value)
        try:
            load_run_campaign(campaign_root, "good")
        except ContractError:
            pass
        else:
            raise AssertionError("duplicate campaign member was accepted")
    finally:
        shutil.rmtree(campaign_root)

    root, events, fingerprint, artifacts, manifest = semantic_fixture()
    try:
        validate_journal_semantics(root, events, fingerprint, artifacts, manifest, "EXPECTED_STATIC_REJECT")
    finally:
        shutil.rmtree(root)
    expect_semantic_rejected("failure_rc_zero", "did not fail", lambda e, f, a, m: e[4]["payload"].__setitem__("returncode", 0))
    expect_semantic_rejected("wrong_argv", "actual argv differs", lambda e, f, a, m: e[3]["payload"].__setitem__("argv", ["/bin/false"]))
    expect_semantic_rejected("wrong_last_step", "ATTEMPT_SEALED payload", lambda e, f, a, m: e[-1]["payload"].__setitem__("last_completed_step_id", "function_contract"))
    expect_semantic_rejected("missing_stderr_seal", "ARTIFACT_SEALED set differs", lambda e, f, a, m: e.pop(6))

    print("EXPECTED_STATIC_REJECT_ADVERSARIAL_PASS mutations=20 positive=3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
