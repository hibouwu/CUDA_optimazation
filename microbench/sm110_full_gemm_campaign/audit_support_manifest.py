#!/usr/bin/env python3
"""Fail-closed audit of the full-GEMM implementation/evidence coverage map."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("support_manifest.json")
REQUIRED = {
    "fp16_f32", "bf16_f32", "tf32_f32", "e4m3_f32", "e5m2_f32",
    "e3m2_f32", "e2m3_f32", "e2m1_f32", "mxfp4_f32", "nvfp4_f32",
    "s8_s32", "u8_s32",
}
STATUSES = {"ready_for_closure_campaign", "partial", "missing"}


def audit() -> list[str]:
    data = json.loads(MANIFEST.read_text())
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    contract = data.get("problem_contract", {})
    expected_contract = {
        "layout": "NN", "epilogue": "none", "beta": 0,
        "output_mode": "accumulator", "work": "2*M*N*K",
        "required_trials": 10,
    }
    if contract != expected_contract:
        errors.append("problem_contract differs from the frozen v1 contract")

    rows = data.get("precisions", [])
    ids = [row.get("precision_id") for row in rows]
    if set(ids) != REQUIRED or len(ids) != len(REQUIRED):
        errors.append(f"precision IDs differ: have={ids}")

    for row in rows:
        pid = str(row.get("precision_id"))
        status = row.get("status")
        if status not in STATUSES:
            errors.append(f"{pid}: invalid status {status}")
            continue
        expected_unit = "operation" if pid in {"s8_s32", "u8_s32"} else "flop"
        if row.get("work_unit") != expected_unit:
            errors.append(f"{pid}: work_unit must be {expected_unit}")
        implementation = row.get("implementation")
        if not isinstance(implementation, dict):
            errors.append(f"{pid}: implementation must be an object")
            continue
        for source in implementation.get("source_paths", []):
            if not (REPO_ROOT / source).is_file():
                errors.append(f"{pid}: missing source path {source}")
        reference = row.get("numerical_reference") or {}
        reference_source = reference.get("source_path")
        if reference_source and not (REPO_ROOT / reference_source).is_file():
            errors.append(
                f"{pid}: missing numerical reference source path "
                f"{reference_source}")

        blockers = row.get("blockers")
        if not isinstance(blockers, list):
            errors.append(f"{pid}: blockers must be a list")
        if status == "ready_for_closure_campaign":
            denominator = row.get("performance_denominator") or {}
            if not row.get("native_mainloop"):
                errors.append(f"{pid}: ready row lacks a native mainloop")
            if not implementation.get("backend_ids"):
                errors.append(f"{pid}: ready row lacks an implementation")
            if not implementation.get("closure_candidate_backend_ids"):
                errors.append(
                    f"{pid}: ready row lacks a closure candidate backend")
            if not reference.get("same_input_precision"):
                errors.append(f"{pid}: ready row lacks same-precision correctness")
            if not reference.get("same_output_type"):
                errors.append(f"{pid}: ready row lacks same-output correctness")
            if not reference_source:
                errors.append(
                    f"{pid}: ready row lacks a numerical reference source")
            if not denominator.get("same_precision") or denominator.get("status") != "ready":
                errors.append(f"{pid}: ready row lacks a same-precision denominator")
            if not denominator.get("backend_id"):
                errors.append(
                    f"{pid}: ready row lacks a denominator backend")
            if blockers:
                errors.append(f"{pid}: ready row must have no blockers")
        else:
            if not blockers:
                errors.append(f"{pid}: non-ready row must explain its blockers")
            denominator = row.get("performance_denominator")
            if denominator and denominator.get("same_precision") is False:
                if denominator.get("status") != "invalid_cross_precision":
                    errors.append(f"{pid}: cross-precision denominator not quarantined")
    return errors


def main() -> int:
    errors = audit()
    rows = json.loads(MANIFEST.read_text())["precisions"]
    summary = {
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "pass": not errors,
        "ready": sorted(r["precision_id"] for r in rows if r["status"] == "ready_for_closure_campaign"),
        "partial": sorted(r["precision_id"] for r in rows if r["status"] == "partial"),
        "missing": sorted(r["precision_id"] for r in rows if r["status"] == "missing"),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
