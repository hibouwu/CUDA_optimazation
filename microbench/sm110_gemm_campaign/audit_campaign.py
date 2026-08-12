#!/usr/bin/env python3
"""Fail-closed audit for a returned Thor compute campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


CAMPAIGN_DIR = Path(__file__).resolve().parent

def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finding(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def audit(run_dir: Path, *, require_ncu: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required_top = (
        "run_spec.json",
        "environment.json",
        "environment_snapshots.jsonl",
        "campaign_status.json",
        "progress.jsonl",
        "summary.json",
        "COMPLETE",
    )
    for name in required_top:
        if not (run_dir / name).is_file():
            rows.append(finding("error", "missing_top_level_artifact", name))
    if rows:
        return rows

    spec = json.loads((run_dir / "run_spec.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    status = json.loads((run_dir / "campaign_status.json").read_text())
    complete_text = (run_dir / "COMPLETE").read_text()
    if summary.get("status") != "complete":
        rows.append(finding("error", "summary_not_complete", str(summary.get("status"))))
    if status.get("status") != "complete":
        rows.append(finding("error", "campaign_status_not_complete", str(status.get("status"))))
    expected_summary_hash = sha256_path(run_dir / "summary.json")
    if f"summary_sha256={expected_summary_hash}" not in complete_text:
        rows.append(finding("error", "complete_hash_mismatch", expected_summary_hash))

    manifest = spec.get("manifest", [])
    result_rows = summary.get("results", [])
    manifest_ids = [entry.get("case_id") for entry in manifest]
    result_ids = [entry.get("case_id") for entry in result_rows]
    if len(manifest_ids) != 72 or len(set(manifest_ids)) != 72:
        rows.append(finding("error", "manifest_cardinality", str(len(manifest_ids))))
    if set(result_ids) != set(manifest_ids):
        rows.append(finding("error", "summary_manifest_mismatch", "case IDs differ"))
    if spec.get("trials", 0) < 10:
        rows.append(finding("error", "insufficient_declared_trials", str(spec.get("trials"))))
    if spec.get("expected_sm_count") != 20:
        rows.append(finding("error", "unexpected_sm_contract", str(spec.get("expected_sm_count"))))
    generator_path = CAMPAIGN_DIR / "run_compute_campaign.py"
    if spec.get("generator_sha256") != sha256_path(generator_path):
        rows.append(finding("error", "generator_hash_mismatch", str(spec.get("generator_sha256"))))
    if not str(spec.get("ptx_primary_source", "")).startswith("https://docs.nvidia.com/"):
        rows.append(finding("error", "non_primary_descriptor_source", str(spec.get("ptx_primary_source"))))
    if spec.get("timed_scope") != "device_globaltimer_mma_issue_to_completion_barrier":
        rows.append(finding("error", "timed_scope_mismatch", str(spec.get("timed_scope"))))
    if spec.get("residency") != "compute_oracle_smem_operands":
        rows.append(finding("error", "residency_mismatch", str(spec.get("residency"))))

    environment = json.loads((run_dir / "environment.json").read_text())
    snapshots = [
        json.loads(line)
        for line in (run_dir / "environment_snapshots.jsonl").read_text().splitlines()
        if line
    ]
    if not snapshots:
        rows.append(finding("error", "missing_environment_snapshots", "no snapshots"))
    identities: set[str] = set()
    for index, snapshot in enumerate(snapshots):
        gpu_text = json.dumps(snapshot)
        if "11.0" not in gpu_text and "Thor" not in gpu_text:
            rows.append(finding("error", "thor_identity_not_found", f"snapshot={index}"))
        power_mode = snapshot.get("power_mode", {})
        if (power_mode.get("returncode") != 0
                or "MAXN" not in str(power_mode.get("output", "")).upper()):
            rows.append(finding("error", "maxn_power_mode_not_proven", f"snapshot={index}"))
        identity = snapshot.get("nvidia_smi_identity_csv", {})
        if identity.get("returncode") != 0 or not str(identity.get("output", "")).strip():
            rows.append(finding("error", "gpu_identity_query_failed", f"snapshot={index}"))
        else:
            identities.add(str(identity.get("output", "")).strip())
    initial_identity = environment.get("nvidia_smi_identity_csv", {}).get("output", "")
    if str(initial_identity).strip():
        identities.add(str(initial_identity).strip())
    if len(identities) != 1:
        rows.append(finding("error", "gpu_identity_changed", str(sorted(identities))))

    ncu_precisions: set[str] = set()
    for result in result_rows:
        case_id = str(result.get("case_id"))
        case_dir = run_dir / "cases" / case_id
        artifacts = {
            "source": case_dir / "source.cu",
            "descriptor": case_dir / "descriptor.json",
            "compile_log": case_dir / "compile.log",
            "compile_command": case_dir / "compile_command.json",
            "binary_hash": case_dir / "binary.sha256",
            "sass": case_dir / "sass.txt",
            "trials": case_dir / "trials.jsonl",
            "result": case_dir / "result.json",
        }
        for label, path in artifacts.items():
            if not path.is_file():
                rows.append(finding("error", "missing_case_artifact", f"{case_id}:{label}"))
        if any(not path.is_file() for path in artifacts.values()):
            continue
        on_disk = json.loads(artifacts["result"].read_text())
        if on_disk != result:
            rows.append(finding("error", "summary_result_mismatch", case_id))
        for key, path in (("source_sha256", artifacts["source"]),
                          ("sass_sha256", artifacts["sass"])):
            if result.get(key) != sha256_path(path):
                rows.append(finding("error", "artifact_hash_mismatch", f"{case_id}:{key}"))
        binary_hash_text = artifacts["binary_hash"].read_text().split()
        if not binary_hash_text or binary_hash_text[0] != result.get("binary_sha256"):
            rows.append(finding("error", "binary_hash_mismatch", case_id))
        descriptor = json.loads(artifacts["descriptor"].read_text())
        manifest_entry = next(
            (entry for entry in manifest if entry.get("case_id") == case_id), None
        )
        if (
            manifest_entry is None
            or descriptor.get("value_u32") != result.get("descriptor_u32")
            or manifest_entry.get("descriptor", {}).get("value_u32")
            != result.get("descriptor_u32")
        ):
            rows.append(finding("error", "descriptor_contract_mismatch", case_id))
        source = artifacts["source"].read_text()
        if f"kIdesc = {result.get('descriptor_u32')}u" not in source:
            rows.append(finding("error", "source_descriptor_mismatch", case_id))
        try:
            compile_command = json.loads(artifacts["compile_command"].read_text())
        except json.JSONDecodeError:
            compile_command = None
        if not isinstance(compile_command, list) or "arch=compute_110a,code=sm_110a" not in compile_command:
            rows.append(finding("error", "compile_command_invalid", case_id))
        if result.get("status") != "ok" or result.get("trial_count") != spec.get("trials"):
            rows.append(finding("error", "case_not_complete", case_id))
        if (result.get("timed_scope") != spec.get("timed_scope")
                or result.get("residency") != spec.get("residency")):
            rows.append(finding("error", "case_semantics_mismatch", case_id))
        if not result.get("source_dense") or not result.get("expected_sass_found"):
            rows.append(finding("error", "instruction_audit_failed", case_id))
        sass = artifacts["sass"].read_text()
        if "tcgen05.mma.sp" in source:
            rows.append(finding("error", "sparse_source_in_dense_campaign", case_id))
        if str(result.get("expected_sass")) not in sass:
            rows.append(finding("error", "expected_sass_missing", case_id))

        trials = [json.loads(line) for line in artifacts["trials"].read_text().splitlines() if line]
        if len(trials) != spec.get("trials"):
            rows.append(finding("error", "trial_file_count_mismatch", case_id))
        audited_rates: list[float] = []
        for trial in trials:
            fields = trial.get("fields", {})
            try:
                rate = float(fields.get("rate_per_second", "nan"))
                elapsed = float(fields.get("elapsed_seconds", "nan"))
                kernel_elapsed = float(fields.get("host_kernel_elapsed_seconds", "nan"))
                nanoseconds = int(fields.get("globaltimer_nanoseconds", "0"))
                start_min = int(fields.get("globaltimer_start_min", "0"))
                stop_max = int(fields.get("globaltimer_stop_max", "0"))
                issued = float(fields.get("issued_work", "nan"))
                blocks = int(fields.get("blocks", "0"))
                unique_smids = int(fields.get("unique_smid_count", "0"))
                sm_count = int(fields.get("sm_count", "0"))
                logical_bits = int(fields.get("logical_input_bits", "0"))
                descriptor_bits = int(fields.get("descriptor_storage_bits", "0"))
                warps = int(fields.get("warps_per_block", "0"))
                iters = int(fields.get("iters", "0"))
            except (TypeError, ValueError):
                rate = math.nan
                elapsed = math.nan
                kernel_elapsed = math.nan
                nanoseconds = 0
                start_min = stop_max = 0
                issued = math.nan
                blocks = warps = iters = 0
                unique_smids = sm_count = 0
                logical_bits = descriptor_bits = 0
            precision = (manifest_entry or {}).get("precision", {})
            expected_unique = (
                int(spec.get("expected_sm_count", 0))
                if (manifest_entry or {}).get("launch") == "full_sm_4warp_block"
                else 1
            )
            expected_logical_bits = int(precision.get("input_bits", 0))
            expected_descriptor_bits = (
                4
                if precision.get("kind") in {"mxf4", "mxf4nvf4"}
                else 16
                if expected_logical_bits == 16
                else 8
            )
            expected_issued = (
                2
                * int((manifest_entry or {}).get("m", 0))
                * int((manifest_entry or {}).get("n", 0))
                * int(precision.get("k", 0))
                * blocks
                * warps
                * iters
            )
            arithmetic_ok = (
                math.isfinite(issued)
                and issued == expected_issued
                and math.isfinite(elapsed)
                and elapsed > 0
                and nanoseconds > 0
                and stop_max > start_min
                and nanoseconds == stop_max - start_min
                and math.isclose(elapsed, nanoseconds * 1e-9, rel_tol=2e-9)
                and math.isclose(rate, issued / elapsed, rel_tol=2e-9)
            )
            if (trial.get("returncode") != 0 or fields.get("case_id") != case_id
                    or fields.get("precision_id") != result.get("precision_id")
                    or fields.get("work_unit") != result.get("work_unit")
                    or not math.isfinite(rate) or rate <= 0
                    or not math.isfinite(elapsed) or elapsed <= 0
                    or not math.isfinite(kernel_elapsed) or kernel_elapsed <= 0
                    or sm_count != spec.get("expected_sm_count")
                    or unique_smids != expected_unique
                    or logical_bits != expected_logical_bits
                    or descriptor_bits != expected_descriptor_bits
                    or not arithmetic_ok):
                rows.append(finding("error", "invalid_trial", f"{case_id}:{trial.get('trial')}"))
            else:
                audited_rates.append(rate)

        if len(audited_rates) == len(trials) and audited_rates:
            aggregates = {
                "rate_per_second_median": statistics.median(audited_rates),
                "rate_per_second_min": min(audited_rates),
                "rate_per_second_max": max(audited_rates),
                "rate_per_second_mean": statistics.fmean(audited_rates),
            }
            for key, value in aggregates.items():
                try:
                    reported = float(result.get(key, "nan"))
                except (TypeError, ValueError):
                    reported = math.nan
                if not math.isclose(reported, value, rel_tol=1e-12):
                    rows.append(finding("error", "aggregate_mismatch", f"{case_id}:{key}"))

        ncu = result.get("ncu", {})
        if ncu.get("selected"):
            ncu_precisions.add(str(result.get("precision_id")))
            report = case_dir / "ncu" / "profile.ncu-rep"
            log = case_dir / "ncu" / "profile.log"
            if (ncu.get("returncode") != 0 or ncu.get("permission_denied")
                    or not report.is_file() or not log.is_file()):
                rows.append(finding("error", "invalid_ncu_evidence", case_id))
            elif (ncu.get("report_sha256") != sha256_path(report)
                  or ncu.get("log_sha256") != sha256_path(log)):
                rows.append(finding("error", "ncu_hash_mismatch", case_id))

    if require_ncu:
        declared_precisions = {
            str(entry.get("precision", {}).get("precision_id")) for entry in manifest
        }
        if ncu_precisions != declared_precisions:
            rows.append(finding(
                "error", "ncu_precision_coverage",
                f"have={sorted(ncu_precisions)} expected={sorted(declared_precisions)}",
            ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-ncu", action="store_true")
    args = parser.parse_args()
    rows = audit(args.run_dir.resolve(), require_ncu=args.require_ncu)
    output = {
        "run_dir": str(args.run_dir.resolve()),
        "finding_count": len(rows),
        "findings": rows,
        "pass": not any(row["severity"] == "error" for row in rows),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
