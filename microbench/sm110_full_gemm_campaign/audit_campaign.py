#!/usr/bin/env python3
"""Independent fail-closed audit for a completed Thor full-GEMM campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import runpy
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXPECTED_SHAPES = {1024, 2048, 4096}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_kv(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        for match in re.finditer(
                r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s,]+)", line):
            fields[match.group(1)] = match.group(2)
    return fields


def sass_blocks(text: str) -> list[str]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if "Function :" in line:
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(block) for block in blocks]


def add(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def finite_positive(value: object) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    errors: list[str] = []
    required = (
        "run_spec.json", "environment.json", "environment_snapshots.jsonl",
        "campaign_status.json", "progress.jsonl", "summary.json", "COMPLETE",
    )
    for name in required:
        add(errors, (root / name).is_file(), f"missing:{name}")
    if errors:
        print(json.dumps({"pass": False, "errors": errors}, indent=2))
        return 1

    spec = json.loads((root / "run_spec.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    add(errors, spec.get("schema_version") == 2, "invalid spec schema")
    add(errors, spec.get("campaign") == "sm110_full_gemm_closure",
        "invalid campaign name")
    add(errors, spec.get("trials") == 10, "trial contract is not 10")
    add(errors, spec.get("trial_timeout_seconds") == 120,
        "trial timeout contract is not 120 seconds")
    add(errors, spec.get("ncu_timeout_seconds") == 300,
        "NCU timeout contract is not 300 seconds")
    add(errors, spec.get("termination_grace_seconds") == 5,
        "termination grace contract is not 5 seconds")
    add(errors, spec.get("static_only") is False,
        "static-only artifact cannot pass the hardware-result audit")
    add(errors, spec.get("problem_contract") == {
        "layout": "NN", "epilogue": "none", "beta": 0,
        "output_mode": "accumulator", "work": "2*M*N*K",
    }, "problem contract changed")
    expected_generator = (REPO / "microbench/sm110_full_gemm_campaign/"
                          "run_full_gemm_campaign.py")
    generator = REPO / str(spec.get("generator", ""))
    add(errors, generator.resolve() == expected_generator.resolve(),
        "generator path is not the canonical campaign runner")
    add(errors, generator.is_file() and digest(generator) == spec.get("generator_sha256"),
        "generator hash mismatch")
    expected_manifest = (REPO / "microbench/sm110_full_gemm_campaign/"
                         "support_manifest.json")
    manifest = REPO / str(spec.get("support_manifest", ""))
    add(errors, manifest.resolve() == expected_manifest.resolve(),
        "support manifest path is not canonical")
    add(errors, manifest.is_file() and digest(manifest) == spec.get("support_manifest_sha256"),
        "support manifest hash mismatch")
    try:
        manifest_data = json.loads(manifest.read_text())
        expected_precisions = {
            row["precision_id"] for row in manifest_data["precisions"]
            if row.get("status") == "ready_for_closure_campaign"
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        expected_precisions = set()
        errors.append("support manifest is not parseable")
    expected_cases = len(expected_precisions) * len(EXPECTED_SHAPES)
    add(errors, bool(expected_precisions), "support manifest has no ready precisions")
    for relative, expected in spec.get("source_dependencies", {}).items():
        path = REPO / relative
        add(errors, path.is_file() and digest(path) == expected,
            f"source dependency hash mismatch:{relative}")

    cases = spec.get("cases", [])
    try:
        canonical_runner = runpy.run_path(str(expected_generator))
        canonical_cases = canonical_runner["CASES"]
        canonical_dependencies = {
            str(path.relative_to(REPO)): digest(path)
            for path in canonical_runner["source_dependencies"]()
        }
    except (OSError, KeyError, RuntimeError, SyntaxError):
        canonical_cases = None
        canonical_dependencies = None
        errors.append("canonical runner cases cannot be loaded")
    add(errors, cases == canonical_cases, "spec cases differ from canonical runner")
    add(errors, spec.get("source_dependencies") == canonical_dependencies,
        "source dependency set differs from canonical runner")
    add(errors, len(cases) == expected_cases, "spec case cardinality mismatch")
    pairs = {(case.get("precision_id"), case.get("n")) for case in cases}
    expected_pairs = {(precision, n) for precision in expected_precisions
                      for n in EXPECTED_SHAPES}
    add(errors, pairs == expected_pairs, "precision/shape matrix is incomplete")
    add(errors, summary.get("status") == "complete", "summary is not complete")
    add(errors, summary.get("schema_version") == 2,
        "invalid summary schema")
    add(errors, summary.get("case_count") == expected_cases,
        "summary case cardinality mismatch")
    status = json.loads((root / "campaign_status.json").read_text())
    add(errors, status.get("status") == "complete", "campaign status is not complete")
    add(errors, f"summary_sha256={digest(root / 'summary.json')}" in
        (root / "COMPLETE").read_text(), "COMPLETE summary hash mismatch")

    snapshots = [json.loads(line) for line in
                 (root / "environment_snapshots.jsonl").read_text().splitlines()
                 if line]
    add(errors, bool(snapshots), "no environment snapshots")
    identities: set[str] = set()
    for snapshot in snapshots:
        identity = str(snapshot.get("gpu_identity", {}).get("output", "")).strip()
        add(errors, bool(identity) and ("11.0" in identity or "Thor" in identity),
            "Thor/SM110 identity not proven")
        add(errors, "MAXN" in str(snapshot.get("power_mode", {})).upper(),
            "MAXN power mode not proven")
        identities.add(identity)
    add(errors, len(identities) == 1, "GPU identity changed during campaign")

    by_id = {row.get("case_id"): row for row in summary.get("results", [])}
    add(errors, len(by_id) == expected_cases, "summary result IDs are not unique")
    for case in cases:
        case_id = str(case["id"])
        result = by_id.get(case_id)
        if not result:
            errors.append(f"{case_id}: missing result")
            continue
        add(errors, result.get("status") == "ok", f"{case_id}: status is not ok")
        add(errors, result.get("trial_count") == 10, f"{case_id}: trial count is not 10")
        for key in ("precision_id", "backend_id", "n", "split", "work_unit",
                    "internal_repeats"):
            add(errors, result.get(key) == case.get(key),
                f"{case_id}: result/spec mismatch for {key}")
        expected_unit = ("operation" if case["precision_id"] in {"s8_s32", "u8_s32"}
                         else "flop")
        add(errors, case.get("work_unit") == expected_unit,
            f"{case_id}: wrong work unit")
        expected_split = "holdout" if case["n"] == 4096 else "calibration"
        add(errors, case.get("split") == expected_split,
            f"{case_id}: wrong calibration/holdout split")

        sass_path = root / str(result.get("sass_path", ""))
        add(errors, sass_path.is_file() and digest(sass_path) == result.get("sass_sha256"),
            f"{case_id}: SASS hash mismatch")
        if sass_path.is_file():
            blocks = [block for block in sass_blocks(sass_path.read_text())
                      if str(case["function_substring"]) in block.splitlines()[0]]
            valid_blocks = [block for block in blocks
                            if all(token in block for token in case["sass_tokens"])]
            add(errors, bool(valid_blocks),
                f"{case_id}: function-scoped SASS evidence missing")
            reported_headers = set(result.get("sass_evidence", {}).get(
                "matching_function_headers", []))
            actual_headers = {block.splitlines()[0].strip() for block in valid_blocks}
            add(errors, reported_headers == actual_headers,
                f"{case_id}: SASS function header evidence mismatch")
        binary_hash_path = root / str(result.get("binary_hash_path", ""))
        binary_hash_fields = (binary_hash_path.read_text().split()
                              if binary_hash_path.is_file() else [])
        add(errors, bool(binary_hash_fields) and
            binary_hash_fields[0] == result.get("binary_sha256"),
            f"{case_id}: binary hash record mismatch")
        compile_path = root / "build" / f"{case['binary']}.compile_command.json"
        try:
            compile_command = json.loads(compile_path.read_text())
        except (OSError, json.JSONDecodeError):
            compile_command = []
        add(errors, "arch=compute_110a,code=sm_110a" in compile_command,
            f"{case_id}: compile target is not sm_110a")
        if case["binary"] == "extended":
            self_test_path = root / str(result.get("self_test_path", ""))
            add(errors, self_test_path.is_file() and
                digest(self_test_path) == result.get("self_test_sha256") and
                "self_test=pass" in self_test_path.read_text(),
                f"{case_id}: extended host self-test evidence missing")
            self_test_command_path = root / str(
                result.get("self_test_command_path", ""))
            try:
                self_test_command = json.loads(self_test_command_path.read_text())
            except (OSError, json.JSONDecodeError):
                self_test_command = None
            add(errors, self_test_command_path.is_file() and
                digest(self_test_command_path) ==
                result.get("self_test_command_sha256") and
                self_test_command == [str(root / "build" / "extended"),
                                      "--self-test"],
                f"{case_id}: extended host self-test command mismatch")

        trials_path = root / "cases" / case_id / "trials.jsonl"
        trials = ([json.loads(line) for line in trials_path.read_text().splitlines()
                   if line] if trials_path.is_file() else [])
        add(errors, len(trials) == 10, f"{case_id}: missing raw trials")
        custom_rates: list[float] = []
        reference_rates: list[float] = []
        for index, trial in enumerate(trials, 1):
            prefix = f"{case_id}/trial{index}"
            trial_dir = root / "cases" / case_id / f"trial_{index:02d}"
            stdout_path = trial_dir / "stdout.log"
            add(errors, stdout_path.is_file(), f"{prefix}: stdout missing")
            stdout = stdout_path.read_text() if stdout_path.is_file() else ""
            add(errors, trial.get("returncode") == 0,
                f"{prefix}: nonzero return code")
            add(errors, trial.get("timeout_seconds") == 120,
                f"{prefix}: timeout contract changed")
            add(errors, trial.get("timed_out") is False,
                f"{prefix}: timed out trial cannot pass")
            add(errors, trial.get("termination_failed") is False,
                f"{prefix}: termination failure cannot pass")
            fields = parse_kv(stdout)
            add(errors, fields == trial.get("fields"), f"{prefix}: parsed fields changed")
            add(errors, fields.get("reference_contract") ==
                case.get("reference_contract"),
                f"{prefix}: reference contract mismatch")
            add(errors, fields.get("numerical_contract") ==
                case.get("numerical_contract"),
                f"{prefix}: numerical contract mismatch")
            for field, expected in (("reference_sample_count", 64),
                                    ("reference_mismatch_count", 0),
                                    ("mismatch_count", 0)):
                try:
                    actual = int(fields.get(field, "-1"))
                except ValueError:
                    actual = -1
                add(errors, actual == expected, f"{prefix}: {field}={actual}")
            csv_name = ({"fp16": "sgemm_sm110_benchmark.csv",
                         "quant": "quant_sm110_benchmark.csv",
                         "extended": "extended_sm110_benchmark.csv"}
                        [str(case["binary"])])
            csv_path = trial_dir / csv_name
            add(errors, csv_path.is_file(), f"{prefix}: CSV missing")
            raw_csv = csv_path.read_text() if csv_path.is_file() else ""
            add(errors, raw_csv == trial.get("raw_csv"), f"{prefix}: raw CSV changed")
            try:
                rows = list(csv.DictReader(io.StringIO(raw_csv)))
                candidates = [row for row in rows
                              if row.get("BackendId") == case["backend_id"]]
                if len(candidates) != 1 or candidates[0].get("Matched") != "1":
                    raise ValueError("candidate row absent or unmatched")
                if candidates[0].get("N") != str(case["n"]):
                    raise ValueError("CSV problem size differs")
                if candidates[0].get("Precision") != case["csv_precision"]:
                    raise ValueError("CSV precision differs")
                if case["binary"] == "fp16":
                    if any(fields.get(axis) != str(case["n"])
                           for axis in ("M", "N", "K")):
                        raise ValueError("stdout GEMM dimensions differ")
                elif fields.get("N") != str(case["n"]):
                    raise ValueError("stdout GEMM dimension differs")
                custom_ms = float(candidates[0]["TimeMs"])
                work = 2 * int(case["n"]) ** 3
                custom_rate = work * 1000.0 / custom_ms
                if case["binary"] == "fp16":
                    references = [row for row in rows
                                  if row.get("BackendId") == "cublas_tc"]
                    if len(references) != 1 or references[0].get("Matched") != "1":
                        raise ValueError("reference row absent")
                    reference_rate = work * 1000.0 / float(references[0]["TimeMs"])
                else:
                    if fields.get("backend_id") != case["backend_id"]:
                        raise ValueError("machine-readable backend differs")
                    if fields.get("work_unit") != case["work_unit"]:
                        raise ValueError("machine-readable unit differs")
                    if fields.get("matched") not in {"1", "true"}:
                        raise ValueError("machine-readable match failed")
                    add(errors, math.isclose(float(fields["time_ms"]), custom_ms,
                                             rel_tol=2e-7),
                        f"{prefix}: custom time fields disagree")
                    reference_rate = (work * 1000.0 /
                                      float(fields["reference_time_ms"]))
                    add(errors, math.isclose(float(fields["rate_per_second"]),
                                             custom_rate, rel_tol=2e-5),
                        f"{prefix}: reported custom rate does not close")
                    add(errors, math.isclose(
                        float(fields["reference_rate_per_second"]),
                        reference_rate, rel_tol=2e-5),
                        f"{prefix}: reported reference rate does not close")
                if not finite_positive(custom_rate) or not finite_positive(reference_rate):
                    raise ValueError("nonpositive rate")
                add(errors, math.isclose(custom_rate,
                                         float(trial["custom_rate_per_second"]),
                                         rel_tol=1e-12),
                    f"{prefix}: custom rate arithmetic mismatch")
                add(errors, math.isclose(reference_rate,
                                         float(trial["reference_rate_per_second"]),
                                         rel_tol=1e-12),
                    f"{prefix}: reference rate arithmetic mismatch")
                add(errors, math.isclose(custom_rate / reference_rate,
                                         float(trial["ratio_to_reference"]),
                                         rel_tol=1e-12),
                    f"{prefix}: ratio arithmetic mismatch")
                custom_rates.append(custom_rate)
                reference_rates.append(reference_rate)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                errors.append(f"{prefix}: invalid trial evidence:{error}")
        if len(custom_rates) == 10 and len(reference_rates) == 10:
            add(errors, math.isclose(statistics.median(custom_rates),
                                     float(result["custom_rate_per_second_median"]),
                                     rel_tol=1e-12), f"{case_id}: custom median mismatch")
            add(errors, math.isclose(statistics.median(reference_rates),
                                     float(result["reference_rate_per_second_median"]),
                                     rel_tol=1e-12), f"{case_id}: reference median mismatch")
            paired_ratio = statistics.median(custom_rates) / statistics.median(reference_rates)
            add(errors, math.isclose(paired_ratio,
                                     float(result["ratio_of_paired_medians"]),
                                     rel_tol=1e-12), f"{case_id}: aggregate ratio mismatch")

    ncu_requested = bool(spec.get("ncu_requested"))
    add(errors, summary.get("ncu_requested") == ncu_requested,
        "NCU request flag mismatch")
    ncu_results = summary.get("ncu_results", [])
    if ncu_requested:
        add(errors, len(ncu_results) == len(expected_precisions),
            f"expected {len(expected_precisions)} NCU reports")
        add(errors, {row.get("case_id") for row in ncu_results} ==
            {case["id"] for case in cases if case["n"] == 4096},
            "NCU precision coverage mismatch")
        for row in ncu_results:
            path = root / str(row.get("report_path", ""))
            add(errors, path.is_file() and path.stat().st_size > 0 and
                digest(path) == row.get("report_sha256"),
                f"{row.get('case_id')}: NCU report hash mismatch")
            add(errors, row.get("returncode") == 0,
                f"{row.get('case_id')}: NCU return code is nonzero")
            add(errors, row.get("timeout_seconds") == 300,
                f"{row.get('case_id')}: NCU timeout contract changed")
            add(errors, row.get("timed_out") is False,
                f"{row.get('case_id')}: timed out NCU cannot pass")
            add(errors, row.get("termination_failed") is False,
                f"{row.get('case_id')}: NCU termination failure cannot pass")
    else:
        add(errors, not ncu_results, "unexpected NCU results")

    output = {"run_dir": str(root), "pass": not errors, "errors": errors}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
