#!/usr/bin/env python3
"""Independent fail-closed audit for SM110 simultaneous read/write evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from microbench.sm110_memory_duplex_campaign.run_memory_duplex_campaign import (
    EXPECTED_SMS, NCU_METRICS, TARGET_SQUARE_SHAPES, TRIALS, cases,
    duplex_sass_block, sha256_text, validate_trial,
)

NCU_BASE_UNITS = {
    "gpu__time_duration.sum": "ns",
    "lts__t_bytes.sum": "byte",
    "lts__t_sectors_op_read.sum": "sector",
    "lts__t_sectors_op_write.sum": "sector",
    "lts__t_sectors_op_read_lookup_hit.sum": "sector",
    "lts__t_sectors_op_read_lookup_miss.sum": "sector",
}
NCU_NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?$"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, errors: list[str], label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: invalid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}: JSON root is not an object")
        return {}
    return value


def ncu_number(value: object) -> float:
    text = str(value).strip()
    if not NCU_NUMBER_RE.fullmatch(text):
        raise ValueError(f"invalid NCU number: {value!r}")
    result = float(text.replace(",", ""))
    if not math.isfinite(result):
        raise ValueError(f"non-finite NCU number: {value!r}")
    return result


def find_ncu_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    for index, header in enumerate(rows):
        if ("ID" not in header or "Kernel Name" not in header
                or not set(NCU_BASE_UNITS).issubset(header)):
            continue
        if index + 1 >= len(rows) or len(rows[index + 1]) != len(header):
            raise ValueError("NCU unit row is missing")
        unit_row = rows[index + 1]
        for name, expected_unit in NCU_BASE_UNITS.items():
            actual_unit = unit_row[header.index(name)]
            if actual_unit != expected_unit:
                raise ValueError(
                    f"NCU metric {name} has unit {actual_unit!r}, "
                    f"expected {expected_unit!r}")
        id_index = header.index("ID")
        kernel_index = header.index("Kernel Name")
        time_index = header.index("gpu__time_duration.sum")
        candidates: list[tuple[float, list[str]]] = []
        for row in rows[index + 1:]:
            if (len(row) != len(header) or not row[id_index].isdigit()
                    or "duplex_kernel" not in row[kernel_index]):
                continue
            try:
                duration = ncu_number(row[time_index])
            except ValueError:
                continue
            if duration >= 0:
                candidates.append((duration, row))
        if candidates:
            return dict(zip(header, candidates[-1][1]))
    raise ValueError("no duplex kernel metric row")


def audit_environment(root: Path, spec: dict[str, object], errors: list[str]) -> None:
    path = root / "environment_snapshots.jsonl"
    try:
        snapshots = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"environment snapshots invalid: {exc}")
        return
    if not snapshots:
        errors.append("environment snapshots are empty")
        return
    identities = set()
    for index, snapshot in enumerate(snapshots):
        identity = str(snapshot.get("gpu_identity", {}).get("output", "")).strip()
        identities.add(identity)
        if not identity or ("11.0" not in identity and "Thor" not in identity):
            errors.append(f"environment {index}: Thor/SM110 identity not proven")
        if "MAXN" not in str(snapshot.get("power_mode", {})).upper():
            errors.append(f"environment {index}: MAXN not proven")
        if str(snapshot.get("git_head", {}).get("output", "")).strip() != spec.get(
                "expected_commit"):
            errors.append(f"environment {index}: commit mismatch")
        if str(snapshot.get("git_status", {}).get("output", "")).strip():
            errors.append(f"environment {index}: tracked worktree was dirty")
    if len(identities) != 1:
        errors.append("GPU identity changed during campaign")


def audit_ncu(root: Path, case: dict[str, object], result: dict[str, object],
              errors: list[str]) -> None:
    cid = str(case["id"])
    ncu = result.get("ncu")
    if not isinstance(ncu, dict):
        errors.append(f"{cid}: NCU summary missing")
        return
    summary_path = root / "cases" / cid / "ncu" / "summary.json"
    independent = load(summary_path, errors, f"{cid}: NCU summary")
    if independent != ncu:
        errors.append(f"{cid}: result and NCU summary differ")
    if ncu.get("returncode") != 0 or set(ncu.get("metrics", [])) != set(NCU_METRICS):
        errors.append(f"{cid}: required NCU metric contract not met")
    if (ncu.get("evidence_contract") != case.get("evidence_contract")
            or ncu.get("external_write_bytes_proven") is not False):
        errors.append(f"{cid}: NCU evidence qualification mismatch")
    case_dir = root / "cases" / cid
    for path_key, hash_key in (("report_path", "report_sha256"),
                               ("raw_path", "raw_sha256"),
                               ("stderr_path", "stderr_sha256")):
        relative = str(ncu.get(path_key, ""))
        path = case_dir / relative
        if not relative or not path.is_file() or digest(path) != ncu.get(hash_key):
            errors.append(f"{cid}: {path_key} hash mismatch")
    raw_path = case_dir / str(ncu.get("raw_path", ""))
    try:
        raw_row = find_ncu_row(raw_path)
        raw_values = {name: ncu_number(raw_row[name]) for name in NCU_METRICS}
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"{cid}: cannot independently parse raw CSV: {exc}")
        raw_values = {}
    try:
        values = {name: float(ncu["values"][name]) for name in NCU_METRICS}
        requested_read = int(ncu["requested_read_bytes"])
        requested_write = int(ncu["requested_write_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{cid}: malformed NCU values: {exc}")
        return
    for name, value in values.items():
        if name in raw_values and not math.isclose(
                value, raw_values[name], rel_tol=1e-12, abs_tol=1e-9):
            errors.append(f"{cid}: summary/raw CSV mismatch for {name}")
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        errors.append(f"{cid}: invalid NCU numeric value")
    if values["lts__t_sectors_op_read.sum"] * 32 < requested_read * 0.90:
        errors.append(f"{cid}: L2 read sectors do not prove requested reads")
    if values["lts__t_sectors_op_write.sum"] * 32 < requested_write * 0.90:
        errors.append(f"{cid}: L2 write sectors do not prove requested writes")
    if case["residency"] == "hot_l2":
        if values["lts__t_sectors_op_read_lookup_hit.sum"] <= values[
                "lts__t_sectors_op_read_lookup_miss.sum"]:
            errors.append(f"{cid}: hot-L2 residency is not counter-proven")
    else:
        miss_proxy_bytes = (
            values["lts__t_sectors_op_read_lookup_miss.sum"] * 32.0)
        if miss_proxy_bytes < requested_read * 0.60:
            errors.append(f"{cid}: cold-DRAM reads are not miss-proxy-proven")
        try:
            stored_proxy = float(ncu["cold_read_miss_proxy_bytes"])
            stored_ratio = float(ncu["cold_read_miss_proxy_to_requested"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{cid}: cold-read proxy summary is malformed")
        else:
            if not math.isclose(stored_proxy, miss_proxy_bytes, rel_tol=1e-12):
                errors.append(f"{cid}: cold-read proxy byte arithmetic mismatch")
            if not math.isclose(
                    stored_ratio, miss_proxy_bytes / requested_read,
                    rel_tol=1e-12):
                errors.append(f"{cid}: cold-read proxy ratio arithmetic mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    errors: list[str] = []
    for name in ("run_spec.json", "environment.json", "environment_snapshots.jsonl",
                 "campaign_status.json", "progress.jsonl", "summary.json", "COMPLETE"):
        if not (root / name).is_file():
            errors.append(f"missing:{name}")
    if errors:
        print(json.dumps({"pass": False, "errors": errors}, indent=2))
        return 1
    spec = load(root / "run_spec.json", errors, "run spec")
    summary = load(root / "summary.json", errors, "summary")
    canonical_cases = cases()
    if spec.get("campaign") != "sm110_memory_duplex_closure":
        errors.append("wrong campaign ID")
    if not re.fullmatch(r"[0-9a-f]{40}", str(spec.get("expected_commit", ""))):
        errors.append("invalid expected commit")
    if spec.get("expected_sm_count") != EXPECTED_SMS or spec.get("trials") != TRIALS:
        errors.append("platform/trial contract mismatch")
    if spec.get("ncu_required") is not True:
        errors.append("NCU was not mandatory")
    expected_cold_contract = {
        "read": "32 * l2_read_lookup_miss_sectors >= 0.60 * requested_read_bytes",
        "write": "32 * l2_write_sectors >= 0.90 * requested_write_bytes",
        "external_write_bytes_proven": False,
        "qualification": "cold_dram_read_plus_write_path_proxy",
    }
    if spec.get("cold_duplex_evidence") != expected_cold_contract:
        errors.append("cold-duplex proxy contract mismatch")
    if spec.get("target_square_shapes") != list(TARGET_SQUARE_SHAPES):
        errors.append("target square-shape contract mismatch")
    for timeout_key in ("trial_timeout_seconds", "ncu_timeout_seconds"):
        value = spec.get(timeout_key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"invalid bounded timeout:{timeout_key}")
    if spec.get("static_only") is not False:
        errors.append("runtime campaign was recorded as static-only")
    if spec.get("cases") != canonical_cases:
        errors.append("case manifest differs from canonical ratio matrix")
    generator = REPO / str(spec.get("generator", ""))
    if not generator.is_file() or digest(generator) != spec.get("generator_sha256"):
        errors.append("generator hash mismatch")
    for relative, expected in spec.get("source_dependencies", {}).items():
        path = REPO / str(relative)
        if not path.is_file() or digest(path) != expected:
            errors.append(f"source dependency hash mismatch:{relative}")
    audit_environment(root, spec, errors)
    if summary.get("status") != "complete" or summary.get("case_count") != len(canonical_cases):
        errors.append("summary is incomplete")
    marker = (root / "COMPLETE").read_text()
    if f"summary_sha256={digest(root / 'summary.json')}" not in marker:
        errors.append("COMPLETE marker hash mismatch")
    by_id = {row.get("case_id"): row for row in summary.get("results", [])
             if isinstance(row, dict)}
    if len(by_id) != len(canonical_cases):
        errors.append("result case cardinality mismatch")
    for case in canonical_cases:
        cid = str(case["id"])
        result = by_id.get(cid, {})
        if result.get("status") != "ok" or result.get("trial_count") != TRIALS:
            errors.append(f"{cid}: result is incomplete")
            continue
        for key in (
            "resource", "residency", "evidence_contract",
            "external_write_bytes_proven",
        ):
            if result.get(key) != case.get(key):
                errors.append(f"{cid}: result/manifest mismatch for {key}")
        source = REPO / str(result.get("source_path", ""))
        sass = root / "build" / "sass.txt"
        function_sass_path = root / str(result.get("function_sass_path", ""))
        if not source.is_file() or digest(source) != result.get("source_sha256"):
            errors.append(f"{cid}: source hash mismatch")
        if not sass.is_file() or digest(sass) != result.get("sass_sha256"):
            errors.append(f"{cid}: SASS hash mismatch")
        else:
            try:
                independent_function_sass = duplex_sass_block(sass.read_text())
            except RuntimeError as exc:
                errors.append(f"{cid}: {exc}")
                independent_function_sass = ""
            if any(token not in independent_function_sass
                   for token in case["sass_tokens"]):
                errors.append(f"{cid}: function-scoped SASS token missing")
            if (not function_sass_path.is_file()
                    or digest(function_sass_path)
                    != result.get("function_sass_sha256")
                    or function_sass_path.read_text() != independent_function_sass):
                errors.append(f"{cid}: function-scoped SASS artifact mismatch")
        expected_fingerprint = sha256_text(
            json.dumps(case, sort_keys=True)
            + str(spec.get("generator_sha256", ""))
            + str(result.get("source_sha256", ""))
            + str(result.get("binary_sha256", ""))
            + str(result.get("sass_sha256", ""))
        )
        if result.get("fingerprint") != expected_fingerprint:
            errors.append(f"{cid}: result fingerprint mismatch")
        trials_path = root / "cases" / cid / "trials.jsonl"
        try:
            trials = [json.loads(line) for line in trials_path.read_text().splitlines()
                      if line]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{cid}: invalid trials: {exc}")
            continue
        rates = []
        if len(trials) != TRIALS:
            errors.append(f"{cid}: trial cardinality mismatch")
        for trial in trials:
            try:
                rate = validate_trial(case, {str(k): str(v)
                                      for k, v in trial["fields"].items()})
                recorded = float(trial["audited_rate_per_second"])
                if not math.isclose(rate, recorded, rel_tol=2e-12):
                    raise ValueError("recorded rate mismatch")
                rates.append(rate)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                errors.append(f"{cid}: invalid trial arithmetic: {exc}")
        if rates and not math.isclose(statistics.median(rates),
                                      float(result["rate_per_second_median"]),
                                      rel_tol=2e-12):
            errors.append(f"{cid}: median mismatch")
        audit_ncu(root, case, result, errors)
    output = {"run_dir": str(root), "pass": not errors, "errors": errors}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
