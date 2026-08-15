#!/usr/bin/env python3
"""Independent fail-closed audit for the Thor TMA payload campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXPECTED_PAYLOADS = {4096, 8192, 16384, 32768, 65536}
EXPECTED_CASE_COUNT = 10
EXPECTED_SMS = 20
EXPECTED_TRIALS = 10
EXPECTED_SLOTS = 2
EXPECTED_THREADS = 128
EXPECTED_RESIDENT_CTAS = 1
REQUIRED_NCU_METRICS = {
    "gpu__time_duration.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "lts__t_bytes.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_kv(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line):
            fields[match.group(1)] = match.group(2)
    return fields


def manifest_arg(case: dict[str, object], option: str) -> str | None:
    args = case.get("args")
    if not isinstance(args, list):
        return None
    try:
        return str(args[args.index(option) + 1])
    except (ValueError, IndexError):
        return None


def find_ncu_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    for index, header in enumerate(rows):
        if "ID" not in header or "Kernel Name" not in header:
            continue
        kernel_index = header.index("Kernel Name")
        time_index = header.index("gpu__time_duration.sum")
        candidates: list[tuple[float, list[str]]] = []
        for row in rows[index + 1 :]:
            if len(row) != len(header) or not row or not row[0].isdigit():
                continue
            if "tma_kernel" not in row[kernel_index]:
                continue
            try:
                candidates.append((float(row[time_index]), row))
            except ValueError:
                continue
        if candidates:
            return dict(zip(header, max(candidates, key=lambda item: item[0])[1]))
    raise ValueError("no TMA kernel metric row")


def load_json(path: Path, errors: list[str], label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: invalid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}: JSON root is not an object")
        return {}
    return value


def integer_field(
    fields: dict[str, object], name: str, errors: list[str], context: str
) -> int | None:
    try:
        return int(fields[name])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{context}: invalid integer field {name}")
        return None


def audit_environment(
    root: Path, spec: dict[str, object], errors: list[str]
) -> None:
    snapshots_path = root / "environment_snapshots.jsonl"
    try:
        snapshots = [
            json.loads(line)
            for line in snapshots_path.read_text().splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"environment snapshots invalid: {exc}")
        return
    if not snapshots:
        errors.append("environment snapshots are empty")
        return
    identities: set[str] = set()
    expected_commit = str(spec.get("expected_commit", ""))
    for index, snapshot in enumerate(snapshots):
        prefix = f"environment snapshot {index}"
        identity = str(snapshot.get("gpu_identity", {}).get("output", "")).strip()
        if not identity or ("11.0" not in identity and "Thor" not in identity):
            errors.append(f"{prefix}: Thor/SM110 identity not proven")
        identities.add(identity)
        if "MAXN" not in str(snapshot.get("power_mode", {})).upper():
            errors.append(f"{prefix}: MAXN mode not proven")
        git_head = str(snapshot.get("git_head", {}).get("output", "")).strip()
        if git_head != expected_commit:
            errors.append(f"{prefix}: Git commit mismatch")
        if str(snapshot.get("git_status", {}).get("output", "")).strip():
            errors.append(f"{prefix}: tracked worktree was dirty")
    if len(identities) != 1:
        errors.append("GPU identity changed during campaign")


def audit_ncu(
    root: Path,
    case_id: str,
    case: dict[str, object],
    result: dict[str, object],
    errors: list[str],
) -> None:
    context = f"{case_id}: NCU"
    ncu = result.get("ncu")
    if not isinstance(ncu, dict):
        errors.append(f"{context}: missing summary")
        return
    ncu_summary_path = root / "cases" / case_id / "ncu" / "summary.json"
    try:
        independent_summary = json.loads(ncu_summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{context}: invalid independent summary: {exc}")
    else:
        if independent_summary != ncu:
            errors.append(f"{context}: result/independent summary mismatch")
    if ncu.get("returncode") != 0 or ncu.get("permission_denied") is not False:
        errors.append(f"{context}: profiler did not complete cleanly")
    if not REQUIRED_NCU_METRICS.issubset(set(ncu.get("metrics", []))):
        errors.append(f"{context}: required metrics absent")
    case_dir = root / "cases" / case_id
    for path_key, hash_key in (
        ("report_path", "report_sha256"),
        ("raw_path", "raw_sha256"),
        ("stderr_path", "stderr_sha256"),
    ):
        relative = str(ncu.get(path_key, ""))
        path = case_dir / relative
        if not relative or not path.is_file() or digest(path) != ncu.get(hash_key):
            errors.append(f"{context}: {path_key} hash mismatch")
    raw_path = case_dir / str(ncu.get("raw_path", ""))
    try:
        raw_row = find_ncu_row(raw_path)
        raw_values = {
            "tma_bytes": float(
                raw_row[
                    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum"
                ]
            ),
            "lts_bytes": float(raw_row["lts__t_bytes.sum"]),
            "l2_hit_sectors": float(
                raw_row["lts__t_sectors_op_read_lookup_hit.sum"]
            ),
            "l2_miss_sectors": float(
                raw_row["lts__t_sectors_op_read_lookup_miss.sum"]
            ),
        }
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"{context}: cannot independently parse raw CSV: {exc}")
        raw_values = {}
    try:
        timed_requested = float(ncu["timed_requested_bytes"])
        expected = float(ncu["expected_counter_bytes"])
        tma_ratio = float(ncu["tma_to_expected"])
        lts_ratio = float(ncu["lts_to_expected"])
        hit_sectors = float(ncu["l2_hit_sectors"])
        miss_sectors = float(ncu["l2_miss_sectors"])
        miss_proxy_ratio = float(ncu["l2_miss_proxy_to_expected"])
        tma_bytes = float(ncu["tma_bytes"])
        lts_bytes = float(ncu["lts_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{context}: invalid numeric summary: {exc}")
        return
    for name, value in (
        ("tma_bytes", tma_bytes),
        ("lts_bytes", lts_bytes),
        ("l2_hit_sectors", hit_sectors),
        ("l2_miss_sectors", miss_sectors),
    ):
        if name in raw_values and not math.isclose(
            value, raw_values[name], rel_tol=1e-12, abs_tol=1e-9
        ):
            errors.append(f"{context}: summary/raw CSV mismatch for {name}")
    ncu_iterations = int(ncu.get("iterations", 0))
    timed_from_contract = (
        int(case["expected_blocks"]) * ncu_iterations * int(case["tile_bytes"])
    )
    # NCU CSV contains one row per launch; the selected row is the timed launch,
    # not the sum of timed and warmup launches.
    expected_from_contract = timed_from_contract
    if timed_requested != timed_from_contract or expected != expected_from_contract:
        errors.append(f"{context}: timed/counter byte contract mismatch")
    if not math.isclose(tma_ratio, tma_bytes / expected, rel_tol=1e-12):
        errors.append(f"{context}: TMA ratio arithmetic mismatch")
    if not math.isclose(lts_ratio, lts_bytes / expected, rel_tol=1e-12):
        errors.append(f"{context}: LTS ratio arithmetic mismatch")
    if not math.isclose(
        miss_proxy_ratio, miss_sectors * 32.0 / expected, rel_tol=1e-12
    ):
        errors.append(f"{context}: miss proxy arithmetic mismatch")
    if tma_ratio < 0.98 or lts_ratio < 0.90:
        errors.append(f"{context}: requested TMA/LTS traffic not confirmed")
    if case.get("residency") == "cold_hbm" and miss_proxy_ratio < 0.70:
        errors.append(f"{context}: DRAM-stream residency not confirmed")
    if case.get("residency") == "hot_l2" and not (hit_sectors > miss_sectors):
        errors.append(f"{context}: L2-hit residency not confirmed")


def audit_case(
    root: Path,
    spec: dict[str, object],
    case: dict[str, object],
    result: dict[str, object] | None,
    errors: list[str],
) -> None:
    case_id = str(case.get("id", ""))
    if not result:
        errors.append(f"{case_id}: missing result")
        return
    if result.get("status") != "ok" or result.get("trial_count") != EXPECTED_TRIALS:
        errors.append(f"{case_id}: incomplete result")
    for key in (
        "resource",
        "residency",
        "tile_bytes",
        "destination_slots",
        "threads_per_cta",
        "resident_ctas_per_sm",
    ):
        if result.get(key) != case.get(key):
            errors.append(f"{case_id}: result/manifest mismatch for {key}")
    source = REPO / str(result.get("source_path", ""))
    if not source.is_file() or digest(source) != result.get("source_sha256"):
        errors.append(f"{case_id}: source hash mismatch")
    sass_path = root / str(result.get("sass_path", ""))
    if not sass_path.is_file() or digest(sass_path) != result.get("sass_sha256"):
        errors.append(f"{case_id}: SASS hash mismatch")
    elif any(token not in sass_path.read_text() for token in case.get("sass_tokens", [])):
        errors.append(f"{case_id}: SASS token mismatch")
    binary_hash_path = root / str(result.get("binary_hash_path", ""))
    fields = binary_hash_path.read_text().split() if binary_hash_path.is_file() else []
    if not fields or fields[0] != result.get("binary_sha256"):
        errors.append(f"{case_id}: binary hash record mismatch")
    expected_fingerprint = hashlib.sha256(
        (
            json.dumps(case, sort_keys=True)
            + str(spec.get("generator_sha256", ""))
            + str(result.get("source_sha256", ""))
            + str(result.get("binary_sha256", ""))
            + str(result.get("sass_sha256", ""))
        ).encode()
    ).hexdigest()
    if result.get("fingerprint") != expected_fingerprint:
        errors.append(f"{case_id}: fingerprint mismatch")

    trials_path = root / "cases" / case_id / "trials.jsonl"
    try:
        trials = [
            json.loads(line)
            for line in trials_path.read_text().splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{case_id}: invalid trials: {exc}")
        trials = []
    rates: list[float] = []
    for index, trial in enumerate(trials):
        context = f"{case_id}: trial {index + 1}"
        trial_fields = trial.get("fields")
        if not isinstance(trial_fields, dict):
            errors.append(f"{context}: missing fields")
            continue
        parsed_stdout = parse_kv(str(trial.get("raw_stdout", "")))
        if parsed_stdout != trial_fields:
            errors.append(f"{context}: raw stdout/fields mismatch")
        sm_count = integer_field(trial_fields, "sm_count", errors, context)
        unique_sms = integer_field(trial_fields, "unique_smid_count", errors, context)
        requested = integer_field(trial_fields, "requested_bytes", errors, context)
        elapsed_ns = integer_field(
            trial_fields, "globaltimer_elapsed_ns", errors, context
        )
        blocks = integer_field(trial_fields, "blocks", errors, context)
        if (sm_count != EXPECTED_SMS
                or unique_sms != int(case["expected_unique_smid_count"])
                or blocks != int(case["expected_blocks"])):
            errors.append(f"{context}: SM/block scope mismatch")
        expected_bytes = (
            int(case["expected_blocks"])
            * int(case.get("iterations", 0)) * int(case["tile_bytes"])
        )
        if requested != expected_bytes:
            errors.append(f"{context}: issued-byte mismatch")
        if not elapsed_ns or elapsed_ns <= 0 or requested is None:
            continue
        recalculated = requested * 1e9 / elapsed_ns
        try:
            stored = float(trial["audited_rate_per_second"])
            reported = float(trial_fields["globaltimer_gbytes_per_second"]) * 1e9
        except (KeyError, TypeError, ValueError):
            errors.append(f"{context}: invalid rate fields")
            continue
        if not math.isclose(recalculated, stored, rel_tol=2e-12):
            errors.append(f"{context}: audited rate mismatch")
        if not math.isclose(recalculated, reported, rel_tol=2e-6, abs_tol=1.0):
            errors.append(f"{context}: reported rate mismatch")
        rates.append(stored)
    if len(rates) != EXPECTED_TRIALS:
        errors.append(f"{case_id}: expected {EXPECTED_TRIALS} valid trials")
    else:
        for key, calculated in (
            ("rate_per_second_median", statistics.median(rates)),
            ("rate_per_second_min", min(rates)),
            ("rate_per_second_max", max(rates)),
        ):
            try:
                stored = float(result[key])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{case_id}: invalid aggregate {key}")
            else:
                if not math.isclose(stored, calculated, rel_tol=1e-12):
                    errors.append(f"{case_id}: aggregate mismatch {key}")
    if spec.get("collect_ncu") is not True:
        errors.append(f"{case_id}: closure requires NCU collection")
    else:
        audit_ncu(root, case_id, case, result, errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    errors: list[str] = []
    required = (
        "run_spec.json",
        "environment.json",
        "environment_snapshots.jsonl",
        "campaign_status.json",
        "progress.jsonl",
        "summary.json",
        "COMPLETE",
        "build/compile_command.json",
        "build/compile.log",
        "build/sass.txt",
        "build/binary.sha256",
    )
    for name in required:
        if not (root / name).is_file():
            errors.append(f"missing:{name}")
    if errors:
        print(json.dumps({"pass": False, "errors": errors}, indent=2))
        return 1
    spec = load_json(root / "run_spec.json", errors, "run_spec")
    summary = load_json(root / "summary.json", errors, "summary")
    status = load_json(root / "campaign_status.json", errors, "campaign_status")
    if spec.get("static_only") is not False:
        errors.append("static-only bundle cannot qualify")
    if (
        spec.get("expected_sm_count") != EXPECTED_SMS
        or spec.get("trials") != EXPECTED_TRIALS
        or spec.get("collect_ncu") is not True
    ):
        errors.append("invalid campaign closure contract")
    expected_commit = str(spec.get("expected_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        errors.append("invalid expected commit")
    generator = REPO / str(spec.get("generator", ""))
    if not generator.is_file() or digest(generator) != spec.get("generator_sha256"):
        errors.append("generator hash mismatch")
    for relative, expected_hash in spec.get("source_dependencies", {}).items():
        path = REPO / str(relative)
        if not path.is_file() or digest(path) != expected_hash:
            errors.append(f"source dependency hash mismatch:{relative}")
    # The compile command is a JSON array, so load it directly here.
    try:
        command = json.loads((root / "build/compile_command.json").read_text())
    except (OSError, json.JSONDecodeError):
        command = []
    if "arch=compute_110a,code=sm_110a" not in command:
        errors.append("compile target mismatch")
    manifest = spec.get("cases", [])
    if not isinstance(manifest, list) or len(manifest) != EXPECTED_CASE_COUNT:
        errors.append("case cardinality mismatch")
        manifest = []
    keys = {
        (case.get("residency"), case.get("tile_bytes"))
        for case in manifest
        if isinstance(case, dict)
    }
    expected_keys = {
        (residency, payload)
        for residency in ("hot_l2", "cold_hbm")
        for payload in EXPECTED_PAYLOADS
    }
    if keys != expected_keys:
        errors.append("payload/residency matrix mismatch")
    for case in manifest:
        if not isinstance(case, dict):
            errors.append("non-object case manifest entry")
            continue
        if (
            case.get("destination_slots") != EXPECTED_SLOTS
            or case.get("threads_per_cta") != EXPECTED_THREADS
            or case.get("resident_ctas_per_sm") != EXPECTED_RESIDENT_CTAS
        ):
            errors.append(f"{case.get('id')}: applicability contract mismatch")
        expected_mode = "l2-hit" if case.get("residency") == "hot_l2" else "dram-stream"
        expected_backing = 16 << 20 if case.get("residency") == "hot_l2" else 256 << 20
        declared_args = {
            "--mode": expected_mode,
            "--bytes": str(expected_backing),
            "--tile-bytes": str(case.get("tile_bytes")),
            "--slots": str(EXPECTED_SLOTS),
            "--iters": str(case.get("iterations")),
            "--warmup-iters": str(case.get("warmup_iterations")),
            "--blocks-per-sm": str(EXPECTED_RESIDENT_CTAS),
            "--threads": str(EXPECTED_THREADS),
        }
        for option, expected_value in declared_args.items():
            if manifest_arg(case, option) != expected_value:
                errors.append(f"{case.get('id')}: argument mismatch for {option}")
        expected_blocks = int(case.get("expected_blocks", 0))
        if case.get("residency") == "hot_l2":
            if expected_blocks != 1 or manifest_arg(case, "--blocks") != "1":
                errors.append(f"{case.get('id')}: per-SM L2-hit isolation missing")
            warmup_bytes = int(case.get("warmup_iterations", 0)) * int(
                case.get("tile_bytes", 0))
            if warmup_bytes < expected_backing:
                errors.append(f"{case.get('id')}: L2 warmup does not cover backing")
        elif expected_blocks != EXPECTED_SMS or manifest_arg(case, "--blocks") is not None:
            errors.append(f"{case.get('id')}: full-GPU DRAM scope mismatch")
    if summary.get("status") != "complete" or summary.get("case_count") != EXPECTED_CASE_COUNT:
        errors.append("summary is not complete")
    if status.get("status") != "complete":
        errors.append("campaign status is not complete")
    if f"summary_sha256={digest(root / 'summary.json')}" not in (root / "COMPLETE").read_text():
        errors.append("COMPLETE summary hash mismatch")
    audit_environment(root, spec, errors)
    by_id = {
        row.get("case_id"): row
        for row in summary.get("results", [])
        if isinstance(row, dict)
    }
    if len(by_id) != EXPECTED_CASE_COUNT:
        errors.append("summary result cardinality mismatch")
    for case in manifest:
        if isinstance(case, dict):
            audit_case(root, spec, case, by_id.get(case.get("id")), errors)
    output = {"run_dir": str(root), "pass": not errors, "errors": errors}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
