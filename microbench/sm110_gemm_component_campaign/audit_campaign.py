#!/usr/bin/env python3
"""Independent fail-closed audit for the Thor component campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXPECTED_CASES = 9


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    errors: list[str] = []
    for name in ("run_spec.json", "environment.json", "environment_snapshots.jsonl",
                 "campaign_status.json", "progress.jsonl", "summary.json", "COMPLETE"):
        if not (root / name).is_file(): errors.append(f"missing:{name}")
    if errors:
        print(json.dumps({"pass": False, "errors": errors}, indent=2)); return 1
    spec = json.loads((root / "run_spec.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if spec.get("trials") != 10 or spec.get("expected_sm_count") != 20:
        errors.append("invalid campaign contract")
    generator = REPO / str(spec.get("generator", ""))
    if not generator.is_file() or digest(generator) != spec.get("generator_sha256"):
        errors.append("generator hash mismatch")
    for relative, expected_hash in spec.get("source_dependencies", {}).items():
        dependency = REPO / relative
        if not dependency.is_file() or digest(dependency) != expected_hash:
            errors.append(f"source dependency hash mismatch:{relative}")
    if len(spec.get("cases", [])) != EXPECTED_CASES or summary.get("case_count") != EXPECTED_CASES:
        errors.append("case cardinality mismatch")
    if summary.get("status") != "complete": errors.append("summary is not complete")
    status = json.loads((root / "campaign_status.json").read_text())
    if status.get("status") != "complete": errors.append("campaign status is not complete")
    if f"summary_sha256={digest(root / 'summary.json')}" not in (root / "COMPLETE").read_text():
        errors.append("COMPLETE summary hash mismatch")
    snapshots = [json.loads(line) for line in (root / "environment_snapshots.jsonl").read_text().splitlines() if line]
    identities = set()
    for snapshot in snapshots:
        if "MAXN" not in str(snapshot.get("power_mode", {})).upper(): errors.append("MAXN not proven")
        identity = str(snapshot.get("gpu_identity", {}).get("output", "")).strip()
        if not identity or ("11.0" not in identity and "Thor" not in identity): errors.append("Thor identity not proven")
        identities.add(identity)
    if len(identities) != 1: errors.append("GPU identity changed")
    by_id = {row.get("case_id"): row for row in summary.get("results", [])}
    for case in spec.get("cases", []):
        cid = case["id"]; result = by_id.get(cid)
        if not result or result.get("status") != "ok" or result.get("trial_count") != 10:
            errors.append(f"{cid}: incomplete result"); continue
        source = REPO / result["source_path"]
        sass = root / "build" / f"{case['binary']}.sass.txt"
        binary_hash = root / str(result.get("binary_hash_path", ""))
        compile_command_path = root / "build" / f"{case['binary']}.compile_command.json"
        if not source.is_file() or digest(source) != result.get("source_sha256"):
            errors.append(f"{cid}: source hash mismatch")
        if not sass.is_file() or digest(sass) != result.get("sass_sha256"):
            errors.append(f"{cid}: SASS hash mismatch")
        elif any(token not in sass.read_text() for token in case["sass"]):
            errors.append(f"{cid}: SASS token mismatch")
        binary_hash_fields = binary_hash.read_text().split() if binary_hash.is_file() else []
        if not binary_hash_fields or binary_hash_fields[0] != result.get("binary_sha256"):
            errors.append(f"{cid}: binary hash record mismatch")
        try:
            compile_command = json.loads(compile_command_path.read_text())
        except (OSError, json.JSONDecodeError):
            compile_command = []
        if "arch=compute_110a,code=sm_110a" not in compile_command:
            errors.append(f"{cid}: compile target mismatch")
        trials_path = root / "cases" / cid / "trials.jsonl"
        trials = [json.loads(line) for line in trials_path.read_text().splitlines()] if trials_path.is_file() else []
        rates = [float(row.get("audited_rate_per_second", math.nan)) for row in trials]
        if len(rates) != 10 or not all(math.isfinite(rate) and rate > 0 for rate in rates):
            errors.append(f"{cid}: invalid trials")
        elif not math.isclose(statistics.median(rates), float(result["rate_per_second_median"]), rel_tol=1e-12):
            errors.append(f"{cid}: aggregate mismatch")
        for row in trials:
            fields = row.get("fields", {})
            if case["resource"].startswith(("tma.", "tmem.")):
                if int(fields.get("sm_count", 0)) != 20 or int(fields.get("unique_smid_count", 0)) != 20:
                    errors.append(f"{cid}: incomplete SM coverage")
                try:
                    if case["resource"].startswith("tma."):
                        recalculated = (int(fields["requested_bytes"]) * 1e9 /
                                        int(fields["globaltimer_elapsed_ns"]))
                    else:
                        recalculated = (int(fields["issued_bytes"]) * 1e9 /
                                        int(fields["globaltimer_elapsed_ns"]))
                    if not math.isclose(recalculated,
                                        float(row["audited_rate_per_second"]),
                                        rel_tol=2e-12):
                        errors.append(f"{cid}: trial arithmetic mismatch")
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    errors.append(f"{cid}: invalid trial arithmetic fields")
            else:
                if int(fields.get("sm_count", 0)) != 20 or int(fields.get("unique_smid_count", 0)) != 20:
                    errors.append(f"{cid}: incomplete SM coverage")
                if int(fields.get("value_mismatches", -1)) or int(fields.get("scale_mismatches", -1)):
                    errors.append(f"{cid}: numerical mismatch")
                try:
                    recalculated = float(fields["gelements_per_second"]) * 1e9
                    if not math.isclose(recalculated,
                                        float(row["audited_rate_per_second"]),
                                        rel_tol=2e-12):
                        errors.append(f"{cid}: trial arithmetic mismatch")
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{cid}: invalid trial arithmetic fields")
    output = {"run_dir": str(root), "pass": not errors, "errors": errors}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
