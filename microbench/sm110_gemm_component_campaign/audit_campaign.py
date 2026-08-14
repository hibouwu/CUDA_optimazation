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
EXPECTED_CASES = 18
MEMORY_RESOURCES = {"hbm.read", "hbm.write", "l2.read", "l2.write"}
EXPECTED_CASE_RESOURCES = {
    "tma_l2_hit_32k": "tma.l2_hit_ingress",
    "tma_dram_stream_32k": "tma.dram_stream_ingress",
    "tma_l2_hit_32k_inflight4": "tma.l2_hit_ingress",
    "tma_dram_stream_32k_inflight4": "tma.dram_stream_ingress",
    "tma_l2_hit_16k_inflight8": "tma.l2_hit_ingress",
    "tma_dram_stream_16k_inflight8": "tma.dram_stream_ingress",
    "tmem_scale_ingress_32x128b_warpx4": "tmem.scale_ingress",
    "hbm_read_aggregate": "hbm.read",
    "hbm_write_aggregate": "hbm.write",
    "l2_read_aggregate": "l2.read",
    "l2_write_aggregate": "l2.write",
    "tmem_ld_32x32b_x8_warps1": "tmem.accumulator_readback",
    "tmem_ld_32x32b_x8_warps4": "tmem.accumulator_readback",
    "tmem_ld_32x32b_x16_warps1": "tmem.accumulator_readback",
    "tmem_ld_32x32b_x16_warps4": "tmem.accumulator_readback",
    "nvfp4_requant_4096x1024_normal": "epilogue.nvfp4_requant",
    "nvfp4_requant_4096x1024_outlier": "epilogue.nvfp4_requant",
    "nvfp4_requant_4096x1024_constant": "epilogue.nvfp4_requant",
}


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
    if (spec.get("trials") != 10 or spec.get("expected_sm_count") != 20
            or not isinstance(spec.get("trial_timeout_seconds"), int)
            or spec.get("trial_timeout_seconds", 0) <= 0):
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
    actual_case_resources = {
        str(case.get("id")): str(case.get("resource"))
        for case in spec.get("cases", [])
    }
    if actual_case_resources != EXPECTED_CASE_RESOURCES:
        errors.append("case/resource matrix mismatch")
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
        if case["resource"] == "epilogue.nvfp4_requant":
            args = case.get("args", [])
            try:
                blocks_per_sm = int(args[args.index("--blocks-per-sm") + 1])
            except (ValueError, IndexError):
                blocks_per_sm = 0
            if blocks_per_sm != 1:
                errors.append(f"{cid}: unsafe or missing blocks-per-sm contract")
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
            if (case["resource"].startswith(("tma.", "tmem."))
                    or case["resource"] in MEMORY_RESOURCES):
                if int(fields.get("sm_count", 0)) != 20 or int(fields.get("unique_smid_count", 0)) != 20:
                    errors.append(f"{cid}: incomplete SM coverage")
                try:
                    if case["resource"].startswith("tma."):
                        args = case.get("args", [])
                        expected_inflight = int(
                            args[args.index("--inflight") + 1])
                        expected_slots = int(
                            args[args.index("--slots") + 1])
                        expected_tile_bytes = int(
                            args[args.index("--tile-bytes") + 1])
                        expected_blocks_per_sm = int(
                            args[args.index("--blocks-per-sm") + 1])
                        if (int(fields.get("inflight", 0))
                                != expected_inflight
                                or int(fields.get("slots", 0))
                                != expected_slots
                                or int(fields.get("tile_bytes", 0))
                                != expected_tile_bytes
                                or int(fields.get("blocks_per_sm", 0))
                                != expected_blocks_per_sm
                                or int(fields.get("blocks", 0))
                                != 20 * expected_blocks_per_sm):
                            raise ValueError("invalid TMA inflight contract")
                        recalculated = (int(fields["requested_bytes"]) * 1e9 /
                                        int(fields["globaltimer_elapsed_ns"]))
                    elif case["resource"] == "tmem.accumulator_readback":
                        recalculated = (int(fields["issued_bytes"]) * 1e9 /
                                        int(fields["globaltimer_elapsed_ns"]))
                    elif case["resource"] == "tmem.scale_ingress":
                        if (fields["case_id"] != cid
                                or int(fields["source_bytes_per_instruction"]) != 512
                                or int(fields["multicast_partitions"]) != 4
                                or int(fields["destination_slots"]) != 32
                                or int(fields["destination_columns_per_copy"]) != 4
                                or int(fields["value_mismatches"]) != 0):
                            raise ValueError("invalid scale S2T contract")
                        recalculated = (int(fields["issued_source_bytes"]) * 1e9 /
                                        int(fields["globaltimer_elapsed_ns"]))
                    else:
                        expected_residency, expected_direction = case["resource"].split(".")
                        expected_bytes = (16 if expected_residency == "l2" else 256) << 20
                        if (fields["case_id"] != cid
                                or fields["residency"] != expected_residency
                                or fields["direction"] != expected_direction
                                or int(fields["blocks_per_sm"]) != 4
                                or int(fields["working_set_bytes"]) != expected_bytes):
                            raise ValueError("invalid memory-path contract")
                        recalculated = (int(fields["requested_bytes"]) * 1e9 /
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
                if int(fields.get("blocks_per_sm", 0)) != 1:
                    errors.append(f"{cid}: blocks-per-sm result mismatch")
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
