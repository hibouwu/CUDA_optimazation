#!/usr/bin/env python3
"""Validate the exhaustive 59-tag SM100 Tensor Core codegen coverage contract."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


EXPECTED_GROUP_COUNTS = {
    "dense": 5,
    "ptr_array_dense": 2,
    "blockwise": 4,
    "planar_complex": 4,
    "fast_fp32": 8,
    "mixed_input": 4,
    "interleaved_complex_tf32": 4,
    "sparse": 2,
    "dense_block_scaled": 10,
    "ptr_array_block_scaled": 8,
    "sparse_block_scaled": 8,
}

EXPECTED_CUTLASS_SHA = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
EXPECTED_TAG_COUNT = 59


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: cannot parse JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be an object")
    return value


def discover_source_tags(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    marker = "struct KernelScheduleSm100"
    if marker not in source:
        raise ValueError(f"{path}: cannot locate {marker}")
    user_schedule_region = source[source.index(marker) :]
    return set(
        re.findall(
            r"struct\s+(Kernel[A-Za-z0-9_]*Sm100)\s+final",
            user_schedule_region,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()

    manifest_path = root / "tests/codegen/sm110a_tensor_schedule_tags.json"
    document_path = root / "docs/02-tensor-core-tcgen05-gemm-codegen-formal-validation.md"
    manifest = load_json(manifest_path)
    document = document_path.read_text(encoding="utf-8")
    errors: list[str] = []

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    target = manifest.get("target_contract", {})
    if target.get("cutlass_git_sha") != EXPECTED_CUTLASS_SHA:
        errors.append("manifest CUTLASS SHA does not match the guide lock")
    if target.get("virtual_arch") != "compute_110a" or target.get("binary_arch") != "sm_110a":
        errors.append("target contract must be compute_110a -> sm_110a")

    policy = manifest.get("coverage_policy", {})
    allowed_statuses = set(policy.get("allowed_statuses", []))
    if policy.get("omission_allowed") is not False:
        errors.append("coverage policy must reject omitted tags")
    if policy.get("out_of_scope_allowed") is not False:
        errors.append("coverage policy must reject OUT_OF_SCOPE as a terminal shortcut")
    if policy.get("static_codegen_required_per_tag") is not True:
        errors.append("every tag must require static codegen evidence")

    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        errors.append("entries must be a list")
        entries = []
    tags = [entry.get("tag") for entry in entries if isinstance(entry, dict)]
    if len(entries) != EXPECTED_TAG_COUNT:
        errors.append(f"expected {EXPECTED_TAG_COUNT} entries, found {len(entries)}")
    if len(tags) != len(set(tags)):
        duplicates = sorted(tag for tag, count in Counter(tags).items() if count > 1)
        errors.append(f"duplicate tags: {duplicates}")
    invalid_names = sorted(
        tag for tag in tags if not isinstance(tag, str) or not tag.startswith("Kernel") or not tag.endswith("Sm100")
    )
    if invalid_names:
        errors.append(f"invalid tag names: {invalid_names}")

    group_counts = Counter(entry.get("group") for entry in entries if isinstance(entry, dict))
    if dict(group_counts) != EXPECTED_GROUP_COUNTS:
        errors.append(f"group counts {dict(group_counts)} != {EXPECTED_GROUP_COUNTS}")

    case_ids = {path.parent.name for path in (root / "cases").glob("*/case.json")}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tag = entry.get("tag", "<missing>")
        status = entry.get("status")
        if status not in allowed_statuses:
            errors.append(f"{tag}: invalid status {status!r}")
        mapped_cases = entry.get("current_case_ids")
        if not isinstance(mapped_cases, list) or not all(isinstance(value, str) for value in mapped_cases):
            errors.append(f"{tag}: current_case_ids must be a string list")
            continue
        missing_cases = sorted(set(mapped_cases) - case_ids)
        if missing_cases:
            errors.append(f"{tag}: unknown current case ids {missing_cases}")
        if status == "HISTORICAL_STATIC_PASS" and not mapped_cases:
            errors.append(f"{tag}: historical PASS requires at least one mapped case")
        if document.count(tag) == 0:
            errors.append(f"{tag}: missing from Tensor Core document")

    source_status = "NOT_AVAILABLE"
    dispatch_policy = root / "third_party/cutlass/include/cutlass/gemm/dispatch_policy.hpp"
    if dispatch_policy.is_file():
        source_status = "CHECKED"
        try:
            source_tags = discover_source_tags(dispatch_policy)
        except ValueError as error:
            errors.append(str(error))
        else:
            manifest_tags = set(tags)
            missing = sorted(source_tags - manifest_tags)
            extra = sorted(manifest_tags - source_tags)
            if missing or extra:
                errors.append(f"fixed-source mismatch: missing={missing}, extra={extra}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    status_counts = Counter(entry["status"] for entry in entries)
    print(
        "SCHEDULE_TAG_COVERAGE_PASS "
        f"tags={len(entries)} unique={len(set(tags))} "
        f"historical={status_counts.get('HISTORICAL_STATIC_PASS', 0)} "
        f"not_checked={status_counts.get('NOT_CHECKED', 0)} "
        f"source={source_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
