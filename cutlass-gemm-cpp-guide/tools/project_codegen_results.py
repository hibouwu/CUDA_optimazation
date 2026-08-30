#!/usr/bin/env python3
"""Project sealed static-codegen results into the mutable coverage ledgers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from codegen_v2.model import (
    ContractError,
    atomic_write_json,
    load_strict_json,
    safe_path,
    sha256_file,
    validate_result,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--summary", required=True)
    parser.add_argument("--require-archive", action="store_true")
    parser.add_argument("--sync-document", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    summary_path = safe_path(root, args.summary, "summary")
    summary = load_strict_json(summary_path)
    refs = summary.get("result_refs")
    if not isinstance(refs, list) or not refs:
        raise ContractError("summary has no current result_refs")

    manifest_path = root / "tests/codegen/sm110a_tensor_schedule_tags.json"
    auto_path = root / "tests/codegen/sm110a_auto_control_inventory.json"
    manifest = load_strict_json(manifest_path)
    auto_inventory = load_strict_json(auto_path)
    explicit_by_tag = {entry["tag"]: entry for entry in manifest["entries"]}
    auto_by_id = {entry["control_id"]: entry for entry in auto_inventory["entries"]}
    seen: set[tuple[str, str]] = set()
    explicit_changed = False
    auto_changed = False

    for ref in refs:
        result_path = safe_path(root, ref.get("path"), "summary.result_ref.path")
        if result_path.stem != ref.get("id") or sha256_file(result_path) != ref.get("sha256"):
            raise ContractError("summary result ref is not sealed")
        result = validate_result(root, result_path, require_archive=args.require_archive)
        subject = result["subject"]
        key = (subject["kind"], subject["id"])
        if key in seen:
            raise ContractError(f"summary repeats subject {key}")
        seen.add(key)
        if subject["kind"] == "explicit_schedule_tag":
            entry = explicit_by_tag.get(subject["id"])
            if entry is None or entry["group"] != subject["group"]:
                raise ContractError(f"unknown explicit result subject {subject}")
            entry["status"] = result["status"]
            entry["result_id"] = result["result_id"]
            explicit_changed = True
        elif subject["kind"] == "kernel_schedule_auto_control":
            entry = auto_by_id.get(subject["id"])
            if entry is None or entry["group"] != subject["group"]:
                raise ContractError(f"unknown Auto result subject {subject}")
            entry["status"] = result["status"]
            entry["result_id"] = result["result_id"]
            auto_changed = True
            if result["status"] == "STATIC_PASS":
                type_item = next(
                    item
                    for item in load_strict_json(root / result["artifact_manifest_ref"]["path"])[
                        "items"
                    ]
                    if item["role"] == "TYPE_WITNESS_OUTPUT"
                )
                type_witness = load_strict_json(root / type_item["path"])
                entry["resolved_builder_specialization"] = type_witness["resolved_types"][
                    "mainloop_builder"
                ]
                entry["resolved_dispatch_policy"] = type_witness["resolved_types"][
                    "dispatch_policy"
                ]
        else:
            raise ContractError(f"unsupported result subject kind {subject['kind']}")

    if explicit_changed:
        atomic_write_json(manifest_path, manifest)
    if auto_changed:
        atomic_write_json(auto_path, auto_inventory)
    if args.sync_document:
        document_path = root / "docs/02-tensor-core-tcgen05-gemm-codegen-formal-validation.md"
        document = document_path.read_text(encoding="utf-8")
        status_by_tag = {entry["tag"]: entry["status"] for entry in manifest["entries"]}
        seen_tags: set[str] = set()

        def replace_tag_row(match: re.Match[str]) -> str:
            tag = match.group(1)
            if tag not in status_by_tag:
                raise ContractError(f"document contains unknown Tag row {tag}")
            seen_tags.add(tag)
            return f"| `{tag}` | `{status_by_tag[tag]}` | {match.group(3)} |"

        document = re.sub(
            r"^\|\s*`(Kernel[A-Za-z0-9_]*Sm100)`\s*\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|$",
            replace_tag_row,
            document,
            flags=re.MULTILINE,
        )
        if seen_tags != set(status_by_tag):
            raise ContractError("document Tag table does not cover the manifest exactly")
        statuses = [
            "NOT_CHECKED",
            "HISTORICAL_STATIC_PASS",
            "STATIC_PASS",
            "EXPECTED_STATIC_REJECT",
            "UNSUPPORTED_SM110A",
            "UNEXPECTED_COMPILE_FAIL",
            "ATTRIBUTION_FAIL",
        ]
        counts = {
            status: sum(entry["status"] == status for entry in manifest["entries"])
            for status in statuses
        }
        for status, count in counts.items():
            document, replacements = re.subn(
                rf"^\|\s*`{status}`\s*\|\s*[0-9]+\s*\|$",
                f"| `{status}` | {count} |",
                document,
                flags=re.MULTILINE,
            )
            if replacements != 1:
                raise ContractError(
                    f"document must contain one status-count row for {status}, found {replacements}"
                )
        document_path.write_text(document, encoding="utf-8")
    print(
        f"CODEGEN_RESULT_PROJECTION_PASS run_id={summary['run_id']} results={len(refs)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, KeyError, OSError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
