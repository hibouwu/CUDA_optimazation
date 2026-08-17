#!/usr/bin/env python3
"""Render or check the v0.1 capability matrix from case manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVIDENCE_COLUMNS = [
    "documented",
    "compile_passed",
    "ptx_verified",
    "sass_verified",
    "runtime_correct",
    "performance_measured",
]


def mark(value: bool) -> str:
    return "PASS" if value else "NOT_RUN"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rows = []
    errors = []
    for path in sorted((args.root / "cases").glob("*/case.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        evidence = data["evidence"]
        if evidence.get("performance_measured"):
            errors.append(f"{data['case_id']}: v0.1 must not claim measured performance")
        rows.append((data["case_id"], data["track"], evidence))

    print("| Case | Track | " + " | ".join(EVIDENCE_COLUMNS) + " |")
    print("|---|---|" + "---|" * len(EVIDENCE_COLUMNS))
    for case_id, track, evidence in rows:
        print("| " + case_id + " | " + track + " | " +
              " | ".join(mark(bool(evidence[column])) for column in EVIDENCE_COLUMNS) + " |")

    if args.check and (errors or len(rows) != 10):
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(rows) != 10:
            print(f"ERROR: expected 10 core manifests, found {len(rows)}", file=sys.stderr)
        return 1
    print(f"COVERAGE_SUMMARY_PASS cases={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
