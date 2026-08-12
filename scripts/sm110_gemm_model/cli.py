#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_capacities, load_hardware, load_schedules, load_workloads
from .coverage import common_resource_coverage, precision_coverage
from .model import audit_inputs, evaluate_manifest, precision_specs
from .observations import summarize_observed_csvs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thor/SM110 three-layer GEMM model")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate", help="evaluate workload/schedule pairs")
    for name in ("hardware", "capacities", "workloads", "schedules"):
        evaluate_parser.add_argument(f"--{name}", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path)

    audit_parser = sub.add_parser("audit", help="audit capacity provenance and semantics")
    audit_parser.add_argument("--capacities", type=Path, required=True)
    audit_parser.add_argument("--repo-root", type=Path, required=True)
    sub.add_parser("list-precisions", help="print the v1 precision contract")
    observed_parser = sub.add_parser(
        "summarize-observed", help="summarize fully matched full-GEMM CSV series"
    )
    observed_parser.add_argument("--repo-root", type=Path, required=True)
    observed_parser.add_argument("--input", type=Path, action="append", required=True)
    observed_parser.add_argument("--minimum-trials", type=int, default=10)
    coverage_parser = sub.add_parser(
        "coverage", help="report precision and common-resource closure gaps"
    )
    coverage_parser.add_argument("--repo-root", type=Path, required=True)
    coverage_parser.add_argument("--capacities", type=Path, required=True)
    coverage_parser.add_argument("--observed-input", type=Path, action="append", required=True)
    coverage_parser.add_argument("--minimum-trials", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list-precisions":
        print(json.dumps({k: v.__dict__ for k, v in precision_specs().items()}, indent=2))
        return 0
    if args.command == "audit":
        findings = audit_inputs(
            load_capacities(args.capacities), repo_root=args.repo_root.resolve()
        )
        print(json.dumps({"findings": findings}, indent=2))
        return 1 if any(row["severity"] == "error" for row in findings) else 0
    if args.command == "summarize-observed":
        rows = summarize_observed_csvs(
            args.input,
            repo_root=args.repo_root,
            minimum_trials=args.minimum_trials,
        )
        print(json.dumps({"observed_best": [row.to_dict() for row in rows]}, indent=2))
        return 0
    if args.command == "coverage":
        capacities = load_capacities(args.capacities)
        observed = summarize_observed_csvs(
            args.observed_input,
            repo_root=args.repo_root,
            minimum_trials=args.minimum_trials,
        )
        rows = precision_coverage(capacities, observed)
        common = common_resource_coverage(capacities)
        print(
            json.dumps(
                {
                    "precision_coverage": [row.to_dict() for row in rows],
                    "common_resource_coverage": common,
                    "all_precisions_closed": all(row.numeric_closure for row in rows),
                    "all_common_resources_closed": all(common.values()),
                },
                indent=2,
            )
        )
        return 0

    hardware = load_hardware(args.hardware)
    capacities = load_capacities(args.capacities)
    rows = []
    schedules = load_schedules(args.schedules)
    for workload in load_workloads(args.workloads):
        rows.append(
            evaluate_manifest(workload, schedules, hardware, capacities).to_dict()
        )
    output = json.dumps({"envelopes": rows}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
