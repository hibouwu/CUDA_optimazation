#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .closure_import import import_closure, import_composite_closure
from .closure_report import build_closure_analysis, write_closure_report
from .io import (
    load_capacities,
    load_closure_inputs,
    load_hardware,
    load_schedules,
    load_workloads,
    read_json,
)
from .coverage import (
    campaign_measurement_coverage,
    common_resource_coverage,
    precision_coverage,
)
from .model import ModelError, audit_inputs, evaluate_manifest, precision_specs
from .observations import audit_observations, summarize_observed_csvs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thor/SM110 three-layer GEMM model")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate", help="evaluate workload/schedule pairs")
    for name in ("hardware", "capacities", "workloads", "schedules"):
        evaluate_parser.add_argument(f"--{name}", type=Path, required=True)
    evaluate_parser.add_argument("--closure-import", type=Path)
    evaluate_parser.add_argument("--output", type=Path)

    audit_parser = sub.add_parser("audit", help="audit capacity provenance and semantics")
    audit_parser.add_argument("--capacities", type=Path, required=True)
    audit_parser.add_argument("--closure-import", type=Path)
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
    coverage_parser.add_argument("--observed-input", type=Path, action="append")
    coverage_parser.add_argument("--closure-import", type=Path)
    coverage_parser.add_argument("--minimum-trials", type=int, default=10)
    import_parser = sub.add_parser(
        "import-closure",
        help="independently audit a returned closure suite and emit model inputs",
    )
    import_parser.add_argument("--repo-root", type=Path, required=True)
    import_parser.add_argument("--suite-id", required=True)
    import_parser.add_argument("--expected-commit", required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    composite_parser = sub.add_parser(
        "import-composite-closure",
        help=("audit base compute/full evidence plus a separately committed "
              "component supplement and emit one model input"),
    )
    composite_parser.add_argument("--repo-root", type=Path, required=True)
    composite_parser.add_argument("--composite-id", required=True)
    composite_parser.add_argument("--base-suite-id", required=True)
    composite_parser.add_argument("--base-expected-commit", required=True)
    composite_parser.add_argument("--component-expected-commit", required=True)
    composite_parser.add_argument("--output", type=Path, required=True)
    report_parser = sub.add_parser(
        "report-closure",
        help="render audited closure capacities, observations, and model comparisons",
    )
    report_parser.add_argument("--closure-import", type=Path, required=True)
    report_parser.add_argument("--repo-root", type=Path, required=True)
    report_parser.add_argument("--capacities", type=Path, required=True)
    report_parser.add_argument("--hardware", type=Path, required=True)
    report_parser.add_argument("--schedules", type=Path, required=True)
    report_parser.add_argument("--output-json", type=Path, required=True)
    report_parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list-precisions":
        print(json.dumps({k: v.__dict__ for k, v in precision_specs().items()}, indent=2))
        return 0
    if args.command == "import-closure":
        try:
            imported = import_closure(
                repo_root=args.repo_root,
                suite_id=args.suite_id,
                expected_commit=args.expected_commit,
            )
        except ModelError as error:
            print(json.dumps({"pass": False, "error": str(error)}, indent=2))
            return 1
        output = json.dumps(imported, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(json.dumps({
            "pass": True,
            "output": str(args.output),
            "qualification": imported["qualification"],
            "capacity_count": len(imported["capacities"]),
            "observation_count": len(imported["observed_best"]),
            "overcurrent_deltas": imported["platform_evidence"]["overcurrent_deltas"],
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "import-composite-closure":
        try:
            imported = import_composite_closure(
                repo_root=args.repo_root,
                composite_id=args.composite_id,
                base_suite_id=args.base_suite_id,
                base_expected_commit=args.base_expected_commit,
                component_expected_commit=args.component_expected_commit,
            )
        except ModelError as error:
            print(json.dumps({"pass": False, "error": str(error)}, indent=2))
            return 1
        output = json.dumps(imported, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(json.dumps({
            "pass": True,
            "output": str(args.output),
            "qualification": imported["qualification"],
            "capacity_count": len(imported["capacities"]),
            "observation_count": len(imported["observed_best"]),
            "campaign_sources": imported["campaign_sources"],
            "overcurrent_deltas": imported["platform_evidence"]
                                              ["overcurrent_deltas"],
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "report-closure":
        try:
            closure_capacities, observations = load_closure_inputs(
                args.closure_import)
            base_capacities = load_capacities(args.capacities)
            input_findings = [
                *audit_inputs(
                    [*base_capacities, *closure_capacities],
                    repo_root=args.repo_root.resolve(),
                ),
                *audit_observations(
                    observations, repo_root=args.repo_root.resolve()),
            ]
            analysis = build_closure_analysis(
                metadata=read_json(args.closure_import),
                base_capacities=base_capacities,
                closure_capacities=closure_capacities,
                observations=observations,
                hardware=load_hardware(args.hardware),
                schedules=load_schedules(args.schedules),
                input_findings=input_findings,
            )
            write_closure_report(
                analysis,
                json_path=args.output_json,
                markdown_path=args.output_markdown,
            )
        except (ModelError, OSError, ValueError, KeyError, TypeError) as error:
            print(json.dumps({
                "pass": False,
                "error": str(error),
            }, indent=2, sort_keys=True))
            return 1
        print(json.dumps({
            "pass": analysis["pass"],
            "output_json": str(args.output_json),
            "output_markdown": str(args.output_markdown),
            "finding_count": len(analysis["findings"]),
        }, indent=2, sort_keys=True))
        return 0 if analysis["pass"] else 1
    if args.command == "audit":
        capacities = load_capacities(args.capacities)
        observations = []
        if args.closure_import:
            imported_capacities, observations = load_closure_inputs(
                args.closure_import)
            capacities.extend(imported_capacities)
        findings = [
            *audit_inputs(capacities, repo_root=args.repo_root.resolve()),
            *audit_observations(observations, repo_root=args.repo_root.resolve()),
        ]
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
        observed = []
        if args.observed_input:
            observed.extend(summarize_observed_csvs(
                args.observed_input,
                repo_root=args.repo_root,
                minimum_trials=args.minimum_trials,
            ))
        if args.closure_import:
            imported_capacities, imported_observations = load_closure_inputs(
                args.closure_import)
            capacities.extend(imported_capacities)
            observed.extend(imported_observations)
        if not observed:
            raise SystemExit(
                "coverage requires --observed-input or --closure-import")
        rows = precision_coverage(capacities, observed)
        common = common_resource_coverage(capacities)
        campaign = campaign_measurement_coverage(capacities, observed)
        print(
            json.dumps(
                {
                    "precision_coverage": [row.to_dict() for row in rows],
                    "common_resource_coverage": common,
                    "campaign_measurement_coverage": campaign,
                    "all_precisions_closed": all(row.numeric_closure for row in rows),
                    "all_declared_precisions_closed": all(
                        row.numeric_closure for row in rows),
                    "all_common_resources_closed": all(common.values()),
                },
                indent=2,
            )
        )
        return 0

    hardware = load_hardware(args.hardware)
    capacities = load_capacities(args.capacities)
    if args.closure_import:
        imported_capacities, _ = load_closure_inputs(args.closure_import)
        capacities.extend(imported_capacities)
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
