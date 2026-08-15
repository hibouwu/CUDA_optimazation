#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .closure_import import import_closure, import_composite_closure
from .causal_import import import_causal_profile
from .resource_import import import_resource_capacities
from .closure_report import build_closure_analysis, write_closure_report
from .io import (
    load_capacities,
    load_closure_inputs,
    load_hardware,
    load_pipeline_profiles,
    load_schedules,
    load_workloads,
    read_json,
)
from .coverage import (
    campaign_measurement_coverage,
    common_resource_coverage,
    precision_coverage,
)
from .model import (
    ModelError,
    audit_inputs,
    audit_pipeline_profiles,
    evaluate_manifest,
    precision_specs,
)
from .observations import audit_observations, summarize_observed_csvs
from .precision_report import (
    build_precision_evidence_analysis,
    render_precision_evidence_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thor/SM110 three-layer GEMM model")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate", help="evaluate workload/schedule pairs")
    for name in ("hardware", "capacities", "workloads", "schedules"):
        evaluate_parser.add_argument(f"--{name}", type=Path, required=True)
    evaluate_parser.add_argument("--closure-import", type=Path)
    evaluate_parser.add_argument("--resource-import", type=Path, action="append")
    evaluate_parser.add_argument("--pipeline-profiles", type=Path)
    evaluate_parser.add_argument("--output", type=Path)

    audit_parser = sub.add_parser("audit", help="audit capacity provenance and semantics")
    audit_parser.add_argument("--capacities", type=Path, required=True)
    audit_parser.add_argument("--closure-import", type=Path)
    audit_parser.add_argument("--resource-import", type=Path, action="append")
    audit_parser.add_argument("--pipeline-profiles", type=Path)
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
    coverage_parser.add_argument("--resource-import", type=Path, action="append")
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
    causal_import_parser = sub.add_parser(
        "import-causal-profile",
        help=("independently audit a returned tc5a causal campaign and emit "
              "independent FP16/BF16 model pipeline profiles"),
    )
    causal_import_parser.add_argument("--repo-root", type=Path, required=True)
    causal_import_parser.add_argument("--run-id", required=True)
    causal_import_parser.add_argument("--expected-commit", required=True)
    causal_import_parser.add_argument("--output", type=Path, required=True)
    resource_import_parser = sub.add_parser(
        "import-resource-capacities",
        help=("independently audit a returned exact-resource suite and emit "
              "54 schedule/stride-qualified model capacities"),
    )
    resource_import_parser.add_argument("--repo-root", type=Path, required=True)
    resource_import_parser.add_argument("--suite-id", required=True)
    resource_import_parser.add_argument("--expected-commit", required=True)
    resource_import_parser.add_argument("--output", type=Path, required=True)
    report_parser = sub.add_parser(
        "report-closure",
        help="render audited closure capacities, observations, and model comparisons",
    )
    report_parser.add_argument("--closure-import", type=Path, required=True)
    report_parser.add_argument("--repo-root", type=Path, required=True)
    report_parser.add_argument("--capacities", type=Path, required=True)
    report_parser.add_argument("--hardware", type=Path, required=True)
    report_parser.add_argument("--schedules", type=Path, required=True)
    report_parser.add_argument("--pipeline-profiles", type=Path)
    report_parser.add_argument("--resource-import", type=Path, action="append")
    report_parser.add_argument("--output-json", type=Path, required=True)
    report_parser.add_argument("--output-markdown", type=Path, required=True)
    precision_report_parser = sub.add_parser(
        "report-precision-closure",
        help=("merge all-precision implementation readiness with audited "
              "numeric evidence"),
    )
    precision_report_parser.add_argument("--repo-root", type=Path, required=True)
    precision_report_parser.add_argument("--capacities", type=Path, required=True)
    precision_report_parser.add_argument("--closure-import", type=Path, required=True)
    precision_report_parser.add_argument("--hardware", type=Path, required=True)
    precision_report_parser.add_argument("--schedules", type=Path, required=True)
    precision_report_parser.add_argument("--pipeline-profiles", type=Path)
    precision_report_parser.add_argument(
        "--resource-import", type=Path, action="append")
    precision_report_parser.add_argument(
        "--support-manifest", type=Path, required=True)
    precision_report_parser.add_argument("--output-json", type=Path, required=True)
    precision_report_parser.add_argument(
        "--output-markdown", type=Path, required=True)
    precision_report_parser.add_argument(
        "--require-all-closed",
        action="store_true",
        help="exit nonzero unless every declared precision is end-to-end closed",
    )
    return parser


def load_resource_imports(paths: list[Path] | None) -> list:
    capacities = []
    for path in paths or []:
        imported, observations = load_closure_inputs(path)
        if observations:
            raise ModelError(
                f"{path}: resource import must not contain GEMM observations"
            )
        capacities.extend(imported)
    return capacities


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
    if args.command == "import-causal-profile":
        try:
            imported = import_causal_profile(
                repo_root=args.repo_root,
                run_id=args.run_id,
                expected_commit=args.expected_commit,
            )
        except ModelError as error:
            print(json.dumps({"pass": False, "error": str(error)}, indent=2))
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(imported, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "pass": True,
            "output": str(args.output),
            "run_id": imported["run_id"],
            "qualification": imported["qualification"],
            "profile_count": imported["profile_count"],
            "profile_qualified_by_precision":
                imported["profile_qualified_by_precision"],
            "profile_qualified": imported["profile_qualified"],
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "import-resource-capacities":
        try:
            imported = import_resource_capacities(
                repo_root=args.repo_root,
                suite_id=args.suite_id,
                expected_commit=args.expected_commit,
            )
        except ModelError as error:
            print(json.dumps({"pass": False, "error": str(error)}, indent=2))
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(imported, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "pass": True,
            "output": str(args.output),
            "suite_id": imported["suite_id"],
            "qualification": imported["qualification"],
            "capacity_count": len(imported["capacities"]),
            "overcurrent_deltas": imported["platform_evidence"]
                                               ["overcurrent_deltas"],
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "report-closure":
        try:
            closure_capacities, observations = load_closure_inputs(
                args.closure_import)
            base_capacities = load_capacities(args.capacities)
            resource_capacities = load_resource_imports(args.resource_import)
            pipeline_profiles = (
                load_pipeline_profiles(args.pipeline_profiles)
                if args.pipeline_profiles else []
            )
            input_findings = [
                *audit_inputs(
                    [
                        *base_capacities,
                        *closure_capacities,
                        *resource_capacities,
                    ],
                    repo_root=args.repo_root.resolve(),
                ),
                *audit_observations(
                    observations, repo_root=args.repo_root.resolve()),
                *audit_pipeline_profiles(
                    pipeline_profiles, repo_root=args.repo_root.resolve()),
            ]
            analysis = build_closure_analysis(
                metadata=read_json(args.closure_import),
                base_capacities=base_capacities,
                closure_capacities=[
                    *closure_capacities,
                    *resource_capacities,
                ],
                observations=observations,
                hardware=load_hardware(args.hardware),
                schedules=load_schedules(args.schedules),
                pipeline_profiles=pipeline_profiles,
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
    if args.command == "report-precision-closure":
        try:
            closure_capacities, observations = load_closure_inputs(
                args.closure_import)
            pipeline_profiles = (
                load_pipeline_profiles(args.pipeline_profiles)
                if args.pipeline_profiles else []
            )
            profile_errors = [
                row for row in audit_pipeline_profiles(
                    pipeline_profiles, repo_root=args.repo_root.resolve()
                )
                if row["severity"] == "error"
            ]
            if profile_errors:
                raise ModelError(
                    f"pipeline profile audit failed: {profile_errors}"
                )
            analysis = build_precision_evidence_analysis(
                capacities=[
                    *load_capacities(args.capacities),
                    *closure_capacities,
                    *load_resource_imports(args.resource_import),
                ],
                observations=observations,
                support_manifest=read_json(args.support_manifest),
                repo_root=args.repo_root.resolve(),
                hardware=load_hardware(args.hardware),
                schedules=load_schedules(args.schedules),
                pipeline_profiles=pipeline_profiles,
                metadata=read_json(args.closure_import),
            )
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(analysis, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args.output_markdown.write_text(
                render_precision_evidence_markdown(analysis),
                encoding="utf-8",
            )
        except (ModelError, OSError, ValueError, KeyError, TypeError) as error:
            print(json.dumps({
                "pass": False,
                "error": str(error),
            }, indent=2, sort_keys=True))
            return 1
        closed = bool(analysis["all_precisions_end_to_end_closed"])
        print(json.dumps({
            "pass": closed,
            "all_precisions_end_to_end_closed": closed,
            "implementation_ready_count": analysis["implementation_ready_count"],
            "numeric_closed_count": analysis["numeric_closed_count"],
            "resource_envelope_closed_count":
                analysis["resource_envelope_closed_count"],
            "causal_pipeline_closed_count":
                analysis["causal_pipeline_closed_count"],
            "end_to_end_closed_count": analysis["end_to_end_closed_count"],
            "output_json": str(args.output_json),
            "output_markdown": str(args.output_markdown),
        }, indent=2, sort_keys=True))
        return 1 if args.require_all_closed and not closed else 0
    if args.command == "audit":
        capacities = load_capacities(args.capacities)
        pipeline_profiles = (
            load_pipeline_profiles(args.pipeline_profiles)
            if args.pipeline_profiles else []
        )
        observations = []
        if args.closure_import:
            imported_capacities, observations = load_closure_inputs(
                args.closure_import)
            capacities.extend(imported_capacities)
        capacities.extend(load_resource_imports(args.resource_import))
        findings = [
            *audit_inputs(capacities, repo_root=args.repo_root.resolve()),
            *audit_observations(observations, repo_root=args.repo_root.resolve()),
            *audit_pipeline_profiles(
                pipeline_profiles, repo_root=args.repo_root.resolve()),
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
        capacities.extend(load_resource_imports(args.resource_import))
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
    capacities.extend(load_resource_imports(args.resource_import))
    rows = []
    schedules = load_schedules(args.schedules)
    pipeline_profiles = (
        load_pipeline_profiles(args.pipeline_profiles)
        if args.pipeline_profiles else []
    )
    for workload in load_workloads(args.workloads):
        rows.append(
            evaluate_manifest(
                workload, schedules, hardware, capacities, pipeline_profiles
            ).to_dict()
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
