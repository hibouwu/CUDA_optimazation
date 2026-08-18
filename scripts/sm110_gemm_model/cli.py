#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .closure_import import import_closure, import_composite_closure
from .causal_import import import_causal_profile
from .resource_import import import_resource_capacities
from .closure_report import build_closure_analysis, write_closure_report
from .campaign_plots import generate_campaign_plots
from .io import (
    load_capacities,
    load_capacity_files,
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
    scenario_coverage,
    workload_manifest_coverage,
)
from .completion import audit_target_completeness
from .model import (
    ModelError,
    audit_inputs,
    audit_pipeline_profiles,
    evaluate_manifest,
    precision_specs,
)
from .observations import (
    audit_observations,
    qualify_observations_for_suite,
    summarize_observed_csvs,
)
from .suite import (
    audit_suite_linkage,
    collect_suite_artifact_paths,
    collect_suite_declared_provenance,
)
from .precision_report import (
    build_precision_evidence_analysis,
    render_precision_evidence_markdown,
)
from .appendix import load_and_render_suite_appendix
from .evidence_import import (
    import_component_campaign,
    import_compute_campaign,
    import_memory_duplex_campaign,
    import_tma_payload_campaign,
)
from .observations import summarize_closure_campaign


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
    closure_campaign_parser = sub.add_parser(
        "summarize-closure-campaign",
        help="audit a full-GEMM bundle and emit scoped observations",
    )
    closure_campaign_parser.add_argument("--repo-root", type=Path, required=True)
    closure_campaign_parser.add_argument("--run-dir", type=Path, required=True)
    closure_campaign_parser.add_argument("--hardware", type=Path, required=True)
    for command, help_text in (
        ("import-compute-campaign", "import scoped compute capacities"),
        ("import-component-campaign", "import scoped component capacities"),
        ("import-tma-payload-campaign", "import scoped TMA payload capacities"),
        ("import-memory-duplex-campaign", "import scoped joint memory capacities"),
    ):
        campaign_import = sub.add_parser(command, help=help_text)
        campaign_import.add_argument("--repo-root", type=Path, required=True)
        campaign_import.add_argument("--run-dir", type=Path, required=True)
        if command == "import-compute-campaign":
            campaign_import.add_argument("--require-ncu", action="store_true")
    appendix_parser = sub.add_parser(
        "render-suite-appendix",
        help="render a deterministic provenance appendix from a suite report",
    )
    appendix_parser.add_argument("--suite-report", type=Path, required=True)
    appendix_parser.add_argument("--repo-root", type=Path, required=True)
    appendix_parser.add_argument("--output", type=Path, required=True)
    coverage_parser = sub.add_parser(
        "coverage", help="report precision and common-resource closure gaps"
    )
    coverage_parser.add_argument("--repo-root", type=Path, required=True)
    coverage_parser.add_argument(
        "--capacities", type=Path, action="append", required=True)
    coverage_parser.add_argument("--hardware", type=Path, required=True)
    coverage_parser.add_argument("--workloads", type=Path, required=True)
    coverage_parser.add_argument("--schedules", type=Path, required=True)
    coverage_parser.add_argument("--pipeline-profiles", type=Path)
    coverage_parser.add_argument("--observed-input", type=Path, action="append")
    coverage_parser.add_argument("--closure-import", type=Path)
    coverage_parser.add_argument("--closure-run-dir", type=Path, action="append")
    coverage_parser.add_argument("--resource-import", type=Path, action="append")
    coverage_parser.add_argument("--minimum-trials", type=int, default=10)
    suite_parser = sub.add_parser(
        "audit-closure-suite",
        help="audit, cross-link, import, and evaluate one three-campaign suite",
    )
    suite_parser.add_argument("--repo-root", type=Path, required=True)
    suite_parser.add_argument("--compute-run-dir", type=Path, required=True)
    suite_parser.add_argument("--component-run-dir", type=Path, required=True)
    suite_parser.add_argument("--full-gemm-run-dir", type=Path, required=True)
    suite_parser.add_argument("--expected-commit", required=True)
    suite_parser.add_argument("--require-ncu", action="store_true")
    suite_parser.add_argument(
        "--base-capacities", type=Path, action="append", required=True)
    suite_parser.add_argument("--hardware", type=Path, required=True)
    suite_parser.add_argument("--workloads", type=Path, required=True)
    suite_parser.add_argument("--schedules", type=Path, required=True)
    suite_parser.add_argument("--pipeline-profiles", type=Path)
    suite_parser.add_argument("--output", type=Path)
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


def _coverage_output(
    capacities,
    hardware,
    workloads,
    schedules,
    observed,
    *,
    repo_root: Path,
    pipeline_profiles=(),
) -> dict[str, object]:
    manifest_rows = workload_manifest_coverage(workloads)
    scenarios = scenario_coverage(
        hardware,
        capacities,
        workloads,
        schedules,
        observed,
        pipeline_profiles=pipeline_profiles,
    )
    rows = precision_coverage(capacities, observed, hardware, scenarios)
    common = common_resource_coverage(capacities, hardware)
    payload: dict[str, object] = {
        "precision_coverage": [row.to_dict() for row in rows],
        "scenario_coverage": [row.to_dict() for row in scenarios],
        "workload_manifest_coverage": [row.to_dict() for row in manifest_rows],
        "common_resource_coverage": common,
        "all_precisions_numerically_closed": all(
            row.numeric_closure for row in rows
        ),
        "all_precisions_absolute_three_layer_closed": all(
            row.absolute_three_layer_closure for row in rows
        ),
        "all_precisions_same_precision_ratio_closed": all(
            row.same_precision_ratio_closure for row in rows
        ),
        "all_common_resources_closed": all(common.values()),
        "all_precisions_workload_manifest_complete": all(
            row.complete for row in manifest_rows
        ),
    }
    payload["target_completion"] = audit_target_completeness(
        repo_root=repo_root,
        hardware=hardware,
        capacities=capacities,
        workloads=workloads,
        schedules=schedules,
        observed=observed,
        coverage=payload,
        pipeline_profiles=pipeline_profiles,
    ).to_dict()
    return payload


def _repo_relative_input(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ModelError(f"model input is outside repo: {path}") from error
    if not resolved.is_file():
        raise ModelError(f"model input is missing: {path}")
    return str(relative)


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list-precisions":
        print(json.dumps({k: v.__dict__ for k, v in precision_specs().items()}, indent=2))
        return 0
    if args.command == "render-suite-appendix":
        output = load_and_render_suite_appendix(
            args.suite_report, repo_root=args.repo_root,
            output_path=args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        return 0
    if args.command == "summarize-closure-campaign":
        rows = summarize_closure_campaign(
            args.run_dir, repo_root=args.repo_root,
            hardware=load_hardware(args.hardware))
        print(json.dumps(
            {"closure_observations": [row.to_dict() for row in rows]},
            indent=2))
        return 0
    if args.command in {
        "import-compute-campaign", "import-component-campaign",
        "import-tma-payload-campaign", "import-memory-duplex-campaign",
    }:
        if args.command == "import-compute-campaign":
            capacities = import_compute_campaign(
                args.run_dir, repo_root=args.repo_root,
                require_ncu=args.require_ncu)
        elif args.command == "import-component-campaign":
            capacities = import_component_campaign(
                args.run_dir, repo_root=args.repo_root)
        elif args.command == "import-tma-payload-campaign":
            capacities = import_tma_payload_campaign(
                args.run_dir, repo_root=args.repo_root)
        else:
            capacities = import_memory_duplex_campaign(
                args.run_dir, repo_root=args.repo_root)
        print(json.dumps(
            {"capacities": [row.to_dict() for row in capacities]},
            indent=2, sort_keys=True))
        return 0
    if args.command == "audit-closure-suite":
        repo_root = args.repo_root.resolve()
        linkage = audit_suite_linkage(
            args.compute_run_dir,
            args.component_run_dir,
            args.full_gemm_run_dir,
            repo_root=repo_root,
            expected_commit=args.expected_commit,
            require_ncu=args.require_ncu,
        )
        imported_capacities = [
            *import_compute_campaign(
                args.compute_run_dir,
                repo_root=repo_root,
                require_ncu=args.require_ncu,
            ),
            *import_component_campaign(
                args.component_run_dir,
                repo_root=repo_root,
            ),
        ]
        hardware = load_hardware(args.hardware)
        observation_snapshots = summarize_closure_campaign(
            args.full_gemm_run_dir,
            repo_root=repo_root,
            hardware=hardware,
        )
        observations = qualify_observations_for_suite(
            observation_snapshots,
            linkage=linkage,
            full_gemm_run_dir=args.full_gemm_run_dir,
            repo_root=repo_root,
            hardware=hardware,
        )
        declared_provenance = collect_suite_declared_provenance(
            (
                args.compute_run_dir,
                args.component_run_dir,
                args.full_gemm_run_dir,
            ),
            repo_root=repo_root,
        )
        campaign_artifacts = collect_suite_artifact_paths(
            (
                args.compute_run_dir,
                args.component_run_dir,
                args.full_gemm_run_dir,
            ),
            repo_root=repo_root,
        )
        capacities = [
            *load_capacity_files(args.base_capacities),
            *imported_capacities,
        ]
        findings = [
            *audit_inputs(capacities, repo_root=repo_root),
            *audit_observations(observations, repo_root=repo_root),
        ]
        if any(row["severity"] == "error" for row in findings):
            raise ModelError(
                "merged suite evidence failed provenance audit: "
                + json.dumps(findings, sort_keys=True)
            )
        workloads = load_workloads(args.workloads)
        schedules = load_schedules(args.schedules)
        pipeline_profiles = (
            load_pipeline_profiles(args.pipeline_profiles)
            if args.pipeline_profiles else []
        )
        coverage = _coverage_output(
            capacities,
            hardware,
            workloads,
            schedules,
            observations,
            repo_root=repo_root,
            pipeline_profiles=pipeline_profiles,
        )
        source_paths = sorted({
            *declared_provenance.source_paths,
            *(
                _repo_relative_input(path, repo_root)
                for path in (
                    *args.base_capacities,
                    args.hardware,
                    args.workloads,
                    args.schedules,
                )
            ),
            *(capacity.source_path for capacity in capacities),
            *(observation.source_path for observation in observations),
        })
        artifact_paths = sorted({
            *campaign_artifacts,
            *(
                path
                for capacity in capacities
                for path in capacity.artifact_paths
            ),
            *(
                path
                for observation in observations
                for path in observation.artifact_paths
            ),
        })
        source_urls = sorted({
            *declared_provenance.source_urls,
            *(
                capacity.source_url
                for capacity in capacities
                if capacity.source_url
            ),
        })
        payload = {
            "suite_linkage": linkage.to_dict(),
            "imported_capacities": [
                capacity.to_dict() for capacity in imported_capacities
            ],
            "closure_observations": [
                observation.to_dict() for observation in observations
            ],
            "capacity_findings": findings,
            "source_paths": source_paths,
            "source_urls": source_urls,
            "artifact_paths": artifact_paths,
            "coverage": coverage,
        }
        output = json.dumps(payload, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
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
            plot_manifest = generate_campaign_plots(
                args.output_json, args.output_json.parent / "figures")
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
            "output_figures": str(args.output_json.parent / "figures"),
            "plot_count": plot_manifest["chart_count"],
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
        capacities = load_capacity_files(args.capacities)
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
        if args.closure_run_dir:
            hardware = load_hardware(args.hardware)
            observed.extend(
                row
                for run_dir in args.closure_run_dir
                for row in summarize_closure_campaign(
                    run_dir,
                    repo_root=args.repo_root,
                    hardware=hardware,
                )
            )
        capacities.extend(load_resource_imports(args.resource_import))
        if not observed:
            raise SystemExit(
                "coverage requires --observed-input, --closure-import, or "
                "--closure-run-dir")
        hardware = load_hardware(args.hardware)
        workloads = load_workloads(args.workloads)
        schedules = load_schedules(args.schedules)
        pipeline_profiles = (
            load_pipeline_profiles(args.pipeline_profiles)
            if args.pipeline_profiles else []
        )
        output = _coverage_output(
            capacities,
            hardware,
            workloads,
            schedules,
            observed,
            repo_root=args.repo_root,
            pipeline_profiles=pipeline_profiles,
        )
        output["campaign_measurement_coverage"] = (
            campaign_measurement_coverage(capacities, observed)
        )
        print(json.dumps(output, indent=2))
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
