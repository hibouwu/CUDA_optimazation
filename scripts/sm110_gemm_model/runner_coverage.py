#!/usr/bin/env python3
"""Adversarial, machine-readable coverage audit for SM110 GEMM evidence runners.

This audit answers whether a runner contract exists.  It deliberately does not
turn a source file, a static preflight, or a historical result into a Thor
measurement, and it does not treat a sustained microbenchmark as a physical
rate upper bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from microbench.sm110_full_gemm_campaign.run_full_gemm_campaign import (
    CASES as FULL_GEMM_CASES,
)
from microbench.sm110_gemm_campaign.run_compute_campaign import make_manifest
from microbench.sm110_gemm_component_campaign.run_component_campaign import (
    CASES as COMPONENT_CASES,
)
from microbench.sm110_memory_duplex_campaign.run_memory_duplex_campaign import (
    HBM_RATIOS,
    L2_RATIOS,
    cases as duplex_cases,
)
from microbench.sm110_tma_payload_campaign.run_tma_payload_campaign import (
    DEFAULT_TARGET_ISSUED_BYTES,
    PAYLOAD_BYTES,
    cases as payload_cases,
)
from scripts.sm110_gemm_model.io import load_capacities, load_schedules
from scripts.sm110_gemm_model.coverage import CAMPAIGN_COMPONENT_CASE_RESOURCES
from scripts.sm110_gemm_model.model import precision_specs


CAPACITIES = REPO / "scripts/sm110_gemm_model/profiles/capacities.json"
SCHEDULES = REPO / "scripts/sm110_gemm_model/examples/schedules.json"
SUPPORT = REPO / "microbench/sm110_full_gemm_campaign/support_manifest.json"


def _paths_exist(paths: list[str]) -> bool:
    return all((REPO / path).is_file() for path in paths)


def audit_runner_coverage() -> dict[str, object]:
    precisions = precision_specs()
    precision_ids = set(precisions)
    capacities = load_capacities(CAPACITIES)
    schedules = load_schedules(SCHEDULES)

    strict_compute = {
        precision_id
        for precision_id, spec in precisions.items()
        if any(
            capacity.resource == spec.compute_resource
            and capacity.evidence_kind.is_rate_upper_bound
            for capacity in capacities
        )
    }
    shared_strict = {
        resource: any(
            capacity.resource == resource
            and capacity.evidence_kind.is_rate_upper_bound
            for capacity in capacities
        )
        for resource in ("hbm.total", "l2.read", "l2.write")
    }

    compute_manifest = make_manifest()
    compute_ids = {
        str(row["precision"]["precision_id"]) for row in compute_manifest
    }
    compute_shapes = {
        (str(row["precision"]["precision_id"]), int(row["m"]), int(row["n"]))
        for row in compute_manifest
        if row["launch"] == "full_sm_4warp_block"
    }
    expected_compute_shapes = {
        (precision_id, 128, n)
        for precision_id in precision_ids
        for n in (64, 128, 256)
    }

    component_resources = set(CAMPAIGN_COMPONENT_CASE_RESOURCES.values())
    required_component = {
        "tma.smem_ingress.per_sm",
        "tma.smem_ingress.per_sm.inflight4",
        "tma.hbm",
        "tma.hbm.inflight4",
        "tmem.scale_ingress",
        "tmem.readback",
        "hbm.read",
        "hbm.write",
        "l2.read",
        "l2.write",
    }

    payload_manifest = payload_cases(DEFAULT_TARGET_ISSUED_BYTES)
    payload_pairs = {
        (int(row["tile_bytes"]), str(row["residency"]))
        for row in payload_manifest
    }
    expected_payload_pairs = {
        (payload, residency)
        for payload in PAYLOAD_BYTES
        for residency in ("hot_l2", "cold_hbm")
    }

    duplex_manifest = duplex_cases()
    hbm_ratio_have = {
        (int(row["read_operations"]), int(row["write_operations"]))
        for row in duplex_manifest
        if row["residency"] == "cold_hbm"
    }
    l2_ratio_have = {
        (int(row["read_operations"]), int(row["write_operations"]))
        for row in duplex_manifest
        if row["residency"] == "hot_l2"
    }

    required_schedule_precision_pairs = {
        (schedule.schedule_id, precision_id)
        for schedule in schedules
        for precision_id in schedule.supported_precisions
    }
    # The component suite has one genuinely schedule-matched independent TMA
    # contract: tc5a, A=16 KiB plus B=32 KiB, four stages/eight requests.
    component_case_ids = {str(case["id"]) for case in COMPONENT_CASES}
    tc5a_defined = {
        "tma_l2_hit_tc5a_ab_inflight8",
        "tma_dram_stream_tc5a_ab_inflight8",
    } <= component_case_ids
    exact_tma_pairs = {
        ("tc5a_m128n256k64_stage4", precision_id)
        for precision_id in ("fp16_f32", "bf16_f32")
        if tc5a_defined
    }
    missing_exact_tma_pairs = sorted(
        required_schedule_precision_pairs - exact_tma_pairs
    )

    support = json.loads(SUPPORT.read_text())
    full_precision_ids = {str(case["precision_id"]) for case in FULL_GEMM_CASES}
    closure_ready_ids = {
        str(row["precision_id"])
        for row in support["precisions"]
        if row["status"] == "ready_for_closure_campaign"
    }
    full_gemm_ready_covered = full_precision_ids == closure_ready_ids

    source_contracts = [
        "microbench/sm110_gemm_campaign/run_compute_campaign.py",
        "microbench/sm110_gemm_component_campaign/run_component_campaign.py",
        "microbench/sm110_tma_payload_campaign/run_tma_payload_campaign.py",
        "microbench/sm110_memory_duplex_campaign/run_memory_duplex_campaign.py",
        "microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py",
    ]

    strict_complete = strict_compute == precision_ids and all(shared_strict.values())
    compute_complete = compute_ids == precision_ids and compute_shapes == expected_compute_shapes
    component_complete = required_component <= component_resources
    payload_complete = payload_pairs == expected_payload_pairs
    duplex_complete = (
        hbm_ratio_have == set(HBM_RATIOS) and l2_ratio_have == set(L2_RATIOS)
    )
    exact_tma_complete = not missing_exact_tma_pairs
    independent_joint_pipeline_runner_defined = False

    # fixed_seconds=0 is a deliberate nonnegative-overhead relaxation for the
    # no-waste upper.  It is not a measured latency parameter and is therefore
    # insufficient for small-GEMM wall-time prediction.
    fixed_cost = {
        "upper_relaxation_defined": all(schedule.fixed_seconds == 0 for schedule in schedules),
        "measured_runner_defined": False,
        "required_for_no_waste_upper": False,
        "required_for_wall_time_prediction": True,
    }

    payload_duplex_complete = payload_complete and duplex_complete
    empirical_runner_complete = (
        compute_complete
        and component_complete
        and payload_duplex_complete
        and exact_tma_complete
        and independent_joint_pipeline_runner_defined
    )
    all_runner_complete = empirical_runner_complete and fixed_cost[
        "measured_runner_defined"
    ]

    return {
        "schema_version": 1,
        "evidence_semantics": {
            "runner_defined_is_measurement": False,
            "microbenchmark_sustained_is_physical_upper": False,
            "static_preflight_is_runtime_evidence": False,
        },
        "strict_upper_sources": {
            "compute_precision_ids": sorted(strict_compute),
            "missing_compute_precision_ids": sorted(precision_ids - strict_compute),
            "shared_resources": shared_strict,
            "complete": strict_complete,
            "note": "strict physical uppers require specification or derivation; runners only falsify or calibrate them",
        },
        "compute_surface": {
            "precision_count": len(compute_ids),
            "full_sm_shape_count": len(compute_shapes),
            "complete": compute_complete,
        },
        "component_surface": {
            "required_resources": sorted(required_component),
            "missing_resources": sorted(required_component - component_resources),
            "complete": component_complete,
        },
        "tma_payload_surface": {
            "case_count": len(payload_manifest),
            "payload_bytes": list(PAYLOAD_BYTES),
            "residencies": ["hot_l2", "cold_hbm"],
            "complete": payload_complete,
            "exact_multi_request_topology": False,
        },
        "memory_duplex_surface": {
            "case_count": len(duplex_manifest),
            "hbm_ratios": [list(value) for value in HBM_RATIOS],
            "l2_ratios": [list(value) for value in L2_RATIOS],
            "complete": duplex_complete,
        },
        "exact_tma_topology_surface": {
            "required_schedule_precision_pair_count": len(required_schedule_precision_pairs),
            "covered_schedule_precision_pairs": [list(value) for value in sorted(exact_tma_pairs)],
            "missing_schedule_precision_pairs": [list(value) for value in missing_exact_tma_pairs],
            "complete": exact_tma_complete,
        },
        "joint_pipeline_surface": {
            "independent_runner_defined": independent_joint_pipeline_runner_defined,
            "full_gemm_can_only_falsify_components": True,
            "complete": independent_joint_pipeline_runner_defined,
        },
        "fixed_cost": fixed_cost,
        "full_gemm_validation": {
            "runner_precision_ids": sorted(full_precision_ids),
            "closure_ready_precision_ids": sorted(closure_ready_ids),
            "all_declared_precision_ids": sorted(precision_ids),
            "ready_paths_covered": full_gemm_ready_covered,
            "all_precisions_covered": full_precision_ids == precision_ids,
        },
        "source_contracts_exist": _paths_exist(source_contracts),
        "payload_duplex_runner_definition_complete": payload_duplex_complete,
        "empirical_parameter_runner_definition_complete": empirical_runner_complete,
        "all_performance_parameter_runner_definition_complete": all_runner_complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-all-performance-parameters",
        action="store_true",
        help="return nonzero unless every runner-defined performance parameter is covered",
    )
    args = parser.parse_args()
    report = audit_runner_coverage()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_all_performance_parameters:
        return 0 if report[
            "all_performance_parameter_runner_definition_complete"
        ] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
