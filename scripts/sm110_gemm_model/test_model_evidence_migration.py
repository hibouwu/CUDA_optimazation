from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from scripts.sm110_gemm_model.coverage import workload_manifest_coverage
from scripts.sm110_gemm_model.io import load_capacity_files
from scripts.sm110_gemm_model.model import (
    Capacity,
    EvidenceKind,
    Hardware,
    ModelError,
    Schedule,
    Workload,
    _select_capacity,
    evaluate,
    evaluate_manifest,
)


class EvidenceMigrationTest(unittest.TestCase):
    def upper(self, **changes: object) -> Capacity:
        row = Capacity(
            capacity_id="scoped_upper",
            resource="tensor.bf16",
            rate_per_second=100.0,
            work_unit="flop",
            evidence_kind=EvidenceKind.DERIVED_UPPER,
            source_id="synthetic",
            source_path="source.json",
            source_locator="upper",
            applicable_sm_counts=(20,),
            applicable_hardware_ids=("thor_t5000_sm110_20sm",),
            applicable_operating_modes=("MAXN",),
            applicable_clock_hz=(1_575_000_000.0,),
            upper_scope="tensor_core_classical",
        )
        return dataclasses.replace(row, **changes)

    def test_capacity_selection_honors_full_hardware_scope(self) -> None:
        workload = Workload(
            "hardware_scope", 128, 128, 64, "bf16_f32",
            residency="compute_oracle",
        )
        schedule = Schedule("s", 128, 128, 64, 2)
        matched = evaluate(
            workload,
            schedule,
            Hardware(
                "thor_t5000_sm110_20sm", 20, 1_575_000_000.0, "MAXN"
            ),
            [self.upper()],
        )
        self.assertEqual(matched.conditional_upper.status, "ok")
        mismatches = (
            Hardware("other_20sm", 20, 1_575_000_000.0, "MAXN"),
            Hardware("thor_t5000_sm110_20sm", 20, 1_575_000_000.0, "30W"),
            Hardware("thor_t5000_sm110_20sm", 20, 1_400_000_000.0, "MAXN"),
        )
        for hardware in mismatches:
            with self.subTest(hardware=hardware):
                result = evaluate(workload, schedule, hardware, [self.upper()])
                self.assertEqual(
                    result.conditional_upper.status,
                    "insufficient_evidence",
                )

    def test_schedule_family_upper_is_not_a_domain_upper(self) -> None:
        family = self.upper(
            upper_scope="schedule_family",
            applicable_mma_shapes=("m128n128k16",),
        )
        result = evaluate_manifest(
            Workload(
                "family", 128, 128, 64, "bf16_f32",
                residency="compute_oracle",
            ),
            [Schedule("s", 128, 128, 64, 2)],
            Hardware(
                "thor_t5000_sm110_20sm", 20, 1_575_000_000.0, "MAXN"
            ),
            [family],
        )
        self.assertIsNotNone(
            result.manifest_conditional_upper.performance_per_second
        )
        self.assertIsNone(
            result.domain_conditional_upper.performance_per_second
        )

    def test_manifest_upper_fails_closed_when_one_legal_schedule_is_unbounded(self) -> None:
        family = self.upper(
            upper_scope="schedule_family",
            applicable_mma_shapes=("m128n128k16",),
        )
        result = evaluate_manifest(
            Workload(
                "manifest", 256, 256, 64, "bf16_f32",
                residency="compute_oracle",
            ),
            [
                Schedule("n128", 128, 128, 64, 2),
                Schedule(
                    "n256", 128, 256, 64, 2,
                    mma_n=256, tmem_columns=256,
                ),
            ],
            Hardware(
                "thor_t5000_sm110_20sm", 20, 1_575_000_000.0, "MAXN"
            ),
            [family],
        )
        self.assertEqual(
            result.manifest_conditional_upper.status,
            "insufficient_evidence",
        )
        self.assertIsNone(result.conditional_schedule_id)

    def test_aggregate_compute_resource_requires_all_classical_scope(self) -> None:
        with self.assertRaisesRegex(
            ModelError, "requires upper_scope=all_classical"
        ):
            self.upper(
                resource="compute.total.bf16_f32",
                upper_scope="tensor_core_classical",
            ).validate()

    def test_domain_upper_rejects_schedule_specific_applicability(self) -> None:
        with self.assertRaisesRegex(
            ModelError, "schedule-specific applicability"
        ):
            self.upper(
                applicable_mma_shapes=("m128n128k16",),
            ).validate()

    def test_joint_pipeline_contract_requires_ncu_artifacts(self) -> None:
        row = Capacity(
            capacity_id="joint",
            resource="pipeline.joint.bf16_f32",
            rate_per_second=100.0,
            work_unit="flop",
            evidence_kind=EvidenceKind.MEASURED_JOINT,
            source_id="run",
            source_path="result.json",
            source_locator="median",
            qualification="closure_qualified",
            trial_count=10,
            artifact_paths=("result.json",),
            applicable_precision_ids=("bf16_f32",),
            applicable_sm_counts=(20,),
            applicable_hardware_ids=("thor_t5000_sm110_20sm",),
            applicable_operating_modes=("MAXN",),
            applicable_residencies=("hot_l2",),
            applicable_threads_per_cta=(128,),
            applicable_resident_ctas_per_sm=(1,),
            applicable_schedule_ids=("s",),
            applicable_workload_ids=("w",),
            timed_scope="device_kernel",
            residency_evidence_qualification="ncu_proven",
        )
        with self.assertRaisesRegex(ModelError, "NCU report and raw CSV"):
            row.validate()
        dataclasses.replace(
            row,
            artifact_paths=(
                "result.json", "cases/c/ncu/profile.ncu-rep",
                "cases/c/ncu/raw.csv",
            ),
        ).validate()

    def test_duplex_ratio_and_proxy_resources_fail_closed(self) -> None:
        l2 = Capacity(
            "l2_joint", "l2.duplex", 100.0, "byte",
            EvidenceKind.MEASURED_JOINT,
            "run", "result.json", "median",
            applicable_read_write_ratios=("3:1",),
        )
        proxy = dataclasses.replace(
            l2, capacity_id="cold_proxy", resource="hbm.duplex.proxy"
        )
        self.assertIsNone(_select_capacity(
            [l2], "l2.duplex", strict=False,
            required_read_write_ratio=(2.0, 1.0),
        ))
        self.assertIs(l2, _select_capacity(
            [l2], "l2.duplex", strict=False,
            required_read_write_ratio=(6.0, 2.0),
        ))
        self.assertIsNone(_select_capacity(
            [proxy], "hbm.duplex", strict=False,
            required_read_write_ratio=(3.0, 1.0),
        ))

    def test_capacity_loader_merges_wrapped_and_bare_payloads(self) -> None:
        row = {
            "capacity_id": "wrapped",
            "resource": "tensor.bf16.m128n128",
            "rate_per_second": 1.0,
            "work_unit": "flop",
            "evidence_kind": "measured_sustained",
            "source_id": "source",
            "source_path": "source.json",
            "source_locator": "median",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapped = root / "wrapped.json"
            bare = root / "bare.json"
            wrapped.write_text(json.dumps({"capacities": [row]}))
            bare.write_text(json.dumps([
                {**row, "capacity_id": "bare"}
            ]))
            capacities = load_capacity_files([wrapped, bare])
        self.assertEqual(
            [capacity.capacity_id for capacity in capacities],
            ["wrapped", "bare"],
        )

    def test_workload_manifest_requires_calibration_and_holdout(self) -> None:
        rows = workload_manifest_coverage([
            Workload(
                "cal", 1024, 1024, 1024, "bf16_f32",
                validation_split="calibration",
            ),
        ])
        bf16 = next(row for row in rows if row.precision_id == "bf16_f32")
        self.assertFalse(bf16.complete)
        self.assertIn("holdout_workload", bf16.missing)


if __name__ == "__main__":
    unittest.main()
