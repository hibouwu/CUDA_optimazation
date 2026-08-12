from __future__ import annotations

import unittest
from pathlib import Path

from scripts.sm110_gemm_model.io import load_capacities
from scripts.sm110_gemm_model.model import (
    Capacity,
    EvidenceKind,
    Hardware,
    Schedule,
    Workload,
    account_work,
    audit_inputs,
    evaluate,
    evaluate_manifest,
    precision_specs,
)
from scripts.sm110_gemm_model.observations import (
    ObservedBest,
    audit_observed_against_upper,
    summarize_observed_csvs,
)
from scripts.sm110_gemm_model.coverage import (
    common_resource_coverage,
    precision_coverage,
)
from scripts.sm110_gemm_model.tcgen05_descriptors import (
    DescriptorError,
    decode_fields,
    encode_block_scaled_fp4,
    encode_unscaled,
)
from microbench.sm110_gemm_campaign.run_compute_campaign import (
    PRECISIONS as CAMPAIGN_PRECISIONS,
    make_manifest as make_compute_campaign_manifest,
    source_for as make_compute_campaign_source,
)


ROOT = Path(__file__).resolve().parents[2]
CAPACITY_PATH = Path(__file__).resolve().parent / "profiles" / "capacities.json"


class WorkAccountingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = Schedule("s", 128, 128, 64, 2, tail_policy="pad")
        self.precisions = precision_specs()

    def test_beta_zero_eliminates_c_read(self) -> None:
        workload = Workload("w", 128, 128, 64, "fp16_f32", beta=0.0)
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        self.assertEqual(work.c_read_bytes_min, 0)

    def test_beta_nonzero_requires_c_read(self) -> None:
        workload = Workload("w", 128, 128, 64, "fp16_f32", beta=1.0)
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        self.assertEqual(work.c_read_bytes_min, 128 * 128 * 4)

    def test_padded_issue_work_is_not_less_than_useful_work(self) -> None:
        workload = Workload("tail", 129, 130, 65, "fp16_f32")
        work = account_work(workload, self.schedule, self.precisions["fp16_f32"])
        self.assertGreater(work.issued_compute_work, work.useful_compute_work)
        self.assertLess(work.shape_efficiency, 1.0)

    def test_fp6_packed_storage_is_three_quarters_byte(self) -> None:
        workload = Workload("fp6", 128, 128, 64, "e3m2_f32")
        work = account_work(workload, self.schedule, self.precisions["e3m2_f32"])
        self.assertEqual(work.input_value_bytes_min, (128 * 64 + 64 * 128) * 0.75)

    def test_fp6_decompression_padding_is_not_charged_as_logical_storage(self) -> None:
        workload = Workload("fp6-cp", 128, 128, 64, "e3m2_f32")
        packed = account_work(
            workload,
            self.schedule,
            self.precisions["e3m2_f32"],
        )
        decompressed = account_work(
            workload,
            Schedule(
                "fp6-cp",
                128,
                128,
                64,
                2,
                tail_policy="pad",
                input_transport_layout="b6x16_p32",
            ),
            self.precisions["e3m2_f32"],
        )
        self.assertEqual(decompressed.input_value_bytes_min, packed.input_value_bytes_min)
        self.assertGreater(decompressed.tma_input_bytes, packed.tma_input_bytes)
        self.assertEqual(
            decompressed.tma_input_bytes,
            128 * 64 + 64 * 128,
        )

    def test_transport_layout_must_match_precision(self) -> None:
        with self.assertRaisesRegex(Exception, "only valid for six-bit"):
            account_work(
                Workload("bad-layout", 128, 128, 64, "fp16_f32"),
                Schedule(
                    "bad-layout",
                    128,
                    128,
                    64,
                    2,
                    input_transport_layout="b6x16_p32",
                ),
                self.precisions["fp16_f32"],
            )

    def test_unmodeled_schedule_families_fail_closed(self) -> None:
        workload = Workload("w", 128, 128, 64, "fp16_f32")
        for schedule, pattern in (
            (Schedule("g2", 128, 128, 64, 2, cta_group=2), "CTA-group-2"),
            (Schedule("split", 128, 128, 64, 2, split_k=2), "split-K"),
            (Schedule("persistent", 128, 128, 64, 2, persistent=True), "persistent"),
        ):
            with self.subTest(schedule=schedule.schedule_id):
                with self.assertRaisesRegex(Exception, pattern):
                    account_work(workload, schedule, self.precisions["fp16_f32"])

    def test_int8_n_shape_uses_table39_nonuniform_rule(self) -> None:
        workload = Workload("i8", 128, 40, 32, "s8_s32")
        with self.assertRaisesRegex(Exception, "MMA N=40 is invalid"):
            account_work(
                workload,
                Schedule("bad-i8-n", 128, 40, 32, 2, mma_n=40, tmem_columns=64),
                self.precisions["s8_s32"],
            )

    def test_quantized_scale_bytes_are_separate_from_values(self) -> None:
        workload = Workload("nv", 128, 128, 64, "nvfp4_f32")
        work = account_work(workload, self.schedule, self.precisions["nvfp4_f32"])
        self.assertGreater(work.input_scale_bytes_min, 0)
        self.assertEqual(work.accumulator_readback_bytes, 128 * 128 * 4)


class Tcgen05DescriptorTest(unittest.TestCase):
    def test_table42_type_codes_cover_all_unscaled_v1_precisions(self) -> None:
        cases = {
            ("f16", "f16"): 0,
            ("f16", "bf16"): 1,
            ("tf32", "tf32"): 2,
            ("f8f6f4", "e4m3"): 0,
            ("f8f6f4", "e5m2"): 1,
            ("f8f6f4", "e2m3"): 3,
            ("f8f6f4", "e3m2"): 4,
            ("f8f6f4", "e2m1"): 5,
            ("i8", "u8"): 0,
            ("i8", "s8"): 1,
        }
        for (kind, element_type), expected in cases.items():
            with self.subTest(kind=kind, element_type=element_type):
                record = encode_unscaled(
                    kind, m=128, n=256, a_type=element_type
                )
                fields = decode_fields(record.value_u32, kind=kind)
                self.assertEqual(fields["atype"], expected)
                self.assertEqual(fields["btype"], expected)
                self.assertEqual(fields["m"], 128)
                self.assertEqual(fields["n"], 256)

    def test_table44_does_not_reuse_table42_e2m1_code(self) -> None:
        raw = encode_unscaled("f8f6f4", m=128, n=256, a_type="e2m1")
        nvfp4 = encode_block_scaled_fp4(
            "mxf4nvf4", m=128, n=256, scale_type="ue4m3"
        )
        self.assertEqual(decode_fields(raw.value_u32, kind="f8f6f4")["atype"], 5)
        nv_fields = decode_fields(nvfp4.value_u32, kind="mxf4nvf4")
        self.assertEqual(nv_fields["atype"], 1)
        self.assertEqual(nv_fields["scale_type"], 0)

    def test_mxfp4_uses_ue8m0_scale_bit(self) -> None:
        mxfp4 = encode_block_scaled_fp4(
            "mxf4", m=128, n=128, scale_type="ue8m0"
        )
        fields = decode_fields(mxfp4.value_u32, kind="mxf4")
        self.assertEqual(fields["scale_type"], 1)
        self.assertEqual(fields["k_is_96"], 0)

    def test_invalid_table44_scale_contract_is_rejected(self) -> None:
        with self.assertRaises(DescriptorError):
            encode_block_scaled_fp4(
                "mxf4", m=128, n=128, scale_type="ue4m3"
            )

    def test_sm110_rejects_sm103a_only_k96(self) -> None:
        with self.assertRaisesRegex(DescriptorError, "sm_103a-only"):
            encode_block_scaled_fp4(
                "mxf4", m=128, n=128, scale_type="ue8m0", k=96
            )

    def test_i8_descriptor_rejects_n40(self) -> None:
        with self.assertRaisesRegex(DescriptorError, "nonuniform Table-39"):
            encode_unscaled("i8", m=128, n=40, a_type="s8")


class ComputeCampaignTest(unittest.TestCase):
    def test_manifest_has_all_72_unique_cases(self) -> None:
        manifest = make_compute_campaign_manifest()
        ids = [row["case_id"] for row in manifest]
        self.assertEqual(len(ids), 72)
        self.assertEqual(len(set(ids)), 72)
        self.assertEqual(
            {row["precision"]["precision_id"] for row in manifest},
            {precision.precision_id for precision in CAMPAIGN_PRECISIONS},
        )

    def test_corrected_nvfp4_source_uses_table44_descriptor(self) -> None:
        precision = next(
            row for row in CAMPAIGN_PRECISIONS if row.precision_id == "nvfp4_f32"
        )
        source, descriptor = make_compute_campaign_source(
            precision, 128, 256, "full_sm_4warp_block", "nv"
        )
        fields = decode_fields(descriptor.value_u32, kind="mxf4nvf4")
        self.assertEqual(fields["atype"], 1)
        self.assertEqual(fields["scale_type"], 0)
        self.assertIn("kind::mxf4nvf4.block_scale.block16", source)
        self.assertNotIn("tcgen05.mma.sp", source)

    def test_fp6_direct_smem_uses_byte_container_descriptor(self) -> None:
        precision = next(
            row for row in CAMPAIGN_PRECISIONS if row.precision_id == "e3m2_f32"
        )
        source, _ = make_compute_campaign_source(
            precision, 128, 128, "single_warp_block", "fp6"
        )
        self.assertIn("kLogicalBits = 6", source)
        self.assertIn("kDescriptorBits = 8", source)
        self.assertIn("smem_desc(a,8,4)", source)


class EvidenceSemanticsTest(unittest.TestCase):
    def test_unimplemented_epilogue_fails_closed(self) -> None:
        with self.assertRaisesRegex(Exception, "work and I/O contract is not implemented"):
            evaluate(
                Workload(
                    "relu",
                    128,
                    128,
                    64,
                    "bf16_f32",
                    epilogue="relu",
                    residency="compute_oracle",
                ),
                Schedule("s", 128, 128, 64, 2),
                Hardware("h", 20, 1.0),
                [],
            )

    def test_measured_capacity_is_not_used_as_strict_upper(self) -> None:
        capacities = [
            Capacity(
                "measured",
                "tensor.bf16",
                100.0,
                "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=1",
            ),
            Capacity(
                "readback",
                "tmem.readback",
                100.0,
                "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=2",
            ),
        ]
        result = evaluate(
            Workload("w", 128, 128, 64, "bf16_f32", residency="compute_oracle"),
            Schedule("s", 128, 128, 64, 2),
            Hardware("h", 20, 1.0),
            capacities,
        )
        self.assertEqual(result.conditional_upper.status, "insufficient_evidence")
        self.assertIsNone(result.conditional_upper.performance_per_second)
        self.assertIsNotNone(result.empirical_envelope.performance_per_second)

    def test_larger_proven_rate_cannot_lower_performance_upper(self) -> None:
        workload = Workload("w", 128, 128, 64, "bf16_f32", residency="compute_oracle")
        schedule = Schedule("s", 128, 128, 64, 2)
        hardware = Hardware("h", 20, 1.0)

        def cap(rate: float) -> Capacity:
            return Capacity(
                f"upper_{rate}",
                "tensor.bf16",
                rate,
                "flop",
                EvidenceKind.DERIVED_UPPER,
                "source",
                "source.csv",
                "row=1",
            )

        slow = evaluate(workload, schedule, hardware, [cap(100.0)])
        fast = evaluate(workload, schedule, hardware, [cap(200.0)])
        self.assertGreaterEqual(
            fast.conditional_upper.performance_per_second,
            slow.conditional_upper.performance_per_second,
        )

    def test_strict_layer_does_not_infer_per_group_rate_from_gpu_rate(self) -> None:
        capacity = Capacity(
            "gpu_upper",
            "tensor.bf16",
            100.0,
            "flop",
            EvidenceKind.DERIVED_UPPER,
            "source",
            "source.csv",
            "row=1",
        )
        measured = Capacity(
            "gpu_measured",
            "tensor.bf16",
            90.0,
            "flop",
            EvidenceKind.MEASURED_SUSTAINED,
            "source",
            "source.csv",
            "row=2",
        )
        readback = Capacity(
            "readback",
            "tmem.readback",
            100.0,
            "byte",
            EvidenceKind.MEASURED_SUSTAINED,
            "source",
            "source.csv",
            "row=3",
        )
        result = evaluate(
            Workload("tiny", 128, 128, 64, "bf16_f32", residency="compute_oracle"),
            Schedule("s", 128, 128, 64, 2),
            Hardware("h", 20, 1.0),
            [capacity, measured, readback],
        )
        self.assertNotIn("parallel_span", result.conditional_upper.resource_seconds)
        self.assertIn("parallel_span", result.empirical_envelope.resource_seconds)

    def test_profiler_peak_requires_condition(self) -> None:
        capacity = Capacity(
            "peak",
            "l2.read",
            100.0,
            "byte",
            EvidenceKind.PROFILER_MODEL_PEAK,
            "source",
            "source.csv",
            "row=1",
        )
        findings = audit_inputs([capacity])
        self.assertIn("conditional_peak_without_condition", {row["code"] for row in findings})

    def test_repo_capacity_sources_exist(self) -> None:
        findings = audit_inputs(load_capacities(CAPACITY_PATH), repo_root=ROOT)
        self.assertEqual(findings, [])

    def test_csv_locator_value_mismatch_is_rejected(self) -> None:
        capacity = Capacity(
            "bad_value",
            "tensor.bf16",
            100.0,
            "flop",
            EvidenceKind.MEASURED_SUSTAINED,
            "source",
            "microbench/mma_compute_only/plots/benchmark_results.csv",
            "precision=BF16,warp_num=4,shape=M128N256K16,tflops",
            original_value=1.0,
            original_unit="TFLOP/s",
        )
        findings = audit_inputs([capacity], repo_root=ROOT)
        self.assertIn("csv_original_value_mismatch", {row["code"] for row in findings})


class SnapshotEvaluationTest(unittest.TestCase):
    def test_bf16_snapshot_exposes_incomplete_empirical_layer(self) -> None:
        result = evaluate(
            Workload("bf16", 1024, 1024, 1024, "bf16_f32"),
            Schedule("s", 128, 128, 64, 2, tail_policy="pad"),
            Hardware("thor", 20, 1.575e9),
            load_capacities(CAPACITY_PATH),
        )
        self.assertIn(result.conditional_upper.status, {"ok", "partial"})
        self.assertEqual(result.empirical_envelope.status, "insufficient_evidence")
        self.assertIsNotNone(result.conditional_upper.performance_per_second)
        self.assertIsNone(result.empirical_envelope.performance_per_second)

    def test_int8_uses_integer_operation_units(self) -> None:
        capacity = Capacity(
            "int8",
            "tensor.s8",
            100.0,
            "operation",
            EvidenceKind.DERIVED_UPPER,
            "source",
            "source.csv",
            "row=1",
        )
        result = evaluate(
            Workload("s8", 128, 128, 32, "s8_s32", residency="compute_oracle"),
            Schedule("s", 128, 128, 32, 2),
            Hardware("thor", 20, 1.575e9),
            [capacity],
        )
        self.assertEqual(result.work.compute_work_unit, "operation")
        self.assertEqual(result.conditional_upper.performance_unit, "operation/s")

    def test_manifest_rejects_unproven_exact_tail_schedule(self) -> None:
        workload = Workload(
            "bf16", 129, 129, 65, "bf16_f32", residency="compute_oracle"
        )
        hardware = Hardware("thor", 20, 1.575e9)
        capacities = [
            Capacity(
                "compute",
                "tensor.bf16",
                100.0,
                "flop",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=1",
            ),
            Capacity(
                "readback",
                "tmem.readback",
                100.0,
                "byte",
                EvidenceKind.MEASURED_SUSTAINED,
                "source",
                "source.csv",
                "row=2",
            ),
        ]
        envelope = evaluate_manifest(
            workload,
            [
                Schedule("padded", 128, 128, 64, 2, tail_policy="pad"),
                Schedule("exact", 128, 128, 64, 2, tail_policy="exact"),
                Schedule("illegal", 128, 128, 63, 2),
            ],
            hardware,
            capacities,
        )
        self.assertEqual(envelope.valid_schedule_count, 1)
        self.assertEqual(envelope.rejected_schedule_count, 2)
        self.assertEqual(envelope.empirical_schedule_id, "padded")
        self.assertTrue(
            any("exact tail requires" in row["reason"] for row in envelope.rejected)
        )


class ObservationTest(unittest.TestCase):
    def test_snapshot_observations_require_ten_matched_trials(self) -> None:
        rows = summarize_observed_csvs(
            [
                ROOT / "results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv",
                ROOT / "results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv",
            ],
            repo_root=ROOT,
        )
        keys = {(row.precision_id, row.n) for row in rows}
        self.assertIn(("fp16_f32", 1024), keys)
        self.assertIn(("e4m3_f32", 1024), keys)
        self.assertIn(("s8_s32", 1024), keys)
        self.assertIn(("nvfp4_f32", 1024), keys)
        nvfp4 = next(row for row in rows if row.precision_id == "nvfp4_f32")
        self.assertEqual(
            nvfp4.performance_reference_relation, "cross_precision_denominator"
        )

    def test_upper_violation_is_an_error(self) -> None:
        row = ObservedBest(
            "obs",
            "bf16_f32",
            128,
            128,
            128,
            "backend",
            "reference",
            "same_precision",
            10,
            10,
            100.0,
            110.0,
            90.0,
            "flop/s",
            "results.csv",
        )
        findings = audit_observed_against_upper(row, 100.0, relative_tolerance=0.01)
        self.assertEqual(findings[0]["code"], "observed_exceeds_conditional_upper")

    def test_snapshot_coverage_exposes_missing_precisions(self) -> None:
        capacities = load_capacities(CAPACITY_PATH)
        observed = summarize_observed_csvs(
            [
                ROOT / "results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv",
                ROOT / "results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv",
            ],
            repo_root=ROOT,
        )
        rows = {row.precision_id: row for row in precision_coverage(capacities, observed)}
        self.assertFalse(rows["tf32_f32"].numeric_closure)
        self.assertIn("empirical_compute_rate", rows["tf32_f32"].missing)
        self.assertFalse(rows["nvfp4_f32"].same_precision_performance_denominator)
        self.assertFalse(rows["e4m3_f32"].numeric_closure)
        self.assertIn(
            "closure_qualified_compute_rate", rows["e4m3_f32"].missing
        )
        common = common_resource_coverage(capacities)
        self.assertFalse(common["tmem.readback"])


if __name__ == "__main__":
    unittest.main()
