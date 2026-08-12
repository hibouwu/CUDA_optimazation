#!/usr/bin/env python3
"""Unit tests for full-GEMM evidence parsing and function-scoped SASS gates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import run_full_gemm_campaign as campaign


class CampaignEvidenceTests(unittest.TestCase):
    def fp16_case(self) -> dict[str, object]:
        return next(case for case in campaign.CASES
                    if case["id"] == "fp16_f32_n1024_tc5b")

    def s8_case(self) -> dict[str, object]:
        return next(case for case in campaign.CASES
                    if case["id"] == "s8_s32_n1024_q15")

    def test_fp16_trial_is_recomputed_from_times(self) -> None:
        case = self.fp16_case()
        stdout = (
            "M=1024, N=1024, K=1024\n"
            "reference_contract=fp16_f32_cpu_samples reference_sample_count=64 "
            "reference_mismatch_count=0\n"
            "numerical_contract=fp_accumulator mismatch_count=0 "
            "max_abs_error=0.01 max_tolerance_ratio=0.5 atol=0.02 rtol=0.002\n"
        )
        csv_text = (
            "BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,RatioToReference,Matched\n"
            "cublas_tc,library,1024,fp16->fp32,library,0.02,1,1,1\n"
            "tc5b,candidate,1024,fp16->fp32,library,0.04,1,0.5,1\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sgemm_sm110_benchmark.csv").write_text(csv_text)
            parsed = campaign.parse_trial(case, root, stdout)
        work = 2 * 1024 ** 3
        self.assertEqual(parsed["custom_rate_per_second"], work * 1000 / 0.04)
        self.assertEqual(parsed["reference_rate_per_second"], work * 1000 / 0.02)
        self.assertEqual(parsed["ratio_to_reference"], 0.5)

    def test_s8_trial_keeps_operation_unit(self) -> None:
        case = self.s8_case()
        work = 2 * 1024 ** 3
        custom_ms, reference_ms = 0.04, 0.02
        stdout = (
            "N=1024, backend=int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol\n"
            "reference_contract=s8_s32_cpu_samples reference_sample_count=64 "
            "reference_mismatch_count=0 reference_time_ms=0.02\n"
            "numerical_contract=s8_s32_exact mismatch_count=0 max_abs_error=0\n"
            "backend_id=int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol "
            f"time_ms={custom_ms} work_unit=operation "
            f"rate_per_second={work * 1000 / custom_ms} "
            f"reference_rate_per_second={work * 1000 / reference_ms} matched=1\n"
        )
        csv_text = (
            "BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,RatioToReference,Matched\n"
            "int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol,candidate,1024,"
            "int8->int32,library,0.04,1,0.5,1\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "quant_sm110_benchmark.csv").write_text(csv_text)
            parsed = campaign.parse_trial(case, root, stdout)
        self.assertEqual(parsed["fields"]["work_unit"], "operation")

    def test_wrong_numerical_contract_is_rejected(self) -> None:
        case = self.s8_case()
        stdout = (
            "N=1024\nreference_contract=s8_s32_cpu_samples "
            "reference_sample_count=64 reference_mismatch_count=0\n"
            "numerical_contract=fp8_e4m3_f32 mismatch_count=0\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "numerical contract mismatch"):
                campaign.parse_trial(case, Path(temporary), stdout)

    def test_sass_tokens_must_be_in_matching_function(self) -> None:
        case = self.s8_case()
        fake = (
            "Function : unrelated\n IMMA.16816.S8.S8\n STG\n"
            f"Function : {case['function_substring']}\n NOP\n"
        )
        with self.assertRaisesRegex(RuntimeError, "no function block"):
            campaign.matching_sass_evidence(fake, case)


if __name__ == "__main__":
    unittest.main()
