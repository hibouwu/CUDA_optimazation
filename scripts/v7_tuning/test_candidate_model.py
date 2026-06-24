#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_model import (
    IMPLEMENTATION_CONSTRAINT,
    LAUNCH_FEASIBILITY,
    RESOURCE_FEASIBILITY,
    CandidateParams,
    derive_candidate,
    validate_lane_coverage,
)


class V7CandidateModelTest(unittest.TestCase):
    def test_baseline_is_valid(self) -> None:
        c = derive_candidate(CandidateParams(128, 128, 16, 64, 64, 8, 4, 4))
        self.assertTrue(c.valid, c.rejection_reason)
        self.assertEqual(c.NumThreads, 128)
        self.assertEqual(c.WMITER, 1)
        self.assertEqual(c.WSUBM, 64)
        self.assertEqual(c.WSUBN, 16)
        self.assertEqual(c.accumulators, 128)
        self.assertEqual(c.reg_m, 8)
        self.assertEqual(c.reg_n, 16)
        self.assertEqual(c.R_static, 152)
        self.assertEqual(c.shared_bytes_model, 32768)

    def test_instantiated_probe_candidates_are_valid(self) -> None:
        params = [
            CandidateParams(128, 128, 16, 64, 64, 8, 4, 4),
            CandidateParams(128, 128, 8, 64, 64, 8, 4, 4),
            CandidateParams(128, 128, 16, 32, 64, 4, 4, 2),
            CandidateParams(128, 128, 16, 128, 32, 8, 4, 2),
            CandidateParams(128, 128, 16, 32, 128, 4, 8, 4),
            CandidateParams(128, 128, 16, 64, 64, 4, 4, 4),
        ]
        for p in params:
            with self.subTest(params=p):
                c = derive_candidate(p)
                self.assertTrue(c.valid, c.rejection_reason)

    def test_wmiter_non_integer_is_rejected(self) -> None:
        c = derive_candidate(CandidateParams(128, 128, 16, 64, 64, 8, 8, 4))
        self.assertFalse(c.valid)
        self.assertEqual(c.rejection_reason, "WMITER_not_integer")

    def test_non_positive_template_parameter_is_safely_rejected(self) -> None:
        c = derive_candidate(CandidateParams(128, 128, 16, 64, 64, 8, 0, 4))
        self.assertFalse(c.valid)
        self.assertEqual(c.rejection_reason, "non_positive_template_parameter")

    def test_lane_coverage_validator_rejects_non_32_coverage(self) -> None:
        self.assertFalse(validate_lane_coverage(16, 16, 4, 4))

    def test_tn_not_multiple_of_4_is_implementation_rejected(self) -> None:
        c = derive_candidate(
            CandidateParams(128, 96, 16, 64, 96, 4, 6, 1),
            allowed_thread_counts=None,
            max_accumulators=None,
        )
        self.assertFalse(c.valid)
        self.assertEqual(c.rejection_stage, IMPLEMENTATION_CONSTRAINT)
        self.assertEqual(c.rejection_reason, "TN_not_multiple_of_4")

    def test_a_copy_remainder_is_rejected(self) -> None:
        c = derive_candidate(
            CandidateParams(128, 96, 16, 32, 32, 4, 4, 1),
            allowed_thread_counts=None,
            max_accumulators=None,
        )
        self.assertFalse(c.valid)
        self.assertEqual(c.rejection_stage, IMPLEMENTATION_CONSTRAINT)
        self.assertEqual(c.rejection_reason, "A_copy_remainder")

    def test_b_copy_remainder_is_rejected(self) -> None:
        c = derive_candidate(
            CandidateParams(256, 128, 8, 64, 32, 4, 4, 1),
            allowed_thread_counts=None,
            max_accumulators=None,
        )
        self.assertFalse(c.valid)
        self.assertEqual(c.rejection_stage, IMPLEMENTATION_CONSTRAINT)
        self.assertEqual(c.rejection_reason, "B_copy_remainder")

    def test_static_shared_limit_rejects_candidate(self) -> None:
        c = derive_candidate(
            CandidateParams(128, 256, 32, 64, 128, 4, 8, 4),
            max_accumulators=None,
        )
        self.assertFalse(c.valid)
        self.assertEqual(c.rejection_stage, RESOURCE_FEASIBILITY)
        self.assertEqual(c.rejection_reason, "static_shared_memory_exceeds_49152")

    def test_launch_thread_limit_is_separately_classified(self) -> None:
        c = derive_candidate(
            CandidateParams(512, 512, 16, 32, 32, 4, 4, 1),
            allowed_thread_counts=None,
            max_accumulators=None,
        )
        self.assertFalse(c.valid)
        self.assertEqual(c.rejection_stage, LAUNCH_FEASIBILITY)
        self.assertEqual(c.rejection_reason, "NumThreads_exceeds_max_threads_per_block")

    def test_num_threads_is_derived_from_tile_shape(self) -> None:
        c = derive_candidate(CandidateParams(128, 128, 16, 32, 64, 4, 4, 2))
        self.assertTrue(c.valid, c.rejection_reason)
        self.assertEqual(c.NumThreads, 256)
        self.assertEqual(c.warps_per_block, 8)

    def test_low_reuse_candidate_can_remain_valid(self) -> None:
        c = derive_candidate(CandidateParams(128, 128, 8, 32, 64, 4, 4, 1))
        self.assertTrue(c.valid, c.rejection_reason)
        self.assertEqual(c.accumulators, 64)
        self.assertLess(c.combined_register_reuse, 4.0)

    def test_32_accumulator_candidate_can_remain_valid(self) -> None:
        c = derive_candidate(CandidateParams(64, 128, 16, 32, 32, 4, 4, 1))
        self.assertTrue(c.valid, c.rejection_reason)
        self.assertEqual(c.accumulators, 32)


if __name__ == "__main__":
    unittest.main()
