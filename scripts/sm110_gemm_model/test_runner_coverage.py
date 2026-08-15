from __future__ import annotations

import unittest

from scripts.sm110_gemm_model.runner_coverage import audit_runner_coverage


class RunnerCoverageAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.report = audit_runner_coverage()

    def test_new_payload_and_duplex_surfaces_are_complete(self) -> None:
        self.assertTrue(self.report["tma_payload_surface"]["complete"])
        self.assertEqual(self.report["tma_payload_surface"]["case_count"], 10)
        self.assertTrue(self.report["memory_duplex_surface"]["complete"])
        self.assertEqual(self.report["memory_duplex_surface"]["case_count"], 21)
        self.assertTrue(self.report["payload_duplex_runner_definition_complete"])

    def test_audit_fails_closed_on_known_topology_and_joint_gaps(self) -> None:
        topology = self.report["exact_tma_topology_surface"]
        self.assertEqual(topology["required_schedule_precision_pair_count"], 28)
        self.assertEqual(len(topology["covered_schedule_precision_pairs"]), 2)
        self.assertFalse(topology["complete"])
        self.assertFalse(self.report["joint_pipeline_surface"]["complete"])
        self.assertFalse(
            self.report["all_performance_parameter_runner_definition_complete"]
        )

    def test_zero_fixed_cost_is_an_upper_relaxation_not_a_measurement(self) -> None:
        fixed = self.report["fixed_cost"]
        self.assertTrue(fixed["upper_relaxation_defined"])
        self.assertFalse(fixed["measured_runner_defined"])
        self.assertFalse(fixed["required_for_no_waste_upper"])
        self.assertTrue(fixed["required_for_wall_time_prediction"])

    def test_full_gemm_runner_only_covers_supported_ready_paths(self) -> None:
        full = self.report["full_gemm_validation"]
        self.assertTrue(full["ready_paths_covered"])
        self.assertFalse(full["all_precisions_covered"])


if __name__ == "__main__":
    unittest.main()
