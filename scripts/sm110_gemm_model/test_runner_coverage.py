from __future__ import annotations

import re
import unittest
from pathlib import Path

from scripts.sm110_gemm_model.runner_coverage import audit_runner_coverage


class RunnerCoverageAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.report = audit_runner_coverage()
        self.repo = Path(__file__).resolve().parents[2]
        self.document_path = (
            self.repo / "Docs/blackwell_tensorcore/"
            "thor_sm110_gemm_microbenchmark_experiment.md"
        )

    def test_new_payload_and_duplex_surfaces_are_complete(self) -> None:
        self.assertTrue(self.report["tma_payload_surface"]["complete"])
        self.assertEqual(self.report["tma_payload_surface"]["case_count"], 10)
        self.assertTrue(self.report["memory_duplex_surface"]["complete"])
        self.assertEqual(self.report["memory_duplex_surface"]["case_count"], 21)
        self.assertTrue(
            self.report["memory_duplex_surface"]["cold_dram_read_proxy_complete"])
        self.assertFalse(
            self.report["memory_duplex_surface"]["cold_external_write_bytes_closed"])
        self.assertEqual(
            self.report["memory_duplex_surface"]["qualification"],
            "cold_dram_read_plus_write_path_proxy",
        )
        self.assertEqual(
            self.report["memory_duplex_surface"]["max_operation_groups"], 128)
        self.assertEqual(
            self.report["memory_duplex_surface"]
                       ["max_required_operation_groups"],
            96,
        )
        self.assertTrue(self.report["payload_duplex_runner_definition_complete"])
        self.assertFalse(self.report["cold_external_write_bytes_closed"])
        self.assertFalse(self.report["physical_memory_duplex_closed"])

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

    def test_experiment_document_tracks_machine_coverage_counts(self) -> None:
        document = self.document_path.read_text()
        for expected in (
            "12/12 precision",
            "10/10 case",
            "21/21 case",
            "2/28 pair",
            "5/12 precision",
            "7/12 precision",
            "runner=0",
        ):
            self.assertIn(expected, document)
        self.assertIn("1:4, 17:64, 9:32, 3:8, 1:2, 1:1, 2:1", document)
        self.assertIn("27:16, 3:1, 27:8, 4:1, 6:1, 27:4, 8:1", document)

    def test_experiment_document_defines_core_symbols_at_first_use(self) -> None:
        document = self.document_path.read_text()
        for symbol in (
            r"\(M\)",
            r"\(Q_r(x,w)\)",
            r"\(S=20\ \mathrm{SM/GPU}\)",
            r"\(p\in\{4,8,16,32,64\}\ \mathrm{KiB/request}\)",
            r"\(r\)",
            r"\(B_R\)",
            r"\(L_r\)",
        ):
            definition = re.search(r"定义\s*(" + re.escape(symbol) + r")", document)
            self.assertIsNotNone(definition, symbol)
            self.assertEqual(
                document.index(symbol),
                definition.start(1),
                symbol,
            )

    def test_experiment_document_local_links_resolve(self) -> None:
        document = self.document_path.read_text()
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", document)
        self.assertGreater(len(links), 15)
        for target in links:
            if target.startswith(("https://", "http://", "#")):
                continue
            relative = target.split("#", 1)[0]
            resolved = (self.document_path.parent / relative).resolve()
            self.assertTrue(resolved.exists(), f"missing link target: {target}")


if __name__ == "__main__":
    unittest.main()
