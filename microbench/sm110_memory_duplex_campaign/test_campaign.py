from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from microbench.sm110_memory_duplex_campaign.run_memory_duplex_campaign import (
    HBM_RATIOS, L2_RATIOS, TARGET_SQUARE_SHAPES, cases,
    derive_ratio_contracts, duplex_sass_block, find_ncu_row, validate_trial,
)


class MemoryDuplexContractTest(unittest.TestCase):
    def test_manifest_covers_all_declared_ratios(self) -> None:
        self.assertEqual((HBM_RATIOS, L2_RATIOS), derive_ratio_contracts())
        self.assertIn((27, 16), L2_RATIOS)
        self.assertIn((27, 8), L2_RATIOS)
        self.assertIn((27, 4), L2_RATIOS)
        self.assertNotIn((13, 2), L2_RATIOS)
        self.assertEqual(TARGET_SQUARE_SHAPES, (1024, 2048, 4096))
        manifest = cases(64 << 20)
        self.assertEqual(len(manifest), len(HBM_RATIOS) + len(L2_RATIOS))
        self.assertEqual(len({case["id"] for case in manifest}), len(manifest))
        self.assertEqual(
            {(case["read_operations"], case["write_operations"])
             for case in manifest if case["residency"] == "cold_hbm"},
            set(HBM_RATIOS),
        )
        self.assertEqual(
            {(case["read_operations"], case["write_operations"])
             for case in manifest if case["residency"] == "hot_l2"},
            set(L2_RATIOS),
        )

    def test_trial_arithmetic_is_recomputed(self) -> None:
        case = cases(64 << 20)[0]
        base_bytes = 20 * 4 * 256 * int(case["iterations"]) * 128
        read_bytes = base_bytes * int(case["read_operations"])
        write_bytes = base_bytes * int(case["write_operations"])
        elapsed_ns = 1_000_000
        rate = (read_bytes + write_bytes) * 1.0e9 / elapsed_ns
        fields = {
            "case_id": str(case["id"]), "direction": "duplex", "residency": "hbm",
            "sm_count": "20", "unique_smid_count": "20", "blocks": "80",
            "blocks_per_sm": "4",
            "threads": "256",
            "read_operations_per_iteration": str(case["read_operations"]),
            "write_operations_per_iteration": str(case["write_operations"]),
            "iterations": str(case["iterations"]), "warmup_iterations": "64",
            "working_set_bytes": str(case["working_set_bytes_per_direction"]),
            "read_working_set_bytes": str(case["working_set_bytes_per_direction"]),
            "write_working_set_bytes": str(case["working_set_bytes_per_direction"]),
            "requested_read_bytes": str(read_bytes),
            "requested_write_bytes": str(write_bytes),
            "requested_bytes": str(read_bytes + write_bytes),
            "globaltimer_elapsed_ns": str(elapsed_ns), "bytes_per_second": str(rate),
        }
        self.assertEqual(validate_trial(case, fields), rate)

    def test_trial_rejects_wrong_ratio(self) -> None:
        case = cases(64 << 20)[0]
        fields = {
            "case_id": str(case["id"]), "direction": "duplex", "residency": "hbm",
            "sm_count": "20", "unique_smid_count": "20", "blocks": "80",
            "blocks_per_sm": "4",
            "threads": "256", "read_operations_per_iteration": "1",
            "write_operations_per_iteration": "1",
            "iterations": str(case["iterations"]), "warmup_iterations": "64",
            "working_set_bytes": str(case["working_set_bytes_per_direction"]),
            "read_working_set_bytes": str(case["working_set_bytes_per_direction"]),
            "write_working_set_bytes": str(case["working_set_bytes_per_direction"]),
            "requested_read_bytes": "100", "requested_write_bytes": "200",
            "requested_bytes": "300", "globaltimer_elapsed_ns": "100",
            "bytes_per_second": "3.0e9",
        }
        with self.assertRaises(RuntimeError):
            validate_trial(case, fields)

    def test_ncu_parser_selects_timed_duplex_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            path.write_text(
                '==PROF==,Connected\n'
                'ID,Kernel Name,gpu__time_duration.sum,lts__t_bytes.sum\n'
                '1,initialize,10,20\n'
                '2,duplex_kernel,100,200\n'
                '3,duplex_kernel,300,400\n'
            )
            row = find_ncu_row(path)
            self.assertEqual(row["ID"], "3")
            self.assertEqual(row["gpu__time_duration.sum"], "300")

    def test_sass_tokens_must_be_in_duplex_function(self) -> None:
        sass = """
Function : unrelated
 LDG.E.128
 STG.E.128
Function : namespace::duplex_kernel
 LDG.E.128
 STG.E.128
Function : later
 NOP
"""
        block = duplex_sass_block(sass)
        self.assertIn("LDG.E.128", block)
        self.assertIn("STG.E.128", block)
        self.assertNotIn("Function : unrelated", block)


if __name__ == "__main__":
    unittest.main()
