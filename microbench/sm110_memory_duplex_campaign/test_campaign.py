from __future__ import annotations

import csv
import hashlib
import json
import unittest
import tempfile
from pathlib import Path

from microbench.sm110_memory_duplex_campaign.run_memory_duplex_campaign import (
    HBM_RATIOS, L2_RATIOS, NCU_BASE_UNITS, NCU_METRICS,
    TARGET_SQUARE_SHAPES, cases,
    derive_ratio_contracts, duplex_sass_block, find_ncu_row, validate_trial,
    ncu_number,
)
from microbench.sm110_memory_duplex_campaign.audit_campaign import (
    audit_ncu,
    find_ncu_row as audit_find_ncu_row,
    ncu_number as audit_ncu_number,
)


class MemoryDuplexContractTest(unittest.TestCase):
    def test_ncu_contract_uses_thor_available_proxy_metrics(self) -> None:
        self.assertNotIn("dram__bytes_op_read.sum", NCU_METRICS)
        self.assertNotIn("dram__bytes_op_write.sum", NCU_METRICS)
        self.assertIn("lts__t_sectors_op_read_lookup_miss.sum", NCU_METRICS)
        self.assertIn("lts__t_sectors_op_write.sum", NCU_METRICS)

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
        cold = [case for case in manifest if case["residency"] == "cold_hbm"]
        self.assertTrue(all(".proxy." in str(case["resource"]) for case in cold))
        self.assertTrue(all(
            case["evidence_contract"]
            == "cold_read_l2_miss_plus_write_l2_issue_proxy"
            and case["external_write_bytes_proven"] is False
            for case in cold
        ))

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
            header = ["ID", "Kernel Name", *NCU_METRICS]
            units = ["", "", *(NCU_BASE_UNITS[name] for name in NCU_METRICS)]
            def row(identifier, kernel, duration, lts_bytes, remainder):
                values = {name: remainder for name in NCU_METRICS}
                values["gpu__time_duration.sum"] = duration
                values["lts__t_bytes.sum"] = lts_bytes
                return [identifier, kernel, *(values[name] for name in NCU_METRICS)]

            rows = [
                row("1", "initialize", "10", "20", "0"),
                row("2", "duplex_kernel", "100,000", "200,000", "1"),
                row("3", "duplex_kernel", "300,000", "400,000", "2"),
            ]
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["==PROF== Connected"])
                writer.writerow(header)
                writer.writerow(units)
                writer.writerows(rows)
            for parser in (find_ncu_row, audit_find_ncu_row):
                row = parser(path)
                self.assertEqual(row["ID"], "3")
                self.assertEqual(row["gpu__time_duration.sum"], "300,000")
            self.assertEqual(ncu_number("400,000"), 400_000.0)
            self.assertEqual(audit_ncu_number("400,000"), 400_000.0)

    def test_ncu_parser_rejects_malformed_grouping_and_scaled_units(self) -> None:
        for parser in (ncu_number, audit_ncu_number):
            with self.assertRaises(ValueError):
                parser("40,00")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            header = ["ID", "Kernel Name", *NCU_METRICS]
            units = ["", "", "ms", *("Mbyte" for _ in NCU_METRICS[1:])]
            values = ["1", "duplex_kernel", *("1" for _ in NCU_METRICS)]
            with path.open("w", newline="") as handle:
                csv.writer(handle).writerows((header, units, values))
            with self.assertRaisesRegex(RuntimeError, "expected 'ns'"):
                find_ncu_row(path)
            with self.assertRaisesRegex(ValueError, "expected 'ns'"):
                audit_find_ncu_row(path)

    def test_independent_auditor_reparses_grouped_raw_metrics(self) -> None:
        case = cases()[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ncu_dir = root / "cases" / str(case["id"]) / "ncu"
            ncu_dir.mkdir(parents=True)
            report = ncu_dir / "profile.ncu-rep"
            raw = ncu_dir / "raw.csv"
            stderr = ncu_dir / "stderr.log"
            report.write_text("NCU")
            stderr.write_text("")

            requested_read = 64 << 20
            requested_write = requested_read * int(case["write_operations"]) // int(
                case["read_operations"])
            values = {
                "gpu__time_duration.sum": 8_000_000.0,
                "lts__t_bytes.sum": float(requested_read + requested_write),
                "lts__t_sectors_op_read.sum": requested_read / 32,
                "lts__t_sectors_op_write.sum": requested_write / 32,
                "lts__t_sectors_op_read_lookup_hit.sum": 0.0,
                "lts__t_sectors_op_read_lookup_miss.sum": requested_read / 32,
            }
            header = ["ID", "Kernel Name", *NCU_METRICS]
            units = ["", "", *(NCU_BASE_UNITS[name] for name in NCU_METRICS)]
            data = [
                "1", "duplex_kernel",
                *(f"{int(values[name]):,}" for name in NCU_METRICS),
            ]
            with raw.open("w", newline="") as handle:
                csv.writer(handle).writerows((header, units, data))

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            ncu = {
                "returncode": 0,
                "metrics": list(NCU_METRICS),
                "iterations": 1,
                "requested_read_bytes": requested_read,
                "requested_write_bytes": requested_write,
                "values": values,
                "evidence_contract": case["evidence_contract"],
                "external_write_bytes_proven": False,
                "cold_read_miss_proxy_bytes": requested_read,
                "cold_read_miss_proxy_to_requested": 1.0,
                "report_path": "ncu/profile.ncu-rep",
                "report_sha256": digest(report),
                "raw_path": "ncu/raw.csv",
                "raw_sha256": digest(raw),
                "stderr_path": "ncu/stderr.log",
                "stderr_sha256": digest(stderr),
            }
            (ncu_dir / "summary.json").write_text(json.dumps(ncu, sort_keys=True))
            result = {"ncu": ncu}
            errors: list[str] = []
            audit_ncu(root, case, result, errors)
            self.assertEqual(errors, [])

            ncu["values"]["lts__t_bytes.sum"] += 1
            (ncu_dir / "summary.json").write_text(json.dumps(ncu, sort_keys=True))
            errors = []
            audit_ncu(root, case, result, errors)
            self.assertTrue(any("summary/raw CSV mismatch" in row for row in errors))

    def test_cold_proxy_rejects_insufficient_read_misses(self) -> None:
        case = next(case for case in cases() if case["residency"] == "cold_hbm")
        requested_read = 1_000_000
        values = {
            name: 0.0 for name in NCU_METRICS
        }
        values["lts__t_sectors_op_read.sum"] = requested_read / 32
        values["lts__t_sectors_op_write.sum"] = requested_read / 32
        values["lts__t_sectors_op_read_lookup_miss.sum"] = (
            requested_read * 0.59 / 32)
        ncu = {
            "returncode": 0,
            "metrics": list(NCU_METRICS),
            "requested_read_bytes": requested_read,
            "requested_write_bytes": requested_read,
            "values": values,
            "evidence_contract": case["evidence_contract"],
            "external_write_bytes_proven": False,
            "cold_read_miss_proxy_bytes": requested_read * 0.59,
            "cold_read_miss_proxy_to_requested": 0.59,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ncu_dir = root / "cases" / str(case["id"]) / "ncu"
            ncu_dir.mkdir(parents=True)
            # Missing artifacts are acceptable in this focused negative test;
            # the residency error must still be emitted independently.
            errors: list[str] = []
            audit_ncu(root, case, {"ncu": ncu}, errors)
            self.assertTrue(any("cold-DRAM reads" in row for row in errors))

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
