from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from microbench.sm110_tma_payload_campaign.audit_campaign import audit_ncu
from microbench.sm110_tma_payload_campaign.run_tma_payload_campaign import (
    DEFAULT_TARGET_ISSUED_BYTES,
    PAYLOAD_BYTES,
    cases,
    validate_trial,
)


class TmaPayloadManifestTest(unittest.TestCase):
    def test_manifest_is_exact_payload_residency_cross_product(self) -> None:
        manifest = cases(DEFAULT_TARGET_ISSUED_BYTES)
        self.assertEqual(len(manifest), 10)
        self.assertEqual(len({row["id"] for row in manifest}), 10)
        self.assertEqual(
            {(row["residency"], row["tile_bytes"]) for row in manifest},
            {
                (residency, payload)
                for residency in ("hot_l2", "cold_hbm")
                for payload in PAYLOAD_BYTES
            },
        )
        for row in manifest:
            self.assertEqual(row["destination_slots"], 2)
            self.assertEqual(row["threads_per_cta"], 128)
            self.assertEqual(row["resident_ctas_per_sm"], 1)
            issued = (int(row["expected_blocks"]) * int(row["iterations"])
                      * int(row["tile_bytes"]))
            self.assertGreaterEqual(issued, DEFAULT_TARGET_ISSUED_BYTES)
            self.assertLess(
                issued,
                DEFAULT_TARGET_ISSUED_BYTES
                + int(row["expected_blocks"]) * int(row["tile_bytes"]),
            )
            if row["residency"] == "hot_l2":
                self.assertEqual(row["expected_blocks"], 1)
                self.assertGreaterEqual(
                    int(row["warmup_iterations"]) * int(row["tile_bytes"]),
                    16 << 20,
                )
            else:
                self.assertEqual(row["expected_blocks"], 20)

    def test_trial_rate_is_recomputed_from_full_grid_interval(self) -> None:
        case = cases(DEFAULT_TARGET_ISSUED_BYTES)[0]
        requested = (int(case["expected_blocks"]) * int(case["iterations"])
                     * int(case["tile_bytes"]))
        elapsed_ns = 1_000_000
        fields = {
            "mode": "l2-hit",
            "sm_count": "20",
            "unique_smid_count": str(case["expected_unique_smid_count"]),
            "blocks": str(case["expected_blocks"]),
            "requested_bytes": str(requested),
            "globaltimer_elapsed_ns": str(elapsed_ns),
            "globaltimer_gbytes_per_second": f"{requested / elapsed_ns:.6f}",
        }
        self.assertEqual(
            validate_trial(case, fields, int(case["iterations"])),
            requested * 1e9 / elapsed_ns,
        )

    def test_trial_rejects_incomplete_sm_coverage(self) -> None:
        case = cases(DEFAULT_TARGET_ISSUED_BYTES)[0]
        fields = {
            "mode": "l2-hit",
            "sm_count": "20",
            "unique_smid_count": "2",
            "blocks": str(case["expected_blocks"]),
            "requested_bytes": str(
                int(case["expected_blocks"]) * int(case["iterations"])
                * int(case["tile_bytes"])
            ),
            "globaltimer_elapsed_ns": "1000000",
            "globaltimer_gbytes_per_second": "1.0",
        }
        with self.assertRaisesRegex(RuntimeError, "SM-scope mismatch"):
            validate_trial(case, fields, int(case["iterations"]))


class TmaPayloadNcuAuditTest(unittest.TestCase):
    def _bundle(self, root: Path, case: dict[str, object]) -> dict[str, object]:
        ncu_dir = root / "cases" / str(case["id"]) / "ncu"
        ncu_dir.mkdir(parents=True)
        for name in ("profile.ncu-rep", "raw.csv", "profile.stderr.log"):
            (ncu_dir / name).write_text(name)
        import hashlib

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        timed_requested = (int(case["expected_blocks"]) * 256
                           * int(case["tile_bytes"]))
        expected = timed_requested
        ncu = {
            "returncode": 0,
            "permission_denied": False,
            "iterations": 256,
            "metrics": [
                "gpu__time_duration.sum",
                "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
                "lts__t_bytes.sum",
                "lts__t_sectors_op_read_lookup_hit.sum",
                "lts__t_sectors_op_read_lookup_miss.sum",
            ],
            "timed_requested_bytes": timed_requested,
            "expected_counter_bytes": expected,
            "tma_bytes": expected,
            "tma_to_expected": 1.0,
            "lts_bytes": expected,
            "lts_to_expected": 1.0,
            "l2_hit_sectors": expected / 32 if case["residency"] == "hot_l2" else 0,
            "l2_miss_sectors": expected / 32 if case["residency"] == "cold_hbm" else 0,
            "l2_miss_proxy_to_expected": 1.0 if case["residency"] == "cold_hbm" else 0.0,
            "report_path": "ncu/profile.ncu-rep",
            "raw_path": "ncu/raw.csv",
            "stderr_path": "ncu/profile.stderr.log",
        }
        raw_header = [
            "ID",
            "Kernel Name",
            "gpu__time_duration.sum",
            "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
            "lts__t_bytes.sum",
            "lts__t_sectors_op_read_lookup_hit.sum",
            "lts__t_sectors_op_read_lookup_miss.sum",
        ]
        raw_values = [
            "1",
            "tma_kernel",
            "1.0",
            str(expected),
            str(expected),
            str(ncu["l2_hit_sectors"]),
            str(ncu["l2_miss_sectors"]),
        ]
        (ncu_dir / "raw.csv").write_text(
            ",".join(raw_header) + "\n" + ",".join(raw_values) + "\n"
        )
        for path_key, hash_key in (
            ("report_path", "report_sha256"),
            ("raw_path", "raw_sha256"),
            ("stderr_path", "stderr_sha256"),
        ):
            ncu[hash_key] = digest(root / "cases" / str(case["id"]) / str(ncu[path_key]))
        (ncu_dir / "summary.json").write_text(json.dumps(ncu, sort_keys=True))
        return {"ncu": ncu}

    def test_ncu_audit_accepts_residency_specific_evidence(self) -> None:
        for case in (
            cases(DEFAULT_TARGET_ISSUED_BYTES)[0],
            cases(DEFAULT_TARGET_ISSUED_BYTES)[-1],
        ):
            with self.subTest(residency=case["residency"]):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    result = self._bundle(root, case)
                    errors: list[str] = []
                    audit_ncu(root, str(case["id"]), case, result, errors)
                    self.assertEqual(errors, [])

    def test_ncu_audit_rejects_unproven_dram_residency(self) -> None:
        case = next(
            row
            for row in cases(DEFAULT_TARGET_ISSUED_BYTES)
            if row["residency"] == "cold_hbm"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._bundle(root, case)
            result["ncu"]["l2_miss_sectors"] = 0
            result["ncu"]["l2_miss_proxy_to_expected"] = 0
            errors: list[str] = []
            audit_ncu(root, str(case["id"]), case, result, errors)
            self.assertTrue(any("DRAM-stream residency" in row for row in errors))


if __name__ == "__main__":
    unittest.main()
