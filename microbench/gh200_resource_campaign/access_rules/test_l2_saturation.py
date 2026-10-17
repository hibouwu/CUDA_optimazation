"""独立地址枚举核对 checksum／请求量，并检查留出与证据边界。"""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import l2_saturation as scan


class L2HostOracleTests(unittest.TestCase):
    def test_checksum_matches_independent_word_enumeration(self):
        # 直接枚举 thread 持有的 vector/word；不复用闭式求和公式。
        for active in (1, 2):
            participants = active * 128
            byte_count = participants * 16 * 3
            for iterations in (1, 2, 16):
                for seed in (3, 0xFFFFFFF0):
                    for thread in (0, 17, participants - 1):
                        expected = 0
                        for _ in range(iterations):
                            for vector in range(thread, byte_count // 16, participants):
                                for lane in range(4):
                                    word = 4 * vector + lane
                                    expected += (17 * word + seed) & 0xFFFFFFFF
                        self.assertEqual(scan.read_checksum_oracle(
                            byte_count, active, iterations, seed, thread), expected & 0xFFFFFFFF)

    def test_work_counts_fixed_array_once_per_iteration(self):
        for active in (1, 8, 32, 132):
            results = [scan.work_oracle(active, chains, "mixed") for chains in (1, 4, 8)]
            for work in results:
                self.assertEqual(work["read_payload_bytes"], 276_824_064)
                self.assertEqual(work["write_payload_bytes"], 276_824_064)
                self.assertEqual(work["work_count"], 553_648_128)
                self.assertEqual(work["aggregate_array_bytes"], 34_603_008)
            self.assertEqual(results[0]["work_count"], results[-1]["work_count"])
        with self.assertRaisesRegex(ValueError, "整除"):
            scan.work_oracle(5, 8, "read")

    def test_finite_dev_and_heldout_are_disjoint(self):
        cases = scan.make_cases()
        development = [c for c in cases if c["split"] == "dev"]
        heldout = [c for c in cases if c["split"] == "heldout"]
        self.assertEqual(len(development), 16)
        self.assertEqual(len(heldout), 4)
        self.assertTrue({c["id"] for c in development}.isdisjoint(c["id"] for c in heldout))
        self.assertEqual(sum(len(c["seeds"]) for c in development), 48)
        self.assertEqual(sum(len(c["seeds"]) for c in heldout), 12)
        for case in cases:
            self.assertEqual(case["seeds"], [3, 22, 41])
            self.assertGreater(case["work"]["batches_per_thread"], 0)

    def test_complete_write_witness_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            words = [(29 * word + 3) & 0xFFFFFFFF for word in range(16)]
            (folder / "output.u32").write_bytes(struct.pack("<16I", *words))
            result = scan.verify_write_witness(folder, {"array_bytes": 64}, 3)
            self.assertEqual(result["independently_checked_words"], 16)
            words[9] ^= 1
            (folder / "output.u32").write_bytes(struct.pack("<16I", *words))
            with self.assertRaisesRegex(ValueError, "word=9"):
                scan.verify_write_witness(folder, {"array_bytes": 64}, 3)

    def test_prepare_is_cpu_only_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            probe = parent / "probe.cu"
            probe.write_text("// stand-alone preparation fixture\n")
            output = parent / "prepared"
            with patch.object(scan, "PROBE", probe), patch.object(scan.subprocess, "Popen") as process:
                scan.prepare(output)
                self.assertEqual(scan.load_plan(output)["GPU_execution"], False)
                process.assert_not_called()
                with self.assertRaises(FileExistsError):
                    scan.prepare(output)
            self.assertFalse((output / "dev").exists())
            self.assertFalse((output / "heldout").exists())

    def test_heldout_rejects_absent_development_freeze(self):
        with self.assertRaisesRegex(ValueError, "--dev-freeze"):
            scan.verify_dev_freeze(Path("/unused"), None)

    def test_run_allocation_check_precedes_any_command(self):
        with patch.dict(scan.os.environ, {}, clear=True), patch.object(scan, "command_record") as command:
            with self.assertRaisesRegex(ValueError, "SLURM_JOB_ID"):
                scan.inspect_allocated_gpu(Path("/unused"))
            command.assert_not_called()


class L2RawValidationTests(unittest.TestCase):
    def fixture(self, active=1):
        case = next(case for case in scan.make_cases() if case["id"] == f"read_a{active}_j1")
        blocks = []
        for sm in range(active):
            blocks.append(dict(active=True, smid_start=sm, smid_end=sm,
                               begin_ns=1000, end_ns=3000,
                               begin_cycle=100, end_cycle=4100))
        row = dict(schema="gh200-l2-saturation/1", status="measured",
                   cache_residency_proven=False, physical_hbm_bytes_proven=False,
                   aggregate_array_bytes=scan.ARRAY_BYTES, **{name: case[name] for name in
                   ("active_ctas", "chains", "mode", "array_bytes", "iterations", "samples")},
                   seed=3, read_payload_bytes=276_824_064, write_payload_bytes=0,
                   work_count=276_824_064,
                   device=dict(name="GH200", uuid="GPU-fixture", sms=132, l2_cache_bytes=62_914_560),
                   resources=dict(threads=128, grid_ctas=active, occupancy_ctas_per_sm=1,
                                  local_size_bytes=0, registers_per_thread=32,
                                  dynamic_smem_bytes=132096, static_smem_bytes=0),
                   warmup=dict(elapsed_ns=[2000] * 8, cv=0.0, converged=True, limit=60),
                   short_check=dict(iterations=[1, 2], errors=0, read_checked=active * 256, write_checked=0),
                   trials=[dict(iterations=16, all_active_windows_overlap=True,
                                blocks_detail=blocks, elapsed_ns=2000, event_ms=0.003,
                                correctness=dict(read_checked=active * 128, write_checked=0, errors=0),
                                read_checksums=[scan.read_checksum_oracle(
                                    scan.ARRAY_BYTES, active, 16, 3, thread) for thread in range(active * 128)])])
        return case, row

    def test_raw_integer_envelope_and_checksum(self):
        case, row = self.fixture()
        result = scan.validate_probe(Path("/unused"), row, case, 3)
        self.assertEqual(result["elapsed_ns"], 2000)
        self.assertEqual(result["rate_numerator"], 276_824_064)
        self.assertEqual(result["active_smids"], [0])
        self.assertFalse(result["physical_hbm_bytes_proven"])
        bad = copy.deepcopy(row)
        bad["trials"][0]["read_checksums"][17] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            scan.validate_probe(Path("/unused"), bad, case, 3)
        bad = copy.deepcopy(row)
        bad["trials"][0]["elapsed_ns"] += 1
        with self.assertRaisesRegex(ValueError, "包络"):
            scan.validate_probe(Path("/unused"), bad, case, 3)

    def test_wrong_active_count_and_work_corruption_are_rejected(self):
        case, row = self.fixture()
        bad = copy.deepcopy(row)
        bad["trials"][0]["blocks_detail"][0]["active"] = False
        with self.assertRaisesRegex(ValueError, "活跃CTA数量"):
            scan.validate_probe(Path("/unused"), bad, case, 3)
        bad = copy.deepcopy(row)
        bad["work_count"] *= 132
        with self.assertRaisesRegex(ValueError, "请求量"):
            scan.validate_probe(Path("/unused"), bad, case, 3)

    def test_sparse_smid_is_accepted_and_duplicate_grid_mapping_is_rejected(self):
        case, row = self.fixture(active=8)
        for block in row["trials"][0]["blocks_detail"]:
            block["smid_start"] = block["smid_start"] * 2 + 200
            block["smid_end"] = block["smid_start"]
        result = scan.validate_probe(Path("/unused"), row, case, 3)
        self.assertEqual(result["active_smids"], list(range(200, 216, 2)))
        bad = copy.deepcopy(row)
        bad["trials"][0]["blocks_detail"][7]["smid_start"] = 200
        bad["trials"][0]["blocks_detail"][7]["smid_end"] = 200
        with self.assertRaisesRegex(ValueError, "未分别位于不同SM"):
            scan.validate_probe(Path("/unused"), bad, case, 3)

    def test_unconverged_warmup_preserves_condition_with_no_performance_qualification(self):
        case, row = self.fixture()
        windows = [2000] * 56 + [1000, 2000, 1000, 2000]
        tail = windows[-5:]
        mean = sum(tail) / 5
        variance = sum((window - mean) ** 2 for window in tail) / 4
        row["warmup"] = dict(elapsed_ns=windows, cv=variance ** 0.5 / mean, converged=False, limit=60)
        result = scan.validate_probe(Path("/unused"), row, case, 3)
        self.assertFalse(result["performance_eligible"])
        self.assertEqual(result["warmup_windows"], 60)


if __name__ == "__main__":
    unittest.main()
