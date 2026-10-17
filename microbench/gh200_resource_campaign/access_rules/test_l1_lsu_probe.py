"""CPU 核对链覆盖、固定工作量、完整 checksum 和准备/采样边界。"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import l1_lsu_probe as scan


class L1ProbeTests(unittest.TestCase):
    def test_chains_cover_same_warp_contiguous_nodes_once(self):
        for kib in (16, 64, 128, 256):
            batches = kib*1024//(128*16)
            for chains in (1, 4):
                for thread in (0, 31, 32, 127):
                    visited = []
                    for chain in range(chains):
                        index = chain*128+thread
                        for _ in range(batches//chains):
                            visited.append(index)
                            index = (((index//128)+chains) % batches)*128+thread
                        self.assertEqual(index, chain*128+thread)
                    self.assertEqual(sorted(visited), list(range(thread, kib*1024//16, 128)))
                    for batch in range(batches):
                        warp = [batch*128+lane for lane in range(32)]
                        self.assertEqual(warp[-1]-warp[0], 31)

    def test_checksum_matches_explicit_pointer_traversal_including_overflow(self):
        for kib in (16, 256):
            batches = kib*1024//(128*16)
            for chains in (1, 4):
                for seed in (3, 0xFFFFFFF0):
                    for thread in (0, 17, 127):
                        for rounds in (1, 2, 256):
                            checksum = 0
                            for chain in range(chains):
                                index = chain*128+thread
                                for _ in range(batches//chains*rounds):
                                    next_index = (((index//128)+chains) % batches)*128+thread
                                    payload = [(17*(4*index+word)+seed) & 0xFFFFFFFF for word in (1, 2, 3)]
                                    checksum += next_index+sum(payload)
                                    index = next_index
                            self.assertEqual(checksum & 0xFFFFFFFF,
                                             scan.checksum_oracle(kib*1024, seed, thread, rounds))

    def test_twenty_conditions_and_fixed_total_work(self):
        cases = scan.make_cases()
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case["id"] for case in cases}), 20)
        self.assertEqual(sum(case["split"] == "dev" for case in cases), 12)
        self.assertEqual(sum(case["split"] == "heldout" for case in cases), 8)
        self.assertEqual(sum(len(case["seeds"]) for case in cases), 60)
        for case in cases:
            self.assertEqual(case["work"]["logical_read_bytes"], 256*case["working_set_bytes"])
            if case["split"] == "heldout": self.assertEqual(case["working_set_kib"], 128)
            elif case["working_set_kib"] == 64: self.assertEqual(case["chains"], 4)
            else: self.assertEqual(case["requested_carveout_percent"], 100)
        for kib in (16, 64, 128, 256):
            self.assertEqual(scan.work_oracle(kib*1024, 1)["loads_per_thread"],
                             scan.work_oracle(kib*1024, 4)["loads_per_thread"])

    def test_prepare_has_no_subprocess_or_cuda_and_keeps_sources(self):
        with tempfile.TemporaryDirectory() as temporary, patch("l2_saturation.subprocess.Popen") as process:
            root = Path(temporary)/"prepared"
            scan.prepare(root)
            plan = scan.load_plan(root)
            self.assertFalse(plan["GPU_execution"])
            self.assertEqual(plan["dev_processes"], 36)
            self.assertEqual(plan["heldout_processes"], 24)
            self.assertIn("l2_saturation.py", plan["source_sha256"])
            process.assert_not_called()
            with self.assertRaises(FileExistsError): scan.prepare(root)

    def fixture(self):
        case = scan.make_cases()[0]
        trials = []
        for rounds in (1, 2, 256):
            trials.append(dict(rounds=rounds, logical_read_bytes=case["working_set_bytes"]*rounds,
                event_ms=0.005, begin_cycle=100, end_cycle=4100, begin_ns=1000, end_ns=3000,
                smid_start=200, smid_end=200, correctness=dict(checked_elements=256, errors=0),
                warm_checksums=[scan.checksum_oracle(case["working_set_bytes"], 3, t, 1) for t in range(128)],
                timed_checksums=[scan.checksum_oracle(case["working_set_bytes"], 3, t, rounds) for t in range(128)]))
        row = dict(schema="gh200-l1-lsu/1", status="measured", seed=3, actual_carveout=None,
            logical_read_bytes=case["work"]["logical_read_bytes"],
            **{key: case[key] for key in ("working_set_bytes", "chains", "policy", "requested_carveout_percent", "rounds")},
            device=dict(name="GH200", uuid="GPU-fixture", sms=132, major=9, minor=0),
            resources=dict(grid_ctas=1, threads=128, dynamic_smem_bytes=0, local_size_bytes=0), trials=trials)
        return case, row

    def test_complete_warm_and_timed_checksums_are_independently_validated(self):
        case, row = self.fixture()
        result = scan.validate(row, case, 3)
        self.assertEqual(result["independently_checked_elements"], 768)
        for name in ("warm_checksums", "timed_checksums"):
            bad = copy.deepcopy(row)
            bad["trials"][0][name][17] ^= 1
            with self.assertRaisesRegex(ValueError, name): scan.validate(bad, case, 3)

    def test_migration_work_change_and_inferred_actual_partition_are_rejected(self):
        case, row = self.fixture()
        for change, message in ((lambda r: r["trials"][-1].update(smid_end=201), "SMID"),
                                (lambda r: r.update(logical_read_bytes=1), "请求量"),
                                (lambda r: r.update(actual_carveout=100), "carveout")):
            bad = copy.deepcopy(row)
            change(bad)
            with self.assertRaisesRegex(ValueError, message): scan.validate(bad, case, 3)

    def test_run_requires_explicit_gpu_and_heldout_predictions_before_allocation(self):
        with patch.object(scan, "inspect_allocated_gpu") as inspect:
            with self.assertRaisesRegex(ValueError, "allow-gpu"):
                scan.run_scan(Path("/unused"), "dev", False, None)
            inspect.assert_not_called()
        with self.assertRaisesRegex(ValueError, "frozen-predictions"):
            scan.frozen_predictions(Path("/unused"), None)


if __name__ == "__main__":
    unittest.main()
