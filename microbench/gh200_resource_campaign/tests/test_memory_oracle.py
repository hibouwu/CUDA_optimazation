"""CPU checks of the v2 probe's address/checksum identities, not GPU evidence."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 32) - 1


class MemoryOracleTests(unittest.TestCase):
    def test_uint4_formula_against_explicit_addresses(self):
        # The closed form is the C++ host oracle. The reference enumerates scalar
        # addresses independently, including wraparound of uint32 input and sum.
        for blocks, per_block, iterations, seed in (
            (1, 256, 1, 3), (3, 768, 64, 193), (528, 32000, 16, 174),
            (2, 65536, 65536, 999999),
        ):
            for block in {0, blocks - 1}:
                for thread in (0, 31, 127, 255):
                    indices = [4 * (block * per_block + j) + lane
                               for j in range(thread, per_block, 256)
                               for lane in range(4)]
                    expected = sum((17 * i + seed) & MASK for i in indices) * iterations & MASK
                    n = per_block // 256
                    first = block * per_block + thread
                    formula = ((272 * (n * first + 256 * n * (n - 1) // 2)
                                + (102 + seed * 4) * n) * iterations) & MASK
                    self.assertEqual(formula, expected)

    def test_global_rounding_covers_every_word_once(self):
        for sms in (1, 132):
            blocks = sms * 4
            for requested in (8, 128, 256):
                requested *= 1024**2
                quantum = blocks * 256 * 16
                allocation = ((requested + quantum - 1) // quantum) * quantum
                self.assertGreaterEqual(allocation, requested)
                self.assertLess(allocation - requested, quantum)
                per_block = allocation // blocks // 16
                self.assertEqual(per_block % 256, 0)
                self.assertEqual(per_block * blocks * 4, allocation // 4)

    def test_shared_mapping_and_poison_are_distinct(self):
        for stride in (1, 2, 4, 8, 16, 32):
            for thread in range(256):
                values = [((((thread + q * 256) * stride) & 8191) * 17 + 3) for q in range(8)]
                self.assertNotEqual(sum(values) & MASK, (8 * 0xdeadbeef) & MASK)
                self.assertEqual((sum(values) * 8192) & MASK,
                                 sum(v * 8192 for v in values) & MASK)
        write_addresses = [thread + q * 256 for thread in range(256) for q in range(8)]
        self.assertEqual(sorted(write_addresses), list(range(2048)))

    def test_frozen_matrix(self):
        spec = json.loads((ROOT / "contracts/memory_baseline.json").read_text())
        self.assertEqual(len(spec["cases"]), 14)
        ids = {c["id"] for c in spec["cases"]}
        self.assertEqual(len(ids), 14)
        controls = [c for c in spec["cases"] if c["work_model"] == "empty_window"]
        self.assertEqual(len(controls), 2)
        self.assertTrue(all(c["exportable"] is False for c in controls))
        self.assertEqual(next(c for c in controls if c["scope"] == "all_gpu")["coverage_policy"],
                         "observed_subset")

    def test_declared_shared_work(self):
        self.assertEqual(256 * 8 * 4 * 8192, 67108864)

    def test_no_reuse_of_v1_warmup_protocol(self):
        source = (ROOT / "probes/memory_baseline.cu").read_text()
        self.assertIn("gh::warmup(execute)", source)
        self.assertNotIn("launch(2);launch(2)", source)
        self.assertIn('ld.volatile.shared.u32', source)
        self.assertIn('st.volatile.shared.u32', source)
        # Regression for an actually observed broken JSON serialization.
        self.assertNotIn(chr(92) * 3 + '"', source)


if __name__ == "__main__":
    unittest.main()
