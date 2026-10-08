"""Adversarial offline checks for the real R16 quota witness archive."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from analyze_r16_quota import replay, sass

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'results/gh200_resource_campaign/access_rules/20261008-R16-quota-dev-job736989'


@unittest.skipUnless(ARCHIVE.exists(), 'legacy evidence archive required')
class QuotaReplayTests(unittest.TestCase):
    def setUp(self):
        self.sample = ARCHIVE / 'samples/delay64_before1/trial-00'
        self.record = json.loads((self.sample / 'result.json').read_text())

    def test_real_values_and_legacy_boundary(self):
        result = replay(self.sample, self.record)
        self.assertEqual(result['checked_values'], 49536)
        self.assertIsNone(result['compute_cycles'])
        facts = sass(ARCHIVE)
        self.assertFalse(facts['compute_boundary_present'])
        self.assertTrue(facts['loads_interleaved_with_reduction'])

    def test_wrong_integer_rejected_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['values.f32', 'integers.u32']:
                data = bytearray(gzip.open(self.sample / (name+'.gz'), 'rb').read())
                if name == 'integers.u32':
                    data[0] ^= 1
                with gzip.open(root / (name+'.gz'), 'wb') as stream:
                    stream.write(data)
                self.record['files'][name]['sha256'] = hashlib.sha256(data).hexdigest()
            with self.assertRaisesRegex(ValueError, 'wrong dependent integer'):
                replay(root, self.record)

    def test_clock_order_and_pool_rejected(self):
        wrong = copy.deepcopy(self.record)
        wrong['stamps'][0][4] = wrong['stamps'][0][1]+1
        with self.assertRaisesRegex(ValueError, 'not before release'):
            replay(self.sample, wrong)
        wrong = copy.deepcopy(self.record)
        wrong['registers'] = 176
        with self.assertRaisesRegex(ValueError, 'pool budget'):
            replay(self.sample, wrong)

    def test_moved_inc_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = (ARCHIVE/'sass.txt').read_text().replace('CS2R R10, SR_CLOCKLO', 'CS2R R10, SRZ', 1)
            (root/'sass.txt').write_text(text)
            with self.assertRaises(ValueError):
                sass(root)


if __name__ == '__main__':
    unittest.main()
