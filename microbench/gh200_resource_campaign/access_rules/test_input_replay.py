"""CPU checks for the shared input contract, including rejected corrupted output."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import analyze_r18 as analysis
import v06_run as common


class InputReplayTests(unittest.TestCase):
    def test_fp16_reference_matches_host_cpp(self):
        # Independently evaluated with the C++ uint32 generator and _Float16 conversion.
        points = [
            (0, 0, 11, -7.204898442141712, 66.62206079345197),
            (17, 63, 74, 2.4200541570317, 61.787989487173036),
            (20261009, 127, 138, 5.205545058008283, 60.673511737026274),
        ]
        for seed, row, col, expected, sum_abs in points:
            values, limits = analysis.random_references([row * 256 + col], 128, 256, 257, seed)
            self.assertEqual(values[0], expected)
            self.assertEqual(limits[0], 2**-20 + 2**-21 * sum_abs)

    def test_nondefault_dyadic_seed(self):
        seed, row, col, k = 9, 31, 23, 39
        expected = sum(((row*7+t*13+seed*3)%17-8) *
                       ((t*5+col*11+seed*5)%17-8) for t in range(k)) / 1024
        self.assertEqual(analysis.reference(row, col, k, seed), expected)

    def test_random_and_zero_replay_reject_wrong_values(self):
        row = dict(id='input_check', config='cfg_b', kind='ordinary', m=64, n=64, k=7,
                   lda=8, ldb=64, ldd=64, storage_m=64, storage_n=64,
                   zero_m=-1, zero_n=-1, swizzle=1, seed=17)
        indices = list(range(4096))
        random_values, _ = analysis.random_references(indices, 64, 64, 7, 17)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = dict(raw='sample.txt.gz', variant='plain', trial=0)
            for mode, values in [('random', random_values), ('zero', [0.0] * 4096)]:
                row['input_mode'] = mode
                setup = dict(row, event='setup', check_atol=2**-20, check_sum_abs_rtol=2**-21)
                call = dict(event='call', warmup_us=[10.0] * 8, elapsed_us=10.0)
                check = dict(event='check', status='ok', padding_errors=0,
                             checked_indices=indices, checked_values=values[:])
                def save():
                    with gzip.open(root / record['raw'], 'wt') as stream:
                        for event in [setup, call, check]:
                            stream.write(json.dumps(event) + '\n')
                    record['raw_sha256'] = common.sha(root / record['raw'])
                save()
                self.assertEqual(analysis.replay(root, record, row)['checked_values'], 4096)
                check['checked_values'][135] += 0.125
                save()
                with self.assertRaisesRegex(ValueError, 'wrong GEMM output'):
                    analysis.replay(root, record, row)


if __name__ == '__main__':
    unittest.main()
