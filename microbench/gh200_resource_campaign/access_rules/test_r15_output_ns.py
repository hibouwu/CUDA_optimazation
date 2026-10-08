#!/usr/bin/env python3
"""CPU fixtures for the proposed R15 profile; no measured GPU data is generated."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from analyze_r15_output_ns import PROFILE, WIDTH, analyze, direct_windows, summarize_call
from analyze_r18 import reference
from analyze_r15 import sha


def fixture():
    row = dict(id='synthetic', config='cfg_c', m=256, n=256, k=64,
        lda=64, ldb=256, ldd=256, storage_m=256, storage_n=256,
        zero_m=-1, zero_n=-1, swizzle=1, kind='ordinary')
    setup = dict(row, event='setup', trace_version=PROFILE, trace_words=WIDTH, grid=[1, 2, 1])
    words = [0] * (2 * WIDTH)
    origin = 2**56  # Subtract integer anchors before floating-point overlap calculations.
    for cta in range(2):
        i = cta * WIDTH
        words[i:i + 16] = [1000, origin, cta + 1, 1200,
            4900, origin + 1000, 1, 5000 + cta * 5000, origin + 1000, 1,
            0, cta + 1, cta + 1, origin + 100 + 100 * cta, origin + 300 + 100 * cta, 0]
        words[i + 16:i + 22] = [1300, 1400, 2500, 3504, 1, cta + 1]
        j = i + 16 + 64 * 6
        words[j:j + 6] = [1300, 1400, 2000, 3500, 1, cta + 1]
    call = dict(event='call', elapsed_us=2.0, warmup_us=[2.0] * 8, trace=words)
    return row, setup, call


class OutputNsTests(unittest.TestCase):
    def test_direct_overlap_ignores_clock_conversion(self):
        _, setup, call = fixture()
        result = summarize_call(direct_windows(setup, call))
        self.assertEqual(result['issuer_store_ns'], 200)
        self.assertEqual(result['overlap_direct'], 1.5)
        self.assertEqual(result['peak_overlap'], 2)
        self.assertEqual(result['issuer_store_cycles'], 1500)
        self.assertEqual(result['merged_store_cycles'], 1004)
        self.assertNotEqual(result['overlap_affine'], result['overlap_direct'])

    def test_old_profile_has_no_direct_fallback(self):
        _, setup, call = fixture()
        setup['trace_version'] = 'r18-work-coordinates'
        with self.assertRaisesRegex(ValueError, 'no affine fallback'):
            direct_windows(setup, call)

    def test_rates_are_paired_before_aggregation(self):
        _, setup, call = fixture()
        call['trace'][WIDTH + 14] -= 100
        result = summarize_call(direct_windows(setup, call))
        self.assertEqual(result['issuer_cycles_per_ns'], (7.5 + 15) / 2)
        self.assertNotEqual(result['issuer_cycles_per_ns'],
                            result['issuer_store_cycles'] / result['issuer_store_ns'])
        self.assertAlmostEqual(result['issuer_to_cta_rate_ratio'], (7.5 / 4 + 15 / 9) / 2)

    def test_second_tile_is_not_silently_omitted(self):
        _, setup, call = fixture()
        call['trace'][6] = call['trace'][9] = 2
        with self.assertRaisesRegex(ValueError, 'exactly one tile'):
            direct_windows(setup, call)

    def test_same_time_end_and_start_do_not_overlap(self):
        _, setup, call = fixture()
        call['trace'][WIDTH + 13] += 100
        result = summarize_call(direct_windows(setup, call))
        self.assertEqual(result['overlap_direct'], 1)
        self.assertEqual(result['peak_overlap'], 1)

    def test_archive_entry_uses_the_raw_global_setup(self):
        row, setup, call = fixture()
        indices = list(range(4096))
        check = dict(event='check', status='ok', padding_errors=0,
            checked_indices=indices, checked_values=[reference(i // row['n'], i % row['n'], row['k']) for i in indices])
        with tempfile.TemporaryDirectory(prefix='r15-output-ns-cpu-') as tmp:
            root = Path(tmp)
            folder = root / 'samples' / row['id']
            folder.mkdir(parents=True)
            (root / 'cases.json').write_text(json.dumps([row]))
            (root / 'static_setup.json').write_text(json.dumps([dict(case=row['id'], setup=dict(setup, trace_version='r18-work-coordinates'))]))
            for variant in ('plain', 'stamped', 'ends', 'global'):
                raw = folder / (variant + '-00.txt.gz')
                with gzip.open(raw, 'wt') as stream:
                    actual_setup = setup if variant == 'global' else dict(setup, trace_version='r18-work-coordinates')
                    for event in (actual_setup, call, check):
                        stream.write(json.dumps(event) + '\n')
                record = dict(variant=variant, trial=0, returncode=0, elapsed_us=2.0,
                    raw=str(raw.relative_to(root)), raw_sha256=sha(raw))
                (folder / (variant + '-00.json')).write_text(json.dumps(record))
            analyze(root, root / 'analysis')
            result = json.loads((root / 'analysis/output-ns.json').read_text())
            self.assertEqual(result['cases'][0]['median']['overlap_direct'], 1.5)
            self.assertEqual(result['cases'][0]['global_relative'], dict(plain=0, stamped=0, ends=0))


if __name__ == '__main__':
    unittest.main()
