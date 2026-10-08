"""CPU checks for R13 observed-SM overlap and exact pitch pairing."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze_r13_sm import cross_scale, overlap_windows, select_attempts, validate_setup
from v08_model import scheduled_work


class R13SmTests(unittest.TestCase):
    def test_union_counts_sm_once_and_keeps_idle_gap(self):
        ctas = [dict(sm=sm, entry_ns=a, end_ns=b, entry_c=a * 2, end_c=b * 2)
                for sm, a, b in [(0, 1, 5), (0, 3, 7), (1, 4, 6), (2, 9, 11)]]
        result = overlap_windows(ctas)
        self.assertEqual(result['duration_by_observed_sms_ns'], {0: 2, 1: 6, 2: 2})
        self.assertEqual(result['peak_observed_sms'], 2)
        self.assertEqual(result['mean_observed_sms'], 1)
        self.assertEqual(result['distinct_smids'], 3)
        self.assertEqual(result['sm_unions'][0]['intervals_ns'], [[1, 7]])

    def test_retry_has_one_success_and_keeps_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'samples' / 'case'
            folder.mkdir(parents=True)
            base = dict(case='case', variant='stamped', trial=0)
            (folder / 'first.json').write_text(json.dumps(dict(base, returncode=1, attempt=0)))
            (folder / 'retry.json').write_text(json.dumps(dict(base, returncode=0, attempt=1)))
            selected, failed, ignored = select_attempts(root, 'case')
            self.assertEqual(len(selected), 1)
            self.assertEqual(len(failed), 1)
            self.assertEqual(selected[0]['record']['attempt'], 1)
            self.assertEqual(ignored, [])
            (folder / 'duplicate.json').write_text(json.dumps(dict(base, returncode=0, attempt=2)))
            with self.assertRaisesRegex(ValueError, 'multiple successful'):
                select_attempts(root, 'case')

    def test_new_uuid_required_legacy_unknown(self):
        self.assertEqual(validate_setup(dict(id='legacy'), {}, {}), 'unknown')
        with self.assertRaisesRegex(ValueError, 'requires sample and static GPU UUID'):
            validate_setup(dict(id='new', sm_count=32), {}, {})
        with self.assertRaisesRegex(ValueError, 'UUID mismatch'):
            validate_setup(dict(id='new', sm_count=32), dict(gpu_uuid='GPU-a'), dict(gpu_uuid='GPU-b'))

    def test_cross_grid_only_matches_same_sequence_total_and_coordinates(self):
        pairs = []
        for sm in (96, 132):
            work = scheduled_work('cfg_a', 2304, 3072, [sm, 1, 1])
            for cta, coords in enumerate(work):
                for j, (mi, ni) in enumerate(coords):
                    pairs.append(dict(pitch='a16', comparison_sm_count=sm, trial=0, cta=cta,
                                      j=j, T=len(coords), mi=mi, ni=ni,
                                      delta_per_kt=2 if sm == 96 else 5,
                                      relative=.02 if sm == 96 else .05, qualified=False))
        details, groups = cross_scale(pairs)
        self.assertEqual(len(details), 48)
        valid = [g for g in groups if g['matched_tiles']]
        self.assertEqual([(g['j'], g['T'], g['matched_tiles']) for g in valid], [(1, 4, 24), (2, 4, 24)])
        self.assertTrue(all(g['delta_penalty_per_kt'] == 3 for g in valid))
        self.assertTrue(all(not g['qualified'] for g in groups))
        self.assertTrue(any(g['status'] == 'no_common_coordinates' for g in groups))


if __name__ == '__main__':
    unittest.main()
