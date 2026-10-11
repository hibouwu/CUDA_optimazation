import unittest
from analyze_v09_gate_audit import error_summary, process_audit, timing_audit


class GateAuditTests(unittest.TestCase):
    def test_opposite_errors_do_not_cancel(self):
        result = error_summary([-.8, .8])
        self.assertEqual(result['signed_median_pct'], 0)
        self.assertEqual(result['median_abs_pct'], 80)

    def test_missing_observation_is_not_zero_error(self):
        result = error_summary([], 2)
        self.assertIsNone(result['median_abs_pct'])
        self.assertEqual(result['invalid_windows'], 2)

    def test_observer_separation_uses_full_empirical_ranges(self):
        rows = [dict(variant=v, trial=i, elapsed_us=t) for v, times in
                [('plain', [10, 12]), ('dual', [13, 15])] for i, t in enumerate(times)]
        result = timing_audit(rows)
        self.assertFalse(result['plain']['complete_trials'])
        self.assertEqual(result['stamped']['processes'], 0)
        self.assertGreater(result['dual']['observed_ratio_range_percent'][0], 0)
        scaled = timing_audit([dict(r, elapsed_us=r['elapsed_us']*2) for r in rows])
        self.assertEqual(result['dual']['observed_ratio_range_percent'],
                         scaled['dual']['observed_ratio_range_percent'])

    def test_invalid_or_duplicate_timing_is_rejected(self):
        row = dict(variant='plain', trial=0, elapsed_us=10)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            timing_audit([row, row])
        with self.assertRaisesRegex(ValueError, 'invalid'):
            timing_audit([dict(row, elapsed_us=0)])

    def example(self):
        ctas = [dict(cta=0, entry_ns=0, prod_ns=10, end_ns=100, tiles_ns=[[20, 40, 50, 80]]),
                dict(cta=1, entry_ns=50, prod_ns=60, end_ns=140, tiles_ns=[[70, 90, 100, 120]])]
        p = dict(critical_cta=0, critical_candidates=[0], cta_events=[
            dict(events=[[20, 40, 50, 80]], supply=20, prefill=10, tail=20),
            dict(events=[[20, 40, 50, 70]], supply=20, prefill=10, tail=20)])
        return ctas, p

    def test_last_endpoint_differs_from_longest_duration(self):
        ctas, p = self.example()
        r = process_audit(ctas, p)['critical']
        self.assertEqual(r['observed_exact_last'], [1])
        self.assertEqual(r['end_regret_ns'], 40)
        self.assertFalse(r['chosen_is_exact_last'])

    def test_observed_ties_preserved_without_widening_prediction(self):
        ctas, p = self.example()
        ctas[0]['end_ns'] = 140
        r = process_audit(ctas, p)['critical']
        self.assertEqual(r['observed_exact_last'], [0, 1])
        self.assertEqual(r['frozen_candidate_count'], 1)

    def test_truncated_cta_or_tile_arrays_are_rejected(self):
        ctas, p = self.example()
        with self.assertRaisesRegex(ValueError, 'CTA count'):
            process_audit(ctas[:1], p)
        p['cta_events'][0]['events'] = []
        with self.assertRaisesRegex(ValueError, 'tile count'):
            process_audit(ctas, p)


if __name__ == '__main__':
    unittest.main()
