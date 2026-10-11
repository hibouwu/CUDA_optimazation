import json
import math
import unittest

from r09_clock_brackets import common_rate_interval, local_clock_witness, observer_perturbation_gate, window_bounds


def exact(cycles, ns, origin=1000):
    return ((0,origin,0),(cycles,origin+ns,cycles))


def observer_pairs():
    return [dict(event_ns_delta=0,cta_pairs=[dict(cta=i,role=role,phase=phase,
        L_ns_delta=0,L_cycles_delta=0) for i in range(2)
        for role,phase in ((0,'first'),(1,'later'))]) for _ in range(10)]


class ClockBrackets(unittest.TestCase):
    def test_exact_common_positive_rate(self):
        result = common_rate_interval([exact(100,50),exact(200,100)],interval_tolerance_ns=0)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['lower_ghz'],2)
        self.assertEqual(result['upper_ghz'],2)
        self.assertEqual(result['extra_interval_slack_ns'],0)

    def test_disjoint_intervals_require_more_than_total_window_budget(self):
        windows = [exact(100,100),exact(200,100)]
        no_budget = common_rate_interval(windows,interval_tolerance_ns=0)
        self.assertFalse(no_budget['feasible'])
        self.assertEqual(no_budget['minimum_interval_slack_fraction'],[100,3])
        self.assertAlmostEqual(no_budget['extra_endpoint_slack_equivalent_ns'],50/3)
        self.assertFalse(common_rate_interval(windows,interval_tolerance_ns=32)['feasible'])
        self.assertTrue(common_rate_interval(windows,interval_tolerance_ns=34)['feasible'])
        self.assertAlmostEqual(common_rate_interval(windows,interval_tolerance_ns=32)['extra_interval_slack_ns'],4/3)

    def test_clock_bracket_can_explain_a_raw_rate_difference(self):
        a = exact(100,100)
        b = ((0,2000,100),(200,2100,200))
        result = common_rate_interval([a,b],interval_tolerance_ns=0)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['lower_ghz'],1)
        self.assertEqual(result['upper_ghz'],1)

    def test_large_integer_timer_origin_is_subtracted_before_float(self):
        origin = (1<<60)+13
        bounds = window_bounds((50,origin,52),(102,origin+33,104))
        self.assertEqual(bounds,dict(cycles_lo=50,cycles_hi=54,ns=33))
        result = common_rate_interval([((50,origin,52),(102,origin+33,104))],interval_tolerance_ns=0)
        self.assertAlmostEqual(result['lower_ghz'],50/33)

    def test_short_unresolved_window_is_unbounded_and_json_finite(self):
        result = common_rate_interval([exact(10,10)],interval_tolerance_ns=128)
        self.assertTrue(result['feasible'])
        self.assertIsNone(result['upper_ghz'])
        self.assertTrue(result['upper_unbounded'])
        json.dumps(result,allow_nan=False)
        # Actual L acceptance must reject this under the prelisted duration
        # gate; the math API does not invent a finite upper frequency.

    def test_bad_stamps_and_allowances_are_rejected(self):
        for window in [((10,100,9),(20,200,21)),((0,100,1),(2,100,3)),
                       ((0,100.,1),(2,200,3)),((10,100,11),(1,200,2))]:
            with self.subTest(window=window),self.assertRaises(ValueError):
                common_rate_interval([window])
        for tolerance in (-1,math.inf,math.nan):
            with self.subTest(tolerance=tolerance),self.assertRaises(ValueError):
                common_rate_interval([exact(100,100)],interval_tolerance_ns=tolerance)

    def test_observer_screen_cannot_cancel_opposite_changes(self):
        pairs = observer_pairs()
        self.assertTrue(observer_perturbation_gate(pairs,expected_ctas=2)['passes'])
        for i,p in enumerate(pairs):
            for x in p['cta_pairs']:
                if x['phase']=='first':x['L_ns_delta']=400 if i%2 else -400
        result = observer_perturbation_gate(pairs,expected_ctas=2)
        self.assertEqual(result['metrics']['first_L_ns_delta']['median_signed'],0)
        self.assertFalse(result['passes'])
        self.assertFalse(result['center_screen_passes'])

    def test_local_outliers_and_event_changes_remain_inconclusive(self):
        pairs = observer_pairs()
        for x in pairs[0]['cta_pairs']:
            if x['phase']=='later':x['L_ns_delta']=1025
        self.assertTrue(observer_perturbation_gate(pairs,expected_ctas=2)['center_screen_passes'])
        for x in pairs[1]['cta_pairs']:
            if x['phase']=='later':x['L_ns_delta']=1025
        self.assertFalse(observer_perturbation_gate(pairs,expected_ctas=2)['center_screen_passes'])
        for p in pairs:
            for x in p['cta_pairs']:x['L_ns_delta']=0
            p['event_ns_delta']=513
        self.assertFalse(observer_perturbation_gate(pairs,expected_ctas=2)['passes'])
        with self.assertRaises(ValueError):
            observer_perturbation_gate(pairs[:9],expected_ctas=2)

    def test_spatial_exchange_does_not_pass_zero_phase_centers(self):
        pairs = observer_pairs()
        for p in pairs:
            for x in p['cta_pairs']:
                x['L_ns_delta']=1024 if x['cta']==0 else -1024
        result = observer_perturbation_gate(pairs,expected_ctas=2)
        self.assertTrue(result['center_screen_passes'])
        self.assertFalse(result['passes'])
        self.assertIn('unresolved',result['spatial_screen_status'])
        first = result['spatial_processes'][0]['phases']['first']['L_ns_delta']
        self.assertEqual(first['center_signed'],0)
        self.assertEqual(first['median_absolute'],1024)
        self.assertEqual(first['maximum_absolute'],1024)
        self.assertEqual(first['signed_values'],[1024,-1024])

    def test_uniform_phase_shift_has_no_spatial_change_but_keeps_center_screen(self):
        pairs = observer_pairs()
        for p in pairs:
            for x in p['cta_pairs']:x['L_ns_delta']=100
        result = observer_perturbation_gate(pairs,expected_ctas=2)
        self.assertTrue(result['exact_spatial_identity'])
        self.assertTrue(result['passes'])
        for p in pairs:
            for x in p['cta_pairs']:x['L_ns_delta']=400
        result = observer_perturbation_gate(pairs,expected_ctas=2)
        self.assertTrue(result['exact_spatial_identity'])
        self.assertFalse(result['center_screen_passes'])
        self.assertFalse(result['passes'])

    def test_missing_or_duplicated_cta_cannot_be_hidden_by_aggregation(self):
        pairs = observer_pairs()
        pairs[0]['cta_pairs'][0]=pairs[0]['cta_pairs'][1]
        with self.assertRaises(ValueError):
            observer_perturbation_gate(pairs,expected_ctas=2)
        with self.assertRaises(KeyError):
            observer_perturbation_gate([dict(event_ns_delta=0) for _ in range(10)],expected_ctas=2)

    def test_local_varying_rate_fits_unequal_interval_rates(self):
        result=local_clock_witness([(0,0),(100,100),(300,200)],declared_rate_cap_ghz=2)
        self.assertEqual(result['minimum_endpoint_slack_ns'],0)
        self.assertEqual(result['segment_rate_ghz'],[1,2])
        # A varying-rate existence witness does not assert one constant rate.
        self.assertFalse(common_rate_interval([exact(100,100),exact(200,100)],interval_tolerance_ns=0)['feasible'])

    def test_local_witness_minimum_is_attained_at_inverted_pair(self):
        result=local_clock_witness([(0,100),(100,50)],declared_rate_cap_ghz=2)
        self.assertEqual(result['minimum_endpoint_slack_ns'],50)
        self.assertEqual(result['adjusted_relative_ns'],[0,50])
        self.assertEqual(result['segment_rate_ghz'],[2])

    def test_local_clock_origins_and_equal_counter_values(self):
        origin=1<<60
        result=local_clock_witness([(1000,origin),(1000,origin+10),(1020,origin+20)],declared_rate_cap_ghz=2)
        self.assertEqual(result['minimum_endpoint_slack_ns'],0)
        self.assertEqual(result['segment_rate_ghz'],[0,2])
        self.assertEqual(result['original_relative_ns'],[0,10,20])


if __name__=='__main__':
    unittest.main()
