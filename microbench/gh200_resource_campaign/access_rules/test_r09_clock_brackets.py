import json
import math
import unittest

from r09_clock_brackets import common_rate_interval, observer_perturbation_gate, window_bounds


def exact(cycles, ns, origin=1000):
    return ((0,origin,0),(cycles,origin+ns,cycles))


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
        keys = ('first_L_ns_delta','later_L_ns_delta','first_L_cycles_delta',
                'later_L_cycles_delta','event_ns_delta')
        pairs = [dict.fromkeys(keys,0) for _ in range(10)]
        self.assertTrue(observer_perturbation_gate(pairs)['passes'])
        for i,p in enumerate(pairs):
            p['first_L_ns_delta']=400 if i%2 else -400
        result = observer_perturbation_gate(pairs)
        self.assertEqual(result['metrics']['first_L_ns_delta']['median_signed'],0)
        self.assertFalse(result['passes'])

    def test_local_outliers_and_event_changes_remain_inconclusive(self):
        keys = ('first_L_ns_delta','later_L_ns_delta','first_L_cycles_delta',
                'later_L_cycles_delta','event_ns_delta')
        pairs = [dict.fromkeys(keys,0) for _ in range(10)]
        pairs[0]['later_L_ns_delta']=1025
        self.assertTrue(observer_perturbation_gate(pairs)['passes'])
        pairs[1]['later_L_ns_delta']=1025
        self.assertFalse(observer_perturbation_gate(pairs)['passes'])
        for p in pairs:
            p['later_L_ns_delta']=0
            p['event_ns_delta']=513
        self.assertFalse(observer_perturbation_gate(pairs)['passes'])
        with self.assertRaises(ValueError):
            observer_perturbation_gate(pairs[:9])


if __name__=='__main__':
    unittest.main()
