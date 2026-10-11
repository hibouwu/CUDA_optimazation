import math
import unittest

from r09_time_contract import constant_rate_slack, cycle_rate_split, relative_point_floor, stratified_floor


class TimeContract(unittest.TestCase):
    def test_minimax_bound_is_tight_and_rms_point_is_stationary(self):
        values = [10,10,15,20]
        result = relative_point_floor(values)
        point = result['minimax_point']
        self.assertAlmostEqual(point/10-1,1-point/20)
        self.assertAlmostEqual(max(abs(point/y-1) for y in values),result['maximum_relative_floor'])
        rms_point = result['rms_point']
        self.assertAlmostEqual(sum((rms_point/y-1)/y for y in values),0)
        for offset in (-1,1):
            self.assertGreater(sum(((rms_point+offset)/y-1)**2 for y in values),
                               sum((rms_point/y-1)**2 for y in values))

    def test_nonpositive_or_nonfinite_data_do_not_form_a_time_bound(self):
        for values in ([],[0,1],[-1,1],[math.inf,1],[math.nan,1]):
            with self.subTest(values=values),self.assertRaises(ValueError):
                relative_point_floor(values)

    def test_repeat_spread_is_separate_from_history_centers(self):
        result = stratified_floor({'early':[9,10,11],'late':[19,20,21]})
        self.assertAlmostEqual(result['history_centers']['maximum_relative_floor'],1/3)
        self.assertGreater(result['all_observations']['maximum_relative_floor'],1/3)
        self.assertGreater(result['between_stratum_fraction'],.9)
        self.assertAlmostEqual(stratified_floor({'a':[1,1],'b':[1,1]})['between_stratum_fraction'],0)

    def test_service_and_rate_opposite_signs_remain_visible(self):
        # More service cycles and higher rate can cancel in total time.
        result = cycle_rate_split(100,100,200,100)
        self.assertEqual(result['delta_ns'],0)
        self.assertGreater(result['cycle_term_ns'],0)
        self.assertLess(result['inverse_rate_term_ns'],0)
        self.assertAlmostEqual(result['cycle_term_ns']+result['inverse_rate_term_ns'],0)
        reverse = cycle_rate_split(200,100,100,100)
        self.assertAlmostEqual(result['cycle_term_ns'],-reverse['cycle_term_ns'])

    def test_constant_rate_slack_is_necessary_and_sufficient_for_interval_bounds(self):
        windows = [(100,100),(200,300),(40,50)]
        result = constant_rate_slack(windows)
        radius = result['interval_slack_ns']
        f = result['best_constant_ghz']
        self.assertAlmostEqual(max(abs(n-c/f) for c,n in windows),radius)
        lower = max((n-radius+1e-6)/c for c,n in windows)
        upper = min((n+radius-1e-6)/c for c,n in windows)
        self.assertGreater(lower,upper)
        self.assertEqual(constant_rate_slack([(100,50),(200,100)])['interval_slack_ns'],0)


if __name__=='__main__':
    unittest.main()
