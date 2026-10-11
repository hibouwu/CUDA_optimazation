import unittest
from dataclasses import replace
from fractions import Fraction as F

from r09_gt_brackets import GtClockStamp, error_intervals, relative_error_status, window_interval


class GtBracketTests(unittest.TestCase):
    def setUp(self):
        self.begin = GtClockStamp('GPU-test', 3, 100, 1000, 108)
        self.end = GtClockStamp('GPU-test', 3, 200, 1150, 212)

    def test_exact_elapsed_rate_and_per_read_budget(self):
        result = window_interval(self.begin, self.end, gt_error_ns=F(2))
        self.assertEqual((result['elapsed_lower_ns'], result['elapsed_upper_ns']), (88, 116))
        self.assertEqual((result['rate_lower_cycle_per_ns'], result['rate_upper_cycle_per_ns']), (F(75,58), F(75,44)))
        self.assertEqual(result['elapsed_upper_ns']-result['elapsed_lower_ns'], 8+12+4*2)
        self.assertFalse(result['observation_qualified'])

    def test_shared_elapsed_error_endpoints_and_interior(self):
        result = error_intervals(self.begin, self.end, gt_error_ns=2,
                                predicted_cycles=144, predicted_rate=F(3,2))
        self.assertEqual(result['cycle_error'], -F(1,25))
        self.assertEqual((result['conversion_error_lower'], result['conversion_error_upper']), (-F(4,29), F(3,22)))
        self.assertEqual((result['free_error_lower'], result['free_error_upper']), (-F(5,29), F(1,11)))
        for elapsed in (F(88),F(177,2),F(96),F(100),F(116)):
            conversion = (F(150)/elapsed)/F(3,2)-1
            free = F(96)/elapsed-1
            self.assertEqual((1+result['cycle_error'])*(1+conversion), 1+free)
            self.assertLessEqual(result['free_error_lower'], free)
            self.assertLessEqual(free, result['free_error_upper'])

    def test_large_origins_removed_as_integers(self):
        shifted_begin = replace(self.begin, gt_before=self.begin.gt_before+2**60,
            gt_after=self.begin.gt_after+2**60, cycle=self.begin.cycle+2**55)
        shifted_end = replace(self.end, gt_before=self.end.gt_before+2**60,
            gt_after=self.end.gt_after+2**60, cycle=self.end.cycle+2**55)
        self.assertEqual(window_interval(self.begin,self.end,gt_error_ns=F(1,2)),
                         window_interval(shifted_begin,shifted_end,gt_error_ns=F(1,2)))

    def test_overlap_has_open_zero_and_unbounded_rate(self):
        end = replace(self.end, gt_before=104, gt_after=112)
        result = error_intervals(self.begin,end,gt_error_ns=0,predicted_cycles=144,predicted_rate=F(3,2))
        self.assertEqual(result['elapsed_lower_ns'],0)
        self.assertTrue(result['elapsed_lower_open'])
        self.assertIsNone(result['rate_upper_cycle_per_ns'])
        self.assertIsNone(result['free_error_upper'])
        self.assertEqual(result['elapsed_upper_ns'],12)

    def test_budget_can_restore_positive_time_but_not_finite_upper_rate(self):
        begin = replace(self.begin,gt_before=100,gt_after=100)
        end = replace(self.end,gt_before=100,gt_after=100)
        with self.assertRaises(ValueError):window_interval(begin,end,gt_error_ns=0)
        result = window_interval(begin,end,gt_error_ns=1)
        self.assertEqual(result['elapsed_upper_ns'],2)
        self.assertIsNone(result['rate_upper_cycle_per_ns'])

    def test_conditional_status_does_not_select_a_favorable_time(self):
        self.assertEqual(relative_error_status(-F(5,29),F(1,11),tolerance=F(1,20))['status'],'undetermined')
        self.assertEqual(relative_error_status(-F(5,29),F(1,11),tolerance=F(1,5))['status'],'within')
        self.assertEqual(relative_error_status(F(1,5),F(2,5),tolerance=F(1,10))['status'],'outside')
        self.assertEqual(relative_error_status(-F(2,5),-F(1,5),tolerance=F(1,10))['status'],'outside')
        self.assertEqual(relative_error_status(-F(1,20),None,tolerance=F(1,10))['status'],'undetermined')
        self.assertEqual(relative_error_status(F(1,5),None,tolerance=F(1,10))['status'],'outside')
        self.assertEqual(relative_error_status(0,0,tolerance=0)['status'],'within')

    def test_counter_domains_and_invalid_native_stamps_rejected(self):
        invalid=(replace(self.end,gpu_uuid='other-card'),replace(self.end,sm=4),
            replace(self.end,cycle=self.begin.cycle),replace(self.end,cycle=999),
            replace(self.end,gt_before=0),replace(self.end,gt_after=199),
            replace(self.end,cycle=2**64),replace(self.end,gt_after=True),replace(self.end,sm=True))
        for end in invalid:
            with self.subTest(end=end),self.assertRaises(ValueError):window_interval(self.begin,end,gt_error_ns=0)

    def test_explicit_exact_budget_and_positive_predictions_required(self):
        for budget in (None,-1,1.0,float('nan'),True):
            with self.subTest(budget=budget),self.assertRaises(ValueError):window_interval(self.begin,self.end,gt_error_ns=budget)
        for cycles,rate in ((0,F(1)),(F(1),0),(-1,F(1)),(144,1.5)):
            with self.subTest(cycles=cycles,rate=rate),self.assertRaises(ValueError):
                error_intervals(self.begin,self.end,gt_error_ns=0,predicted_cycles=cycles,predicted_rate=rate)

    def test_counter_unit_rescaling_preserves_composed_time_error(self):
        before = error_intervals(self.begin,self.end,gt_error_ns=0,predicted_cycles=144,predicted_rate=F(3,2))
        after = error_intervals(replace(self.begin,cycle=2*self.begin.cycle),
            replace(self.end,cycle=2*self.end.cycle),gt_error_ns=0,predicted_cycles=288,predicted_rate=3)
        for key in ('cycle_error','conversion_error_lower','conversion_error_upper','free_error_lower','free_error_upper','predicted_ns'):
            self.assertEqual(before[key],after[key])


if __name__=='__main__':unittest.main()
