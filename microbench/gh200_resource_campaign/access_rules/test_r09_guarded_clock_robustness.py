from fractions import Fraction
import unittest

from r09_guarded_clock_robustness import bracket_graph, minimum_clock_allowance, cycle_lower_bound


class GuardedClockRobustness(unittest.TestCase):
    def test_same_timer_distinct_cycles_need_half_cycle_even_without_rate_cap(self):
        points={0:[(0,0,0),(1,0,1)]}
        result=minimum_clock_allowance(points,0,None)
        self.assertEqual(result['minimum_extra_clock_cycles_fraction'],[1,2])
        self.assertTrue(result['attained'])
        lower=result['separating_cycles'][-1]
        self.assertEqual(cycle_lower_bound(lower['graph'],lower['certificate']),Fraction(1,2))
        self.assertEqual(minimum_clock_allowance(points,1,None)['minimum_extra_clock_cycles'],0)

    def test_analytic_capped_interval_and_widened_native_bracket(self):
        narrow={0:[(0,0,0),(200,100,200)]}
        self.assertEqual(minimum_clock_allowance(narrow,0,1)['minimum_extra_clock_cycles'],50)
        wide={0:[(0,0,100),(200,100,200)]}
        self.assertEqual(minimum_clock_allowance(wide,0,1)['minimum_extra_clock_cycles'],0)

    def test_equal_windows_with_375_cycle_gap_have_quarter_gap_minimum(self):
        points={3:[(1000,0,1000),(1100,100,1100)],7:[(9000,0,9000),(9475,100,9475)]}
        self.assertEqual(minimum_clock_allowance(points,0,None)['minimum_extra_clock_cycles_fraction'],[375,4])
        self.assertEqual(minimum_clock_allowance(points,0,1.98)['minimum_extra_clock_cycles_fraction'],[277,2])

    def test_independent_epochs_and_timer_translation_preserve_minimum(self):
        points={3:[(0,0,0),(100,100,100)],7:[(0,25,0),(200,75,200)]}
        shifted={sm:[(a+10**18*(sm+1),n+10**18,z+10**18*(sm+1)) for a,n,z in values]
                 for sm,values in points.items()}
        a=minimum_clock_allowance(points,4,2)
        b=minimum_clock_allowance(shifted,4,2)
        self.assertEqual(a['minimum_extra_clock_cycles_fraction'],b['minimum_extra_clock_cycles_fraction'])
        self.assertEqual(a['feasible_certificate'],b['feasible_certificate'])

    def test_bad_allowances_and_reversed_brackets_fail(self):
        for allowance in (True,-1,.5):
            with self.subTest(allowance=allowance),self.assertRaises(ValueError):
                bracket_graph({0:[(1,1,2)]},0,allowance)
        with self.assertRaises(ValueError):bracket_graph({0:[(3,1,2)]},0,0)


if __name__=='__main__':
    unittest.main()
