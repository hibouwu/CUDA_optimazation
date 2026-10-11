import unittest

from r09_clock_brackets import observer_L_window, common_rate_interval


def tile():
    return [100,6500,6600,6700,1,1,10000,14000,14100,14200,108,6508]


class ClockObserverWords(unittest.TestCase):
    def test_guarded_field_mapping_and_native_bounds(self):
        result=observer_L_window(tile(),'guarded_bracket')
        self.assertEqual(result['cycles'],6400)
        self.assertEqual(result['ns'],4000)
        self.assertEqual(result['begin'],(100,10000,108))
        self.assertEqual(result['end'],(6500,14000,6508))
        self.assertEqual(result['bounds'],dict(cycles_lo=6392,cycles_hi=6408,ns=4000))
        self.assertEqual(result['work_coordinate'],[0,0])

    def test_matched_baseline_and_guarded_arm_cannot_fallback(self):
        baseline=tile();baseline[10:]=[0,0]
        result=observer_L_window(baseline,'matched_dual')
        self.assertFalse(result['clock_bracket_present'])
        self.assertNotIn('begin',result)
        with self.assertRaises(ValueError):observer_L_window(baseline,'guarded_bracket')
        with self.assertRaises(ValueError):observer_L_window(tile(),'matched_dual')

    def test_bad_width_arm_and_non_uint64_fail(self):
        for values,arm in [(tile()[:10],'guarded_bracket'),(tile(),'plain'),(tile(),'dual')]:
            with self.subTest(arm=arm),self.assertRaises(ValueError):observer_L_window(values,arm)
        for value in (True,100.0,-1,2**64):
            bad=tile();bad[0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):observer_L_window(bad,'guarded_bracket')

    def test_missing_endpoints_or_work_cannot_pass(self):
        for index in range(12):
            bad=tile();bad[index]=0
            with self.subTest(index=index),self.assertRaises(ValueError):observer_L_window(bad,'guarded_bracket')

    def test_read_and_original_event_order_are_checked(self):
        for index,value in [(10,99),(10,6501),(11,6499),(11,6601),(2,6499),(7,9999)]:
            bad=tile();bad[index]=value
            with self.subTest(index=index,value=value),self.assertRaises(ValueError):observer_L_window(bad,'guarded_bracket')

    def test_decoded_windows_reach_the_existing_exact_rate_test(self):
        a=observer_L_window(tile(),'guarded_bracket')
        b=observer_L_window(tile(),'guarded_bracket')
        self.assertTrue(common_rate_interval([(a['begin'],a['end']),(b['begin'],b['end'])])['feasible'])
        changed=tile();changed[1:4]=[9500,9600,9700];changed[11]=9508
        b=observer_L_window(changed,'guarded_bracket')
        self.assertFalse(common_rate_interval([(a['begin'],a['end']),(b['begin'],b['end'])])['feasible'])


if __name__=='__main__':
    unittest.main()
