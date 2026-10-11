import unittest
from analyze_v09_crosscard import compare_static


class StaticCrosscardTests(unittest.TestCase):
    def test_reference_label_is_not_a_workload_difference(self):
        self.assertEqual(compare_static(dict(config='cfg_a', set='calib'),
                                        dict(config='cfg_a', set='reference')), {})

    def test_changed_history_and_allocation_are_not_silently_matched(self):
        delta = compare_static(dict(config='cfg_a', alloc_lda=1024),
                               dict(config='cfg_a', alloc_lda=1032, history_calls=16))
        self.assertEqual(set(delta), {'alloc_lda', 'history_calls'})

    def test_missing_field_is_not_assumed_to_have_default_value(self):
        self.assertIn('seed', compare_static(dict(config='cfg_a'), dict(config='cfg_a', seed=17)))


if __name__ == '__main__':
    unittest.main()
