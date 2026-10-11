"""A source-guarantee stamp must not become an exact source-completion time."""
import copy
import unittest

from analyze_r15_lifecycle import output_source_windows, output_cycle_bounds


class OutputLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for role in (1,2):
            for tile,(main,permit,release) in enumerate([(10,12,110),(100,102,150)]):
                self.rows.append(dict(role=role,seq=tile,main_end_cycle=main,
                                      epi_permit_cycle=permit,source_release_cycle=release if role==2 else None))

    def test_middle_interval_contains_next_output_and_terminal_does_not(self):
        windows=output_source_windows(self.rows,2)
        self.assertEqual(windows[0]['permit_to_release_cycles'],98)
        self.assertEqual(windows[0]['carried_to_next_permit_cycles'],90)
        self.assertEqual(windows[0]['after_next_permit_cycles'],8)
        self.assertEqual(windows[0]['guarantee_token'],'r0.d4.acquire_after')
        self.assertEqual(windows[1]['terminal_service_upper_cycles'],48)
        self.assertEqual(windows[1]['carried_to_next_permit_cycles'],0)
        self.assertEqual(windows[1]['guarantee_token'],'r0.store_tail_read0')

    def test_guarantee_does_not_force_actual_source_read_to_wait_for_next_output(self):
        windows=output_source_windows(self.rows,2)
        setup=dict(stages=6,k=128,tiles_per_cta=2)
        intervals=output_cycle_bounds(setup,self.rows,windows)
        self.assertLess(intervals[0]['lower_cycles'],windows[0]['guarantee_lower_cycle']-10)
        self.assertLessEqual(intervals[0]['upper_cycles'],100)
        self.assertEqual(intervals[0]['observed_guarantee_upper_cycles'],100)

    def test_reject_middle_guarantee_before_next_permit_and_missing_issuer(self):
        rows=copy.deepcopy(self.rows)
        rows[2]['source_release_cycle']=101
        with self.assertRaisesRegex(ValueError,'next output'):
            output_source_windows(rows,2)
        with self.assertRaisesRegex(ValueError,'issuer tile'):
            output_source_windows(self.rows[:-1],2)


if __name__=='__main__':
    unittest.main()
