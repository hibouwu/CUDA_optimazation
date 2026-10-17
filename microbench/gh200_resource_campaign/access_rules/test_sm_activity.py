"""活动报告保留服务、排队与复合窗的不同计账边界。"""
import unittest

from sm_activity import activity_windows, overlap_ns


class SmActivityTests(unittest.TestCase):
    def test_interval_unions_count_overlap_once_and_use_half_open_boundaries(self):
        self.assertEqual(overlap_ns([[0,10],[5,20]],[[8,16],[12,18]]),10)
        self.assertEqual(overlap_ns([[0,10],[10,20]],[[20,30],[3,3]]),0)

    def test_queued_compute_is_not_counted_twice_and_input_keeps_destination_wait(self):
        actor = dict(starts=[0.,5.],service_frontier=[10.,20.],
            outputs=[dict(prepare_begin=[3.],prepare_end=[5.],write_begin=[8.],write_end=[16.])])
        deliveries = [dict(arrival_ns=0.,ready_ns=10.),dict(arrival_ns=0.,ready_ns=10.),
                      dict(arrival_ns=5.,ready_ns=15.)]
        report = activity_windows(actor,deliveries,prefix_ns=50.,detailed=True)
        self.assertEqual(report['composite_compute_service_ns'],20.)
        self.assertEqual(report['compute_write_overlap_ns'],8.)
        self.assertEqual(report['compute_prepare_overlap_ns'],2.)
        self.assertEqual(report['input_pending_peak'],3)
        self.assertEqual(report['windows_ns']['composite_compute_service'],[[50.,60.],[60.,70.]])
        self.assertEqual(report['windows_ns']['input_pending'][-1],[55.,65.])

    def test_compact_summary_and_zero_duration_windows(self):
        actor = dict(starts=[0.],service_frontier=[1.],
            outputs=[dict(prepare_begin=[1.],prepare_end=[1.],write_begin=[1.],write_end=[1.])])
        report = activity_windows(actor,[dict(arrival_ns=0.,ready_ns=0.)])
        self.assertNotIn('windows_ns',report)
        self.assertEqual(report['input_pending_peak'],0)
        self.assertEqual(report['epilogue_write_ns'],0.)
        self.assertEqual(report['compute_write_overlap_ns'],0.)


if __name__ == '__main__':
    unittest.main()
