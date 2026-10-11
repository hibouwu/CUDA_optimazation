"""Interval bands, observer gates and failure retention; no GPU timing claim."""
import copy
import unittest

from analyze_r13_release_interpretation import (interval_classification,
    paired_observer_gates, release_process)


def sparse(before, after, acquire=1230, enabled=True):
    return dict(clocks=dict(refill_begin=900,release_begin=before[0],
        refill_end=acquire,full_wait_begin=500,wait1_end=800),
        release_warps=dict(pre=before,post=after,enabled=enabled))


def paired_fixture():
    return [dict(stages=s,trial=t,observer=f,gpu_uuid='fixture',sm=1,
        work=[0,0],slot=80%s,phase=(80//s+1)%2,elapsed_us=50.,
        diagnostic=dict(leader_pre_to_acquire_end_cycles=200,
            H16_to_wait1_return_cycles=400,reasons=[]))
        for s in (4,6) for t in range(10) for f in (0,1)]


class ReleaseInterpretationTests(unittest.TestCase):
    def test_resolution_band_boundaries(self):
        for low,high,expected in ((0,64,'compatible'),(65,80,'excluded'),(64,65,'unresolved')):
            self.assertEqual(interval_classification(dict(lower=low,upper=high)),expected)
        with self.assertRaises(ValueError):
            interval_classification(dict(lower=80,upper=65))

    def test_distinct_feasible_release_explanations(self):
        late = release_process(sparse([1000,1180,1170,1160],[1010,1190,1180,1170],1220))
        post = release_process(sparse([1000,980,980,980],[1010,990,990,990]))
        self.assertEqual(late['conditional_classification'],dict(
            late_required_arrivals='compatible',post_arrival_path='excluded'))
        self.assertEqual(post['conditional_classification'],dict(
            late_required_arrivals='excluded',post_arrival_path='compatible'))

    def test_mixed_and_unresolved_intervals_remain(self):
        mixed = release_process(sparse([1000,1100,1100,1100],[1010,1130,1130,1130],1200))
        self.assertEqual(set(mixed['conditional_classification'].values()),{'excluded'})
        unresolved = release_process(sparse([1000,1100,1100,1100],[1010,1120,1120,1120],1180))
        self.assertEqual(unresolved['conditional_classification']['late_required_arrivals'],'unresolved')

    def test_wide_brackets_and_late_waiter_are_retained(self):
        result = release_process(sparse([1000,980,980,980],[1033,990,990,990]))
        self.assertIn('warp_bracket_exceeds_32_cycles',result['reasons'])
        self.assertIsNotNone(result['conditional_classification'])
        off = sparse([1000,0,0,0],[1033,0,0,0],enabled=False)
        off['clocks']['refill_begin'] = 1000
        result = release_process(off)
        self.assertEqual(set(result['reasons']),{
            'warp_bracket_exceeds_32_cycles','producer_not_waiting_before_leader_release'})
        self.assertIsNone(result['conditional_classification'])

    def test_complete_unperturbed_pairs_pass(self):
        result = paired_observer_gates(paired_fixture())
        self.assertTrue(all(s['gate_passed'] for s in result))
        self.assertTrue(all(s['median_convention']=='median_absolute' for s in result))

    def test_gates_are_inclusive_at_original_limits(self):
        processes = paired_fixture()
        for p in processes:
            if p['observer']:
                p['diagnostic']['leader_pre_to_acquire_end_cycles'] += 128 if p['trial']==9 else 32
                p['diagnostic']['H16_to_wait1_return_cycles'] += 192 if p['trial']==9 else 64
                p['elapsed_us'] = 52.5 if p['trial']==9 else 51.
        self.assertTrue(all(s['gate_passed'] for s in paired_observer_gates(processes)))

    def test_signed_cancellation_cannot_pass(self):
        processes = paired_fixture()
        for p in processes:
            if p['observer']:
                p['diagnostic']['leader_pre_to_acquire_end_cycles'] += 60 if p['trial']%2 else -60
        for stage in paired_observer_gates(processes):
            metric = stage['metrics']['local_cycles']
            self.assertEqual(metric['absolute_median'],0)
            self.assertEqual(metric['median_absolute'],60)
            self.assertEqual(metric['status'],'fail')
            self.assertFalse(stage['gate_passed'])

    def test_single_maximum_failure_cannot_hide_in_median(self):
        processes = paired_fixture()
        processes[1]['diagnostic']['leader_pre_to_acquire_end_cycles'] += 129
        stage = paired_observer_gates(processes)[0]
        self.assertEqual(stage['metrics']['local_cycles']['median_absolute'],0)
        self.assertFalse(stage['gate_passed'])

    def test_missing_or_wrong_identity_pairs_remain_invalid(self):
        processes = paired_fixture()
        processes.pop(1)
        stage = paired_observer_gates(processes)[0]
        self.assertIn('missing_paired_arm',stage['pairs'][0]['reasons'])
        self.assertFalse(stage['gate_passed'])
        processes = paired_fixture()
        processes[1]['sm'] = 2
        stage = paired_observer_gates(processes)[0]
        self.assertIn('paired_identity_differs_sm',stage['pairs'][0]['reasons'])
        self.assertFalse(stage['gate_passed'])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            paired_observer_gates(processes+[copy.deepcopy(processes[0])])

    def test_invalid_trace_and_small_sample_cannot_qualify(self):
        processes = paired_fixture()
        processes[1]['diagnostic']['reasons'].append('warp_bracket_exceeds_32_cycles')
        self.assertFalse(paired_observer_gates(processes)[0]['gate_passed'])
        processes[1]['decode_error'] = 'invalid clock64'
        self.assertFalse(paired_observer_gates(processes)[0]['gate_passed'])
        self.assertTrue(all(not s['gate_passed'] for s in paired_observer_gates(processes[:2])))


if __name__=='__main__':
    unittest.main()
