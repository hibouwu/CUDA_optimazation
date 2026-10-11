"""Interval bands, observer gates and failure retention; no GPU timing claim."""
import copy
import unittest

from analyze_r13_release_interpretation import (interval_classification,
    paired_observer_gates, release_process, release_thread_process)
from analyze_r13_release import release_arrival_bounds, thread_release_arrival_bounds


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
    def test_unobserved_nonleader_can_reverse_four_leader_classification(self):
        threads = list(range(256,384))
        before,after = [100]*128,[110]*128
        before[1],after[1] = 200,210
        four = release_arrival_bounds([100]*4,[110]*4,230)
        all_threads = thread_release_arrival_bounds(threads,before,after,230)
        self.assertEqual(interval_classification(four['after_last_emit_to_acquire_end']),'excluded')
        self.assertEqual(interval_classification(all_threads['after_last_emit_to_acquire_end']),'compatible')
        self.assertEqual(all_threads['last_arrival_emit'],dict(lower=200,upper=210))
        reverse = thread_release_arrival_bounds(threads[::-1],before[::-1],after[::-1],230)
        self.assertEqual(reverse,all_threads)

    def test_required_thread_coverage_is_exact(self):
        threads = list(range(256,384))
        for ids in (threads[:4],threads[:-1]+[382],threads[:-1]+[384],threads[:-1]+[True]):
            with self.assertRaisesRegex(ValueError,'128 distinct'):
                thread_release_arrival_bounds(ids,[100]*len(ids),[110]*len(ids),230)
        with self.assertRaises(ValueError):
            thread_release_arrival_bounds(threads,[100]*127,[110]*128,230)

    def test_required_thread_clock_safety_and_impossible_acquire(self):
        threads = list(range(256,384))
        for invalid in (0,-1,True,1.,float('nan'),float('inf'),2**64):
            for field in ('pre','post','acquire'):
                before,after,acquire = [100]*128,[110]*128,230
                if field=='pre':before[1]=invalid
                elif field=='post':after[1]=invalid
                else:acquire=invalid
                with self.assertRaises(ValueError):
                    thread_release_arrival_bounds(threads,before,after,acquire)
        with self.assertRaisesRegex(ValueError,'ordered'):
            thread_release_arrival_bounds(threads,[120]*128,[110]*128,230)
        with self.assertRaisesRegex(ValueError,'producer returned'):
            thread_release_arrival_bounds(threads,[100]*128,[110]*128,99)

    def test_v2_never_silently_uses_four_compatibility_leaders(self):
        parsed = sparse([1000]*4,[1010]*4,1230)
        parsed['clocks']['release_end'] = 1010
        parsed['release_threads'] = dict(enabled=True,physical_threads=list(range(256,384)),
            pre=[1000]*128,post=[1010]*128,
            conditional_on_sass_instruction_order=True,requires_warp_simultaneity=False)
        parsed['release_threads']['pre'][1] = 1200
        parsed['release_threads']['post'][1] = 1210
        with self.assertRaisesRegex(ValueError,'drop required arrivals'):
            release_process(parsed)
        result = release_thread_process(parsed)
        self.assertEqual(result['bounds']['last_arrival_emit'],dict(lower=1200,upper=1210))
        self.assertEqual(result['conditional_classification']['late_required_arrivals'],'compatible')
        parsed['release_threads']['requires_warp_simultaneity'] = True
        with self.assertRaisesRegex(ValueError,'contract'):
            release_thread_process(parsed)

    def test_v2_off_does_not_claim_full_thread_observation(self):
        parsed = sparse([1000,0,0,0],[1010,0,0,0],enabled=False)
        parsed['clocks']['release_end'] = 1010
        parsed['release_threads'] = dict(enabled=False,physical_threads=list(range(256,384)),
            pre=[1000]+[0]*127,post=[1010]+[0]*127,
            conditional_on_sass_instruction_order=True,requires_warp_simultaneity=False)
        result = release_thread_process(parsed)
        self.assertFalse(result['required_thread_brackets'])
        self.assertIsNone(result['conditional_classification'])

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
