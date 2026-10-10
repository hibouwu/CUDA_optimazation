"""Check the measurement reductions and the stage max-branch diagnosis."""
import copy
import unittest

from analyze_r13_supply import (
    _failure_pair_summary, _failure_pair_windows, _failure_stage_response,
    _failure_k_scaling,
    _finite_stage_cycles,
    _commit_paced_stage_cycles,
    _loop_paced_stage_cycles,
    _stage_event_intervals,
    _stage_progress_intervals,
    _stage_candidate_progress,
    _stage_handoff_windows,
)


def cta(index, length):
    return dict(cta=index, work=[[index, 0], [index, 1]], entry_ns=0, prod_ns=2,
        end_ns=2*length+40,
        tiles_ns=[[10, 10+length, 11+length, 15+length],
                  [20+length, 20+2*length, 21+2*length, 25+2*length]],
        tiles=[[20, 20+2*length, 22+2*length, 30+2*length],
               [40+2*length, 40+4*length, 42+4*length, 50+4*length]])


class PairedFailureTests(unittest.TestCase):
    def test_pairing_uses_identity_and_coordinates_not_list_order(self):
        a = [cta(0, 10), cta(1, 20)]
        b = [cta(1, 30), cta(0, 20)]
        rows = _failure_pair_windows(a, b)
        self.assertEqual([(r['cta'], r['j']) for r in rows], [(0,0),(0,1),(1,0),(1,1)])
        self.assertEqual([r['L_ns_delta'] for r in rows], [10,10,10,10])
        self.assertEqual(rows[0]['L_rate_ratio'], 0)
        b[0]['work'][1] = [99, 1]
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            _failure_pair_windows(a, b)

    def test_ratios_are_paired_before_reduction(self):
        a = [cta(0,1), cta(1,10), cta(2,100)]
        b = [cta(0,2), cta(1,100), cta(2,100)]
        control = [dict(trial=t,row=dict(id='base'),observed=dict(ctas=a)) for t in range(10)]
        variant = [dict(trial=t,row=dict(id='variant'),observed=dict(ctas=b)) for t in range(10)]
        summary, positions = _failure_pair_summary('synthetic', control, variant)
        self.assertEqual(summary['phases']['first']['L_ns_ratio']['median'], 1)
        self.assertEqual(summary['phases']['later']['L_ns_ratio']['median'], 1)
        self.assertEqual(len(positions), 20)
        # Ratio of condition medians would be 9, and is not our paired statistic.
        self.assertNotEqual(summary['phases']['first']['L_ns_ratio']['median'], 100/10-1)
        with self.assertRaisesRegex(ValueError, 'ten successful trials'):
            _failure_pair_summary('synthetic', control[:-1], variant)

    def test_compute_max_can_hide_a_changed_stage_feature(self):
        row = dict(id='base',config='cfg_b',m=2816,n=1536,k=4096,swizzle=1)
        setup = dict(grid=[1,132,1],stages=6,smem=215040,max_active_ctas_per_sm=1)
        model = dict(projection=[[1],[0],[0],[0],[0],[0],[0]], parameters=[10,.1])
        calibration = dict(first_supply=dict(cfg_b=model),supply=dict(cfg_b=model))
        prediction = dict(frequency_ghz=1.,window_us=1.,plain_us=1.,cta_duration_ns=[1.],cta_events=[])
        control = dict(row=row,setup=setup,calibration=calibration,prediction=prediction)
        variant = copy.deepcopy(control)
        variant['setup'].update(stages=4,smem=149504)
        result = _failure_stage_response(control, variant)
        first6, later6 = result['stage6']['positions']
        first4, later4 = result['stage4']['positions']
        self.assertEqual(first6['source_KiB'], first4['source_KiB'])
        self.assertNotEqual(later6['source_KiB'], later4['source_KiB'])
        self.assertEqual(later6['L_ns'], later4['L_ns'])
        self.assertEqual(later4['supply_active_windows'], 0)
        self.assertTrue(all(result['frozen_outputs_equal'].values()))

    def test_K_scaling_separates_fixed_from_per_tile_difference(self):
        def positions(value):
            return [dict(trial=t,j=0,T=2,windows=132,L_ns_delta=value,L_cycles_delta=2*value)
                for t in range(10)]
        fixed = _failure_k_scaling(positions(100),positions(100),16,64)['positions'][0]['statistics']
        sustained = _failure_k_scaling(positions(100),positions(400),16,64)['positions'][0]['statistics']
        self.assertEqual(fixed['fixed_only_residual_ns']['median'],0)
        self.assertNotEqual(fixed['per_ktile_only_residual_ns']['median'],0)
        self.assertNotEqual(sustained['fixed_only_residual_ns']['median'],0)
        self.assertEqual(sustained['per_ktile_only_residual_ns']['median'],0)

    def test_finite_slots_release_previous_group_after_wait1(self):
        result = _finite_stage_cycles(16,4,100,2200,512,detail=True)
        for i,issue in enumerate(result['issued']):
            self.assertEqual(result['ready'][i],issue+2200)
            if i>=4:
                self.assertGreaterEqual(issue,result['released'][i-4])
            self.assertGreaterEqual(result['released'][i],result['completed'][i])
            if i%16!=15:
                self.assertEqual(result['released'][i],max(result['first_mma_or_commit'][i+1],result['completed'][i]))
        self.assertEqual(result['released'][15],result['completed'][15])

    def test_pingpong_handoff_precedes_previous_tail_completion(self):
        result = _finite_stage_cycles(16,6,0,0,512,detail=True)
        self.assertLess(result['first_mma_or_commit'][16],result['completed'][15])
        self.assertEqual(result['windows'][0],16*512)
        # The second FIRST_MMA is before the previous last group completes.
        self.assertEqual(result['windows'][1],17*512)
        delayed = _finite_stage_cycles(16,6,0,200,512,detail=True)
        self.assertEqual(delayed['windows'][0],result['windows'][0])
        self.assertEqual(delayed['first_mma_or_commit'][0],200)

    def test_sparse_mainloop_closure_uses_mma_tail_not_exit(self):
        sparse=dict(intervals=dict(mainloop_before_tail=1000,handoff_before_tail=20,mma_tail=60,L=1080),
            clocks=dict(first_mma=0,mma_return=1000,release_end=305,
                full_wait_begin=150,full_wait_end=200,commit_end=250,wait1_end=270,
                issue_end=100,release_begin=300,acquire_end=80,refill_begin=250,refill_end=310))
        result=_stage_event_intervals(sparse)
        self.assertEqual(result['closure_cycles'],0)
        self.assertEqual(result['issue_to_observed_full'],100)
        self.assertEqual(result['slot_live_until_release_begin'],220)
        self.assertEqual(result['consumer_wait_to_commit'],100)
        self.assertEqual(result['consumer_wait_to_wait1'],120)
        self.assertEqual(result['before_selected_wait'],150)
        self.assertEqual(result['after_selected_wait1_to_mma_return'],730)
        self.assertEqual(result['wait1_to_next_release_begin'],30)
        self.assertEqual(result['refill_after_release_end'],5)
        sparse['intervals']['mma_tail']=61
        with self.assertRaisesRegex(ValueError,'does not close'):
            _stage_event_intervals(sparse)

    def test_commit_pacing_separates_throughput_from_residual_tail(self):
        trace=_commit_paced_stage_cycles(16,6,10,200,400,detail=True)
        self.assertTrue(all(b-a>=512 for a,b in zip(trace['commits'],trace['commits'][1:])))
        self.assertTrue(all(d-c==400 for c,d in zip(trace['commits'],trace['completed'])))
        self.assertEqual(trace['wait1_return'][5]-trace['commits'][5],0)
        self.assertEqual(trace['windows'][0],15*512+400)
        for i in range(6,len(trace['issued'])):
            self.assertGreaterEqual(trace['issued'][i],trace['released'][i-6])

    def test_zero_bookkeeping_commit_model_forces_combined_window_floor(self):
        for stages in (4,6):
            for parameters in ((10,200,100),(500,1500,800),(100,4000,1500)):
                trace=_commit_paced_stage_cycles(64,stages,*parameters,detail=True)
                # The observed Ktile16 combined interval is inside the output,
                # so prologue/tail exceptions cannot remove this model bound.
                self.assertGreaterEqual(trace['wait1_return'][80]-trace['consumer_available'][80],512-1e-9)

    def test_progress_partitions_and_interval_counts(self):
        clocks=dict(first_mma=100,commit1=500,full_wait_begin16=8000,commit16=8300,
            wait1_end16=8306,full_wait_begin17=8400,commit17=8700,commit63=31700,
            mma_return=31800,main_end=32200)
        progress=dict(clocks=clocks,intervals=dict(L=32100))
        result=_stage_progress_intervals(progress)
        self.assertEqual(result['commit16_to17'],400)
        self.assertEqual(result['loop_advance_16_to17'],94)
        self.assertEqual(result['average_commit1_to16'],520)
        self.assertEqual(result['average_commit17_to63'],500)
        self.assertEqual(result['adjacent_closure_cycles'],0)
        self.assertEqual(result['L_closure_cycles'],0)
        progress['intervals']['L']=32101
        with self.assertRaisesRegex(ValueError,'disagrees with clocks'):
            _stage_progress_intervals(progress)

    def test_candidate_progress_exposes_zero_loop_gap_without_refitting(self):
        for evaluate in (_finite_stage_cycles,_commit_paced_stage_cycles):
            for stages in (4,6):
                result=_stage_candidate_progress(evaluate,(500.75,1766.02,523.65),stages)
                self.assertEqual(result['loop_advance_16_to17'],0)
                self.assertAlmostEqual(result['adjacent_closure_cycles'],0)
                self.assertAlmostEqual(result['L_closure_cycles'],0)
                trace=evaluate(64,stages,500.75,1766.02,523.65,detail=True)
                self.assertAlmostEqual(result['L'],trace['windows'][1])

    def test_shared_loop_gap_does_not_delay_the_output_handoff(self):
        zero=_loop_paced_stage_cycles(16,6,100,1200,600,0,detail=True)
        original=_commit_paced_stage_cycles(16,6,100,1200,600,detail=True)
        self.assertEqual(zero,original)
        trace=_loop_paced_stage_cycles(16,6,100,1200,600,75,detail=True)
        for i in range(1,32):
            expected=0 if i%16==0 else 75
            self.assertEqual(trace['consumer_available'][i]-trace['wait1_return'][i-1],expected)



class AdmissionStageTests(unittest.TestCase):
    def test_handoff_bounds_preserve_signed_overlap_and_clock_domains(self):
        c=dict(tiles=[[100,1000,1020,1200],[920,1800,1810,1900]],
               tiles_ns=[[10,100,102,120],[110,190,191,195]])
        r=_stage_handoff_windows(c)
        self.assertEqual(r['cycles']['first1_minus_main0'],-80)
        self.assertEqual(r['cycles']['mma0_to_first1_lower_bound'],0)
        self.assertEqual(r['ns']['mma0_to_first1_lower_bound'],10)

    def test_ready_and_handoff_are_not_identified_by_stage6_progress(self):
        from supply_model import admission_stage_events
        def observation(stage,latency,delay):
            t=admission_stage_events(64,stage,50,latency,407,93,
                wait_overhead=27,loop_advance=89.5,return_overhead=13,handoff_delay=delay)
            c=t['commits'];b=t['first_mma_or_commit']
            return (c[65]-b[64],(c[80]-c[65])/15,(c[127]-c[81])/46,t['windows'][1])
        self.assertEqual(observation(6,2257.5,0),observation(6,700,196.5))
        self.assertEqual(observation(6,700,196.5),observation(6,1858,196.5))
        self.assertNotEqual(observation(4,700,196.5),observation(4,1858,196.5))
        # Consumer recovery is not tail completion, and cannot run backwards.
        t=admission_stage_events(16,6,0,0,407,93,handoff_delay=196.5)
        self.assertEqual(t['consumer_available'][16]-t['mma_returns'][0],196.5)
        self.assertLess(t['consumer_available'][16],t['main_ends'][0])
        with self.assertRaises(ValueError):
            admission_stage_events(16,6,0,0,407,93,handoff_delay=-1)

    def test_group_service_overlaps_commit_and_keeps_slot_lifetimes(self):
        from supply_model import admission_stage_events
        t=admission_stage_events(16,6,50,700,430,70,issue_floor=64,
            wait_overhead=27,loop_advance=90,return_overhead=13,reuse_delay=200)
        for i in range(32):
            self.assertLessEqual(t['first_mma_or_commit'][i],t['commits'][i])
            self.assertLessEqual(t['commits'][i],t['service_frontier'][i])
            self.assertLessEqual(t['service_frontier'][i],t['completed'][i])
            if i>=6:self.assertGreaterEqual(t['issued'][i],t['released'][i-6]+200)
            if i%16:self.assertGreaterEqual(t['released'][i-1],t['completed'][i-1])
        self.assertEqual(t['slot'][16],4)
        self.assertEqual(t['wait1_return'][16],t['commits'][16])
        self.assertGreaterEqual(t['main_ends'][0],t['mma_returns'][0])
        self.assertGreaterEqual(t['main_ends'][1],t['completed'][-1])

    def test_handoff_does_not_reset_service_or_wait_for_tail(self):
        from supply_model import admission_stage_events
        t=admission_stage_events(16,6,0,0,200,100,wait_overhead=20,
            loop_advance=80,return_overhead=20)
        self.assertLess(t['first_mma_or_commit'][16],t['main_ends'][0])
        self.assertGreaterEqual(t['service_frontier'][16],t['service_frontier'][15]+512)
        self.assertEqual(t['consumer_available'][16],t['mma_returns'][0])

    def test_total_time_does_not_uniquely_identify_commit_tail(self):
        from supply_model import admission_stage_events
        a=admission_stage_events(16,6,0,0,100,100,wait_overhead=20,loop_advance=80)
        b=admission_stage_events(16,6,0,0,200,100,wait_overhead=20,loop_advance=80)
        self.assertEqual(a['windows'][0],b['windows'][0])
        self.assertNotEqual(a['completed'][15]-a['commits'][15],b['completed'][15]-b['commits'][15])

    def test_completion_and_return_are_both_respected(self):
        from supply_model import admission_stage_events
        t=admission_stage_events(8,4,0,0,700,0,issue_floor=100,
            wait_overhead=600,loop_advance=90,return_overhead=300)
        for j in range(2):
            self.assertEqual(t['main_ends'][j],max(t['mma_returns'][j],t['completed'][(j+1)*8-1]))
        with self.assertRaises(ValueError):admission_stage_events(8,1,0,0,0,0)
        with self.assertRaises(ValueError):admission_stage_events(8,4,0,0,0,0,issue_floor=513)

    def test_reuse_cycle_necessary_bound_holds_in_each_output(self):
        from supply_model import admission_stage_events
        for stage in (4,6):
            for g,q in ((0,430),(105,500),(105,700)):
                t=admission_stage_events(64,stage,100,2200,q,50,issue_floor=g,
                    wait_overhead=27,loop_advance=90,return_overhead=13,reuse_delay=200)
                c=t['commits'];jump=2200+512-q+27+200
                for j in (0,1):
                    for k in range(stage,64):
                        i=j*64+k
                        self.assertGreaterEqual(c[i]-c[i-stage+1]+1e-9,jump)
                self.assertGreaterEqual((c[127]-c[81])/46, (46//(stage-1))*max(0,jump)/46)


if __name__ == '__main__':
    unittest.main()
