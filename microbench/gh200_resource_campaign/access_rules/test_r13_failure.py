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
    def test_ready_latency_reuse_delay_translation_leaves_internal_windows_invariant(self):
        from supply_model import admission_stage_events
        import random
        rng=random.Random(739793)
        # This is a parameter symmetry, not evidence of zero physical reuse cost.
        # The first consumer root stays zero; its initial ready wait absorbs the shift.
        shifted_fields=('ready','first_mma_or_commit','service_frontier','commits',
            'completed','wait1_return','released','mma_returns','main_ends')
        for case in range(64):
            c=rng.uniform(1.,700.)
            latency=rng.uniform(0.,8*c)
            reuse=rng.uniform(0.,4*c)
            params=dict(producer_interval=rng.uniform(0.,6*c),ready_latency=latency,
                commit_lead=rng.uniform(0.,5*c),completion_lag=rng.uniform(0.,4*c),
                service_cycles=c,issue_floor=rng.uniform(0.,c),
                wait_overhead=rng.uniform(0.,3*c),loop_advance=rng.uniform(0.,3*c),
                return_overhead=rng.uniform(0.,3*c),reuse_delay=reuse,
                handoff_delay=rng.uniform(0.,6*c),producer_boundary_delay=rng.uniform(0.,6*c))
            translated=dict(params,ready_latency=latency+reuse,reuse_delay=0.)
            for stage in (4,6):
                for prefill in (False,True):
                    with self.subTest(case=case,stage=stage,prefill=prefill):
                        original=admission_stage_events(64,stage,**params,second_prefill=prefill)
                        shifted=admission_stage_events(64,stage,**translated,second_prefill=prefill)
                        for field in ('issued','windows'):
                            for a,b in zip(original[field],shifted[field]):
                                self.assertAlmostEqual(a,b,delta=1e-7,msg=field)
                        self.assertEqual(original['slot'],shifted['slot'])
                        self.assertEqual(original['consumer_available'][0],0.)
                        self.assertEqual(shifted['consumer_available'][0],0.)
                        for a,b in zip(original['consumer_available'][1:],shifted['consumer_available'][1:]):
                            self.assertAlmostEqual(b,a+reuse,delta=1e-7,msg='consumer_available')
                        for field in shifted_fields:
                            for a,b in zip(original[field],shifted[field]):
                                self.assertAlmostEqual(b,a+reuse,delta=1e-7,msg=field)
                        if prefill:
                            first=original['second_prefill'];second=shifted['second_prefill']
                            for key in ('begin','end'):
                                self.assertAlmostEqual(second[key],first[key]+reuse,delta=1e-7,msg=key)
                            self.assertEqual(first['waited_stages'],second['waited_stages'])
                            self.assertAlmostEqual(second['end']-second['begin'],
                                first['end']-first['begin'],delta=1e-7)

    def test_wide_second_prefill_first_to_commit1_relation(self):
        from supply_model import admission_stage_events
        import random
        rng=random.Random(739791)
        # Independent endpoint equation; finite regression samples are not a proof.
        for case in range(64):
            c=rng.uniform(1.,700.)
            q=rng.uniform(0.,5*c)
            g=rng.uniform(0.,c)
            h=rng.uniform(0.,3*c)
            params=dict(producer_interval=rng.uniform(0.,6*c),ready_latency=rng.uniform(0.,8*c),
                commit_lead=q,completion_lag=rng.uniform(0.,4*c),service_cycles=c,
                issue_floor=g,wait_overhead=rng.uniform(0.,3*c),loop_advance=h,
                return_overhead=rng.uniform(0.,3*c),reuse_delay=rng.uniform(0.,4*c),
                handoff_delay=rng.uniform(0.,6*c),producer_boundary_delay=rng.uniform(0.,6*c))
            G=max(g,c-q)
            expected=max(G+h+g,max(G+h,c)+c-q)
            for stage in (4,6):
                with self.subTest(case=case,stage=stage):
                    trace=admission_stage_events(64,stage,**params,second_prefill=True)
                    actual=trace['commits'][65]-trace['first_mma_or_commit'][64]
                    self.assertAlmostEqual(actual,expected,delta=1e-7)

    def test_wide_late_window_obeys_two_step_retirement_bound(self):
        from supply_model import admission_stage_events
        import random
        rng=random.Random(739792)
        # C_i>=H_i+G, E_i-C_i>=c-G and H_i>=E_(i-2)+lag+w+h
        # give C_i-C_(i-2)>=lag+w+h+c. The 46-interval window has 23 such steps.
        # Sampling only protects this necessary relation against code regressions.
        for case in range(64):
            c=rng.uniform(1.,700.)
            lag=rng.uniform(0.,4*c)
            w=rng.uniform(0.,3*c)
            h=rng.uniform(0.,3*c)
            params=dict(producer_interval=rng.uniform(0.,6*c),ready_latency=rng.uniform(0.,8*c),
                commit_lead=rng.uniform(0.,5*c),completion_lag=lag,service_cycles=c,
                issue_floor=rng.uniform(0.,c),wait_overhead=w,loop_advance=h,
                return_overhead=rng.uniform(0.,3*c),reuse_delay=rng.uniform(0.,4*c),
                handoff_delay=rng.uniform(0.,6*c),producer_boundary_delay=rng.uniform(0.,6*c))
            for stage in (4,6):
                for prefill in (False,True):
                    with self.subTest(case=case,stage=stage,prefill=prefill):
                        trace=admission_stage_events(64,stage,**params,second_prefill=prefill)
                        for output in (0,1):
                            start=output*64
                            late=trace['commits'][start+63]-trace['commits'][start+17]
                            self.assertGreaterEqual(late+1e-7,23*(lag+w+h+c))
                            G=max(params['issue_floor'],c-params['commit_lead'])
                            slot_cost=params['ready_latency']+params['reuse_delay']+w+G
                            self.assertGreaterEqual(late+1e-7,(46//(stage-1))*slot_cost)

    def test_second_prefill_reduced_closed_form_and_window_bound(self):
        from supply_model import admission_stage_events
        import random
        rng=random.Random(739790)
        # Regression coverage for the reduced subclass, not a proof by sampling.
        for case in range(64):
            c=rng.uniform(1.,700.)
            q=rng.uniform(0.,c)
            w=rng.uniform(0.,q)
            h=rng.uniform(0.,q-w)
            latency=rng.uniform(0.,5*c)
            reuse=rng.uniform(0.,3*c)
            params=dict(producer_interval=rng.uniform(0.,c),ready_latency=latency,
                commit_lead=q,completion_lag=rng.uniform(0.,c-q),service_cycles=c,
                issue_floor=rng.uniform(0.,c-q),wait_overhead=w,loop_advance=h,
                return_overhead=rng.uniform(0.,4*c),reuse_delay=reuse,
                handoff_delay=rng.uniform(0.,8*c),producer_boundary_delay=rng.uniform(0.,8*c),
                second_prefill=True)
            for stage in (4,6):
                with self.subTest(case=case,stage=stage):
                    trace=admission_stage_events(64,stage,**params)
                    commits=trace['commits'][64:]
                    residual=max(0.,latency+reuse+c-q+w-(stage-1)*c)
                    for j in range(1,64):
                        expected=commits[0]+j*c+((j-1)//(stage-1))*residual
                        self.assertAlmostEqual(commits[j],expected,delta=1e-7)
                    early=commits[16]-commits[1]
                    late=commits[63]-commits[17]
                    self.assertAlmostEqual(late,c+3*early,delta=1e-7)
                    self.assertLessEqual(late,(46/15)*early+1e-7)

    def test_second_prefill_window_bound_does_not_cover_wide_parameters(self):
        from supply_model import admission_stage_events
        # Deliberately violates reduced assumptions (including p<=c and q+lag<=c).
        # This is a structural counterexample, not a fit to measured data.
        params=dict(producer_interval=1700,ready_latency=2900,commit_lead=430,
            completion_lag=240,service_cycles=512,issue_floor=160,wait_overhead=245,
            loop_advance=90,return_overhead=290,reuse_delay=320,handoff_delay=1860,
            producer_boundary_delay=2740,second_prefill=True)
        for stage,expected_early in ((4,24183),(6,20783)):
            with self.subTest(stage=stage):
                commits=admission_stage_events(64,stage,**params)['commits'][64:]
                early=commits[16]-commits[1]
                late=commits[63]-commits[17]
                self.assertEqual(early,expected_early)
                self.assertEqual(late,78200)
                self.assertGreater(late-(46/15)*early,4000)

    def test_second_prefill_default_preserves_original_trace(self):
        from supply_model import admission_stage_events
        args=(3,2,2,7,3,4)
        params=dict(service_cycles=10,issue_floor=1,wait_overhead=2,loop_advance=1,
            return_overhead=3,reuse_delay=5,handoff_delay=2,producer_boundary_delay=4)
        # Captured from the original recurrence before introducing the intervention.
        expected=dict(windows=[45.,45.],issued=[0.,2.,31.,52.,57.,83.],
            ready=[7.,9.,38.,59.,64.,90.],consumer_available=[0.,15.,27.,52.,67.,79.],
            first_mma_or_commit=[7.,15.,38.,59.,67.,90.],
            service_frontier=[17.,27.,48.,69.,79.,100.],commits=[14.,24.,45.,66.,76.,97.],
            completed=[21.,31.,52.,73.,83.,104.],wait1_return=[14.,26.,47.,66.,78.,99.],
            released=[26.,47.,52.,78.,99.,104.],mma_returns=[50.,102.],
            main_ends=[52.,104.],slot=[0,1,0,1,0,1])
        self.assertEqual(admission_stage_events(*args,**params),expected)
        self.assertEqual(admission_stage_events(*args,**params,second_prefill=False),expected)

    def test_second_prefill_without_new_wait_leaves_all_events_unchanged(self):
        from supply_model import admission_stage_events
        for stage in (4,6):
            for latency,handoff in ((0.,0.),(900.,10000.)):
                with self.subTest(stage=stage,latency=latency):
                    args=(8,stage,0.,latency,0.,0.)
                    params=dict(service_cycles=64.,handoff_delay=handoff)
                    baseline=admission_stage_events(*args,**params)
                    filled=admission_stage_events(*args,**params,second_prefill=True)
                    wait=filled.pop('second_prefill')
                    self.assertEqual(wait['begin'],wait['end'])
                    self.assertEqual(wait['waited_stages'],stage)
                    self.assertEqual(filled,baseline)

    def test_second_prefill_waits_full_ring_without_releasing_or_resetting_it(self):
        from supply_model import admission_stage_events
        kt=8
        for stage in (4,6):
            with self.subTest(stage=stage):
                args=(kt,stage,9.,900.,40.,12.)
                params=dict(service_cycles=64.,wait_overhead=4.,loop_advance=3.,
                    return_overhead=2.,reuse_delay=17.,handoff_delay=5.,producer_boundary_delay=23.)
                baseline=admission_stage_events(*args,**params)
                filled=admission_stage_events(*args,**params,second_prefill=True)
                wait=filled['second_prefill']
                self.assertEqual(wait['begin'],baseline['consumer_available'][kt])
                self.assertGreater(wait['end'],wait['begin'])
                self.assertEqual(wait['end'],max(wait['begin'],*filled['ready'][kt:kt+stage]))
                self.assertEqual(filled['first_mma_or_commit'][kt],wait['end'])
                self.assertLessEqual(filled['main_ends'][0],wait['end'])
                for key in ('issued','ready'):
                    self.assertEqual(filled[key][:kt+stage],baseline[key][:kt+stage])
                for key in ('commits','completed','service_frontier','wait1_return','released'):
                    self.assertEqual(filled[key][:kt],baseline[key][:kt])
                self.assertEqual(filled['service_frontier'][kt],
                    max(filled['first_mma_or_commit'][kt],filled['service_frontier'][kt-1])+64.)
                for i in range(kt,2*kt):
                    release=filled['main_ends'][1] if i==2*kt-1 else filled['wait1_return'][i+1]
                    self.assertEqual(filled['released'][i],release)
                    self.assertGreaterEqual(filled['released'][i],baseline['released'][i])
                    self.assertGreaterEqual(filled['issued'][i],filled['released'][i-stage]+17.)
                # The next group beyond the initial fill still needs a real release.
                self.assertGreaterEqual(filled['issued'][kt+stage],filled['wait1_return'][kt+1]+17.)

    def test_second_prefill_rejects_missing_groups_and_observed_entry_cut(self):
        from supply_model import admission_stage_events
        with self.assertRaisesRegex(ValueError,'Ktile>=stages'):
            admission_stage_events(4,6,0,0,0,0,second_prefill=True)
        with self.assertRaisesRegex(ValueError,'observed entry cut'):
            admission_stage_events(8,4,0,0,0,0,second_prefill=True,observed_second_entry=(0,0))

    def test_wide_retirement_inequalities_allow_different_ready_histories(self):
        from supply_model import admission_stage_events
        import random
        rng=random.Random(739718);active=0
        for _ in range(200):
            c=rng.uniform(100,700);g=rng.uniform(0,c);q=rng.uniform(0,3*c)
            lag=rng.uniform(0,2*c);w=rng.uniform(0,70);r=rng.uniform(0,100)
            traces=[];advances=[]
            # Sharing service parameters does not require sharing readiness
            # or entry history across the two independent stage processes.
            for stage in (4,6):
                h=rng.uniform(0,160);advances.append(h)
                traces.append(admission_stage_events(64,stage,rng.uniform(0,2*c),
                    rng.uniform(0,5*c),q,lag,service_cycles=c,issue_floor=g,
                    wait_overhead=w,loop_advance=h,return_overhead=r,
                    reuse_delay=rng.uniform(0,c),handoff_delay=rng.uniform(0,5*c),
                    producer_boundary_delay=rng.uniform(0,20*c)))
            a,b=traces;G=max(g,c-q);L=lag-G
            C,E,W,H=(a[k] for k in ('commits','service_frontier','wait1_return','consumer_available'))
            X=W[80]-H[80];Y=W[80]-C[80]
            Z=b['mma_returns'][1]-b['commits'][81]
            self.assertGreaterEqual(Z+1e-7,46*c+L+w+r)
            self.assertGreaterEqual(Z+1e-7,45*c)
            if Y>w+1e-7:
                active+=1
                self.assertLessEqual(X,c+max(0,L)-advances[0]+1e-7)
                if X>c-advances[0]:
                    self.assertGreaterEqual(Z-X+1e-7,45*c+advances[0]+w+r)
        self.assertGreater(active,20)

    def test_producer_boundary_gap_has_only_a_single_commit_impulse(self):
        from supply_model import admission_stage_events,stage_reuse_window
        import random
        rng=random.Random(739717)
        for stage in (4,6):
            for _ in range(30):
                c=rng.uniform(500,525);q=rng.uniform(200,480)
                w=rng.uniform(0,40);h=rng.uniform(10,100)
                params=dict(producer_interval=rng.uniform(0,c),ready_latency=rng.uniform(0,2500),
                    commit_lead=q,completion_lag=rng.uniform(0,c-q),service_cycles=c,
                    issue_floor=rng.uniform(0,c-q),wait_overhead=w,loop_advance=h,
                    return_overhead=rng.uniform(0,80),reuse_delay=rng.uniform(0,300),
                    handoff_delay=rng.uniform(0,40))
                base=admission_stage_events(64,stage,**params)
                m=stage-1;Z=params['ready_latency']+params['reuse_delay']+c-q
                extra=max(0,Z+w-m*c)
                for gap in (0,c/2,64*c):
                    result=admission_stage_events(64,stage,**params,producer_boundary_delay=gap)
                    for field in ('issued','ready','commits','released'):
                        self.assertEqual(result[field][:64],base[field][:64])
                    actual=result['commits'];old=base['commits']
                    for j in range(1,64):
                        # The first long edge out of the new output is 1 -> S.
                        path=j*c+((j-1)//m)*extra
                        expected=max(old[64+j],actual[64]+path)
                        self.assertAlmostEqual(actual[64+j],expected,places=7)
                    bound=stage_reuse_window(15,stage,c,Z,w)[1]
                    self.assertLessEqual(actual[80]-actual[65],bound+1e-8)

    def test_reuse_reduction_and_full_inheritance_early_bound(self):
        from supply_model import admission_stage_events,stage_reuse_window
        import random
        rng=random.Random(738686)
        for stage in (4,6):
            for _ in range(40):
                c=rng.uniform(500,525);q=rng.uniform(200,480)
                w=rng.uniform(0,40);h=rng.uniform(10,100)
                r=rng.uniform(0,80);delta=rng.uniform(0,40)
                latency=rng.uniform(100,2500);reuse=rng.uniform(0,300)
                t=admission_stage_events(64,stage,rng.uniform(0,c),latency,q,
                    rng.uniform(0,c-q),service_cycles=c,issue_floor=rng.uniform(0,c-q),
                    wait_overhead=w,loop_advance=h,return_overhead=r,reuse_delay=reuse,
                    handoff_delay=delta)
                C=t['commits'];Z=latency+reuse+c-q
                for j in range(stage,64):
                    i=64+j
                    self.assertAlmostEqual(C[i],max(C[i-1]+c,C[i-stage+1]+Z+w),places=8)
                lo,hi=stage_reuse_window(46,stage,c,Z,w)
                self.assertGreaterEqual(C[127]-C[81]+1e-8,lo)
                self.assertLessEqual(C[127]-C[81]-1e-8,hi)
                # Here q >= w+r+delta. The previous final-slot release is
                # earlier than a continued ordinary release and cannot delay C1.
                early=stage_reuse_window(15,stage,c,Z,w)[1]
                self.assertLessEqual(C[80]-C[65]-1e-8,early)

    def test_internal_stall_bounds_do_not_need_first_output_or_q_cap(self):
        from supply_model import admission_stage_events
        for stage in (4,6):
            for p,latency,q,lag,g in ((50,700,407,93,0),(520,2400,900,20,200),
                                     (0,0,1500,0,400),(300,1800,20,900,100)):
                t=admission_stage_events(64,stage,p,latency,q,lag,issue_floor=g,
                    wait_overhead=43,loop_advance=97,return_overhead=100)
                E,C,B=(t[k] for k in ('service_frontier','commits','first_mma_or_commit'))
                for i in (81,127):
                    self.assertGreaterEqual(E[i]-B[i],512)
                    self.assertLessEqual(E[i]-B[i],1024)
                self.assertLessEqual(abs((E[127]-C[127])-(E[81]-C[81])),512)
                stalls=sum(max(0,B[i]-E[i-1]) for i in range(82,128))
                self.assertEqual(stalls,E[127]-E[81]-46*512)
                self.assertGreaterEqual(stalls,max(0,C[127]-C[81]-47*512))
                self.assertGreaterEqual(C[127]-C[81],45*512)
                self.assertGreaterEqual(C[127]-C[81],(46//(stage-1))*latency)

    def test_conditioned_entry_is_one_lipschitz_after_the_cut(self):
        from supply_model import admission_stage_events
        kw=dict(wait_overhead=43,loop_advance=89,return_overhead=100)
        a=admission_stage_events(64,6,50,2257.5,379,105,**kw,observed_second_entry=(134,239))
        b=admission_stage_events(64,6,50,2257.5,379,105,**kw,observed_second_entry=(139,240))
        for field in ('commits','service_frontier','completed','wait1_return','issued'):
            for x,y in zip(a[field][64:],b[field][64:]):
                self.assertGreaterEqual(y-x,0)
                self.assertLessEqual(y-x,1)

    def test_observed_entry_preserves_state_and_rejects_early_readiness(self):
        from supply_model import admission_stage_events
        args=(64,6,50,700,407,93)
        kw=dict(wait_overhead=27,loop_advance=89.5,return_overhead=13)
        free=admission_stage_events(*args,**kw)
        t=admission_stage_events(*args,**kw,handoff_delay=900,
            observed_second_entry=(135,239))
        for key in ('issued','ready','commits','completed','service_frontier'):
            self.assertEqual(t[key][:64],free[key][:64])
        self.assertEqual(t['consumer_available'][64]-t['mma_returns'][0],135)
        self.assertEqual(t['first_mma_or_commit'][64]-t['mma_returns'][0],239)
        self.assertLess(t['first_mma_or_commit'][64],t['completed'][63])
        self.assertGreaterEqual(t['service_frontier'][64],t['service_frontier'][63]+512)
        self.assertEqual(t['slot'][64],4)
        for i in range(64,128):
            self.assertGreaterEqual(t['issued'][i],t['released'][i-6])
        with self.assertRaisesRegex(ValueError,'H gap'):
            admission_stage_events(*args,**kw,observed_second_entry=(240,239))
        with self.assertRaisesRegex(ValueError,'precedes modeled readiness'):
            admission_stage_events(64,4,50,2257.5,407,93,**kw,
                observed_second_entry=(135,239))

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
