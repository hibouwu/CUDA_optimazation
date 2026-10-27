"""Dependency and accounting checks for the development-only shared server."""
from collections import Counter
import random
import unittest

from supply_model import admission_stage_events, shared_stage_events


class SharedStageTests(unittest.TestCase):
    def test_sync_issue_return_only_delays_producer_successors(self):
        import copy

        results={}
        for mode in ('legacy_no_return','late_ready_immediate_return','same_ready_delayed_return'):
            def source(t,cta,index,atom,weight):
                served=dict(source_id=index*3+atom,source_key=(cta,index,atom),merged=False,
                    source_ready=100.,source_busy_ns=0.,service_start=t,service_end=t)
                if mode!='legacy_no_return':
                    returned=10. if mode=='same_ready_delayed_return' and index==0 and atom==0 else \
                        20. if mode=='same_ready_delayed_return' and index==0 and atom==2 else t
                    served['issue_return_ns']=returned
                return served

            results[mode]=shared_stage_events(2,2,1,output_counts=[1],trace_requests=True,
                input_source_service=source,**self.parameters(destination_service_ns=0.))

        legacy=results['legacy_no_return'];immediate=results['late_ready_immediate_return']
        delayed=results['same_ready_delayed_return']
        stripped=copy.deepcopy(immediate)
        for collection in ('input_demands','deliveries'):
            for row in stripped[collection]:row.pop('issue_return_ns',None)
        self.assertEqual(stripped,legacy)
        for collection in ('input_demands','deliveries'):
            self.assertTrue(all('issue_return_ns' not in row for row in legacy[collection]))
        self.assertEqual(legacy['ctas'][0]['issued_a'],[0.,5.])
        self.assertEqual(legacy['ctas'][0]['issued_b0'],[1.,6.])
        self.assertEqual(legacy['ctas'][0]['issued_b1'],[2.,7.])
        self.assertEqual(delayed['ctas'][0]['issued_a'],[0.,23.])
        self.assertEqual(delayed['ctas'][0]['issued_b0'],[11.,24.])
        self.assertEqual(delayed['ctas'][0]['issued_b1'],[12.,25.])
        self.assertEqual(delayed['ctas'][0]['issued_b'][0],12.)
        b1=next(row for row in delayed['input_demands'] if row['index']==0 and row['kind']=='issue_b1')
        self.assertEqual(b1['issue_return_ns'],20.)
        self.assertEqual([row['source_ready_ns'] for row in delayed['input_demands']],
                         [row['source_ready_ns'] for row in legacy['input_demands']])
        self.assertEqual([row['ready_ns'] for row in delayed['deliveries']],
                         [row['ready_ns'] for row in legacy['deliveries']])
        self.assertEqual(delayed['ctas'][0]['ready'],[100.,100.])
        self.synthetic_issue_return_check=dict(
            cases={mode:dict(issued_a=result['ctas'][0]['issued_a'],
                issued_b0=result['ctas'][0]['issued_b0'],issued_b1=result['ctas'][0]['issued_b1'],
                issued_b=result['ctas'][0]['issued_b'],ready=result['ctas'][0]['ready'],
                delivery_ready=[row['ready_ns'] for row in result['deliveries']])
                for mode,result in results.items()},default_fields_preserved=True,
            immediate_return_matches_legacy_after_optional_metadata_removal=True,
            first_A_delayed_return_ns=10.,first_B1_delayed_return_ns=20.,
            issued_b_is_issue_time_not_return=True,source_and_destination_ready_preserved=True)

    def test_output_completion_lag_prices_completed_minus_service_frontier(self):
        phases = [dict(compute_ns=256.,commit_lead_ns=128.,completion_lag_ns=47.5),
                  dict(compute_ns=512.,commit_lead_ns=256.),
                  dict(compute_ns=128.,commit_lead_ns=64.,completion_lag_ns=23.75)]
        result = shared_stage_events(4,4,2,output_counts=[3,2],compute_phases_ns=phases,
            **self.parameters(completion_lag_ns=8.))
        for actor in result['ctas']:
            for i,(completed,frontier) in enumerate(zip(actor['completed'],actor['service_frontier'])):
                self.assertEqual(completed-frontier,(47.5,8.,23.75)[i//4])

    def test_local_exclusion_preserves_work_and_blocks_only_its_own_intervals(self):
        params=self.parameters(epilogue_prepare_ns=71.,epilogue_write_ns=83.,
            service_ns=0.,read_latency_ns=31.,output_latency_ns=43.)
        for mode in ('producer','consumer','both'):
            with self.subTest(mode=mode):
                model=shared_stage_events(8,4,2,local_epilogue_exclusion=mode,trace_requests=True,**params)
                self.assertEqual(model['input_delivery_count'],96)
                self.assertEqual(model['input_delivery_work'],64.)
                self.assertEqual(len([r for r in model['requests'] if r['kind']=='store']),32)
                for cta,a in enumerate(model['ctas']):
                    intervals=model['local_epilogue_intervals'][cta]
                    self.assertEqual(len(intervals),32)
                    self.assertAlmostEqual(sum(r['end_ns']-r['start_ns'] for r in intervals),16*(71+83))
                    fields=(['issued_a','issued_b0','issued_b1'] if mode in ('producer','both') else [])+(['starts'] if mode in ('consumer','both') else [])
                    for field in fields:
                        for t in a[field]:
                            # Equal-time events preserve insertion order; a zero-time
                            # issue already processed at reservation start is allowed.
                            self.assertFalse(any(r['start_ns']<t<r['end_ns'] for r in intervals),(mode,field,t))
                active=[model['local_epilogue_blocked_ns'][r] for r in ('producer','consumer') if mode in (r,'both')]
                self.assertGreater(sum(active),0.)

    def test_zero_duration_local_exclusion_preserves_the_legacy_graph(self):
        params=self.parameters(epilogue_prepare_ns=0.,epilogue_write_ns=0.)
        baseline=shared_stage_events(8,4,2,**params)
        for mode in ('producer','consumer','both'):
            self.assertEqual(shared_stage_events(8,4,2,local_epilogue_exclusion=mode,**params),baseline)

    def test_all_E0_deferred_exclusion_cannot_change_input_or_compute_for_any_scope(self):
        for stage in (2,4,6):
            for prefill in (False,True):
                params=self.parameters(epilogue_prepare_ns=173.,epilogue_write_ns=257.)
                baseline=shared_stage_events(8,stage,3,epilogue_defer=2,second_prefill=prefill,**params)
                for mode in ('producer','consumer','both'):
                    model=shared_stage_events(8,stage,3,epilogue_defer=2,second_prefill=prefill,
                        local_epilogue_exclusion=mode,**params)
                    self.assertEqual(model,baseline)

    def source_keys(self, kt, ctas):
        return [[[(cta,i,atom) for atom in range(3)] for i in range(2*kt)] for cta in range(ctas)]

    def parameters(self, **overrides):
        values=dict(service_ns=2.,read_latency_ns=17.,compute_ns=64.,
            producer_gap_ns=3.,operand_gap_ns=1.,reuse_delay_ns=4.,
            commit_lead_ns=40.,completion_lag_ns=8.,issue_floor_ns=5.,
            wait_overhead_ns=4.,loop_advance_ns=6.,return_overhead_ns=3.,
            handoff_delay_ns=2.,output_latency_ns=31.,output_weight=3.,
            epilogue_prepare_ns=7.,epilogue_write_ns=5.,
            epilogue_control_ns=2.,epilogue_tail_ns=3.)
        values.update(overrides)
        return values

    def test_bijective_source_rename_preserves_all_events(self):
        # The hypothesis sees equality classes, not numeric addresses. A future
        # address/placement control must not be fitted via opaque key labels.
        kt,ctas=8,6
        keys=[[[(('A',cta//3,i),('B0',cta%3,i),('B1',cta%3,i))[atom]
                for atom in range(3)] for i in range(2*kt)] for cta in range(ctas)]
        unique=list(dict.fromkeys(k for actor in keys for group in actor for k in group))
        rename={key:('renamed',len(unique)-i) for i,key in enumerate(unique)}
        renamed=[[[rename[k] for k in group] for group in actor] for actor in keys]
        for stage in (4,6):
            for prefill in (False,True):
                args=dict(share_inflight_sources=True,second_prefill=prefill,
                    epilogue_defer=2,**self.parameters(destination_service_ns=3.))
                original=shared_stage_events(kt,stage,ctas,source_keys=keys,**args)
                changed=shared_stage_events(kt,stage,ctas,source_keys=renamed,**args)
                self.assertEqual(original,changed)

    def test_inflight_source_is_served_once_but_delivered_twice(self):
        kt,ctas=4,2
        keys=self.source_keys(kt,ctas)
        keys[1][0][0]=keys[0][0][0]
        params=self.parameters(destination_service_ns=3.)
        result=shared_stage_events(kt,4,ctas,share_inflight_sources=True,source_keys=keys,
            trace_requests=True,**params)
        first=[d for d in result['input_demands'] if d['kind']=='issue_a' and d['index']==0]
        self.assertEqual(len(first),2)
        self.assertEqual(first[0]['source_id'],first[1]['source_id'])
        self.assertEqual([d['merged'] for d in first],[False,True])
        copies=[d for d in result['deliveries'] if d['source_id']==first[0]['source_id']]
        self.assertEqual({d['cta'] for d in copies},{0,1})
        self.assertEqual(len(copies),2)
        self.assertEqual(result['input_demand_count'],ctas*2*kt*3)
        self.assertEqual(result['input_delivery_count'],result['input_demand_count'])
        self.assertEqual(result['input_source_count']+result['input_merged_count'],result['input_demand_count'])
        self.assertEqual(result['input_source_work']+result['input_merged_work'],ctas*2*kt*2)
        self.assertEqual(result['input_demand_work'],result['input_delivery_work'])
        stores=[r for r in result['requests'] if r['kind']=='store']
        self.assertEqual(len(stores),ctas*16)
        self.assertEqual(result['request_count'],result['input_source_count']+len(stores))
        self.assertEqual(result['source_busy_ns'],params['service_ns']*result['input_source_work'])
        self.assertEqual(result['shared_busy_ns'],result['source_busy_ns']+
            params['service_ns']*ctas*8*params['output_weight'])
        self.assertEqual(result['destination_busy_ns'],params['destination_service_ns']*result['input_demand_work'])
        sources={s['source_id']:s for s in result['sources']}
        for cta in range(ctas):
            previous_end=0.
            deliveries=[d for d in result['deliveries'] if d['cta']==cta]
            for d in deliveries:
                self.assertEqual(d['source_ready_ns'],sources[d['source_id']]['ready_ns'])
                self.assertEqual(d['start_ns'],max(d['source_ready_ns'],previous_end))
                self.assertEqual(d['service_end_ns']-d['start_ns'],params['destination_service_ns']*d['work'])
                previous_end=d['service_end_ns']
            for i in range(2*kt):
                group=[d for d in deliveries if d['index']==i]
                self.assertEqual({d['kind'] for d in group},{'issue_a','issue_b0','issue_b1'})
                self.assertEqual(result['ctas'][cta]['ready'][i],max(d['ready_ns'] for d in group))

    def test_source_expiry_is_independent_of_delivery_and_generation_cleanup(self):
        keys=self.source_keys(2,3)
        for cta in range(3):keys[cta][0][0]='revisited-A'
        # At offset 10 the new issue was queued before the old source_done event.
        # Its replacement must survive cleanup so CTA2 at 15 can subscribe to it.
        for second,third in ((10.,15.),(20.,25.)):
            with self.subTest(second=second):
                result=shared_stage_events(2,2,3,share_inflight_sources=True,source_keys=keys,
                    diagnostic_producer_offsets_ns=[0.,second,third],trace_requests=True,
                    **self.parameters(service_ns=10.,read_latency_ns=0.,operand_gap_ns=1000.,
                        destination_service_ns=100.))
                first=sorted((d for d in result['input_demands'] if d['kind']=='issue_a' and d['index']==0),
                    key=lambda d:d['cta'])
                self.assertEqual(first[0]['source_ready_ns'],10.)
                self.assertNotEqual(first[0]['source_id'],first[1]['source_id'])
                self.assertEqual(first[1]['source_id'],first[2]['source_id'])
                self.assertEqual([d['merged'] for d in first],[False,False,True])
                old_delivery=next(d for d in result['deliveries'] if d['demand_id']==first[0]['demand_id'])
                self.assertGreater(old_delivery['ready_ns'],first[1]['arrival_ns'])

    def test_service_boundary_closes_admission_before_ready_without_losing_delivery(self):
        keys=self.source_keys(2,3)
        for actor in keys:actor[0][0]='same-A'
        params=self.parameters(service_ns=10.,read_latency_ns=100.,operand_gap_ns=1000.,
            destination_service_ns=3.)
        for boundary,expected in [('ready',[False,True,True]),
                                  ('service_end',[False,True,False]),
                                  ('service_start',[False,False,False])]:
            with self.subTest(boundary=boundary):
                result=shared_stage_events(2,2,3,share_inflight_sources=True,source_keys=keys,
                    source_join_boundary=boundary,diagnostic_producer_offsets_ns=[0.,5.,10.],
                    trace_requests=True,**params)
                first=sorted((d for d in result['input_demands'] if d['kind']=='issue_a' and d['index']==0),key=lambda d:d['cta'])
                self.assertEqual([d['merged'] for d in first],expected)
                self.assertEqual(result['input_delivery_count'],36)
                self.assertEqual(result['input_demand_work'],result['input_delivery_work'])
                sources={s['source_id']:s for s in result['sources']}
                for d in result['input_demands']:
                    if d['merged']:self.assertLess(d['arrival_ns'],sources[d['source_id']]['join_end_ns'])
                    self.assertEqual(d['source_ready_ns'],sources[d['source_id']]['ready_ns'])

    def test_service_closed_old_generation_cannot_remove_a_new_waiting_generation(self):
        keys=self.source_keys(2,4)
        for actor in keys:actor[0][0]='same-A'
        result=shared_stage_events(2,2,4,share_inflight_sources=True,source_keys=keys,
            source_join_boundary='service_end',diagnostic_producer_offsets_ns=[0.,109.,110.,115.],
            trace_requests=True,**self.parameters(service_ns=10.,read_latency_ns=100.,operand_gap_ns=1000.))
        first=sorted((d for d in result['input_demands'] if d['kind']=='issue_a' and d['index']==0),key=lambda d:d['cta'])
        self.assertEqual([d['merged'] for d in first],[False,False,True,True])
        self.assertEqual(len({d['source_id'] for d in first[1:]}),1)

    def test_repeated_source_has_new_generations_across_slots_and_outputs(self):
        kt=4;keys=self.source_keys(kt,1)
        for group in keys[0]:group[0]='same-A-every-group'
        result=shared_stage_events(kt,2,1,share_inflight_sources=True,source_keys=keys,
            trace_requests=True,**self.parameters(service_ns=2.,read_latency_ns=5.,
                operand_gap_ns=20.,output_weight=0.,destination_service_ns=5.))
        a_demands=[d for d in result['input_demands'] if d['kind']=='issue_a']
        self.assertEqual(len({d['source_id'] for d in a_demands}),2*kt)
        self.assertTrue(all(not d['merged'] for d in a_demands))
        identities={(d['cta'],d['index'],d['kind']) for d in result['deliveries']}
        self.assertEqual(len(identities),2*kt*3)
        self.assertEqual(result['ctas'][0]['ready_count'],[3]*(2*kt))
        self.assertEqual(result['input_source_count'],result['input_demand_count'])

    def test_source_keys_reject_different_weights_and_incomplete_shape(self):
        keys=self.source_keys(2,1);keys[0][0][1]=keys[0][0][0]
        with self.assertRaisesRegex(ValueError,'weight'):
            shared_stage_events(2,2,1,share_inflight_sources=True,source_keys=keys,**self.parameters())
        with self.assertRaisesRegex(ValueError,'shape'):
            shared_stage_events(2,2,1,share_inflight_sources=True,source_keys=[],**self.parameters())

    def test_zero_source_service_can_still_share_remaining_read_latency(self):
        keys=self.source_keys(2,2);keys[1][0][0]=keys[0][0][0]
        result=shared_stage_events(2,2,2,share_inflight_sources=True,source_keys=keys,
            diagnostic_producer_offsets_ns=[0.,5.],trace_requests=True,
            **self.parameters(service_ns=0.,read_latency_ns=10.,operand_gap_ns=100.))
        demands=[d for d in result['input_demands'] if d['kind']=='issue_a' and d['index']==0]
        self.assertEqual([d['source_ready_ns'] for d in demands],[10.,10.])
        self.assertEqual([d['merged'] for d in demands],[False,True])

    def test_zero_service_prefill_matches_admission_only_on_cta_zero(self):
        kt=8
        params=self.parameters(service_ns=0.,destination_service_ns=0.,operand_gap_ns=0.,
            read_latency_ns=900.,producer_gap_ns=9.,reuse_delay_ns=17.,commit_lead_ns=40.,
            completion_lag_ns=12.,issue_floor_ns=0.,wait_overhead_ns=4.,loop_advance_ns=3.,
            return_overhead_ns=2.,handoff_delay_ns=5.)
        mapping={'issued_a':'issued','issued_b0':'issued','issued_b1':'issued','ready':'ready',
            'starts':'first_mma_or_commit','service_frontier':'service_frontier','commits':'commits',
            'completed':'completed','wait1_return':'wait1_return','released':'released',
            'mma_returns':'mma_returns','main_ends':'main_ends'}
        for stage in (4,6):
            for sharing in (False,True):
                with self.subTest(stage=stage,sharing=sharing):
                    # Distinct keys isolate the dependency graph: subscribing to
                    # another CTA's remaining read latency would change readiness.
                    keys=self.source_keys(kt,2)
                    result=shared_stage_events(kt,stage,2,second_prefill=True,
                        share_inflight_sources=sharing,source_keys=keys,**params)
                    for cta in (0,1):
                        prior=admission_stage_events(kt,stage,9.,900.,40.,12.,service_cycles=64.,
                            wait_overhead=4.,loop_advance=3.,return_overhead=2.,reuse_delay=17.,
                            handoff_delay=5.,second_prefill=cta==0)
                        for field,old in mapping.items():self.assertEqual(result['ctas'][cta][field],prior[old],field)
                        if cta==0:
                            gate=result['second_prefill']
                            self.assertEqual(gate['begin_ns'],prior['second_prefill']['begin'])
                            self.assertEqual(gate['end_ns'],prior['second_prefill']['end'])
                            self.assertEqual(gate['waited_stages'],stage)
                            a=result['ctas'][0]
                            self.assertGreaterEqual(gate['end_ns'],a['main_ends'][0])
                            self.assertEqual(a['ready_count'][kt:kt+stage],[3]*stage)
                            for i in range(kt,2*kt-1):
                                self.assertEqual(a['released'][i],a['wait1_return'][i+1])

    def test_second_prefill_waits_destination_delivery_not_source_completion(self):
        kt,stage=8,4
        result=shared_stage_events(kt,stage,2,share_inflight_sources=True,
            source_keys=self.source_keys(kt,2),destination_service_ns=40.,second_prefill=True,
            trace_requests=True,**self.parameters(service_ns=1.,read_latency_ns=0.))
        gate=result['second_prefill'];actor=result['ctas'][0]
        initial=[d for d in result['deliveries'] if d['cta']==0 and kt<=d['index']<kt+stage]
        self.assertEqual(len(initial),3*stage)
        self.assertEqual(gate['begin_ns'],actor['mma_returns'][0]+2.)
        self.assertEqual(gate['end_ns'],max(gate['begin_ns'],max(d['ready_ns'] for d in initial)))
        self.assertGreater(gate['end_ns'],max(d['source_ready_ns'] for d in initial))
        self.assertEqual(actor['starts'][kt],gate['end_ns'])
        self.assertGreaterEqual(gate['end_ns'],actor['main_ends'][0])

    def test_zero_service_matches_existing_two_output_recurrence(self):
        # Includes a broad lead/retirement regime, not only the reduced subclass.
        for stages in (2,4,6):
            for q,lag in ((19.,3.),(83.,15.)):
                with self.subTest(stages=stages,q=q):
                    params=self.parameters(service_ns=0.,operand_gap_ns=0.,
                        read_latency_ns=29.,producer_gap_ns=7.,reuse_delay_ns=3.,
                        commit_lead_ns=q,completion_lag_ns=lag,issue_floor_ns=9.,
                        wait_overhead_ns=4.,loop_advance_ns=5.,return_overhead_ns=6.,
                        handoff_delay_ns=7.,epilogue_prepare_ns=47.,output_latency_ns=80.)
                    result=shared_stage_events(8,stages,3,**params)
                    prior=admission_stage_events(8,stages,7.,29.,q,lag,service_cycles=64.,
                        issue_floor=9.,wait_overhead=4.,loop_advance=5.,return_overhead=6.,
                        reuse_delay=3.,handoff_delay=7.)
                    mapping={'issued_a':'issued','issued_b0':'issued','issued_b1':'issued',
                        'issued_b':'issued','ready':'ready',
                        'starts':'first_mma_or_commit','service_frontier':'service_frontier',
                        'commits':'commits','completed':'completed','wait1_return':'wait1_return',
                        'released':'released','mma_returns':'mma_returns','main_ends':'main_ends'}
                    for cta in result['ctas']:
                        for new,old in mapping.items():
                            self.assertEqual(cta[new],prior[old],new)
                        self.assertEqual([cta['main_ends'][j]-cta['starts'][8*j] for j in range(2)],prior['windows'])

    def test_three_input_atoms_readiness_slot_identity_and_tail_release(self):
        kt,stages=8,4
        params=self.parameters(operand_gap_ns=5.)
        result=shared_stage_events(kt,stages,3,trace_requests=True,**params)
        requests={(r['cta'],r['kind'],r['index']):r for r in result['requests']}
        for cta,a in enumerate(result['ctas']):
            for i in range(2*kt):
                ra,rb0,rb1=(requests[cta,kind,i] for kind in ('issue_a','issue_b0','issue_b1'))
                self.assertEqual((ra['work'],rb0['work'],rb1['work']),(1.,.5,.5))
                self.assertEqual(a['issued_b0'][i]-a['issued_a'][i],params['operand_gap_ns'])
                self.assertEqual(a['issued_b1'][i]-a['issued_b0'][i],params['operand_gap_ns'])
                self.assertEqual(a['issued_b'][i],a['issued_b1'][i])
                self.assertEqual(a['ready_count'][i],3)
                self.assertEqual(a['ready'][i],max(ra['ready_ns'],rb0['ready_ns'],rb1['ready_ns']))
                self.assertGreaterEqual(a['starts'][i],a['ready'][i])
                if i:self.assertGreaterEqual(a['issued_a'][i],a['issued_b'][i-1]+params['producer_gap_ns'])
                if i>=stages:
                    self.assertGreaterEqual(a['issued_a'][i],a['released'][i-stages]+params['reuse_delay_ns'])
                self.assertGreaterEqual(a['released'][i],a['completed'][i])
                if i%kt!=kt-1:self.assertEqual(a['released'][i],a['wait1_return'][i+1])
                else:self.assertEqual(a['released'][i],a['main_ends'][i//kt])
            # This next-output slot specifically belongs to the previous WG tail.
            self.assertGreaterEqual(a['issued_a'][kt+stages-1],a['main_ends'][0]+params['reuse_delay_ns'])

    def test_math_handoff_does_not_wait_for_previous_tail_or_epilogue(self):
        params=self.parameters(service_ns=0.,operand_gap_ns=0.,read_latency_ns=0.,
            producer_gap_ns=0.,compute_ns=512.,commit_lead_ns=400.,completion_lag_ns=100.,
            issue_floor_ns=0.,wait_overhead_ns=20.,loop_advance_ns=10.,
            return_overhead_ns=30.,handoff_delay_ns=0.,reuse_delay_ns=0.)
        a=shared_stage_events(8,4,1,**params)['ctas'][0]
        self.assertEqual(a['starts'][8],a['mma_returns'][0])
        self.assertLess(a['starts'][8],a['main_ends'][0])
        self.assertGreaterEqual(a['outputs'][0]['begin'],a['main_ends'][0])
        self.assertGreaterEqual(a['outputs'][1]['begin'],a['outputs'][0]['end'])
        self.assertGreaterEqual(a['outputs'][1]['begin'],a['main_ends'][1])

    def test_delayed_store_preparation_and_two_buffer_identity(self):
        params=self.parameters(output_latency_ns=150.)
        result=shared_stage_events(8,4,3,**params)
        blocked=0
        for a in result['ctas']:
            for e in a['outputs']:
                for j in range(8):
                    self.assertLessEqual(e['prepare_begin'][j],e['prepare_end'][j])
                    self.assertLessEqual(e['prepare_end'][j],e['write_begin'][j])
                    self.assertLessEqual(e['write_begin'][j],e['write_end'][j])
                    self.assertGreaterEqual(e['submitted'][j],e['write_end'][j])
                    self.assertGreaterEqual(e['source_done'][j],e['submitted'][j])
                    if j:
                        self.assertGreaterEqual(e['prepare_begin'][j],e['write_end'][j-1])
                        self.assertGreaterEqual(e['submitted'][j-1],e['prepare_end'][j])
                        self.assertGreaterEqual(e['write_begin'][j],e['submitted'][j-1])
                    if j>=2:
                        self.assertGreaterEqual(e['write_begin'][j],e['source_done'][j-2])
                        blocked+=e['source_done'][j-2]>e['submitted'][j-1]
                    pending=sum(e['source_done'][k]>e['submitted'][j] for k in range(j+1))
                    self.assertLessEqual(pending,2)
                self.assertGreaterEqual(e['submitted'][7],e['write_end'][7])
                self.assertGreaterEqual(e['end'],max(e['source_done'])+params['epilogue_tail_ns'])
        self.assertGreater(blocked,0,'test must exercise an actual source-buffer acquire wait')

    def test_treatment_scope_and_zero_output_work_do_not_change_math(self):
        params=self.parameters(output_weight=0.,output_latency_ns=300.,
            epilogue_prepare_ns=17.,epilogue_write_ns=11.)
        runs=[shared_stage_events(8,4,3,epilogue_defer=flag,**params) for flag in (0,1,2)]
        for flag,result in enumerate(runs):
            for i,a in enumerate(result['ctas']):
                for key in ('issued_a','issued_b0','issued_b1','issued_b','ready','starts','commits','completed',
                            'wait1_return','released','mma_returns','main_ends'):
                    self.assertEqual(a[key],runs[0]['ctas'][i][key],(flag,i,key))
                deferred=flag==2 or (flag==1 and i==0)
                if deferred:
                    self.assertGreaterEqual(a['outputs'][0]['begin'],a['mma_returns'][1])
                else:
                    self.assertEqual(a['outputs'][0]['begin'],a['main_ends'][0])
                    self.assertLess(a['outputs'][0]['begin'],a['mma_returns'][1])

    def test_fifo_serial_service_and_latency_do_not_double_charge(self):
        params=self.parameters(service_ns=1.5,read_latency_ns=100.)
        result=shared_stage_events(8,6,4,trace_requests=True,**params)
        previous_arrival=previous_finish=0.
        for r in result['requests']:
            self.assertGreaterEqual(r['arrival_ns'],previous_arrival)
            self.assertEqual(r['start_ns'],max(r['arrival_ns'],previous_finish))
            self.assertEqual(r['service_end_ns']-r['start_ns'],params['service_ns']*r['work'])
            latency=params['output_latency_ns'] if r['kind']=='store' else params['read_latency_ns']
            self.assertEqual(r['ready_ns']-r['service_end_ns'],latency)
            previous_arrival,previous_finish=r['arrival_ns'],r['service_end_ns']
        self.assertLess(result['requests'][1]['start_ns'],result['requests'][0]['ready_ns'])
        self.assertEqual(result['last_service_end_ns'],previous_finish)
        self.assertEqual(result['shared_busy_ns'],sum(params['service_ns']*r['work'] for r in result['requests']))
        self.assertLessEqual(result['shared_busy_ns'],result['last_service_end_ns'])
        self.assertLessEqual(result['last_service_end_ns'],result['makespan_ns'])

    def test_uniform_diagnostic_offsets_translate_all_events(self):
        delta=123.5
        actor_times=('issued_a','issued_b0','issued_b1','issued_b','ready','starts',
            'service_frontier','commits','completed','wait1_return','released',
            'mma_returns','main_ends')
        output_times=('prepare_begin','prepare_end','write_begin','write_end',
            'submitted','source_done')
        request_times=('arrival_ns','start_ns','service_end_ns','ready_ns')
        for stages in (4,6):
            for flag in (0,1,2):
                with self.subTest(stages=stages,flag=flag):
                    params=self.parameters(epilogue_defer=flag,trace_requests=True)
                    base=shared_stage_events(8,stages,3,**params)
                    zero=shared_stage_events(8,stages,3,
                        diagnostic_producer_offsets_ns=[0.]*3,**params)
                    shifted=shared_stage_events(8,stages,3,
                        diagnostic_producer_offsets_ns=[delta]*3,**params)
                    self.assertEqual(zero,base)
                    for key in ('makespan_ns','last_service_end_ns'):
                        self.assertEqual(shifted[key],base[key]+delta,key)
                    for key in ('shared_busy_ns','request_count'):
                        self.assertEqual(shifted[key],base[key],key)
                    self.assertEqual(len(shifted['requests']),len(base['requests']))
                    for actual,original in zip(shifted['requests'],base['requests']):
                        for key in ('cta','kind','index','work'):
                            self.assertEqual(actual[key],original[key],key)
                        for key in request_times:
                            self.assertEqual(actual[key],original[key]+delta,key)
                    for actual,original in zip(shifted['ctas'],base['ctas']):
                        for key in actor_times:
                            self.assertEqual(actual[key],[t+delta for t in original[key]],key)
                        for out in range(2):
                            self.assertEqual(actual['main_ends'][out]-actual['starts'][8*out],
                                original['main_ends'][out]-original['starts'][8*out])
                        for e,old_e in zip(actual['outputs'],original['outputs']):
                            for key in ('begin','end'):
                                self.assertEqual(e[key],old_e[key]+delta,key)
                            for key in output_times:
                                self.assertEqual(e[key],[t+delta for t in old_e[key]],key)
                            self.assertEqual(e['end']-e['begin'],old_e['end']-old_e['begin'])

    def test_diagnostic_offsets_gate_each_first_producer_issue(self):
        offsets=[0.,2000.,37.5]
        result=shared_stage_events(8,4,3,diagnostic_producer_offsets_ns=offsets,
            trace_requests=True,**self.parameters())
        first_requests={r['cta']:r for r in result['requests']
            if r['kind']=='issue_a' and r['index']==0}
        self.assertEqual(len(first_requests),len(offsets))
        for i,offset in enumerate(offsets):
            self.assertGreaterEqual(result['ctas'][i]['issued_a'][0],offset)
            self.assertEqual(first_requests[i]['arrival_ns'],result['ctas'][i]['issued_a'][0])
            self.assertGreaterEqual(first_requests[i]['start_ns'],offset)

    def test_homogeneous_fifo_preserves_root_order(self):
        # Regression samples protect the proved class; they are not an exhaustive proof.
        # Flag 1 deliberately has a different CTA0 graph and is outside this property.
        rng=random.Random(739095)
        for case in range(64):
            kt,stages,count=8,rng.choice((2,4,6)),rng.randrange(2,7)
            params={key:float(rng.choice((0,0,1,3,17,63,150)))
                for key in self.parameters()}
            params['compute_ns']=float(rng.randrange(1,129))
            params['issue_floor_ns']=float(rng.randrange(int(params['compute_ns'])+1))
            params['output_weight']=rng.choice((0.,.5,1.,4.))
            offsets=[float(rng.choice((0,0,7,91,1000))) for _ in range(count)]
            offsets[-1]=1000.  # Every sample has a nonzero root; ties remain common.
            order=sorted(range(count),key=lambda i:(offsets[i],i))
            for flag in (0,2):
                with self.subTest(case=case,stages=stages,flag=flag):
                    actors=shared_stage_events(kt,stages,count,epilogue_defer=flag,
                        diagnostic_producer_offsets_ns=offsets,**params)['ctas']
                    for first,second in zip(order,order[1:]):
                        for key in ('ready','starts','commits','mma_returns','main_ends'):
                            for i,(a,b) in enumerate(zip(actors[first][key],actors[second][key])):
                                self.assertLessEqual(a,b,(key,i,first,second,offsets))
            with self.subTest(case=case,stages=stages,flag=2,roots='all zero'):
                actors=shared_stage_events(kt,stages,count,epilogue_defer=2,**params)['ctas']
                first=actors[0]
                begin,end=first['starts'][kt],first['mma_returns'][1]
                for other in actors[1:]:
                    self.assertLessEqual(end,other['mma_returns'][1])
                    e=other['outputs'][0]
                    overlap=max(0.,min(end,e['end'])-max(begin,e['begin']))
                    self.assertEqual(overlap,0.)

    def test_earliest_root_need_not_be_cta_zero(self):
        actors=shared_stage_events(8,4,3,epilogue_defer=2,
            diagnostic_producer_offsets_ns=[10000.,0.,100.],**self.parameters())['ctas']
        self.assertLess(actors[1]['starts'][0],actors[0]['starts'][0])
        self.assertLess(actors[1]['mma_returns'][1],actors[0]['mma_returns'][1])
        self.assertEqual(min(range(3),key=lambda i:actors[i]['mma_returns'][1]),1)

    def test_full_declared_geometry_request_counts_and_work_budget(self):
        params=self.parameters(service_ns=.25,output_weight=2.)
        result=shared_stage_events(64,4,132,trace_requests=True,**params)
        counts=Counter(r['kind'] for r in result['requests'])
        self.assertEqual(counts,dict(issue_a=16896,issue_b0=16896,issue_b1=16896,store=2112))
        self.assertEqual(result['request_count'],52800)
        self.assertEqual(len(result['requests']),result['request_count'])
        expected_work=33792+1056*params['output_weight']
        self.assertEqual(sum(r['work'] for r in result['requests']),expected_work)
        self.assertEqual(result['shared_busy_ns'],params['service_ns']*expected_work)
        self.assertEqual(len(result['ctas']),132)
        self.assertEqual(sum(len(a['outputs']) for a in result['ctas']),264)
        self.assertTrue(all(all(x is not None for x in e['submitted'])
            for a in result['ctas'] for e in a['outputs']))


if __name__=='__main__':unittest.main()
