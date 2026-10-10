"""Dependency and accounting checks for the development-only shared server."""
from collections import Counter
import random
import unittest

from supply_model import admission_stage_events, shared_stage_events


class SharedStageTests(unittest.TestCase):
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
