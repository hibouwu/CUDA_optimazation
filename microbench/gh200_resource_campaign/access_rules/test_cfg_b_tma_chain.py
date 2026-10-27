"""D原生TMA短链的地址、数值与无时间依赖合同；不执行GPU。"""
import unittest

import numpy as np

import cfg_b_handoff_chain as handoff
import cfg_b_tma_chain as transfer
from mbarrier_service import BarrierPhase
from test_q02_async_admission import KnownAdmissionSource
from test_r13_async_source import DeferredNotifySource, OneCreditSource, ReadyCreditSource, SharedDestination


def generation_matrix(ktiles, stages, seed, stale_operand=None):
    """独立整数点积；漏填槽时保留该物理槽的首代正值。"""
    rows = np.arange(128, dtype=np.int64)[:, None]
    inner = np.arange(64, dtype=np.int64)[None, :]
    a_base = 1+(7*rows+13*inner+3*(seed % 8)) % 8
    b_base = 1+(11*rows+5*inner+5*(seed % 8)) % 8
    total = np.zeros((128, 128), dtype=np.int64)
    for tile in range(ktiles):
        a_source = tile % stages if stale_operand == 'A' else tile
        b_source = tile % stages if stale_operand == 'B' else tile
        a = a_base+a_source
        b = b_base+b_source
        if stale_operand == 'B1':
            b[64:] = b_base[64:]+tile % stages
        total += a @ b.T
    return (total/1024).astype(np.float32)


def ancestors(nodes, node_id):
    found = set()
    pending = list(nodes[node_id]['predecessors'])
    while pending:
        predecessor = pending.pop()
        if predecessor not in found:
            found.add(predecessor)
            pending.extend(nodes[predecessor]['predecessors'])
    return found


class TmaChainTests(unittest.TestCase):
    @staticmethod
    def synthetic_source(latencies, first_return=0.):
        def source(t, request):
            atom = {'A':0, 'B0':1, 'B1':2}[request['operand']]
            return dict(source_id=3*request['tile']+atom,
                source_key=(request['operand'], request['source_k_tile']), merged=False,
                source_ready=t+latencies[atom], source_busy_ns=0., service_start=t,
                service_end=t, issue_return_ns=t+(first_return if atom == 0 else 0.))
        return source

    def test_timed_full_waits_for_each_of_three_completions(self):
        for latencies, ready in (((100,140,180),180), ((220,140,180),220)):
            report = transfer.timed_lifecycle(4, 6,
                request_service=self.synthetic_source(latencies), compute_ns=512., commit_lead_ns=256.)
            self.assertEqual(report['timeline'][0]['full_ready_ns'], ready)
            self.assertEqual(report['timeline'][0]['compute_begin_ns'], ready)
            self.assertEqual(len(report['requests']), 12)
            self.assertTrue(all(slot['full']['pending_tx_bytes'] == 0 for slot in report['slots']))

    def test_timed_issue_return_advances_next_request_independently(self):
        report = transfer.timed_lifecycle(4, 6,
            request_service=self.synthetic_source((100,140,180), first_return=50.),
            compute_ns=512., commit_lead_ns=256.)
        self.assertEqual([r['arrival_ns'] for r in report['requests'][:3]], [0.,50.,50.])
        self.assertEqual(report['requests'][0]['source_ready_ns'], 100.)
        self.assertEqual(report['timeline'][0]['full_ready_ns'], 230.)

    def test_async_admission_advances_producer_before_source_ready(self):
        for ready in (30.,100.):
            report=transfer.timed_lifecycle(4,6,request_service=None,
                input_source_async=KnownAdmissionSource(ready),compute_ns=8.,commit_lead_ns=4.,
                producer_gap_ns=3.,operand_gap_ns=1.,source_tiles=32,source_profile='generation')
            self.assertEqual([r['arrival_ns'] for r in report['requests'][:4]],[0.,11.,12.,15.])
            self.assertEqual(report['requests'][0]['issue_return_ns'],10.)
            self.assertEqual(report['requests'][0]['source_ready_ns'],ready)
            self.assertEqual(report['timeline'][0]['full_ready_ns'],max(ready,32.))
            self.assertEqual(len(report['input_demands']),12)

    def test_preparation_precedes_source_and_waits_for_admission(self):
        hardware = self.synthetic_source((100.,100.,100.), first_return=10.)
        class Source:
            def issue_with_admission(self,t,cta,index,atom,weight,demand_id,notify_ready,notify_admitted):
                served=hardware(t,dict(tile=index,source_k_tile=index,operand=('A','B0','B1')[atom]))
                notify_admitted(demand_id,served['issue_return_ns'])
                notify_ready(demand_id,served['source_ready'],served)
            def destination_done(self,t,demand_id):
                pass
        args=dict(compute_ns=64.,commit_lead_ns=32.,producer_control_ns=(7.,5.,3.))
        sync=transfer.timed_lifecycle(4,6,request_service=hardware,**args)
        asynchronous=transfer.timed_lifecycle(4,6,request_service=None,input_source_async=Source(),**args)
        self.assertEqual([q['arrival_ns'] for q in sync['requests'][:4]],[7.,22.,25.,32.])
        self.assertEqual(sync['requests'][0]['issue_return_ns'],17.)
        self.assertEqual(sync['requests'][0]['source_ready_ns'],107.)
        self.assertEqual(sync['timeline'][0]['full_ready_ns'],125.)
        self.assertEqual([(w['start_ns'],w['end_ns']) for w in sync['producer_control_windows'][:3]],
                         [(0.,7.),(17.,22.),(22.,25.)])
        self.assertEqual(asynchronous['timeline'],sync['timeline'])
        self.assertEqual(asynchronous['producer_control_windows'],sync['producer_control_windows'])
        self.assertEqual([q['arrival_ns'] for q in asynchronous['requests']],[q['arrival_ns'] for q in sync['requests']])

    def test_preparation_after_empty_matches_shared_cta_entry(self):
        hardware=self.synthetic_source((100.,100.,100.),first_return=10.)
        args=dict(compute_ns=512.,commit_lead_ns=256.,completion_lag_ns=95.,
            loop_advance_ns=133.,release_visibility_ns=275.,producer_control_ns=(7.,5.,3.))
        local=transfer.timed_lifecycle(13,4,request_service=hardware,**args)
        shared=transfer.timed_cta_group(13,4,[0,0],source_batches=1,
            request_service=lambda t,cta,q:hardware(t,q),**args)
        for tile in range(4,13):
            row=local['timeline'][tile]
            self.assertGreaterEqual(row['producer_acquire_ns'],local['timeline'][tile-4]['empty_complete_ns'])
            self.assertEqual(row['request_A_issue_ns']-row['producer_acquire_ns'],7.)
        for cta in shared['ctas']:
            self.assertEqual(cta['timeline'],local['timeline'])
        legacy=transfer.timed_lifecycle(4,6,request_service=hardware,compute_ns=64.,commit_lead_ns=32.)
        self.assertEqual(transfer.timed_lifecycle(4,6,request_service=hardware,compute_ns=64.,
            commit_lead_ns=32.,producer_control_ns=(0.,0.,0.)),legacy)

    def test_async_credit_release_and_software_slot_release_are_separate(self):
        for source_type,start,full in ((ReadyCreditSource,11.,134.),(OneCreditSource,112.,336.)):
            source=source_type()
            report=transfer.timed_lifecycle(13,4,request_service=None,input_source_async=source,
                input_source_bind_events=getattr(source,'bind_scheduler',None),
                input_destination_service=SharedDestination(100.),
                compute_ns=8.,commit_lead_ns=4.,release_visibility_ns=7.,
                source_tiles=32,source_profile='generation')
            self.assertEqual(source.sources[1]['start'],start)
            self.assertEqual(source.completed[0][0],112.)
            self.assertEqual(report['timeline'][0]['full_ready_ns'],full)
            self.assertEqual(len(source.completed),39)
            self.assertEqual(len(source.released),39)
            self.assertEqual(source.by_demand,{})
            self.assertEqual(source.current,None)
            for row in report['timeline'][4:]:
                predecessor=report['timeline'][row['tile']-4]
                self.assertGreaterEqual(row['request_A_issue_ns'],predecessor['empty_complete_ns'])
            self.assertTrue(all(slot['full']['pending_tx_bytes']==0 for slot in report['slots']))

    def test_async_notifications_preserve_logical_request_metadata_order(self):
        source=DeferredNotifySource()
        report=transfer.timed_lifecycle(4,6,request_service=None,input_source_async=source,
            input_source_bind_events=source.bind_scheduler,compute_ns=8.,commit_lead_ns=4.,
            source_tiles=32,source_profile='generation')
        self.assertNotEqual(source.notified,list(range(12)))
        self.assertEqual([q['operand'] for q in report['requests']],['A','B0','B1']*4)
        self.assertEqual([q['source_k_tile'] for q in report['requests']],[i//3 for i in range(12)])
        self.assertEqual([q['source_ready_ns'] for q in report['requests'][:3]],[20.,1.,1.])
        self.assertEqual(report['timeline'][0]['full_ready_ns'],20.)

    def test_destination_a_can_finish_after_last_source(self):
        def destination(t, cta, tile, atom, weight):
            end=t+(100. if atom==0 else 0.)
            return dict(start_ns=t,service_end_ns=end,ready_ns=end+5.)
        report=transfer.timed_lifecycle(4,6,
            request_service=self.synthetic_source((10,20,30)),compute_ns=8.,commit_lead_ns=4.,
            fast_return_ns=5.,blocked_return_ns=7.,input_destination_service=destination)
        self.assertEqual([r['source_ready_ns'] for r in report['requests'][:3]],[10.,20.,30.])
        first={r['kind']:r for r in report['deliveries'] if r['index']==0}
        self.assertEqual(first['issue_a']['service_end_ns'],110.)
        self.assertEqual(first['issue_a']['ready_ns'],115.)
        self.assertEqual(report['timeline'][0]['full_ready_ns'],115.)
        self.assertEqual(report['timeline'][0]['compute_begin_ns'],122.)

    def test_destination_b1_completion_controls_full(self):
        def destination(t, cta, tile, atom, weight):
            return dict(start_ns=t,service_end_ns=t,ready_ns=t+(200. if atom==2 else 0.))
        report=transfer.timed_lifecycle(4,6,
            request_service=self.synthetic_source((10,20,30)),compute_ns=8.,commit_lead_ns=4.,
            input_destination_service=destination)
        self.assertEqual(report['timeline'][0]['full_ready_ns'],230.)
        self.assertEqual(report['timeline'][0]['compute_begin_ns'],230.)

    def test_destination_fifo_keeps_slot_release_dependency(self):
        frontier=0.
        def destination(t, cta, tile, atom, weight):
            nonlocal frontier
            start=max(t,frontier);frontier=start+10.*weight
            return dict(start_ns=start,service_end_ns=frontier,ready_ns=frontier)
        report=transfer.timed_lifecycle(13,4,
            request_service=self.synthetic_source((0,0,0)),compute_ns=8.,commit_lead_ns=4.,
            release_visibility_ns=7.,input_destination_service=destination)
        rows=report['timeline']
        self.assertEqual(rows[0]['full_ready_ns'],20.)
        self.assertEqual(rows[0]['compute_begin_ns'],20.)
        self.assertEqual(rows[0]['empty_complete_ns'],rows[1]['wait1_return_ns']+7.)
        self.assertEqual(rows[4]['request_A_issue_ns'],rows[0]['empty_complete_ns'])
        for row in rows:
            done=max(r['ready_ns'] for r in report['deliveries'] if r['index']==row['tile'])
            self.assertEqual(row['full_ready_ns'],done)

    def test_timed_reuse_waits_for_prior_wait1_release(self):
        report = transfer.timed_lifecycle(13, 4,
            request_service=self.synthetic_source((0,0,0)), compute_ns=4., commit_lead_ns=2.)
        rows = report['timeline']
        self.assertEqual(rows[0]['release_ns'], rows[1]['wait1_return_ns'])
        self.assertEqual(rows[4]['request_A_issue_ns'], rows[0]['release_ns'])
        self.assertEqual(rows[-1]['release_ns'], report['tail_wait0_and_release_ns'])
        self.assertEqual([slot['generation'] for slot in report['slots']], [4,3,3,3])

    def test_empty_completion_delays_reuse_but_not_consumer_continuation(self):
        arguments=dict(request_service=self.synthetic_source((0,0,0)),
                       compute_ns=4.,commit_lead_ns=2.)
        before=transfer.timed_lifecycle(13,4,**arguments)
        delayed=transfer.timed_lifecycle(13,4,release_visibility_ns=10.,**arguments)
        rows=delayed['timeline']
        self.assertEqual(rows[0]['release_emit_ns'],rows[1]['wait1_return_ns'])
        self.assertEqual(rows[0]['empty_complete_ns'],rows[0]['release_emit_ns']+10.)
        self.assertEqual(rows[4]['request_A_issue_ns'],rows[0]['empty_complete_ns'])
        self.assertEqual(rows[1]['compute_begin_ns'],before['timeline'][1]['compute_begin_ns'])
        self.assertLess(rows[2]['compute_begin_ns'],rows[0]['empty_complete_ns'])
        self.assertEqual(delayed['tail_wait0_and_release_ns'],delayed['tail_wait0_return_ns']+10.)
        self.assertEqual(rows[-1]['empty_complete_ns'],delayed['tail_wait0_and_release_ns'])

    def test_fixed_generation_source_has_same_prefix_across_stage_and_length(self):
        reports = [transfer.timed_lifecycle(length, stages,
            request_service=self.synthetic_source((0,0,0)), compute_ns=512., commit_lead_ns=256.,
            source_tiles=32, source_profile='generation')
            for stages, length in ((6,4), (6,13), (4,13), (6,32))]
        for report in reports:
            self.assertEqual(report['source_tiles'], 32)
            self.assertEqual(report['descriptors'], transfer.expected_descriptor(32))
            for request in report['requests']:
                self.assertEqual(request['source_k_tile'], request['tile'])
                self.assertEqual(request['source_row_stride_bytes'],
                                 4096 if request['operand'] == 'A' else 256)
        fields = ('operand', 'source_k_tile', 'source_coord', 'source_byte_offset',
                  'source_row_stride_bytes', 'bytes')
        prefixes = [[{key:request[key] for key in fields} for request in report['requests'][:12]]
                    for report in reports]
        self.assertTrue(all(prefix == prefixes[0] for prefix in prefixes))

    def test_initial_consumer_delay_keeps_producer_and_source_timing(self):
        arguments=dict(request_service=self.synthetic_source((5000,5000,5000)),
                       compute_ns=512.,commit_lead_ns=256.)
        baseline=transfer.timed_lifecycle(4,6,**arguments)
        early=transfer.timed_lifecycle(4,6,consumer_entry_delay_ns=4096.,**arguments)
        late=transfer.timed_lifecycle(4,6,consumer_entry_delay_ns=6000.,**arguments)
        for report,entry,returned in ((early,4096.,5000.),(late,6000.,6000.)):
            self.assertEqual(report['timeline'][0]['full_enter_ns'],entry)
            self.assertEqual(report['timeline'][0]['full_return_ns'],returned)
            self.assertEqual(report['requests'],baseline['requests'])
        self.assertEqual(early['tail_wait0_and_release_ns'],baseline['tail_wait0_and_release_ns'])
        self.assertEqual(late['tail_wait0_and_release_ns'],baseline['tail_wait0_and_release_ns']+1000.)

    def test_batch_131_matches_both_operand_matrix_addresses(self):
        descriptor = transfer.expected_descriptor(32, 132)
        self.assertEqual(descriptor['A']['global_dim'], [2048,128,132])
        self.assertEqual(descriptor['B']['global_dim'], [128,2048,132])
        for stage in (4,6):
            for request in transfer.requests(12,stage,32,'generation',source_batch=131,source_batches=132):
                self.assertEqual(request['source_coord'][2],131)
                first = 131*128*2048
                if request['operand']=='A':
                    matrix = 2*(first+np.arange(128)[:,None]*2048+12*64+np.arange(64)[None,:])
                else:
                    column = 64 if request['operand']=='B1' else 0
                    matrix = 2*(first+(12*64+np.arange(64)[:,None])*128+column+np.arange(64)[None,:])
                self.assertTrue(np.array_equal(transfer.source_offsets(request,32),matrix))

    def test_batch_address_reaches_source_service(self):
        reports = []
        for batch in (0,131):
            seen = []
            def source(t, request):
                seen.append(request)
                served = self.synthetic_source((10,10,10))(t,request)
                served['source_ready'] += request['source_coord'][2]*5
                return served
            reports.append(transfer.timed_lifecycle(13,6,request_service=source,
                compute_ns=8.,commit_lead_ns=4.,source_tiles=32,source_profile='generation',
                source_batch=batch,source_batches=132))
            self.assertEqual([q['source_coord'][2] for q in seen],[batch]*39)
            self.assertEqual(reports[-1]['source_batch'],batch)
        self.assertEqual([r['timeline'][0]['full_ready_ns'] for r in reports],[10.,665.])

    def test_cta_group_single_actor_matches_local_endpoints(self):
        for stage,gap,operand in ((4,0.,0.),(6,0.,0.),(4,7.,11.),(6,7.,11.)):
            kwargs = dict(compute_ns=8.,commit_lead_ns=4.,completion_lag_ns=2.,
                loop_advance_ns=3.,release_visibility_ns=7.,source_tiles=32,source_profile='generation',
                producer_gap_ns=gap,operand_gap_ns=operand)
            single = transfer.timed_lifecycle(13,stage,
                request_service=self.synthetic_source((10,20,30)),**kwargs)
            source = self.synthetic_source((10,20,30))
            group = transfer.timed_cta_group(13,stage,[0],source_batches=1,
                request_service=lambda t,cta,q:source(t,q),**kwargs)
            self.assertEqual(group['ctas'][0]['timeline'],single['timeline'])
            self.assertEqual(group['ctas'][0]['slots'],single['slots'])
            self.assertEqual(group['tail_wait0_and_release_ns'],single['tail_wait0_and_release_ns'])
            self.assertEqual([{k:v for k,v in q.items() if k!='cta'} for q in group['requests']],single['requests'])

    def test_cta_group_async_admission_keeps_control_intervals(self):
        for ready in (30.,100.):
            report=transfer.timed_cta_group(4,6,[0,1],source_batches=2,request_service=None,
                input_source_async=KnownAdmissionSource(ready),compute_ns=8.,commit_lead_ns=4.,
                producer_gap_ns=3.,operand_gap_ns=1.,source_tiles=32,source_profile='generation')
            for cta in (0,1):
                requests=[q for q in report['requests'] if q['cta']==cta]
                self.assertEqual([q['arrival_ns'] for q in requests[:4]],[0.,11.,12.,15.])
                self.assertEqual(requests[0]['issue_return_ns'],10.)
                self.assertEqual(requests[0]['source_ready_ns'],ready)
                self.assertEqual(report['ctas'][cta]['timeline'][0]['full_ready_ns'],max(ready,32.))

    def test_cta_group_uses_one_shared_source_object(self):
        counts = []
        for batches in ([0,0],[0,1]):
            active = {}
            def source(t,cta,q):
                key = (q['operand'],tuple(q['source_coord']))
                merged = key in active
                if not merged:
                    active[key] = len(active)
                return dict(source_id=active[key],source_key=key,merged=merged,
                    source_ready=t+100.,source_busy_ns=0.,service_start=t,service_end=t,
                    issue_return_ns=t)
            group = transfer.timed_cta_group(4,4,batches,source_batches=132,
                request_service=source,compute_ns=8.,commit_lead_ns=4.,source_tiles=32,source_profile='generation')
            counts.append(group['accounting']['input_merged_count'])
            self.assertEqual(len(group['requests']),24)
            self.assertEqual([c['source_batch'] for c in group['ctas']],batches)
        self.assertEqual(counts,[12,0])

    def test_cta_group_async_destination_credit_carries_cta_identity(self):
        source = OneCreditSource()
        report = transfer.timed_cta_group(4,4,[0,1],source_batches=132,
            request_service=None,input_source_async=source,
            input_destination_service=SharedDestination(100.),compute_ns=8.,commit_lead_ns=4.,
            source_tiles=32,source_profile='generation')
        self.assertEqual(len(source.completed),24)
        self.assertEqual(len(source.released),24)
        self.assertEqual({q['cta'] for q in report['requests']},{0,1})
        self.assertTrue(all(q['source_coord'][2]==q['cta'] for q in report['requests']))
        self.assertTrue(all(slot['full']['pending_tx_bytes']==0 for actor in report['ctas'] for slot in actor['slots']))

    def test_eight_shapes_and_exact_work(self):
        rows = transfer.cases()
        self.assertEqual(len(rows), 8)
        self.assertEqual({(row['stages'], row['ktiles'], row['trace']) for row in rows},
            {(stages, ktiles, trace) for stages, ktiles in ((6,4), (6,13), (6,32), (4,13))
             for trace in (False, True)})
        self.assertEqual(sum(row['split'] == 'dev' for row in rows)*3, 18)
        self.assertEqual(sum(row['split'] == 'heldout' for row in rows)*3, 6)
        for row in rows:
            self.assertEqual(row['wgmma_count'], 8*row['ktiles'])
            self.assertEqual(row['work_flop'], 2097152*row['ktiles'])

    def test_three_requests_and_distinct_global_destination_offsets(self):
        for stages in (4, 6):
            for profile, source_tiles in (('ring', stages), ('generation', 13)):
                for tile in range(13):
                    requests = transfer.requests(tile, stages, source_tiles, profile)
                    self.assertEqual([request['operand'] for request in requests], ['A','B0','B1'])
                    self.assertEqual([request['bytes'] for request in requests], [16384,8192,8192])
                    self.assertEqual(sum(request['bytes'] for request in requests), 32768)
                    source = tile % stages if profile == 'ring' else tile
                    a, b0, b1 = requests
                    for request in requests:
                        self.assertEqual(request['source_k_tile'], source)
                        self.assertEqual(request['slot'], tile % stages)
                    self.assertEqual(a['source_coord'], [64*source,0,0])
                    self.assertEqual(b0['source_coord'], [0,64*source,0])
                    self.assertEqual(b1['source_coord'], [64,64*source,0])
                    self.assertEqual(a['source_byte_offset'], 128*source)
                    self.assertEqual(b0['source_byte_offset'], 16384*source)
                    self.assertEqual(b1['source_byte_offset']-b0['source_byte_offset'], 128)
                    self.assertEqual(a['source_row_stride_bytes'], 128*source_tiles)
                    self.assertEqual(b0['source_row_stride_bytes'], 256)
                    self.assertEqual(b1['source_row_stride_bytes'], 256)
                    self.assertEqual(a['destination_byte_offset'], (tile % stages)*16384)
                    self.assertEqual(b0['destination_byte_offset'], 98304+(tile % stages)*16384)
                    self.assertEqual(b1['destination_byte_offset']-b0['destination_byte_offset'], 8192)

    def test_half_offsets_cover_each_source_once_and_split_b_columns(self):
        for stages in (4, 6):
            for profile, source_tiles in (('ring', stages), ('generation', 13)):
                a_offsets, b_offsets = [], []
                for tile in range(source_tiles):
                    a, b0, b1 = transfer.requests(tile, stages, source_tiles, profile)
                    a_actual = transfer.source_offsets(a, source_tiles)
                    b0_actual = transfer.source_offsets(b0, source_tiles)
                    b1_actual = transfer.source_offsets(b1, source_tiles)
                    source = tile
                    a_expected = 2*(np.arange(128)[:,None]*(64*source_tiles)+
                                    64*source+np.arange(64)[None,:])
                    b_expected = 2*((64*source+np.arange(64)[:,None])*128+
                                    np.arange(64)[None,:])
                    np.testing.assert_array_equal(a_actual, a_expected)
                    np.testing.assert_array_equal(b0_actual, b_expected)
                    np.testing.assert_array_equal(b1_actual, b_expected+128)
                    a_offsets.append(a_actual.reshape(-1))
                    b_offsets.extend((b0_actual.reshape(-1), b1_actual.reshape(-1)))
                expected = np.arange(128*64*source_tiles, dtype=np.int64)*2
                np.testing.assert_array_equal(np.sort(np.concatenate(a_offsets)), expected)
                np.testing.assert_array_equal(np.sort(np.concatenate(b_offsets)), expected)

    def test_ring_reference_is_identical_to_handoff(self):
        for case in transfer.cases():
            for seed in handoff.SEEDS:
                np.testing.assert_array_equal(
                    transfer.reference(case['ktiles'], case['stages'], seed, profile='ring'),
                    handoff.reference(case['ktiles'], case['stages'], seed))

    def test_positive_generation_reference_matches_scalar_integer_dot(self):
        row, column = handoff.output_coordinates()
        for stages in (4, 6):
            for length in (4, 13, 32):
                actual = transfer.reference(length, stages, 22, profile='generation')
                matrix = generation_matrix(length, stages, 22)
                np.testing.assert_array_equal(actual, matrix[row,column])
                for position in (0, 63, 64, 8191, 16383):
                    total = 0
                    for tile in range(length):
                        for inner in range(64):
                            a = 1+(7*int(row[position])+13*inner+3*(22 % 8)) % 8+tile
                            b = 1+(11*int(column[position])+5*inner+5*(22 % 8)) % 8+tile
                            total += a*b
                    self.assertEqual(actual[position], total/1024)

    def test_missing_a_b_or_b1_refill_strictly_lowers_positive_witness(self):
        for stages in (4, 6):
            correct = generation_matrix(13, stages, 41)
            for operand in ('A', 'B'):
                wrong = generation_matrix(13, stages, 41, stale_operand=operand)
                self.assertTrue(np.all(wrong < correct), operand)
            wrong_b1 = generation_matrix(13, stages, 41, stale_operand='B1')
            np.testing.assert_array_equal(wrong_b1[:,:64], correct[:,:64])
            self.assertTrue(np.all(wrong_b1[:,64:] < correct[:,64:]))

    def test_descriptor_metadata_is_three_dimensional(self):
        for source_tiles in (4, 6, 13, 32):
            descriptor = transfer.expected_descriptor(source_tiles)
            k = 64*source_tiles
            for operand, dims, strides, box in (
                    ('A', [k,128,1], [2*k,256*k], [64,128,1]),
                    ('B', [128,k,1], [256,256*k], [64,64,1])):
                fields = descriptor[operand]
                self.assertEqual(fields['global_dim'], dims)
                self.assertEqual(fields['byte_strides'], strides)
                self.assertEqual(fields['box'], box)
                self.assertEqual(fields['swizzle_bytes'], 128)
                self.assertEqual(fields['rank'], 3)
                self.assertEqual(fields['encoding_type'], 'UINT16')
                self.assertEqual(fields['numeric_type'], 'FP16')
                self.assertEqual(fields['element_strides'], [1,1,1])
                self.assertEqual(fields['interleave'], 'NONE')
                self.assertEqual(fields['l2_promotion_bytes'], 128)
                self.assertEqual(fields['oob_fill'], 'NONE')

    def test_all_ktile_transfers_gate_full_and_preserve_program_order(self):
        for stages, length in ((6,4), (6,13), (6,32), (4,13)):
            program = transfer.lifecycle(length, stages)
            nodes = program['nodes']
            by_kind = {}
            for node in nodes:
                self.assertTrue(all(previous < node['id'] for previous in node['predecessors']))
                self.assertEqual(node['slot'], node['tile'] % stages)
                self.assertEqual(node['generation'], node['tile']//stages)
                by_kind.setdefault(node['kind'], []).append(node)
            self.assertEqual(len(by_kind['tma_issue']), 3*length)
            self.assertEqual(len(by_kind['tma_complete']), 3*length)
            self.assertEqual(len(by_kind['producer_acquire']), length)
            self.assertEqual(len(by_kind['empty_release']), length)
            for tile in range(length):
                select = lambda kind: [node for node in by_kind[kind] if node['tile'] == tile]
                expect = select('expect_tx')[0]
                self.assertEqual(expect['bytes'], 32768)
                issues, completions = select('tma_issue'), select('tma_complete')
                self.assertEqual({node['operand'] for node in issues}, {'A','B0','B1'})
                for records in (issues, completions):
                    self.assertEqual({node['operand']:node['bytes'] for node in records},
                                     dict(A=16384, B0=8192, B1=8192))
                full = select('full_ready')[0]
                consumer = select('consumer_full_wait')[0]
                for issue in issues:
                    self.assertIn(expect['id'], ancestors(nodes, issue['id']))
                    self.assertTrue(issue['in_timed_window'])
                for completion in completions:
                    self.assertIn(completion['id'], ancestors(nodes, full['id']))
                self.assertIn(full['id'], ancestors(nodes, consumer['id']))
                acquire = select('producer_acquire')[0]
                if tile >= stages:
                    release = next(node for node in by_kind['empty_release'] if node['tile'] == tile-stages)
                    self.assertIn(release['id'], ancestors(nodes, acquire['id']))
                if tile:
                    prior_issues = [node for node in by_kind['tma_issue'] if node['tile'] == tile-1]
                    for prior_issue in prior_issues:
                        self.assertIn(prior_issue['id'], ancestors(nodes, acquire['id']))
                    prior_control = next(node for node in by_kind['commit' if tile == 1 else 'empty_release']
                                         if node['tile'] == (0 if tile == 1 else tile-2))
                    self.assertIn(prior_control['id'], ancestors(nodes, consumer['id']))
                release = select('empty_release')[0]
                self.assertEqual(release['arrivals'], 128)
                predecessor = nodes[release['predecessors'][0]]
                self.assertEqual(predecessor['kind'], 'tail_wait0' if tile == length-1 else 'wait1')
            if length == 4:
                self.assertEqual(len(by_kind['tma_issue']), 12)

    def test_full_requires_three_completions_and_empty_all_arrivers(self):
        for stages in (4, 6):
            full = [BarrierPhase(1) for _ in range(stages)]
            empty = [BarrierPhase(128) for _ in range(stages)]
            for tile in range(13):
                slot, generation = tile % stages, tile//stages
                self.assertTrue(empty[slot].wait_complete((generation+1) % 2))
                self.assertFalse(full[slot].arrive_and_expect_tx(32768))
                self.assertFalse(full[slot].complete_tx(16384))
                self.assertFalse(full[slot].wait_complete(generation % 2))
                self.assertFalse(full[slot].complete_tx(8192))
                self.assertFalse(full[slot].wait_complete(generation % 2))
                self.assertTrue(full[slot].complete_tx(8192))
                self.assertTrue(full[slot].wait_complete(generation % 2))
                for _ in range(127):
                    self.assertFalse(empty[slot].arrive())
                self.assertFalse(empty[slot].wait_complete((generation+2) % 2))
                self.assertTrue(empty[slot].arrive())
                self.assertTrue(empty[slot].wait_complete((generation+2) % 2))
                self.assertEqual(full[slot].completed_phases, generation+1)
                self.assertEqual(empty[slot].completed_phases, generation+1)


if __name__ == '__main__':
    unittest.main()
