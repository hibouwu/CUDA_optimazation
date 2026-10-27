"""Source constraints and diagnostic interval validity, without GPU service fitting."""
import math
import unittest

from output_model import cfg_a_rank_dependencies, cfg_a_output_source_guarantees, cfg_a_empty_arrival_routes, cfg_a_empty_reuse_interval, dependency_bounds, dependency_order
from analyze_r18 import rank_dependency_anchors, validate_rank_setup
from supply_model import shared_stage_events


class OutputCopyMaskTests(unittest.TestCase):
    def parameters(self, **changes):
        result=dict(service_ns=4.,read_latency_ns=7.,compute_ns=10.,
            output_latency_ns=9.,output_weight=2.,epilogue_prepare_ns=2.,
            epilogue_write_ns=3.,epilogue_control_ns=1.,epilogue_tail_ns=5.,
            trace_requests=True,local_epilogue_exclusion='both')
        result.update(changes)
        return result

    def test_default_and_all_enabled_are_identical(self):
        baseline=shared_stage_events(8,4,3,**self.parameters())
        for mask in (None,[True]*3):
            with self.subTest(mask=mask):
                actual=shared_stage_events(8,4,3,output_tma_enabled=mask,**self.parameters())
                self.assertEqual(actual,baseline)

    def test_disabling_instant_copies_preserves_all_sts_and_control_times(self):
        parameters=self.parameters(service_ns=0.,output_latency_ns=0.)
        baseline=shared_stage_events(8,4,3,**parameters)
        masked=shared_stage_events(8,4,3,output_tma_enabled=[True,False,False],**parameters)
        self.assertEqual(masked['ctas'],baseline['ctas'])
        self.assertEqual(masked['local_epilogue_intervals'],baseline['local_epilogue_intervals'])
        self.assertEqual(masked['makespan_ns'],baseline['makespan_ns'])

    def test_only_cta_zero_copies_charge_server_and_notify_observer(self):
        observed=[]
        result=shared_stage_events(8,4,3,output_tma_enabled=[True,False,False],
            output_source_observer=lambda t,c,j:observed.append((t,c,j)),**self.parameters())
        copies=[r for r in result['requests'] if r['kind']=='store']
        self.assertEqual(len(copies),16)
        self.assertEqual({r['cta'] for r in copies},{0})
        self.assertEqual({r['index'] for r in copies},set(range(16)))
        self.assertEqual(sorted(observed),sorted((r['ready_ns'],0,r['index']) for r in copies))
        self.assertEqual(result['request_count'],len(result['requests']))
        self.assertEqual(result['request_count'],3*2*8*3+16)
        self.assertEqual(result['shared_busy_ns'],
                         sum(r['service_end_ns']-r['start_ns'] for r in result['requests']))
        self.assertEqual(result['shared_busy_ns']-result['source_busy_ns'],16*4.)
        self.assertEqual(result['output_tma_enabled'],[True,False,False])
        for actor in result['ctas'][1:]:
            for output in actor['outputs']:
                begin=output['begin']
                self.assertEqual(output['prepare_begin'],[begin]+[begin+5+6*j for j in range(7)])
                self.assertEqual(output['write_begin'],[begin+2]+[begin+8+6*j for j in range(7)])
                self.assertEqual(output['submitted'],[begin+8+6*j for j in range(7)]+[begin+48])
                self.assertEqual(output['source_done'],output['submitted'])
                self.assertEqual(output['end'],begin+53)

    def test_mask_shape_and_boolean_values_match_ctas(self):
        for mask in ([True],[True,False,1]):
            with self.subTest(mask=mask),self.assertRaisesRegex(ValueError,'one boolean per CTA'):
                shared_stage_events(8,4,3,output_tma_enabled=mask,**self.parameters())


class RankDependencyTests(unittest.TestCase):
    def setUp(self):
        self.graph = cfg_a_rank_dependencies(6, 8, 2)
        self.nodes = self.graph['nodes']

    def test_destination_bytes_and_four_empty_arrivals(self):
        for rank in range(2):
            transfers = [t for t in self.graph['transfers']
                         if t['group'] == 0 and t['destination_rank'] == rank]
            self.assertEqual(sum(t['bytes'] for t in transfers), 32768)
            self.assertEqual({t['source_rank'] for t in transfers if t['operand'] == 'B'}, {0, 1})
            self.assertTrue(all(t['completion'] in self.nodes[f'r{rank}.g0.full'] for t in transfers))
            arrivals = [n for n in self.nodes[f'r{rank}.g6.acquire'] if 'empty_to' in n]
            self.assertEqual(set(arrivals), {f'r{s}.w{w}.g0.empty_to_r{rank}'
                                            for s in range(2) for w in range(2)})
        self.assertFalse(any('empty_to' in n for n in self.nodes['r0.g5.acquire']))

    def test_effective_arrival_threads_cover_each_destination_exactly_four_times(self):
        routes=cfg_a_empty_arrival_routes()
        for source in range(2):
            selected=[r for r in routes if r['source_rank']==source]
            self.assertEqual([(r['physical_thread'],r['destination_rank']) for r in selected],
                             [(128,0),(168,1),(256,0),(296,1)])
        for destination in range(2):
            selected=[r for r in routes if r['destination_rank']==destination]
            self.assertEqual(sum(r['arrival_count'] for r in selected),4)
            self.assertEqual({(r['source_rank'],r['consumer_wg']) for r in selected},
                             {(0,0),(0,1),(1,0),(1,1)})
            for route in selected:
                self.assertIn(route['visibility_token_template'].format(group=0),self.nodes[f'r{destination}.g6.acquire'])
                self.assertIn(route['emit_token_template'].format(group=0),self.nodes)
        self.assertTrue(all(r['lane']==8 and r['warp_in_wg']==1 for r in routes if r['destination_rank']==1))

    def test_slot_generation_continues_across_output_boundary(self):
        group = self.graph['groups'][8]
        self.assertEqual((group['tile'], group['k'], group['slot'], group['generation']), (1, 0, 2, 1))
        self.assertEqual((group['empty_wait_phase'], group['full_wait_phase']), (0, 1))
        self.assertIn('r1.w1.g2.empty_to_r0', self.nodes['r0.g8.acquire'])
        self.assertIn('r0.w0.t0.DONE', self.nodes['r0.w0.g8.wait_return'])

    def test_release_requires_next_commit_and_prior_retirement(self):
        self.assertEqual(self.nodes['r0.w0.g6.safe_release'],
                         ['r0.w0.g7.commit', 'r0.w0.g6.retired'])
        self.assertEqual(self.nodes['r0.w0.g7.safe_release'], ['r0.w0.g7.tail_wait0'])
        self.assertIn('r0.w0.g7.retired', self.nodes['r0.w0.g7.tail_wait0'])
        self.assertEqual(self.nodes['r0.w0.g7.empty_to_r1'], ['r0.w0.g7.empty_emit_to_r1'])
        self.assertEqual(self.nodes['r0.w0.g7.empty_emit_to_r1'], ['r0.w0.g7.release_issue'])
        self.assertNotIn('r0.w0.g7.empty_to_r1', self.nodes['r0.w0.t0.MAIN'])

    def emitter_packet(self):
        acquire=dict(invocation_id='raw-process:target0',cluster_id=3,destination_rank=1,acquire_group=22,stages=6,slot=4,generation=3,
                     empty_wait_phase=0,acquire_end_ns=180)
        senders=[]
        for i,route in enumerate(r for r in cfg_a_empty_arrival_routes() if r['destination_rank']==1):
            senders.append(dict(invocation_id=acquire['invocation_id'],cluster_id=3,source_rank=route['source_rank'],consumer_wg=route['consumer_wg'],
                physical_thread=route['physical_thread'],destination_rank=1,release_group=16,
                slot=4,generation=2,consumer_phase=0,release_pre_ns=100+10*i,release_post_ns=104+10*i))
        return senders,acquire

    def test_actual_sender_bounds_keep_emit_and_remote_processing_separate(self):
        senders,acquire=self.emitter_packet()
        result=cfg_a_empty_reuse_interval(senders,acquire)
        self.assertEqual(result['last_emit_ns'],[130,134])
        self.assertEqual(result['post_last_emit_to_acquire_end_ns'],[46,50])
        acquire['acquire_end_ns']=132
        self.assertEqual(cfg_a_empty_reuse_interval(senders,acquire)['last_emit_ns'],[130,132])
        self.assertGreater(senders[-1]['release_post_ns'],acquire['acquire_end_ns'])

    def test_reject_incomplete_wrong_sender_or_wrong_generation(self):
        senders,acquire=self.emitter_packet()
        with self.assertRaises(ValueError):cfg_a_empty_reuse_interval(senders[:-1],acquire)
        for field,value in [('physical_thread',256),('release_group',10),('consumer_phase',1),
                            ('cluster_id',99),('invocation_id','another-call')]:
            changed=[dict(r) for r in senders];changed[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                cfg_a_empty_reuse_interval(changed,acquire)
        changed=[dict(r) for r in senders];changed[-1]=dict(changed[0])
        with self.assertRaises(ValueError):cfg_a_empty_reuse_interval(changed,acquire)

    def test_reject_noninteger_or_inconsistent_globaltimer_observations(self):
        senders,acquire=self.emitter_packet()
        for bad in (math.nan,math.inf,True,130.5,-1,0,2**64):
            changed=[dict(r) for r in senders];changed[0]['release_pre_ns']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                cfg_a_empty_reuse_interval(changed,acquire)
        with self.assertRaises(ValueError):
            cfg_a_empty_reuse_interval(senders,dict(acquire,acquire_end_ns=129))

    def test_producer_acquire_return_is_after_expect_tx(self):
        self.assertEqual(self.nodes['r0.g6.acquire_return'],['r0.g6.arrive_expect_tx'])
        self.assertEqual(self.nodes['r0.g6.A_issue'],['r0.g6.acquire_return'])

    def test_reuse_packet_keeps_global_generation_across_output_boundary(self):
        senders,acquire=self.emitter_packet()
        acquire.update(acquire_group=69,slot=3,generation=11,empty_wait_phase=0)
        for sender in senders:
            sender.update(release_group=63,slot=3,generation=10,consumer_phase=0)
        result=cfg_a_empty_reuse_interval(senders,acquire)
        self.assertEqual((result['release_group'],result['acquire_group']),(63,69))

    def test_exit_keeps_source_read_and_global_visibility_separate(self):
        tail = self.nodes['r0.producer_tail']
        self.assertEqual(len([n for n in tail if 'empty_to' in n]), 6 * 4)
        self.assertIn('r0.store_tail_read0', self.nodes['r0.w1.FINAL'])
        self.assertIn('r0.output_source_read', self.nodes['r0.store_tail_read0'])
        self.assertNotIn('r0.D_visible', self.nodes['r0.w1.FINAL'])
        bounds = dependency_bounds(self.graph, {'r0.w1.FINAL': (100, 100)})
        self.assertEqual(bounds['r0.D_visible'][1], None)
        self.assertEqual(bounds['cuda_complete'][1], None)
        self.assertEqual(len(dependency_order(self.graph)[0]), len(self.nodes))

    def test_delayed_output_store_and_buffer_state_cross_tile_boundary(self):
        outputs = [o for o in self.graph['outputs'] if o['rank'] == 0]
        self.assertEqual(sum(o['nominal_bytes'] for o in outputs if o['tile'] == 0), 65536)
        self.assertEqual((outputs[4]['buffer'], outputs[4]['generation']), (0, 2))
        self.assertIn('r0.d1.register_prepare', self.nodes['r0.d0.store_issue'])
        self.assertIn('r0.d0.read_retired', self.nodes['r0.d1.acquire_after'])
        self.assertIn('r0.d3.acquire_after', self.nodes['r0.d4.r2s'])
        self.assertIn('r0.d3.read_retired', self.nodes['r0.d4.acquire_after'])
        self.assertNotIn('r0.d3.source_read', self.nodes['r0.w1.t0.DONE'])
        self.assertIn('r0.d3.acquire_after', self.nodes['r0.w1.t0.DONE'])

    def test_observed_guarantee_maps_to_next_output_acquire_and_terminal_read0(self):
        guarantees = cfg_a_output_source_guarantees(8)
        first = guarantees[0]
        self.assertEqual((first['read_through_count'], first['committed_count'], first['pending_batches']), (4, 5, 1))
        self.assertEqual((first['last_subtile'], first['acquire_subtile'], first['reuse_subtile']), (3, 4, 5))
        self.assertEqual(first['guarantee_token'], 'r0.d4.acquire_after')
        self.assertEqual((first['protected_buffer'], first['source_generation']), (1, 1))
        last = guarantees[7]
        self.assertEqual(last['guarantee_token'], 'r0.store_tail_read0')
        self.assertEqual((last['read_through_count'], last['pending_batches']), (32, 0))
        self.assertEqual(last['committed_count'],32)
        graph = cfg_a_rank_dependencies(6, 64, 8)
        for entry in guarantees:
            self.assertIn(entry['source_completion'], graph['nodes'])
            self.assertIn(entry['guarantee_token'], graph['nodes'])
        for invalid in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                cfg_a_output_source_guarantees(invalid)

    def test_anchor_subtraction_preserves_large_integer_globaltimer(self):
        origin = 10**18
        ctas = []
        for rank in range(2):
            roles = [dict(tiles_ns=[[origin+100+rank, origin+200+rank,
                                    origin+201+rank, origin+300+wg+rank]],
                          final_ns=origin+400+wg+rank) for wg in range(2)]
            ctas.append(dict(entry_ns=origin+rank, work=[(rank, 0)], roles=roles))
        observed_origin, anchors = rank_dependency_anchors(ctas)
        self.assertEqual(observed_origin, origin)
        self.assertEqual(anchors['r1.w1.t0.DONE'], (302, 302))
        self.assertEqual(anchors['r1.g0.acquire'], (1, 101))
        bounds = dependency_bounds(cfg_a_rank_dependencies(6, 2, 1), anchors)
        self.assertEqual(bounds['r1.w1.FINAL'], [402, 402])
        self.assertEqual(bounds['r1.D_visible'][1], None)

    def test_graph_geometry_and_gpu_must_match_this_process(self):
        setup = dict(config='cfg_a', grid=[132, 1, 1], stages=6, cluster=[2, 1],
                     tile=[128, 128, 64], epilogue_tile=[128, 32], gpu_uuid='device1')
        validate_rank_setup(setup, dict(setup))
        for field,value in [('stages', 4), ('gpu_uuid', 'device2'), ('cluster', [1, 1])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_rank_setup(setup, dict(setup, **{field: value}))
        other = dict(setup, epilogue_tile=[64, 32])
        with self.assertRaises(ValueError):
            validate_rank_setup(other, other)

    def test_bounds_reject_invalid_observations_and_allow_negative_origin(self):
        graph = {'nodes': {'a': [], 'b': ['a']}}
        for interval in ((math.nan, math.nan), (0, math.nan), (-math.inf, 2),
                         (0, -math.inf), (3, 2), (1,), '12', (0, '2')):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                dependency_bounds(graph, {'a': interval})
        self.assertEqual(dependency_bounds(graph, {'a': (-10, -10)}, {'b': 2})['b'], [-8, None])
        self.assertEqual(dependency_bounds(graph, {'a': (1, math.inf)})['a'], [1, None])
        with self.assertRaises(ValueError):
            dependency_bounds(graph, {'a': (2, 2), 'b': (1, 1)})
        for latency in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                dependency_bounds(graph, latencies_ns={'b': latency})
        with self.assertRaises(ValueError):
            dependency_order({'nodes': {'a': ['missing']}})
        with self.assertRaises(ValueError):
            dependency_order({'nodes': {'a': ['b'], 'b': ['a']}})

    def test_large_integer_equal_bounds_and_one_ns_reversal(self):
        graph={'nodes':{'a':[],'b':['a']}}
        # Both downward and upward float rounding occurred in the old comparison.
        for stamp in (1791000000000000001,1791000000000000129):
            with self.subTest(stamp=stamp):
                bounds=dependency_bounds(graph,{'a':(stamp,stamp),'b':(stamp,stamp)})
                self.assertEqual(bounds,{'a':[stamp,stamp],'b':[stamp,stamp]})
                self.assertTrue(all(type(value) is int for pair in bounds.values() for value in pair))
                with self.assertRaisesRegex(ValueError,'infeasible'):
                    dependency_bounds(graph,{'a':(stamp+1,stamp+1),'b':(stamp,stamp)})

    def test_integer_anchored_bounds_are_translation_invariant(self):
        graph={'nodes':{'a':[],'b':['a'],'c':['b'],'unknown':[]}}
        anchors={'a':(2,5),'c':(20,21)};lags={'b':3,'c':7}
        expected=dependency_bounds(graph,anchors,lags)
        for shift in (1791000000000000001,-1791000000000000001):
            translated={name:(low+shift,high+shift) for name,(low,high) in anchors.items()}
            actual=dependency_bounds(graph,translated,lags)
            unshifted={name:[None if value is None else value-shift for value in interval]
                       for name,interval in actual.items()}
            self.assertEqual(unshifted,expected)

    def test_fractional_feasibility_tolerance_is_preserved(self):
        graph={'nodes':{'a':[],'b':['a']}}
        for reversal in (0.0,5e-10,1e-9):
            with self.subTest(reversal=reversal):
                dependency_bounds(graph,{'a':(0,0),'b':(-reversal,-reversal)})
        with self.assertRaisesRegex(ValueError,'infeasible'):
            dependency_bounds(graph,{'a':(0,0),'b':(-2e-9,-2e-9)})

    def test_coarse_endpoints_do_not_localize_a_synthetic_shared_delivery_delay(self):
        graph = cfg_a_rank_dependencies(6, 64, 2)
        prices = {n: (300 if n.endswith('.retired') else 10 if
                      n.endswith('.A_visible') or '.B_to_r' in n else
                      200 if n.endswith('.DONE') else 1) for n in graph['nodes']}
        baseline = dependency_bounds(graph, latencies_ns=prices)
        predictions = []
        for group in (6, 63):
            changed = dict(prices)
            for rank in range(2):
                changed[f'r1.g{group}.B_to_r{rank}'] += 4000
            bounds = dependency_bounds(graph, latencies_ns=changed)
            endpoints = [bounds[f'r{r}.w1.t{t}.{event}'][0]
                         for r in range(2) for t in range(2) for event in ('FIRST', 'MAIN')]
            predictions.append(endpoints)
            for rank in range(2):
                def length(values, tile):
                    return values[f'r{rank}.w1.t{tile}.MAIN'][0] - values[f'r{rank}.w1.t{tile}.FIRST'][0]
                self.assertGreater(length(bounds, 0), length(baseline, 0))
                self.assertEqual(length(bounds, 1), length(baseline, 1))
        self.assertEqual(predictions[0], predictions[1])


if __name__ == '__main__':
    unittest.main()
