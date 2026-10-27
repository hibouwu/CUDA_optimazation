"""请求服务与共同程序的绑定、跨 CTA 共享和旧候选反例的 CPU 回归。"""
import copy
from graphlib import TopologicalSorter
import importlib.util
import unittest
from unittest.mock import patch

import cfg_b_composition as aggregate
import cfg_b_program
import cfg_b_request_model as request_model
import supply_model as supply
import tma_tensor_multicta as tensor


def without_full_query_graph(graph):
    graph = copy.deepcopy(graph)
    nodes = graph['nodes']
    queries = {name: nodes.pop(name) for name in list(nodes) if name.endswith('.full_query')}
    for name, predecessors in nodes.items():
        nodes[name] = [prior for p in predecessors for prior in queries.get(p, [p])]
    graph['topological_order'] = list(TopologicalSorter(nodes).static_order())
    return graph


def without_full_query_report(report):
    report = copy.deepcopy(report)
    for detail in report['details']:
        times = detail['node_times_ns']
        queries = [name for name in times if name.endswith('.full_query')]
        for name in queries:
            del times[name]
        detail['dependency_edges_checked'] -= len(queries)
        report['dependency_edges_checked'] -= len(queries)
        detail['control_nodes_at_predecessor_max'].sort()
    return report


class RequestCompositionTests(unittest.TestCase):
    def program(self, stages=4, m=768, n=256, k=320):
        return cfg_b_program.build_program(dict(config='cfg_b', m=m, n=n, k=k, stages=stages),
                                           dict(grid=[4, 1, 1], stages=stages))

    def parameters(self):
        return dict(service_ns=2., read_latency_ns=17., compute_ns=64.,
            producer_gap_ns=3., operand_gap_ns=1., reuse_delay_ns=4.,
            commit_lead_ns=40., completion_lag_ns=8., issue_floor_ns=5.,
            wait_overhead_ns=4., loop_advance_ns=6., return_overhead_ns=3.,
            handoff_delay_ns=2., output_latency_ns=31., output_weight=3.,
            epilogue_prepare_ns=7., epilogue_write_ns=5., epilogue_control_ns=2.,
            epilogue_tail_ns=3., destination_service_ns=3.)

    def test_source_fold_changes_only_actual_a_requests(self):
        row = dict(config='cfg_b', m=2816, n=1536, k=4096, stages=6,
                   lda=4104, alloc_lda=4104, seed=29)
        setup = dict(grid=[1, 132, 1], stages=6)
        original = cfg_b_program.build_program(row, setup)
        folded = cfg_b_program.build_program(row, dict(setup, fold_a_source=1))
        self.assertEqual(folded['work'], original['work'])
        self.assertEqual(folded['outputs'], original['outputs'])
        changed = []
        for old, new in zip(original['source_atoms'], folded['source_atoms']):
            if old == new:
                continue
            changed.append(new)
            self.assertEqual(old['operand'], 'A')
            self.assertEqual(new, dict(old, origin=[old['origin'][0]-2176, old['origin'][1]]))
            self.assertGreaterEqual(old['origin'][0], 2176)
        self.assertEqual(len(changed), 60)
        self.assertEqual({a['origin'][0] for a in changed}, {128*m for m in range(5)})
        for name, value in original['totals'].items():
            if name == 'distinct_source_boxes':
                self.assertEqual(folded['totals'][name], value - 5*64)
            elif name == 'distinct_valid_source_KiB':
                self.assertEqual(folded['totals'][name], value - 5*64*16)
            else:
                self.assertEqual(folded['totals'][name], value)
        before = cfg_b_program.expand_cta(original, 72)
        after = cfg_b_program.expand_cta(folded, 72)
        self.assertEqual(before['nodes'], after['nodes'])
        for old, new in zip(before['requests'], after['requests']):
            if old['output'] == 1 and old['atom'] == 'A':
                self.assertEqual(new['source']['origin'], [0, 64*new['k']])
                self.assertEqual(new['source']['shape'], [128, 64])
            else:
                self.assertEqual(new, old)
        self.assertEqual(cfg_b_program.expand_cta(original, 0), cfg_b_program.expand_cta(folded, 0))
        self.assertEqual(original, cfg_b_program.build_program(row, setup))
        self.assertEqual(folded['runtime_intervention']['fold_a_source'], 1)
        for changed_row, changed_setup in (
                (dict(row, input_mode='random'), setup), (dict(row, zero_m=2176), setup),
                (dict(row, m=2944), setup), (dict(row, swizzle=8), setup),
                (dict(row, stages=4), dict(setup, stages=4)),
                (row, dict(setup, active_stages=4)), (row, dict(setup, wgmma_repeats=3))):
            with self.assertRaisesRegex(ValueError, 'fold_a_source requires'):
                cfg_b_program.build_program(changed_row, dict(changed_setup, fold_a_source=1))

    def test_output_copy_gate_preserves_local_work_and_control(self):
        row = dict(config='cfg_b', m=768, n=256, k=320, stages=4)
        setup = dict(grid=[4, 1, 1], stages=4)
        original = cfg_b_program.build_program(row, setup)
        self.assertNotIn('runtime_intervention', original)
        self.assertNotIn('output_tma_enabled', original)
        explicit = cfg_b_program.build_program(row, dict(setup, fold_a_source=0, store_target_ctas=4))
        explicit['setup'] = dict(setup)
        self.assertEqual(explicit, original)
        for limit in (0, 1, 3):
            gated = cfg_b_program.build_program(row, dict(setup, store_target_ctas=limit))
            self.assertEqual(gated['output_tma_enabled'], [cta < limit for cta in range(4)])
            self.assertEqual(gated['runtime_intervention']['store_target_ctas'], limit)
            self.assertEqual(gated['work'], original['work'])
            self.assertEqual(gated['source_atoms'], original['source_atoms'])
            for cta in range(4):
                graph = cfg_b_program.expand_cta(gated, cta)
                self.assertEqual(graph.pop('output_tma_enabled'), cta < limit)
                self.assertEqual(graph, cfg_b_program.expand_cta(original, cta))
            demand = cfg_b_program.smem_demands(gated)
            old_demand = cfg_b_program.smem_demands(original)
            self.assertEqual(demand['paths'][:3], old_demand['paths'][:3])
            self.assertEqual(demand['paths'][3]['total_bytes'], limit*3*65536)
        for bad in (-1, 5, 1.5):
            with self.assertRaisesRegex(ValueError, 'store_target_ctas'):
                cfg_b_program.build_program(row, dict(setup, store_target_ctas=bad))
        with self.assertRaisesRegex(ValueError, 'fold_a_source must'):
            cfg_b_program.build_program(row, dict(setup, fold_a_source=2))

    def test_private_sources_keep_program_and_use_allocation_stride(self):
        row = dict(config='cfg_b', m=2816, n=1536, k=4096, lda=4104, alloc_lda=4104, storage_m=2944)
        setup = dict(grid=[1, 132, 1], stages=6)
        original = cfg_b_program.build_program(row, setup)
        private = cfg_b_program.build_program(row, dict(setup, private_source=1))
        explicit = cfg_b_program.build_program(row, dict(setup, private_source=0))
        explicit['setup'] = setup
        self.assertEqual(explicit, original)
        self.assertEqual(private['outputs'], original['outputs'])
        self.assertEqual(private['work'], original['work'])
        for cta in (0, 1, 131):
            old, new = (cfg_b_program.expand_cta(p, cta) for p in (original, private))
            self.assertEqual(old['nodes'], new['nodes'])
            for a, b in zip(old['requests'], new['requests']):
                stride = 2944*4104*2 if a['atom'] == 'A' else 4096*1536*2
                self.assertEqual(b, dict(a, source=dict(a['source'],
                    base_offset_bytes=a['source']['base_offset_bytes']+cta*stride)))
        for key in original['totals']:
            if not key.startswith('distinct_'):
                self.assertEqual(private['totals'][key], original['totals'][key])
        self.assertGreater(private['totals']['distinct_source_boxes'], original['totals']['distinct_source_boxes'])

    def test_same_s6_runtime_slots_and_repeated_compute(self):
        for ktiles, outputs, expected in ((4, 1, 192), (13, 1, 624), (4, 2, 480)):
            program = cfg_b_program.build_program(
                dict(config='cfg_b', m=128*outputs, n=128, k=64*ktiles),
                dict(grid=[1, 1, 1], stages=6, smem=215040,
                     active_stages=4, wgmma_repeats=3, epilogue_alpha=0.3333333433))
            costs = dict(service_ns=0., read_latency_ns=0., compute_ns=16., commit_lead_ns=8.)
            if outputs == 2:
                costs['compute_phases_ns'] = [dict(compute_ns=16., commit_lead_ns=8.),
                                              dict(compute_ns=24., commit_lead_ns=12.)]
            original = copy.deepcopy(costs)
            report = request_model.simulate(program, costs, detailed_ctas=(0,))
            self.assertEqual(costs, original)
            self.assertEqual(program['stages'], 6)
            self.assertEqual(program['setup']['smem'], 215040)
            detail = report['details'][0]
            times = detail['node_times_ns']
            self.assertEqual(times[f'j{outputs-1}.main_end'], expected)
            for group in detail['groups']:
                self.assertEqual(group['slot'], group['group'] % 4)
                self.assertEqual(group['wgmma_count'], 24)
                self.assertGreaterEqual(times[group['release']], times[group['compute_completion']])
                if group['previous_group'] is not None:
                    self.assertGreaterEqual(times[f'g{group["group"]}.acquire'],
                                            times[f'g{group["previous_group"]}.release'])
            self.assertEqual(report['accounting']['input_delivery_count'], 3*ktiles*outputs)
        historical = cfg_b_program.build_program(
            dict(config='cfg_b', m=128, n=128, k=256),
            dict(grid=[1, 1, 1], stages=6, active_stages=4))
        self.assertEqual(historical['runtime_intervention']['epilogue_alpha'], 1.)
        self.assertEqual(historical['runtime_intervention']['wgmma_repeats'], 1)

    def test_preparation_acquire_requests_and_critical_path_share_coordinates(self):
        import cfg_b_tma_chain as transfer
        from mbarrier_service import ConditionSyncService
        program=cfg_b_program.build_program(dict(config='cfg_b',m=128,n=128,k=832,stages=4),
            dict(grid=[1,1,1],stages=4))
        def hardware(t,q):
            atom=('A','B0','B1').index(q['operand'])
            returned=t+(10. if q['tile']==atom==0 else 0.)
            return dict(source_id=3*q['tile']+atom,source_key=(q['tile'],atom),merged=False,
                source_ready=returned+100.,source_busy_ns=0.,service_start=returned,
                service_end=returned,issue_return_ns=returned)
        costs=dict(service_ns=0.,read_latency_ns=0.,compute_ns=512.,commit_lead_ns=256.,
            completion_lag_ns=95.,loop_advance_ns=133.,input_release_delay_ns=275.,
            producer_gap_ns=3.,operand_gap_ns=1.,producer_control_ns=(7.,5.,3.))
        local=transfer.timed_lifecycle(13,4,request_service=hardware,compute_ns=512.,
            commit_lead_ns=256.,completion_lag_ns=95.,loop_advance_ns=133.,release_visibility_ns=275.,
            producer_gap_ns=3.,operand_gap_ns=1.,producer_control_ns=(7.,5.,3.))
        report=request_model.simulate(program,costs,detailed_ctas=(0,),producer_prefix_ns=13.,
            input_source_service=lambda t,cta,i,atom,weight:hardware(t,dict(tile=i,operand=('A','B0','B1')[atom])),
            input_sync_service=ConditionSyncService(4,0.,0.),critical_path=True)
        self.assertEqual(report['producer_control_windows'],
            [dict(w,start_ns=w['start_ns']+13.,end_ns=w['end_ns']+13.) for w in local['producer_control_windows']])
        detail=report['details'][0]
        self.assertEqual([q['service']['arrival_ns'] for q in detail['requests']],
                         [q['arrival_ns']+13. for q in local['requests']])
        self.assertEqual([q['service']['arrival_ns'] for q in detail['requests'][:4]],[20.,36.,40.,50.])
        for row in local['timeline']:
            self.assertEqual(detail['node_times_ns'][f'g{row["tile"]}.acquire'],row['producer_acquire_ns']+13.)
        cp=report['critical_path']
        self.assertAlmostEqual(cp['target_ns'],report['consumer_span_ns'])
        self.assertAlmostEqual(sum(cp['breakdown_ns'].values()),cp['target_ns'])
        self.assertEqual(cp['breakdown_ns'].get('epilogue_exclusion',0.),0.)
        for stages in (4,6):
            costs=dict(self.parameters(),producer_control_ns=(7.,5.,3.),input_release_delay_ns=11.)
            complete=request_model.simulate(self.program(stages),costs,detailed_ctas=range(4),critical_path=True)
            for cta in complete['details']:
                self.assertAlmostEqual(cta['critical_path']['target_ns'],cta['consumer_exit_ns'])
                for group in cta['groups']:
                    g=group['group'];times=cta['node_times_ns']
                    self.assertEqual(times[f'g{g}.A.issue']-times[f'g{g}.acquire'],7.)
                    if group['previous_group'] is not None:
                        self.assertGreaterEqual(times[f'g{g}.acquire'],times[f'g{group["previous_group"]}.release'])

    def test_common_work_sources_and_every_dependency(self):
        for stages in (4, 6):
            for sharing in (False, True):
                program = self.program(stages)
                original = copy.deepcopy(program)
                keys, counts = request_model.request_inputs(program)
                legacy = supply.cfg_b_source_program(program['case'], program['setup'])
                self.assertEqual(keys, legacy['source_keys'])
                self.assertEqual(counts, [3, 3, 3, 3])
                with patch.object(supply, 'shared_stage_events', wraps=supply.shared_stage_events) as engine:
                    report = request_model.simulate(program, self.parameters(),
                        share_inflight_sources=sharing, detailed_ctas=range(4))
                self.assertEqual(engine.call_count, 1)
                self.assertEqual(program, original)
                accounting = report['accounting']
                self.assertEqual(accounting['input_demand_count'], 12*5*3)
                self.assertEqual(accounting['input_delivery_count'], 12*5*3)
                self.assertEqual(accounting['input_source_count'] + accounting['input_merged_count'], 12*5*3)
                self.assertEqual(report['consumer_span_ns'], report['program_end_ns'])
                for cta in report['details']:
                    graph = cfg_b_program.expand_cta(program, cta['cta'])
                    times = cta['node_times_ns']
                    self.assertEqual(set(times), set(graph['nodes']))
                    for node, predecessors in graph['nodes'].items():
                        for predecessor in predecessors:
                            self.assertGreaterEqual(times[node], times[predecessor])
                    for r in cta['requests']:
                        self.assertEqual(r['bytes'], 16384 if r['atom'] == 'A' else 8192)
                        if sharing:
                            self.assertEqual(request_model.source_key(r['source']), r['service']['source_key'])
                        self.assertEqual(times[r['completion']], r['service']['ready_ns'])
                    for g in cta['groups']:
                        if g['previous_group'] is not None:
                            self.assertGreaterEqual(times[f'g{g["group"]}.A.issue'],
                                                    times[f'g{g["previous_group"]}.release'])
                        self.assertGreaterEqual(times[g['release']], times[g['compute_completion']])

    def test_deferred_first_output_matches_engine_and_future_math_dependency(self):
        program = self.program()
        keys, counts = request_model.request_inputs(program)
        for flag in (1, 2):
            parameters = dict(self.parameters(), epilogue_defer=flag)
            report = request_model.simulate(program, parameters, register_quota=True,
                                            detailed_ctas=range(4))
            legacy = supply.shared_stage_events(program['ktiles'], program['stages'], 4,
                output_counts=counts, source_keys=keys, trace_requests=True, **parameters)
            for detail, actor in zip(report['details'], legacy['ctas']):
                cta = detail['cta']
                graph = cfg_b_program.expand_cta(program, cta, register_quota=True,
                                                 epilogue_defer=flag)
                selected = flag == 2 or cta == 0
                self.assertEqual('j1.math_handoff' in graph['nodes']['j0.epilogue_permit'], selected)
                times = detail['node_times_ns']
                if selected:
                    self.assertGreaterEqual(times['j0.epilogue_permit'], times['j1.math_handoff'])
                self.assertEqual(detail['output_service'], actor['outputs'])
                for group in detail['groups']:
                    g = group['group']
                    self.assertEqual(times[f'g{g}.commit'], actor['commits'][g])
                    self.assertEqual(times[f'g{g}.release'], actor['released'][g])
                for node, predecessors in graph['nodes'].items():
                    self.assertTrue(all(times[node] >= times[p] for p in predecessors))

    def test_explicit_zero_defer_preserves_default_report(self):
        program = self.program()
        baseline = request_model.simulate(program, self.parameters(), detailed_ctas=range(4))
        explicit = request_model.simulate(program, dict(self.parameters(), epilogue_defer=0),
                                         detailed_ctas=range(4))
        explicit['parameters'].pop('epilogue_defer')
        self.assertEqual(explicit, baseline)

    def test_services_change_events_without_changing_program(self):
        program = self.program()
        parameters = self.parameters()
        baseline = request_model.simulate(program, parameters)
        for field in ('read_latency_ns', 'compute_ns', 'epilogue_tail_ns'):
            with self.subTest(service=field):
                changed = request_model.simulate(program, dict(parameters, **{field: 1000.}))
                self.assertGreater(changed['consumer_span_ns'], baseline['consumer_span_ns'])
                self.assertEqual(changed['accounting']['input_demand_count'],
                                 baseline['accounting']['input_demand_count'])
        shared = request_model.simulate(program, parameters, share_inflight_sources=True)
        self.assertGreater(shared['accounting']['input_merged_count'], 0)
        self.assertLess(shared['accounting']['input_source_count'], baseline['accounting']['input_source_count'])
        self.assertEqual(shared['accounting']['input_delivery_count'], baseline['accounting']['input_delivery_count'])

    def test_producer_prefix_matches_delayed_producer_without_delaying_consumer_permit(self):
        program = self.program()
        parameters = self.parameters()
        for sharing in (False, True):
            report = request_model.simulate(program, parameters, producer_prefix_ns=1608.,
                share_inflight_sources=sharing, detailed_ctas=range(4))
            keys, counts = request_model.request_inputs(program)
            delayed = supply.shared_stage_events(5, 4, 4, output_counts=counts, source_keys=keys,
                diagnostic_producer_offsets_ns=[1608.]*4, share_inflight_sources=sharing,
                trace_requests=True, **parameters)
            self.assertEqual(report['consumer_span_ns'], delayed['makespan_ns'])
            for cta in report['details']:
                actor = delayed['ctas'][cta['cta']]
                times = cta['node_times_ns']
                self.assertEqual(times['entry'], 0.)
                self.assertEqual(times['j0.math_permit'], 0.)
                self.assertEqual(times['producer_first_work'], 1608.)
                self.assertEqual(times['g0.acquire'], 1608.)
                for group in cta['groups']:
                    g = group['group']
                    self.assertEqual(times[f'g{g}.consumer_wait'], actor['starts'][g])
                    self.assertEqual(times[f'g{g}.release'], actor['released'][g])
                for request in cta['requests']:
                    self.assertEqual(times[request['issue']], request['service']['arrival_ns'])
                    self.assertEqual(times[request['completion']], request['service']['ready_ns'])
                self.assertEqual(cta['output_service'], actor['outputs'])

    def test_phase_clock_changes_compute_without_changing_request_work(self):
        program = cfg_b_program.build_program(dict(config='cfg_b', m=128, n=256, k=320, stages=4),
                                             dict(grid=[1, 1, 1], stages=4))
        parameters = dict(service_ns=0., read_latency_ns=0., compute_ns=64.)
        calibration = dict(dual_event_extra_ns=dict(cfg_b=100.),
                           plain_transfer=dict(cfg_b=dict(F_us=2., kappa=1.1)))
        with patch.object(request_model.phase_frequency, 'predict_phases', return_value=[2., 1.]) as clock:
            report = request_model.predict(program, parameters, dict(schema='test-static-clock'),
                calibration, commit_lead_cycles=0., detailed_ctas=(0,))
        clock.assert_called_once_with(dict(schema='test-static-clock'), program['case'], program['work'])
        times = report['details'][0]['node_times_ns']
        self.assertEqual(times['j0.main_end']-times['g0.compute_batch'], 5*256.)
        self.assertEqual(times['j1.main_end']-times['g5.compute_batch'], 5*512.)
        self.assertEqual(report['consumer_span_ns'], 3840.)
        self.assertEqual(report['dual_event_us'], 3.94)
        self.assertEqual(report['plain_us'], 2.+1.1*3.84)
        self.assertEqual(report['accounting']['input_delivery_count'], 30)

    def test_producer_prefix_maps_existing_request_path_timestamps(self):
        program = cfg_b_program.build_program(dict(config='cfg_b', m=128, n=128, k=128, stages=4),
                                             dict(grid=[1, 1, 1], stages=4))
        parameters = dict(service_ns=2., read_latency_ns=17., compute_ns=64.,
            destination_service_ns=3., input_path_service_ns=2., input_min_latency_ns=10.,
            destination_latency_ns=5.)
        report = request_model.simulate(program, parameters, producer_prefix_ns=1608., detailed_ctas=(0,))
        served = report['details'][0]['requests'][0]['service']
        expected = dict(arrival_ns=1608., source_ready_ns=1627., input_path_start_ns=1608.,
            input_path_end_ns=1610., destination_arrival_ns=1632.,
            input_earliest_destination_ns=1620., start_ns=1632., ready_ns=1635.)
        self.assertEqual({key: served[key] for key in expected}, expected)

    def test_constant_phase_service_preserves_scalar_engine_events(self):
        program = self.program()
        parameters = self.parameters()
        scalar = request_model.simulate(program, parameters, detailed_ctas=range(4))
        phases = [dict(compute_ns=parameters['compute_ns'],
                       commit_lead_ns=parameters['commit_lead_ns']) for _ in range(3)]
        phased = request_model.simulate(program, dict(parameters, compute_phases_ns=phases),
                                       detailed_ctas=range(4))
        self.assertEqual(phased['details'], scalar['details'])
        self.assertEqual(phased['accounting'], scalar['accounting'])

    def test_tensor_service_uses_gemm_addresses_and_keeps_slot_dependencies(self):
        program = self.program()
        source = dict(candidate='banked_resident_fifo', parameters=dict(service=4., latency=1000.,
            resident_service=2., resident_latency=300., destination=16., source_partitions=16))
        report = request_model.simulate(program, self.parameters(), tensor_source=source,
                                        detailed_ctas=range(4))
        self.assertTrue(report['tensor_resident_path'])
        self.assertEqual(report['accounting']['input_delivery_count'], 180)
        for cta in report['details']:
            graph = cfg_b_program.expand_cta(program, cta['cta'])
            for node, predecessors in graph['nodes'].items():
                for predecessor in predecessors:
                    self.assertGreaterEqual(cta['node_times_ns'][node], cta['node_times_ns'][predecessor])
            for request in cta['requests']:
                self.assertEqual(request_model.source_key(request['source']), request['service']['source_key'])

    def test_probe_and_gemm_adapter_agree_on_all_prefill_deliveries(self):
        probe_program = tensor.program(3, 4, 'independent', groups=4, source_ring=8, ldb=400)
        costs = dict(service=4., latency=1000., resident_service=2., resident_latency=300.,
                     destination=16., source_partitions=16, issue=2., consumer=3.)
        keys = []
        for cta in range(3):
            groups = []
            for group in range(4):
                atoms = []
                for name, shape, _ in tensor.ATOMS:
                    operand = 'A' if name == 'A' else 'B'
                    source = dict(operand=operand, shape=shape,
                        origin=[cta*128 + (64 if name == 'B1' else 0), group*64],
                        extents=[384, 512], base_offset_bytes=0,
                        stride_bytes=[1024, 2] if operand == 'A' else [2, 800])
                    atoms.append(request_model.source_key(source))
                groups.append(tuple(atoms))
            keys.append(groups)
        adapter = request_model.CfgBTensorSource(
            dict(totals=dict(distinct_valid_source_KiB=probe_program['source_union_bytes']/1024)),
            keys, 'banked_resident_fifo', costs)
        probe = tensor.simulate(probe_program, 'banked_resident_fifo', trace=True, **costs)
        gemm = supply.shared_stage_events(4, 4, 3, service_ns=0., read_latency_ns=0.,
            compute_ns=.01, producer_gap_ns=2., operand_gap_ns=2., destination_service_ns=16.,
            output_counts=[1]*3, input_source_service=adapter, trace_requests=True)
        names = {'issue_a': 'A', 'issue_b0': 'B0', 'issue_b1': 'B1'}
        actual = {(r['cta'], r['index'], names[r['kind']]): r for r in gemm['deliveries']}
        for request in probe['request_events']:
            delivery = actual[request['cta'], request['group'], request['atom']]
            self.assertEqual(request['issue'], delivery['arrival_ns'])
            self.assertEqual(request['source_ready'], delivery['source_ready_ns'])
            self.assertEqual(request['ready'], delivery['ready_ns'])

    def test_engine_domain_is_not_silently_extended(self):
        for program in (self.program(m=769), self.program(k=64), self.program(m=128, n=128)):
            with self.assertRaises(ValueError):
                request_model.simulate(program, self.parameters())
        with self.assertRaisesRegex(ValueError, 'not expanded'):
            request_model.simulate(self.program(), dict(self.parameters(), second_prefill=True))

    def test_binding_detects_completion_before_issue(self):
        program = self.program()
        keys, counts = request_model.request_inputs(program)
        result = supply.shared_stage_events(5, 4, 4, output_counts=counts,
            source_keys=keys, trace_requests=True, **self.parameters())
        atoms = {'issue_a': 'A', 'issue_b0': 'B0', 'issue_b1': 'B1'}
        deliveries = {(r['index'], atoms[r['kind']]): dict(r)
                      for r in result['deliveries'] if r['cta'] == 0}
        deliveries[0, 'A']['ready_ns'] = -1.
        with self.assertRaisesRegex(ValueError, 'service violates g0.A.complete'):
            request_model.bind_cta(cfg_b_program.expand_cta(program, 0), result['ctas'][0], deliveries)

    def test_critical_path_and_slack_match_bound_timeline(self):
        for stages in (4, 6):
            program = self.program(stages)
            baseline = request_model.simulate(program, self.parameters(), producer_prefix_ns=120.,
                                              detailed_ctas=range(4))
            reported = request_model.simulate(program, self.parameters(), producer_prefix_ns=120.,
                                              detailed_ctas=range(4), critical_path=True)
            top_cp = reported.pop('critical_path')
            self.assertEqual(top_cp['cta'], reported['critical_candidates'][0])
            self.assertAlmostEqual(top_cp['target_ns'], reported['consumer_span_ns'], places=6)
            self.assertAlmostEqual(sum(top_cp['breakdown_ns'].values()), top_cp['target_ns'], places=6)
            for cta_detail in reported['details']:
                cp = cta_detail.pop('critical_path')
                self.assertAlmostEqual(cp['target_ns'], cta_detail['consumer_exit_ns'], places=6)
                self.assertAlmostEqual(sum(cp['breakdown_ns'].values()), cp['target_ns'], places=6)
                self.assertEqual(cp['path'][0]['from_node'], 'entry')
                self.assertEqual(cp['path'][-1]['to_node'], 'consumer_exit')
                for step in cp['path']:
                    self.assertAlmostEqual(cp['slack_ns'][step['from_node']], 0., places=6)
                    self.assertAlmostEqual(cp['slack_ns'][step['to_node']], 0., places=6)
            self.assertEqual(reported, baseline)

    def test_full_query_uses_consumer_control_before_readiness(self):
        from mbarrier_service import ConditionSyncService
        for stages in (4, 6):
            program = self.program(stages)
            for entry in (0., 37.):
                costs = dict(self.parameters(), consumer_entry_delay_ns=entry,
                             input_release_delay_ns=11.)
                for sync in (False, True):
                    report = request_model.simulate(program, costs, producer_prefix_ns=120.,
                        detailed_ctas=range(4), critical_path=True,
                        input_sync_service=ConditionSyncService(stages,0.,0.) if sync else None)
                    for cta in report['details']:
                        times = cta['node_times_ns']
                        for group in cta['groups']:
                            g, j, k = group['group'], group['output'], group['k']
                            if g == 0:
                                expected = 0. if entry == 0. else entry + 120.
                            elif k == 0:
                                expected = times[f'j{j-1}.math_handoff'] + costs['handoff_delay_ns']
                                if j >= 2:
                                    expected = max(expected,times[f'j{j-2}.epilogue_handoff'])
                            else:
                                previous = f'g{g-1}.commit' if k == 1 else f'g{g-1}.wait1'
                                expected = times[previous] + costs['loop_advance_ns']
                            self.assertAlmostEqual(times[f'g{g}.full_query'], expected)
                            self.assertEqual(times[f'g{g}.consumer_wait'],
                                max(times[f'g{g}.full_query'],times[f'g{g}.full_ready']))
                            if sync:
                                self.assertEqual(times[f'g{g}.full_query'],
                                    cta['full_wait_windows'][g]['enter_ns'])
                        if entry == 0.:
                            self.assertLess(times['g0.full_query'],times['g0.full_ready'])
                        self.assertAlmostEqual(cta['critical_path']['target_ns'],cta['consumer_exit_ns'])

    def test_consumer_slack_excludes_later_producer_tail(self):
        program = cfg_b_program.build_program(dict(config='cfg_b', m=128, n=128, k=256, stages=4),
                                             dict(grid=[1, 1, 1], stages=4))
        costs = dict(service_ns=0., read_latency_ns=10., compute_ns=64.,
                     commit_lead_ns=32., input_release_delay_ns=1000.)
        report = request_model.simulate(program, costs, detailed_ctas=(0,), critical_path=True)
        detail = report['details'][0]
        self.assertEqual(report['consumer_span_ns'], 266.)
        self.assertEqual(report['program_end_ns'], 1266.)
        slack = report['critical_path']['slack_ns']
        self.assertEqual(slack['consumer_exit'], 0.)
        self.assertEqual(detail['node_times_ns']['g3.release'], 1266.)
        for node in ('g3.release', 'producer_tail', 'program_end'):
            self.assertNotIn(node, slack)


class HistoricalCandidateTests(unittest.TestCase):
    def test_phase_frequency_inference_preserves_development_snapshot(self):
        root = aggregate.DEFAULT_ARCHIVE.parents[2]/'Q09-frequency/frequency-mechanism/20261015-phase-frequency-development-001'
        model = aggregate.load_json(root/'fit-001/model.json')
        rows = aggregate.load_json(root/'rows.json')['rows']
        audit = aggregate.load_json(root/'audit.json')['cases']
        spec = importlib.util.spec_from_file_location('frozen_phase_frequency', root/'phase_frequency.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        self.assertEqual(len(rows), 176)
        for row in rows:
            case = row['case']
            work = audit[row['archive']+'/'+case['id']]['work']
            expected = original.predict(model, case, work, row['output'])
            actual = request_model.phase_frequency.predict(model, case, work, row['output'])
            self.assertEqual(actual.hex(), expected.hex())
        with self.assertRaisesRegex(ValueError, 'dyadic and zero'):
            request_model.phase_frequency.predict(model, dict(case, input_mode='random'), work, 0)

    def test_existing_candidate_remains_outside_all_measured_spans(self):
        root = aggregate.DEFAULT_ARCHIVE
        replay_path = request_model.EXAMPLES.with_name('B-source-candidate-replay.json')
        if not replay_path.exists():
            self.skipTest('local historical candidate evidence required')
        examples = aggregate.load_json(request_model.EXAMPLES)['examples']
        measured = aggregate.load_json(replay_path)['processes']
        rows = aggregate.load_json(root/'cases.json')
        setups = {r['case']: r['setup'] for r in aggregate.load_json(root/'static_setup.json')}
        # 复用历史 second_prefill 引擎分支做旧反例回归；共同程序尚未展开该分支。
        for stages, case in [(4, 'v09d_cfg_b_G3_stage4'), (6, 'v09d_cfg_b_G3_base')]:
            row = next(r for r in rows if r['id'] == case)
            program = cfg_b_program.build_program(row, setups[case])
            keys, counts = request_model.request_inputs(program)
            for candidate, example in examples.items():
                for flag in (False, True):
                    old = next(r for r in example['rows'] if r['stage'] == stages and r['flag'] == flag)
                    result = supply.shared_stage_events(64, stages, 132, source_keys=keys,
                        output_counts=counts, share_inflight_sources=candidate == 'shared',
                        epilogue_defer=2, second_prefill=flag, **example['parameters'])
                    ends = [cta['mma_returns'][1] for cta in result['ctas']]
                    self.assertAlmostEqual(max(ends)-min(ends), old['span'], places=6)
                    lower = min(p['M_span'] for p in measured
                                if p['stage'] == stages and p['flag'] == flag)
                    self.assertLess(max(ends)-min(ends), lower)


if __name__ == '__main__':
    unittest.main()
