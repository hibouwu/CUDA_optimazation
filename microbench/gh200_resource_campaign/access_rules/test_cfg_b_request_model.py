"""请求服务与共同程序的绑定、跨 CTA 共享和旧候选反例的 CPU 回归。"""
import copy
import importlib.util
import unittest
from unittest.mock import patch

import cfg_b_composition as aggregate
import cfg_b_program
import cfg_b_request_model as request_model
import supply_model as supply
import tma_tensor_multicta as tensor


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


class HistoricalCandidateTests(unittest.TestCase):
    def test_phase_frequency_inference_preserves_development_snapshot(self):
        root = aggregate.DEFAULT_ARCHIVE.parent/'20261015-phase-frequency-development-001'
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
        # 复用历史受干预引擎做旧反例回归；共同程序接线只接未干预分支。
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
