"""同步源／输出观察回调透传和JSON声明；使用真实HBM适配器，CPU only。"""
import json
import unittest

import cfg_b_program
import cfg_b_request_model as model
from hbm_service import SharedExternalMemory
from mbarrier_service import ConditionSyncService
import supply_model
from tma_stateful_source import ProgramGemmSource


class SourceCallbackTests(unittest.TestCase):
    def test_zero_price_prefix_wait_and_release_keep_real_float_bits(self):
        # The G3 full-request regression exposed this exact time/prefix pair.
        time, prefix = 3008.1879500193613, 1608.
        self.assertEqual((time+prefix)-prefix, 3008.187950019361)
        self.assertNotEqual((time+prefix)-prefix, time)
        service = ConditionSyncService(4, 0., 0.)
        wrapped = model.PrefixSyncService(service, prefix)
        blocked = wrapped.full_wait(0., time, 0, 0)
        self.assertEqual(blocked['path'], 'blocked')
        self.assertEqual(blocked['ready_ns'], time)
        self.assertEqual(blocked['eligibility_ns'], time)
        self.assertEqual(blocked['wait_return_ns'], time)
        self.assertEqual(wrapped.release(time, 0, 0), time)
        self.assertEqual(service.slot_snapshot(0, 0)['generation'], 1)
        ready = wrapped.full_wait(time, time-8., 0, 1)
        self.assertEqual(ready['path'], 'ready')
        self.assertEqual(ready['enter_ns'], time)
        self.assertEqual(ready['wait_return_ns'], time)
        self.assertEqual(wrapped.release(time, 0, 1), time)

    def test_nonzero_price_prefix_wait_keeps_existing_conversion(self):
        time, prefix = 3008.1879500193613, 1608.
        service = ConditionSyncService(4, 2., 5.)
        wrapped = model.PrefixSyncService(service, prefix)
        blocked = wrapped.full_wait(0., time, 0, 0)
        self.assertEqual(blocked['wait_return_ns'], (time+prefix+5.)-prefix)
        ready = wrapped.full_wait(time, time-8., 0, 1)
        self.assertEqual(ready['wait_return_ns'], (time+prefix+2.)-prefix)

    def program(self, ktiles=4, stages=4):
        return cfg_b_program.build_program(
            dict(config='cfg_b', m=768, n=256, k=64*ktiles, stages=stages),
            dict(grid=[4, 1, 1], stages=stages))

    def parameters(self):
        # The supplied source must replace this deliberately long built-in
        # input latency; output service and its consumer remain unchanged.
        return dict(service_ns=2., read_latency_ns=100000., compute_ns=7.,
                    commit_lead_ns=2., output_latency_ns=19.,
                    epilogue_prepare_ns=3., epilogue_write_ns=4.)

    def source(self, program):
        memory = SharedExternalMemory(128., 64.)
        source = ProgramGemmSource(program, [0, 1, 4, 5], [1., 10., 1.],
            output_base=64*1024*1024, cache_bytes=8*1024*1024, hit_ns=5.,
            home_ns=20., link_ns=10., hbm_service_ns=128., external_memory_service=memory)
        source.begin(0, 0., groups=max(map(len, program['outputs']))*program['ktiles'])
        return source, memory

    def test_six_programs_export_json_and_preserve_existing_engine_events(self):
        for ktiles in (4, 13, 32):
            for stages in (4, 6):
                with self.subTest(ktiles=ktiles, stages=stages):
                    program, parameters = self.program(ktiles, stages), self.parameters()
                    source, memory = self.source(program)
                    report = model.simulate(program, parameters, input_source_service=source.input,
                        output_source_observer=source.output, register_quota=True,
                        ordinary_lsu=True, sm_activity=True, detailed_ctas=range(4))
                    json.dumps(report, allow_nan=False)
                    self.assertEqual(report['parameters'], parameters)
                    declaration = report['source_callbacks']
                    self.assertEqual(declaration['input_source']['class_name'], 'ProgramGemmSource')
                    self.assertEqual(declaration['input_source']['callable'], 'input')
                    self.assertEqual(declaration['output_source_observer']['callable'], 'output')
                    self.assertEqual(report['input_services']['source'], 'ProgramGemmSource')
                    reference_source, reference_memory = self.source(program)
                    keys, counts = model.request_inputs(program)
                    reference = supply_model.shared_stage_events(ktiles, stages, len(counts),
                        output_counts=counts, source_keys=keys, trace_requests=True,
                        input_source_service=reference_source.input,
                        output_source_observer=reference_source.output, **parameters)
                    deliveries = [{} for _ in counts]
                    names = {'issue_a': 'A', 'issue_b0': 'B0', 'issue_b1': 'B1'}
                    for delivery in reference['deliveries']:
                        deliveries[delivery['cta']][delivery['index'], names[delivery['kind']]] = delivery
                    for detail, actor in zip(report['details'], reference['ctas']):
                        bound = model.bind_cta(cfg_b_program.expand_cta(program, detail['cta'],
                            register_quota=True), actor, deliveries[detail['cta']])
                        for field in ('node_times_ns', 'requests', 'output_service', 'observations'):
                            self.assertEqual(detail[field], bound[field])
                        self.assertTrue(detail['register_quota']['startup_complete'])
                    self.assertEqual(report['consumer_span_ns'], reference['makespan_ns'])
                    self.assertEqual(report['internal_window_ns'], reference['makespan_ns'])
                    self.assertEqual(report['accounting'], {key: reference[key] for key in report['accounting']})
                    self.assertLess(report['details'][0]['requests'][0]['service']['ready_ns'], 100000.)
                    self.assertEqual(source.caches, reference_source.caches)
                    self.assertEqual(memory.read_bytes, reference_memory.read_bytes)
                    self.assertEqual(memory.read_bytes, source.hbm_lines*128)
                    self.assertGreater(memory.read_bytes, 0)
                    self.assertEqual(memory.write_bytes, 0)
                    self.assertEqual(source.output_bytes, program['totals']['output_tiles']*65536)
                    self.assertEqual(source.output_bytes, reference_source.output_bytes)

    def test_legacy_parameter_callbacks_match_independent_kwargs_without_json_objects(self):
        program, parameters = self.program(), self.parameters()
        source, _ = self.source(program)
        independent = model.simulate(program, parameters, input_source_service=source.input,
                                    output_source_observer=source.output, detailed_ctas=(0,))
        source, _ = self.source(program)
        legacy = model.simulate(program, dict(parameters, input_source_service=source.input,
            output_source_observer=source.output), detailed_ctas=(0,))
        self.assertEqual(legacy, independent)
        self.assertEqual(legacy['parameters'], parameters)
        json.dumps(legacy, allow_nan=False)

    def test_output_only_observer_retains_default_events_without_source_lifecycle_access(self):
        program = self.program()
        parameters = dict(service_ns=2., read_latency_ns=10., compute_ns=7., commit_lead_ns=2.)
        baseline = model.simulate(program, parameters, detailed_ctas=range(4))
        observed = []
        report = model.simulate(program, parameters, detailed_ctas=range(4),
            output_source_observer=lambda time, cta, subtile: observed.append((time, cta, subtile)))
        self.assertEqual(len(observed), program['totals']['output_tiles']*8)
        self.assertNotIn('input_sources', report)
        self.assertNotIn('input_services', report)
        declaration = report.pop('source_callbacks')
        self.assertEqual(report.pop('internal_window_ns'), baseline['consumer_span_ns'])
        self.assertIsNone(declaration['input_source'])
        self.assertEqual(report, baseline)

    def test_source_lifecycle_uses_the_existing_common_prefix_translation(self):
        program, parameters = self.program(), self.parameters()

        def run(prefix):
            source, _ = self.source(program)
            return model.simulate(program, parameters, input_source_service=source.input,
                output_source_observer=source.output, producer_prefix_ns=prefix)

        original, shifted = run(0.), run(50.)
        self.assertEqual(shifted['internal_window_ns'], original['internal_window_ns'])
        self.assertEqual(shifted['consumer_span_ns'], original['consumer_span_ns']+50.)
        self.assertEqual(len(original['input_sources']), len(shifted['input_sources']))
        for before, after in zip(original['input_sources'], shifted['input_sources']):
            for field in ('arrival_ns', 'start_ns', 'service_end_ns', 'ready_ns', 'last_destination_ready_ns'):
                if before.get(field) is not None:
                    self.assertEqual(after[field], before[field]+50.)


if __name__ == '__main__':
    unittest.main()
