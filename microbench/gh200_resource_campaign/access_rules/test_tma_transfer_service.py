"""cfg_b源、目的、条件等待与槽复用接口的CPU回归。"""
import json
import unittest

import cfg_b_program
import cfg_b_request_model as model
from mbarrier_service import ConditionSyncService
import supply_model
from test_r13_async_source import OneCreditSource, ReadyCreditSource, SharedDestination


class TransferServiceTests(unittest.TestCase):
    def program(self, ktiles=13, stages=4):
        return cfg_b_program.build_program(
            dict(config='cfg_b', m=768, n=256, k=64*ktiles, stages=stages),
            dict(grid=[4, 1, 1], stages=stages))

    def parameters(self):
        return dict(service_ns=0., read_latency_ns=0., compute_ns=7., commit_lead_ns=2.)

    def test_six_programs_bind_existing_async_engine_without_json_objects(self):
        for ktiles in (4, 13, 32):
            for stages in (4, 6):
                with self.subTest(ktiles=ktiles, stages=stages):
                    program = self.program(ktiles, stages)
                    source = OneCreditSource()
                    report = model.simulate(program, self.parameters(), detailed_ctas=range(4),
                        input_source_async=source, input_destination_service=SharedDestination(2.),
                        input_sync_service=ConditionSyncService(stages, 3., 5.))
                    json.dumps(report, allow_nan=False)
                    keys, counts = model.request_inputs(program)
                    reference = supply_model.shared_stage_events(ktiles, stages, 4,
                        output_counts=counts, source_keys=keys, trace_requests=True,
                        input_source_async=OneCreditSource(), input_destination_service=SharedDestination(2.),
                        input_sync_service=ConditionSyncService(stages, 3., 5.), **self.parameters())
                    self.assertEqual(report['consumer_span_ns'], reference['makespan_ns'])
                    self.assertEqual(report['accounting']['input_delivery_count'], 4*3*ktiles*3)
                    self.assertEqual(len(source.completed), 4*3*ktiles*3)
                    self.assertEqual(source.by_demand, {})
                    for detail in report['details']:
                        actor = reference['ctas'][detail['cta']]
                        times = detail['node_times_ns']
                        for group in detail['groups']:
                            g = group['group']
                            self.assertEqual(times[f'g{g}.compute_batch'], actor['starts'][g])
                            self.assertEqual(times[f'g{g}.release'], actor['released'][g])
                            ready = max(r['service']['ready_ns'] for r in detail['requests'] if r['group'] == g)
                            self.assertEqual(detail['full_wait_windows'][g]['ready_ns'], ready)

    def test_two_credit_returns_preserve_distinct_destination_and_slot_events(self):
        program = self.program(4, 4)
        profiles = []
        for source_type in (OneCreditSource, ReadyCreditSource):
            source = source_type()
            bind = source.bind_scheduler if isinstance(source, ReadyCreditSource) else None
            report = model.simulate(program, self.parameters(), detailed_ctas=(0,),
                input_source_async=source, input_source_bind_events=bind,
                input_destination_service=SharedDestination(100.),
                input_sync_service=ConditionSyncService(4, 3., 5.))
            first = report['input_sources'][0]
            released = next(t for t, source_id in source.released if source_id == first['source_id'])
            profiles.append((first, released, report))
            self.assertLess(first['ready_ns'], first['last_destination_ready_ns'])
            self.assertGreater(report['details'][0]['node_times_ns']['g0.release'],
                               first['last_destination_ready_ns'])
        self.assertEqual(profiles[0][1], profiles[0][0]['last_destination_ready_ns'])
        self.assertEqual(profiles[1][1], profiles[1][0]['ready_ns'])
        self.assertLess(profiles[1][1], profiles[0][1])

    def test_prefix_shifts_source_and_destination_records(self):
        program = self.program(4, 4)
        def run(prefix):
            source = ReadyCreditSource()
            return model.simulate(program, self.parameters(), producer_prefix_ns=prefix,
                input_source_async=source, input_source_bind_events=source.bind_scheduler,
                input_destination_service=SharedDestination(2.))
        original, shifted = run(0.), run(1608.)
        self.assertEqual(len(original['input_sources']), len(shifted['input_sources']))
        for before, after in zip(original['input_sources'], shifted['input_sources']):
            self.assertEqual(before['source_id'], after['source_id'])
            self.assertEqual(before['source_key'], after['source_key'])
            for field in ('arrival_ns', 'start_ns', 'service_end_ns', 'ready_ns',
                          'last_destination_ready_ns'):
                self.assertEqual(after[field], before[field]+1608.)

    def test_destination_only_keeps_existing_tensor_source_identity(self):
        program = self.program(4, 4)
        source = dict(candidate='shared_fifo', parameters=dict(service=2., latency=10., destination=0.))
        report = model.simulate(program, self.parameters(), tensor_source=source,
            input_destination_service=SharedDestination(2.), detailed_ctas=(0,))
        for request in report['details'][0]['requests']:
            self.assertEqual(request['service']['source_key'], model.source_key(request['source']))
        self.assertEqual(report['tensor_source'], source)
        json.dumps(report, allow_nan=False)

    def test_prefix_shifts_existing_address_unit_completion_metadata(self):
        program = self.program(4, 4)
        keys, _ = model.request_inputs(program)
        units = {key: (0,) for cta in keys for group in cta for key in group}
        parameters = dict(service_ns=2., read_latency_ns=17., compute_ns=7.,
            source_address_units=units, source_unit_service_ns=2.)
        def run(prefix):
            return model.simulate(program, parameters, producer_prefix_ns=prefix,
                share_inflight_sources=True, input_destination_service=SharedDestination(2.))
        before, after = run(0.), run(1608.)
        observed = set()
        for original, shifted in zip(before['input_sources'], after['input_sources']):
            for field in ('batch_ready_ns', 'joined_last_ready_ns'):
                if original.get(field) is not None:
                    observed.add(field)
                    self.assertEqual(shifted[field], original[field]+1608.)
            self.assertEqual(original['source_service_ns'], shifted['source_service_ns'])
        self.assertEqual(observed, {'batch_ready_ns', 'joined_last_ready_ns'})

    def test_rejects_two_simultaneous_source_replacements(self):
        with self.assertRaisesRegex(ValueError, 'both replace'):
            model.simulate(self.program(), self.parameters(), input_source_async=OneCreditSource(),
                tensor_source=dict(candidate='shared_fifo', parameters=dict(service=2., latency=10., destination=0.)))

    def test_default_report_has_no_external_service_fields(self):
        report = model.simulate(self.program(4, 4), self.parameters())
        self.assertNotIn('input_services', report)
        self.assertNotIn('input_sources', report)


if __name__ == '__main__':
    unittest.main()
