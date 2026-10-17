"""同步计数、条件返回和 cfg_b 槽反压的 CPU 回归。"""
import copy
import unittest

import cfg_b_program
import cfg_b_request_model as request_model
from mbarrier_service import BarrierPhase, ConditionSyncService


class BarrierTests(unittest.TestCase):
    def test_all_consumer_arrivals_are_required(self):
        barrier = BarrierPhase(128)
        self.assertFalse(barrier.arrive(127))
        self.assertEqual(barrier.snapshot()['pending_arrivals'], 1)
        self.assertFalse(barrier.wait_complete(0))
        self.assertTrue(barrier.arrive())
        self.assertTrue(barrier.wait_complete(0))
        self.assertEqual(barrier.snapshot()['pending_arrivals'], 128)

    def test_full_wait_requires_all_three_transaction_atoms(self):
        barrier = BarrierPhase(1)
        barrier.arrive_and_expect_tx(32768)
        for size in (16384, 8192):
            self.assertFalse(barrier.complete_tx(size))
            self.assertFalse(barrier.wait_complete(0))
        self.assertTrue(barrier.complete_tx(8192))
        self.assertTrue(barrier.wait_complete(0))

    def test_completion_can_precede_transaction_registration(self):
        barrier = BarrierPhase(1)
        barrier.complete_tx(32768)
        barrier.arrive()
        self.assertEqual(barrier.snapshot()['pending_tx_bytes'], -32768)
        self.assertFalse(barrier.wait_complete(0))
        self.assertTrue(barrier.expect_tx(32768))

    def test_three_generations_retain_absolute_identity(self):
        barrier = BarrierPhase(1)
        for generation in range(3):
            self.assertEqual(barrier.phase, generation % 2)
            barrier.arrive()
        self.assertEqual(barrier.phase, 1)
        self.assertEqual(barrier.snapshot()['completed_phases'], 3)

    def test_release_matches_waited_generation(self):
        service = ConditionSyncService(4, 3., 5.)
        service.full_wait(0., 10., 0, 0)
        with self.assertRaises(ValueError):
            service.release(14., 0, 0)
        service.release(15., 0, 0)
        with self.assertRaises(ValueError):
            service.release(15., 0, 0)
        next_window = service.full_wait(16., 20., 0, 4)
        self.assertEqual((next_window['slot'], next_window['generation']), (0, 1))
        self.assertEqual((next_window['full_wait_phase'], next_window['empty_wait_phase']), (1, 0))


class CompositionTests(unittest.TestCase):
    def program(self, ktiles, stages):
        return cfg_b_program.build_program(
            dict(config='cfg_b', m=768, n=256, k=64*ktiles, stages=stages),
            dict(grid=[4, 1, 1], stages=stages))

    def parameters(self):
        return dict(service_ns=2., read_latency_ns=3., compute_ns=7., commit_lead_ns=2.)

    def test_zero_price_preserves_six_programs_and_slot_lifetimes(self):
        for ktiles in (4, 13, 32):
            for stages in (4, 6):
                with self.subTest(ktiles=ktiles, stages=stages):
                    program = self.program(ktiles, stages)
                    baseline = request_model.simulate(program, self.parameters(), detailed_ctas=range(4))
                    service = ConditionSyncService(stages, 0., 0.)
                    report = request_model.simulate(program, self.parameters(),
                        detailed_ctas=range(4), input_sync_service=service)
                    stripped = copy.deepcopy(report)
                    del stripped['modules']['sync']
                    for detail in stripped['details']:
                        del detail['full_wait_windows']
                    self.assertEqual(stripped, baseline)
                    for detail in report['details']:
                        windows = detail['full_wait_windows']
                        total = len(windows)
                        self.assertEqual(total, 3*ktiles)
                        for g, window in enumerate(windows):
                            self.assertEqual(window['generation'], g//stages)
                            self.assertEqual(window['wait_return_ns'],
                                detail['node_times_ns'][f'g{g}.consumer_wait'])
                        for slot in range(min(stages, total)):
                            state = service.slot_snapshot(detail['cta'], slot)
                            generations = len(range(slot, total, stages))
                            self.assertEqual(state['generation'], generations)
                            self.assertEqual(state['full']['completed_phases'], generations)
                            self.assertEqual(state['empty']['completed_phases'], generations)
                            self.assertIsNone(state['group'])

    def test_nonzero_return_delays_compute_release_and_next_issue(self):
        program = self.program(13, 4)
        parameters = dict(service_ns=0., read_latency_ns=0., compute_ns=7., commit_lead_ns=2.)
        baseline = request_model.simulate(program, parameters, detailed_ctas=range(4))
        service = ConditionSyncService(4, 3., 5.)
        report = request_model.simulate(program, parameters, detailed_ctas=range(4),
            input_sync_service=service)
        before = baseline['details'][0]['node_times_ns']
        after = report['details'][0]['node_times_ns']
        self.assertEqual((before['g0.compute_batch'], after['g0.compute_batch']), (0., 3.))
        self.assertGreater(after['g0.release'], before['g0.release'])
        self.assertGreater(after['g4.A.issue'], before['g4.A.issue'])
        self.assertGreater(report['consumer_span_ns'], baseline['consumer_span_ns'])

    def test_prefix_keeps_first_consumer_entry_at_common_origin(self):
        program = self.program(13, 4)
        parameters = dict(service_ns=0., read_latency_ns=0., compute_ns=7., commit_lead_ns=2.)
        service = ConditionSyncService(4, 3., 5.)
        report = request_model.simulate(program, parameters, producer_prefix_ns=1608.,
            detailed_ctas=range(4), input_sync_service=service)
        for detail in report['details']:
            first = detail['full_wait_windows'][0]
            self.assertEqual(first['enter_ns'], 0.)
            self.assertEqual(first['ready_ns'], 1608.)
            self.assertEqual(first['path'], 'blocked')
            self.assertEqual(first['wait_return_ns'], 1613.)
            self.assertEqual(detail['node_times_ns']['g0.compute_batch'], 1613.)
            self.assertTrue(all(w['enter_ns'] >= 1608.
                                for w in detail['full_wait_windows'][1:]))
        zero = request_model.simulate(program, parameters, producer_prefix_ns=1608.,
            input_sync_service=ConditionSyncService(4, 0., 0.))
        original = request_model.simulate(program, parameters, producer_prefix_ns=1608.)
        self.assertEqual(zero['consumer_span_ns'], original['consumer_span_ns'])

    def test_return_and_mma_remain_distinct_during_local_exclusion(self):
        program = cfg_b_program.build_program(
            dict(config='cfg_b', m=128, n=640, k=128, stages=4),
            dict(grid=[1, 1, 1], stages=4))
        parameters = dict(service_ns=0., read_latency_ns=0., compute_ns=64.,
            commit_lead_ns=32., epilogue_prepare_ns=20., epilogue_write_ns=20.,
            local_epilogue_exclusion='consumer')
        report = request_model.simulate(program, parameters, detailed_ctas=(0,),
            input_sync_service=ConditionSyncService(4, 3., 17.))
        detail = report['details'][0]
        for window in detail['full_wait_windows']:
            self.assertEqual(detail['node_times_ns'][f'g{window["group"]}.consumer_wait'],
                             window['wait_return_ns'])
        self.assertTrue(any(detail['node_times_ns'][f'g{w["group"]}.compute_batch'] > w['wait_return_ns']
                            for w in detail['full_wait_windows']))


if __name__ == '__main__':
    unittest.main()
