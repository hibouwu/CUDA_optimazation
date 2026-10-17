"""共享外存计账及真实cfg_b程序接线；价格均为显式CPU例子。"""
import unittest

import cfg_b_program
import cfg_b_request_model
from hbm_service import SharedExternalMemory
from supply_model import shared_stage_events
from tma_stateful_source import ProgramGemmSource, StatefulTensorSource


def source(memory=None, price=128.):
    return StatefulTensorSource([[(0, 0)], [(0, 1)]], [0, 1], [1., 10., 1.],
        cache_bytes=4096, hit_ns=5., home_ns=20., link_ns=10., hbm_service_ns=price,
        external_memory_service=memory)


def program_example(ktiles, stages, external):
    program = cfg_b_program.build_program(
        dict(config='cfg_b', m=768, n=256, k=64*ktiles, stages=stages),
        dict(grid=[4, 1, 1], stages=stages))
    memory = SharedExternalMemory(128., 64.) if external else None
    inputs = ProgramGemmSource(program, [0, 1, 4, 5], [1., 10., 1.],
        output_base=64*1024*1024, cache_bytes=8*1024*1024, hit_ns=5., home_ns=20.,
        link_ns=10., hbm_service_ns=128., external_memory_service=memory)
    inputs.begin(0, 0., groups=3*ktiles)
    keys, counts = cfg_b_request_model.request_inputs(program)
    events = shared_stage_events(ktiles, stages, 4, output_counts=counts, source_keys=keys,
        input_source_service=inputs.input, trace_requests=True,
        service_ns=0., read_latency_ns=0., compute_ns=7., commit_lead_ns=2.)
    names = {'issue_a': 'A', 'issue_b0': 'B0', 'issue_b1': 'B1'}
    deliveries = [dict() for _ in counts]
    for row in events['deliveries']:
        deliveries[row['cta']][row['index'], names[row['kind']]] = row
    bound = [cfg_b_request_model.bind_cta(cfg_b_program.expand_cta(program, cta),
        actor, deliveries[cta]) for cta, actor in enumerate(events['ctas'])]
    return program, inputs, memory, events, bound


class HbmServiceTests(unittest.TestCase):
    def test_directions_compete_in_one_gpu_frontier(self):
        memory = SharedExternalMemory(128., 64.)
        read = memory.submit(0., read_bytes=256)
        write = memory.submit(0., write_bytes=256)
        mixed = memory.submit(1., read_bytes=128, write_bytes=64)
        idle = memory.submit(20., read_bytes=128)
        self.assertEqual([(r['service_start_ns'], r['service_end_ns'])
            for r in (read, write, mixed, idle)], [(0., 2.), (2., 6.), (6., 8.), (20., 21.)])
        self.assertEqual((memory.read_bytes, memory.write_bytes, memory.count, memory.busy_ns),
            (512, 320, 4, 9.))

    def test_out_of_order_events_need_event_solver_order(self):
        memory = SharedExternalMemory(128., 64.)
        memory.submit(10., read_bytes=128)
        with self.assertRaisesRegex(ValueError, 'event-time order'):
            memory.submit(0., read_bytes=128)
        self.assertEqual((memory.count, memory.frontier_ns, memory.read_bytes), (1, 11., 128))

    def test_explicit_writeback_delays_read_miss_without_another_read_charge(self):
        memory = SharedExternalMemory(128., 64.)
        memory.submit(0., write_bytes=256)
        inputs = source(memory, price=12800.)
        self.assertEqual(inputs.read(0., 0, 0), 25.)
        self.assertEqual(inputs.read(1., 0, 0), 25.)
        self.assertEqual(inputs.read(2., 1, 0), 35.)
        self.assertEqual(inputs.read(40., 0, 0), 45.)
        self.assertEqual((memory.count, memory.read_bytes, memory.write_bytes), (2, 128, 256))
        self.assertEqual((inputs.hbm_lines, inputs.pending_lines, inputs.local_lines), (1, 1, 1))

    def test_cache_stores_do_not_invent_external_writebacks(self):
        memory = SharedExternalMemory(128., 64.)
        inputs = source(memory)
        inputs.store(0., 0, 0, 4096)
        self.assertEqual((memory.count, memory.write_bytes, memory.frontier_ns), (0, 0, 0.))
        self.assertEqual(inputs.read(1., 0, 0), 6.)
        self.assertEqual(memory.count, 0)

    def test_old_128_line_price_is_replaced_exactly(self):
        memory = SharedExternalMemory(128., 64.)
        legacy, replacement = source(), source(memory)
        for t, domain, line in ((0., 0, 0), (0., 0, 1), (1., 0, 0),
                                (2., 1, 1), (40., 0, 0), (50., 1, 2)):
            self.assertEqual(legacy.read(t, domain, line), replacement.read(t, domain, line))
        for field in ('hbm_frontier', 'hbm_lines', 'pending_lines', 'local_lines',
                      'remote_lines', 'hbm_pending', 'caches'):
            self.assertEqual(getattr(legacy, field), getattr(replacement, field))
        self.assertEqual(memory.read_bytes, replacement.hbm_lines*128)

    def test_service_and_cache_survive_next_output(self):
        memory = SharedExternalMemory(128., 64.)
        inputs = source(memory)
        inputs.begin(0, 0., probe_metadata=False)
        self.assertEqual(inputs.read(0., 0, 0), 21.)
        inputs.begin(1, 100., probe_metadata=False)
        self.assertEqual(inputs.read(100., 0, 0), 105.)
        self.assertEqual(inputs.read(100., 0, 1), 121.)
        self.assertEqual((memory.count, memory.read_bytes), (2, 256))

    def test_six_cfg_b_programs_keep_events_accounting_and_every_dependency(self):
        for ktiles in (4, 13, 32):
            for stages in (4, 6):
                with self.subTest(ktiles=ktiles, stages=stages):
                    _, legacy, _, before, old_bound = program_example(ktiles, stages, False)
                    _, inputs, memory, after, bound = program_example(ktiles, stages, True)
                    self.assertEqual(before, after)
                    self.assertEqual(old_bound, bound)
                    self.assertEqual(inputs.caches, legacy.caches)
                    self.assertEqual(memory.read_bytes, inputs.hbm_lines*128)
                    self.assertEqual(memory.write_bytes, 0)
                    self.assertEqual(after['input_delivery_count'], 4*3*ktiles*3)
                    self.assertGreater(memory.count, 0)
                    self.assertGreater(sum(c['dependency_edges_checked'] for c in bound), 0)


if __name__ == '__main__':
    unittest.main()
