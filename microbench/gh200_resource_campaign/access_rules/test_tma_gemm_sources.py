import unittest

import tma_gemm_sources as model
from tma_gpc_service import GroupedTensorSource


class SourceCoordinatesTest(unittest.TestCase):
    def source(self, candidate):
        return model.GemmSource([(2, 3), (2, 4), (5, 3)], [0, 1, 0], candidate, [20., 100., 30.])

    def test_operand_coordinates_and_scope(self):
        source = self.source('global_ready')
        first = source.submit(0., 0, (0, 7, 'A'), 1.)
        joined = source.submit(1., 1, (1, 7, 'A'), 1.)
        self.assertTrue(joined['merged'])
        self.assertEqual(first['source_key'], joined['source_key'])
        self.assertEqual(first['source_ready'], joined['source_ready'])
        b0 = source.submit(2., 0, (0, 7, 'B0'), .5)
        self.assertFalse(source.submit(3., 1, (1, 7, 'B0'), .5)['merged'])
        self.assertEqual(b0['source_id'], source.submit(4., 2, (2, 7, 'B0'), .5)['source_id'])
        self.assertFalse(source.submit(5., 2, (2, 7, 'B1'), .5)['merged'])
        self.assertFalse(source.submit(6., 0, (0, 8, 'A'), 1.)['merged'])
        local = self.source('gpc_ready')
        local.submit(0., 0, (0, 7, 'A'), 1.)
        self.assertFalse(local.submit(1., 1, (1, 7, 'A'), 1.)['merged'])

    def test_subscription_expiration(self):
        for candidate, boundary in [('gpc_ready', 120.), ('gpc_service_end', 20.)]:
            source = self.source(candidate)
            source.submit(0., 0, (0, 7, 'B0'), .5)
            # B0 costs half of A, so each boundary is 10 ns earlier.
            self.assertTrue(source.submit(boundary-11, 2, (2, 7, 'B0'), .5)['merged'])
            self.assertFalse(source.submit(boundary-10, 2, (2, 7, 'B0'), .5)['merged'])

    def test_record_observer_preserves_program_timing(self):
        import tma_tensor_multicta as tensor
        program = tensor.program(2, 2, 'independent', groups=4, source_ring=4)
        prices = dict(service=20., latency=100., issue=10., destination=30., consumer=20., trace=True)
        baseline = tensor.simulate(program, 'shared_fifo', **prices)
        events = []
        observed = tensor.simulate(program, 'shared_fifo', **prices,
            memory_observer=lambda *event: events.append(event))
        self.assertEqual(observed, baseline)
        self.assertEqual([e[0] for e in events], sorted(e[0] for e in events))
        for section, expected in [('issue_record', 8), ('release_record', 8), ('cta_record', 2)]:
            self.assertEqual(sum(e[3] == section for e in events), expected)

    def test_unmerged_preserves_grouped_service(self):
        source = self.source('unmerged')
        reference = GroupedTensorSource([0, 1, 0], 2, 20., 100.)
        for t, cta, atom, weight in [(0., 0, 'A', 1.), (0., 1, 'A', 1.), (1., 2, 'B1', .5), (2., 0, 'A', 1.)]:
            served = source.submit(t, cta, (cta, 0, atom), weight)
            self.assertIn(served.pop('source_key')[0], ('A', 'B'))
            self.assertEqual(served, reference.submit(t, cta, None, weight))


if __name__ == '__main__':
    unittest.main()
