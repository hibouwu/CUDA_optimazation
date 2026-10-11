"""Source constraints and diagnostic interval validity, without GPU service fitting."""
import math
import unittest

from output_model import cfg_a_rank_dependencies, dependency_bounds, dependency_order
from analyze_r18 import rank_dependency_anchors, validate_rank_setup


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
        self.assertEqual(self.nodes['r0.w0.g7.empty_to_r1'], ['r0.w0.g7.release_issue'])
        self.assertNotIn('r0.w0.g7.empty_to_r1', self.nodes['r0.w0.t0.MAIN'])

    def test_exit_keeps_source_read_and_global_visibility_separate(self):
        tail = self.nodes['r0.producer_tail']
        self.assertEqual(len([n for n in tail if 'empty_to' in n]), 6 * 4)
        self.assertIn('r0.output_source_read', self.nodes['r0.w1.FINAL'])
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
