"""Cooperative source atoms, rank ownership and aggregate-token refinement."""
import copy
import unittest

from supply_model import (cfg_a_atomic_source_graph, cfg_a_source_bindings,
    input_slot_state)


def rank_fixture(stages=6, kt=4, outputs=2):
    """Small C input-token contract; integration audit uses C's full graph."""
    nodes, groups, transfers = {}, [], []
    for g in range(kt*outputs):
        state = input_slot_state(g, stages)
        groups.append(dict(state, tile=g//kt, k=g%kt,
            expected_full_bytes=32768, empty_arrivals=4))
        for r in range(2):
            prefix = f'r{r}.g{g}.'
            nodes[prefix+'arrive_expect_tx'] = []
            nodes[prefix+'A_issue'] = [prefix+'arrive_expect_tx']
            nodes[prefix+'B_issue'] = [prefix+'A_issue']
            nodes[prefix+'A_visible'] = [prefix+'A_issue']
            transfers.append(dict(group=g, operand='A', source_rank=r,
                destination_rank=r, bytes=16384, completion=prefix+'A_visible'))
            for d in range(2):
                name = prefix+f'B_to_r{d}'
                nodes[name] = [prefix+'B_issue']
                transfers.append(dict(group=g, operand='B', source_rank=r,
                    destination_rank=d, bytes=8192, completion=name))
        for r in range(2):
            prefix = f'r{r}.g{g}.'
            nodes[prefix+'full'] = [prefix+'arrive_expect_tx', prefix+'A_visible'] + [
                f'r{s}.g{g}.B_to_r{r}' for s in range(2)]
    return dict(stages=stages, k_tiles=kt, output_tiles=outputs,
        nodes=nodes, groups=groups, transfers=transfers)


class CfgASupplyTests(unittest.TestCase):
    def binding(self, m=256, n=256, k=4096, **extra):
        row = dict(config='cfg_a', m=m, n=n, k=k, swizzle=1)
        row.update(extra)
        setup = dict(grid=[2, 1, 1], cluster=[2, 1], stages=6)
        return cfg_a_source_bindings(row, setup)

    def test_B_atoms_and_remote_destination_ownership(self):
        binding = self.binding()
        cluster = binding['clusters'][0]
        self.assertEqual(cluster['ctas'], [0, 1])
        for r in range(2):
            output = cluster['rank_outputs'][r][0]
            self.assertEqual([s['payload_KiB'] for s in output['sources']], [16, 4, 4])
            self.assertEqual([s['origin'][1] for s in output['sources']], [0, r*32, r*32])
            self.assertEqual([s['destination_ranks'] for s in output['sources']], [[r], [0, 1], [0, 1]])
            self.assertEqual(len(output['destination']['B']['source_atoms']), 4)
        self.assertEqual(binding['totals']['source_payload_KiB'], 48*64*2)
        self.assertEqual(binding['totals']['destination_valid_KiB'], 64*64*2)

    def test_second_output_ring_is_not_reset(self):
        binding = self.binding()
        for outputs in binding['clusters'][0]['rank_outputs']:
            state = outputs[1]['first_input_state']
            self.assertEqual((state['group'], state['slot'], state['generation']), (64, 4, 10))

    def test_OOB_fill_and_explicit_zero_keep_same_destination_payload(self):
        oob = self.binding(m=192, n=128, input_map_m=192, storage_m=256)
        zero = self.binding(m=192, n=128, input_map_m=256, storage_m=256)
        a = oob['clusters'][0]['rank_outputs'][1][0]
        b = zero['clusters'][0]['rank_outputs'][1][0]
        self.assertEqual((a['sources'][0]['valid_KiB'], a['sources'][0]['fill_KiB']), (8, 8))
        self.assertEqual((b['sources'][0]['valid_KiB'], b['sources'][0]['fill_KiB']), (16, 0))
        self.assertEqual(a['output'], b['output'])
        self.assertEqual(a['output']['valid_KiB'], 32)
        for binding in (oob, zero):
            total = binding['totals']
            self.assertEqual(total['destination_valid_KiB']+total['destination_fill_KiB'], 64*64)

    def test_physical_cluster_is_stable_under_raster_and_swizzle(self):
        for swizzle in (1, 8):
            for raster in ('along_m', 'along_n'):
                row = dict(config='cfg_a', m=1024, n=1536, k=128,
                    swizzle=swizzle, raster_order=raster)
                setup = dict(grid=[8, 1, 1] if raster=='along_m' else [4, 2, 1], stages=6)
                binding = cfg_a_source_bindings(row, setup)
                self.assertEqual([c['ctas'] for c in binding['clusters']], [[0, 1], [2, 3], [4, 5], [6, 7]])
                self.assertEqual(binding['totals']['output_valid_KiB'], 1024*1536*4/1024)
        # The inherited scheduler's arbitrary multirow AlongM grid can pair
        # logical partners in different physical clusters. Refuse that mapping.
        row = dict(config='cfg_a', m=1024, n=1536, k=128, swizzle=1, raster_order='along_m')
        with self.assertRaisesRegex(ValueError, 'physical cluster'):
            cfg_a_source_bindings(row, dict(grid=[4, 2, 1], stages=6))

    def test_idle_cluster_has_no_fabricated_lifecycle(self):
        row = dict(config='cfg_a', m=128, n=128, k=128, swizzle=1)
        binding = cfg_a_source_bindings(row, dict(grid=[4, 1, 1], stages=6))
        self.assertEqual(len(binding['clusters']), 1)
        self.assertEqual(binding['idle_clusters'], [dict(cluster=[1, 0], ctas=[2, 3])])

    def test_scope_and_observed_timestamp_rejection_boundary(self):
        base = self.binding()
        poisoned = self.binding(L_ns=-1e30, MAIN_END=1e30, measured_frequency=.001)
        self.assertEqual(base, poisoned)
        with self.assertRaisesRegex(ValueError, 'cfg_a'):
            cfg_a_source_bindings(dict(config='cfg_b'), {})
        with self.assertRaisesRegex(ValueError, 'two-dimensional'):
            cfg_a_source_bindings(dict(config='cfg_a'), dict(grid=[2, 1, 2]))

    def test_refinement_preserves_original_edges_and_does_not_mutate_C(self):
        original = rank_fixture()
        saved = copy.deepcopy(original)
        refined = cfg_a_atomic_source_graph(original)
        self.assertEqual(original, saved)
        for node, predecessors in original['nodes'].items():
            self.assertTrue(set(predecessors).issubset(refined['nodes'][node]))
        self.assertEqual(refined['totals'], dict(source_requests=48,
            source_payload_bytes=48*1024*8, destination_atoms=80,
            destination_payload_bytes=64*1024*8))

    def test_B0_source_can_progress_before_B1_issue(self):
        graph = cfg_a_atomic_source_graph(rank_fixture())
        ancestors = set()
        pending = ['r0.g0.B0_delivered_r1']
        while pending:
            node = pending.pop()
            if node not in ancestors:
                ancestors.add(node)
                pending.extend(graph['nodes'][node])
        self.assertNotIn('r0.g0.B1_issue', ancestors)
        self.assertIn('r0.g0.B0_issue', ancestors)
        self.assertIn('r0.g0.B0_issue', graph['nodes']['r0.g0.B_issue'])
        self.assertIn('r0.g0.B1_issue', graph['nodes']['r0.g0.B_issue'])
        self.assertEqual(graph['nodes']['r0.g0.B1_issue'], ['r0.g0.A_issue'])
        self.assertIn('r0.g0.B0_delivered_r1', graph['nodes']['r0.g0.B_to_r1'])
        self.assertIn('r0.g0.B1_delivered_r1', graph['nodes']['r0.g0.B_to_r1'])

    def test_incompatible_C_contract_is_rejected(self):
        for field, value in (('stages', True), ('k_tiles', 1), ('output_tiles', 1.5)):
            graph = rank_fixture()
            graph[field] = value
            with self.assertRaises(ValueError):
                cfg_a_atomic_source_graph(graph)
        for field, value in (('slot', 99), ('tile', 99), ('expected_full_bytes', 16384), ('empty_arrivals', 1)):
            graph = rank_fixture()
            graph['groups'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'contract'):
                cfg_a_atomic_source_graph(graph)
        graph = rank_fixture()
        graph['groups'].pop()
        with self.assertRaisesRegex(ValueError, 'group count'):
            cfg_a_atomic_source_graph(graph)
        graph = rank_fixture()
        graph['transfers'][1]['bytes'] = 4096
        with self.assertRaisesRegex(ValueError, 'transfer'):
            cfg_a_atomic_source_graph(graph)
        graph = rank_fixture()
        graph['nodes']['r0.g0.full'].pop()
        with self.assertRaisesRegex(ValueError, 'full prerequisites'):
            cfg_a_atomic_source_graph(graph)


if __name__ == '__main__':
    unittest.main()
