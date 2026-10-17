"""cooperative 实例绑定：物理伙伴、连续槽代次及两个消费者的原始记录。"""
import copy
import gzip
import json
from pathlib import Path
import unittest

import analyze_r18
import cooperative_program as program
import output_model
import supply_model


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011'
OLD = ROOT/'results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/main-cfg-c-rank-dependencies-v2'
CASE = 'cfg_c_r13_comp_k1024_aligned'


class CooperativeProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = next(r for r in json.loads((RUN/'cases.json').read_text()) if r['id'] == CASE)
        cls.setup = next(r['setup'] for r in json.loads((RUN/'static_setup.json').read_text()) if r['case'] == CASE)
        cls.program = program.build_program(cls.row, cls.setup)
        cls.graph = program.expand_cluster(cls.program, 0)
        record = json.loads((RUN/'samples'/CASE/'dual-00.json').read_text())
        cls.observed = analyze_r18.replay(RUN, record, cls.row, retain_roles=True)

    def test_existing_templates_preserved(self):
        for count in (1, 2):
            with gzip.open(OLD/f'cfg_c-S4-K16-T{count}-rank-dependencies.json.gz', 'rt') as stream:
                self.assertEqual(self.program['templates'][f'K16-T{count}'], json.load(stream))

    def test_actual_work_and_totals(self):
        self.assertEqual(len(self.program['clusters']), 66)
        self.assertEqual(sum(len(work) for work in self.program['work']), 216)
        total = self.program['totals']
        self.assertEqual((total['source_requests'], total['destination_atoms'], total['output_nominal_subtiles']),
                         (10368, 13824, 1728))
        self.assertEqual((total['source_payload_KiB'], total['destination_valid_KiB']), (110592, 165888))
        self.assertEqual(total['output_valid_KiB'], self.row['m'] * self.row['n'] * 4 / 1024)
        for cluster in self.program['clusters']:
            self.assertEqual(cluster['ctas'][1] - cluster['ctas'][0], 66)

    def test_physical_requests_and_last_ktile_origin(self):
        requests = self.graph['requests']
        for rank, cta in enumerate(self.graph['ctas']):
            atoms = [r for r in requests if r['group'] == 0 and r['source_rank'] == rank]
            self.assertEqual([r['bytes'] for r in atoms], [16384, 8192, 8192])
            self.assertEqual([r['destination_ctas'] for r in atoms], [self.graph['ctas'], [cta], [cta]])
            last = [r for r in requests if r['group'] == 15 and r['source_rank'] == rank]
            self.assertEqual([r['logical_source']['origin'][1] for r in last], [960, 960, 960])
            for atom in atoms:
                self.assertEqual(atom['bytes'], atom['valid_bytes'] + atom['fill_bytes'])
        self.assertEqual(len({r['id'] for r in requests}), len(requests))

    def test_persistent_slot_and_four_remote_local_arrivals(self):
        self.assertEqual((self.graph['groups'][16]['slot'], self.graph['groups'][16]['generation']), (0, 4))
        parents = self.graph['nodes']['c0.r0.g16.acquire']
        self.assertEqual({p for p in parents if 'empty_to' in p},
                         {f'c0.r{rank}.w{wg}.g12.empty_to_r0' for rank in (0, 1) for wg in (0, 1)})
        for route in self.graph['empty_arrival_routes']:
            self.assertIn(route['emit_token_template'].format(group=0), self.graph['nodes'])
            self.assertIn(route['visibility_token_template'].format(group=0), self.graph['nodes'])
        self.assertEqual(len(self.graph['producer_tail_waits']), 8)

    def test_two_warpgroups_share_every_output(self):
        endpoints = self.graph['observations']
        for cta in self.graph['ctas']:
            for j in (0, 1):
                self.assertEqual({e['consumer_warpgroup'] for e in endpoints
                                  if e['cta'] == cta and e.get('output') == j}, {0, 1})
        self.assertEqual(len(self.graph['output_subtiles']), 32)
        self.assertNotIn('c0.r0.D_visible', self.graph['nodes']['c0.source_safe_exit'])
        self.assertIn('c0.r0.D_visible', self.graph['nodes']['c0.cuda_complete'])

    def test_cluster_names_are_disjoint(self):
        other = program.expand_cluster(self.program, 1)
        self.assertFalse(set(self.graph['nodes']) & set(other['nodes']))
        self.assertEqual(len(self.graph['topological_order']), len(self.graph['nodes']))

    def test_trace_preserves_roles_and_rejects_wrong_work(self):
        bound = program.bind_trace(self.program, self.observed)
        self.assertEqual(sum(len(c['observations']) for c in bound), 1992)
        endpoint = bound[0]['observations'][0]
        self.assertEqual(endpoint['cycle'], self.observed['ctas'][0]['roles'][0]['tiles'][0][0])
        self.assertEqual(endpoint['ns'], self.observed['ctas'][0]['roles'][0]['tiles_ns'][0][0])
        changed = copy.deepcopy(self.observed)
        changed['ctas'][0]['role_work'][1].reverse()
        with self.assertRaisesRegex(ValueError, 'trace work differs'):
            program.bind_trace(self.program, changed)

    def test_same_work_but_different_K_trace_is_rejected(self):
        case = 'cfg_c_r13_comp_k4096_aligned'
        row = next(r for r in json.loads((RUN/'cases.json').read_text()) if r['id'] == case)
        record = json.loads((RUN/'samples'/case/'dual-00.json').read_text())
        observed = analyze_r18.replay(RUN, record, row, retain_roles=True)
        self.assertEqual(observed['ctas'][0]['work'], self.observed['ctas'][0]['work'])
        with self.assertRaisesRegex(ValueError, 'trace case differs'):
            program.bind_trace(self.program, observed)

    def test_explicit_support_boundary(self):
        small = dict(config='cfg_c', m=256, n=256, k=1024, swizzle=1)
        for config, stages, k in [('cfg_b', 6, 1024), ('cfg_c', 6, 1024), ('cfg_c', 4, 4096)]:
            row = dict(small, config=config, stages=stages, k=k)
            with self.subTest(config=config, stages=stages, k=k), self.assertRaisesRegex(ValueError, 'supports cfg_c S4/K1024'):
                program.build_program(row, dict(grid=[1, 2, 1], stages=stages))
        with self.assertRaisesRegex(ValueError, 'idle cluster'):
            program.build_program(small, dict(grid=[66, 2, 1], stages=4))
        with self.assertRaisesRegex(ValueError, 'complete cfg_c'):
            program.build_program(dict(small, n=128), dict(grid=[1, 2, 1], stages=4))


class CfgAProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = 'cfg_a_r13_comp_k1024_aligned'
        cls.row = next(r for r in json.loads((RUN/'cases.json').read_text()) if r['id'] == case)
        cls.setup = next(r['setup'] for r in json.loads((RUN/'static_setup.json').read_text()) if r['case'] == case)
        cls.program = program.build_program(cls.row, cls.setup)
        cls.graph = program.expand_cluster(cls.program, 0)

    def test_cfg_a_work_and_source_multicast_direction(self):
        self.assertEqual((self.program['tile'], self.program['cluster_shape'], self.program['stages']),
                         ([128, 128, 64], [2, 1], 6))
        self.assertEqual(len(self.program['clusters']), 66)
        self.assertEqual(sum(map(len, self.program['work'])), 432)
        self.assertEqual(self.program['totals']['source_requests'], 20736)
        self.assertEqual(self.program['totals']['destination_atoms'], 34560)
        for cluster in self.program['clusters']:
            self.assertEqual(cluster['ctas'][1] - cluster['ctas'][0], 1)
        for rank, cta in enumerate(self.graph['ctas']):
            atoms = [r for r in self.graph['requests'] if r['group'] == 0 and r['source_rank'] == rank]
            self.assertEqual([r['bytes'] for r in atoms], [16384, 4096, 4096])
            self.assertEqual([r['destination_ctas'] for r in atoms], [[cta], self.graph['ctas'], self.graph['ctas']])
            self.assertEqual([r['logical_source']['shape'] for r in atoms], [[128, 64], [64, 32], [64, 32]])
            self.assertEqual([r['logical_source']['origin'][1] for r in atoms], [0, 32 * rank, 32 * rank])
            incoming = [d for d in self.graph['deliveries'] if d['group'] == 0 and d['destination_rank'] == rank]
            self.assertEqual(sum(d['bytes'] for d in incoming), 32768)
        self.assertEqual(len(self.graph['output_subtiles']), 32)

    def test_stage4_carries_slots_across_outputs(self):
        run = ROOT/'results/gh200_resource_campaign/access_rules/20261015-c1-graph-protocol-expansion-001/returned'
        row = next(r for r in json.loads((run/'cases.json').read_text()) if r['id'] == 'd014_cfg_a_s4')
        setup = next(r['setup'] for r in json.loads((run/'static_setup.json').read_text()) if r['case'] == row['id'])
        bound = program.build_program(row, setup)
        graph = program.expand_cluster(bound, 0)
        self.assertEqual((bound['stages'], bound['ktiles']), (4, 10))
        self.assertEqual((graph['groups'][10]['slot'], graph['groups'][10]['generation']), (2, 2))
        self.assertEqual({p for p in graph['nodes']['c0.r0.g4.acquire'] if 'empty_to' in p},
                         {f'c0.r{rank}.w{wg}.g0.empty_to_r0' for rank in (0, 1) for wg in (0, 1)})
        self.assertEqual(len(graph['producer_tail_waits']), 8)
        original = program.build_program(dict(row, stages=6), dict(setup, stages=6))
        self.assertEqual(bound['work'], original['work'])
        self.assertEqual(bound['totals'], original['totals'])

    def test_cfg_a_preserves_existing_dependencies_and_advances_six_slots(self):
        for key, graph in self.program['templates'].items():
            old = supply_model.cfg_a_atomic_source_graph(output_model.cfg_a_rank_dependencies(6, 16, graph['output_tiles']))
            for node, parents in old['nodes'].items():
                if node.endswith('.producer_tail'):
                    continue
                self.assertTrue(set(parents) <= set(graph['nodes'][node]), node)
        state = self.graph['groups'][16]
        self.assertEqual((state['slot'], state['generation'], state['full_wait_phase']), (4, 2, 0))
        parents = self.graph['nodes']['c0.r0.g16.acquire']
        self.assertEqual({p for p in parents if 'empty_to' in p},
            {f'c0.r{rank}.w{wg}.g10.empty_to_r0' for rank in (0, 1) for wg in (0, 1)})
        waits = [w for w in self.graph['producer_tail_waits'] if w['rank'] == 0]
        self.assertEqual([w['slot'] for w in waits], [4, 5, 0, 1, 2, 3])
        self.assertEqual([w['release_group'] for w in waits], list(range(58, 64)))
        for previous, current in zip(waits, waits[1:]):
            self.assertIn(previous['event'], self.graph['nodes'][current['event']])

    def test_cfg_a_unused_tail_slots_and_n128_geometry(self):
        row = dict(config='cfg_a', m=256, n=128, k=256, stages=6, swizzle=1)
        bound = program.build_program(row, dict(grid=[2, 1, 1], stages=6))
        graph = program.expand_cluster(bound, 0)
        self.assertEqual(len(graph['output_subtiles']), 8)
        waits = [w for w in graph['producer_tail_waits'] if w['rank'] == 0]
        self.assertEqual([w['slot'] for w in waits], [4, 5, 0, 1, 2, 3])
        self.assertEqual([w['release_group'] for w in waits], [None, None, 0, 1, 2, 3])
        self.assertEqual([w['initial_empty'] for w in waits], [True, True, False, False, False, False])
        for change in (dict(k=192+16), dict(stages=3)):
            with self.assertRaisesRegex(ValueError, 'cfg_a S4/S6'):
                program.build_program(dict(row, **change), dict(grid=[2, 1, 1], stages=change.get('stages', 6)))

    def test_boundary_rank_keeps_multicast_and_full_slot_lifecycle(self):
        row = dict(config='cfg_a', m=320, n=192, k=256, stages=6, swizzle=1)
        model = program.build_program(row, dict(grid=[8, 1, 1], stages=6))
        graphs = [program.expand_cluster(model, i) for i in range(len(model['clusters']))]
        self.assertEqual(sum(x['valid_bytes'] for g in graphs for x in g['output_subtiles']), 320*192*4)
        index = next(i for i, c in enumerate(model['clusters'])
                     if c['rank_outputs'][1][0]['coord'] == [3, 1])
        graph = graphs[index]
        atoms = {x['atom']: x for x in graph['requests'] if x['source_rank'] == 1 and x['group'] == 0}
        self.assertEqual((atoms['A']['valid_bytes'], atoms['A']['fill_bytes']), (0, 16384))
        self.assertEqual((atoms['B0']['valid_bytes'], atoms['B0']['destination_ctas']), (4096, graph['ctas']))
        self.assertEqual((atoms['B1']['valid_bytes'], atoms['B1']['fill_bytes']), (0, 4096))
        self.assertTrue(all(x['expected_full_bytes'] == 32768 for x in graph['groups']))
        empty_output = [x for x in graph['output_subtiles'] if x['rank'] == 1]
        self.assertEqual(len(empty_output), 4)
        self.assertEqual(sum(x['valid_bytes'] for x in empty_output), 0)
        self.assertTrue(all(x['store_issue'] in graph['nodes'] for x in empty_output))

    def test_cfg_a_native_trace_and_second_warpgroup_work(self):
        record = json.loads((RUN/'samples'/self.row['id']/'dual-00.json').read_text())
        observed = analyze_r18.replay(RUN, record, self.row, retain_roles=True)
        bound = program.bind_trace(self.program, observed)
        self.assertEqual(sum(len(c['observations']) for c in bound), 3720)
        endpoint = next(e for e in bound[0]['observations'] if e['consumer_warpgroup'] == 1 and e['event'] == 'MAIN_END')
        self.assertEqual(endpoint['cycle'], observed['ctas'][0]['roles'][1]['tiles'][0][1])
        observed['ctas'][0]['role_work'][1].reverse()
        with self.assertRaisesRegex(ValueError, 'trace work differs'):
            program.bind_trace(self.program, observed)


if __name__ == '__main__':
    unittest.main()
