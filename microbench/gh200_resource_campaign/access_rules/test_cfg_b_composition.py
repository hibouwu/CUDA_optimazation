"""冻结数值、程序生命周期、历史工作映射以及服务替换的 CPU 验收。"""
import copy
import json
import unittest
from unittest.mock import patch

import cfg_b_composition as composition
import cfg_b_program as program_model


class ProgramTests(unittest.TestCase):
    def program(self, k=320, stages=4, m=512, n=128):
        row = dict(config='cfg_b', m=m, n=n, k=k, stages=stages)
        return program_model.build_program(row, dict(grid=[1, 1, 1], stages=stages))

    def test_ring_wrap_and_pingpong_handoff(self):
        # 每输出 5 个 Ktile，S4；跨输出后第一个槽应为 1，而非重新归零。
        graph = program_model.expand_cta(self.program(), 0)
        groups, nodes = graph['groups'], graph['nodes']
        self.assertEqual(len(groups), 20)
        self.assertEqual((groups[5]['slot'], groups[5]['generation'], groups[5]['consumer_warpgroup']), (1, 1, 1))
        self.assertEqual((groups[10]['slot'], groups[10]['generation'], groups[10]['consumer_warpgroup']), (2, 2, 0))
        position = {node: i for i, node in enumerate(graph['topological_order'])}
        for group in groups:
            g = group['group']
            if g >= 4:
                self.assertLess(position[f'g{g-4}.release'], position[f'g{g}.acquire'])
            self.assertEqual(group['full_wait_phase'], (g//4) % 2)
            self.assertEqual(group['empty_wait_phase'], 1 ^ ((g//4) % 2))
        self.assertIn('j0.math_handoff', nodes['j1.math_permit'])
        self.assertNotIn('j0.main_end', nodes['j1.math_permit'])
        self.assertIn('j0.epilogue_handoff', nodes['j2.math_permit'])
        self.assertEqual(nodes['g0.release'], ['g1.wait1'])
        self.assertEqual(nodes['g4.release'], ['j0.wait0'])
        self.assertIn('g0.compute_complete', nodes['g1.wait1'])
        self.assertIn('g4.compute_complete', nodes['j0.wait0'])
        self.assertEqual(graph['observations'][0]['nodes'],
            ['g0.consumer_wait', 'j0.main_end', 'j0.epilogue_permit', 'j0.store_tail'])

    def test_single_ktile_uses_tail_release(self):
        for stages in (4, 6):
            graph = program_model.expand_cta(self.program(k=64, stages=stages), 0)
            self.assertFalse(any(n.endswith('.wait1') for n in graph['nodes']))
            for j in range(4):
                self.assertEqual(graph['nodes'][f'g{j}.release'], [f'j{j}.wait0'])

    def test_partial_output_requests_conserve_payload_and_fill(self):
        program = self.program(k=128, m=129, n=136)
        graph = program_model.expand_cta(program, 0)
        self.assertEqual(program['totals']['output_valid_KiB'] * 1024, 129 * 136 * 4)
        self.assertEqual(len(graph['requests']), 4 * 2 * 3)
        for g in range(8):
            requests = [r for r in graph['requests'] if r['group'] == g]
            self.assertEqual([r['bytes'] for r in requests], [16384, 8192, 8192])
            self.assertEqual(sum(r['bytes'] for r in requests), 32768)
            for request in requests:
                self.assertEqual(request['valid_bytes'] + request['fill_bytes'], request['bytes'])
                self.assertIn(request['completion'], graph['nodes'][f'g{g}.full_ready'])
            self.assertEqual(requests[2]['source']['origin'][0] - requests[1]['source']['origin'][0], 64)
            self.assertEqual(requests[2]['destination']['offset_in_operand_buffer'] -
                             requests[1]['destination']['offset_in_operand_buffer'], 8192)
        self.assertTrue(any(r['fill_bytes'] for r in graph['requests']))


class FrozenCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = composition.DEFAULT_ARCHIVE
        if not cls.root.exists():
            raise unittest.SkipTest('local frozen V09 archive required')
        cls.identity = composition.archive_identity(cls.root)
        cls.calibration = composition.load_json(cls.root/'frozen/calibration.json')
        cls.frozen = composition.load_json(cls.root/'frozen/predictions.json')['predictions']
        cls.setups = {r['case']: r['setup'] for r in composition.load_json(cls.root/'static_setup.json')}
        cls.rows = [r for r in composition.load_json(cls.root/'cases.json')
                    if r['config'] == 'cfg_b' and r['set'] == 'heldout'
                    and cls.frozen[r['id']]['status'] == 'predicted']

    def build(self, row):
        return program_model.build_program(row, self.setups[row['id']], self.identity)

    def test_frozen_frequency_replacements(self):
        from pathlib import Path
        import cfg_b_predict
        base = Path('results/gh200_resource_campaign/access_rules')
        for run in ('20261015-cfg-b-time-heldout-001', '20261015-cfg-b-time-heldout-002'):
            root = base/run
            with self.subTest(run=run):
                result = cfg_b_predict.predict_cases(
                    composition.load_json(root/'cases.json'), composition.load_json(root/'expected_setup.json'),
                    composition.load_json(root/'calibration.json'), composition.load_json(root/'frequency_model.json'))
                frozen = composition.load_json(root/'predictions.json')['predictions']
                self.assertEqual(json.dumps(result, sort_keys=True), json.dumps(
                    [{key: row[key] for key in ('id', 'prediction')} for row in frozen], sort_keys=True))

    def test_all_frozen_fields_and_work_owners(self):
        self.assertEqual(len(self.rows), 12)
        # 独立组合路径不调用旧完整预测器，也不委托旧 CTA 递推。
        with patch('v09_model.predict_components', side_effect=AssertionError('old predictor called')), \
             patch('v08_model.cta_cycles', side_effect=AssertionError('old recurrence called')):
            for row in self.rows:
                with self.subTest(case=row['id']):
                    program = self.build(row)
                    predicted = composition.predict(program, self.calibration)
                    expected = {k: v for k, v in self.frozen[row['id']].items() if k != 'status'}
                    self.assertEqual(len(expected), 16)
                    self.assertEqual(json.dumps(predicted, sort_keys=True), json.dumps(expected, sort_keys=True))
                    trace = composition.compare_trace(self.root, row, program, predicted)
                    self.assertEqual(trace[0]['tiles'], program['totals']['output_tiles'])
                    self.assertEqual(trace[0]['endpoint_error_ns']['FIRST_MMA']['count'], trace[0]['tiles'])

    def test_s6_s4_graphs_and_binary_requests(self):
        for case, build in [('v09d_cfg_b_G3_base', 'cfg_b_plain'),
                            ('v09d_cfg_b_G3_stage4', 'cfg_b_s4_plain')]:
            row = next(r for r in self.rows if r['id'] == case)
            program = self.build(row)
            for cta, work in enumerate(program['work']):
                graph = program_model.expand_cta(program, cta)
                self.assertEqual(len(graph['groups']), 128)
                self.assertEqual(len(graph['requests']), 384)
                self.assertEqual(work, [tuple(o['coord']) for o in graph['observations']])
            sass = (self.root/f'build/{build}.sass').read_text()
            self.assertEqual(sass.count('UTMALDG.3D'), 3)
            log = (self.root/f'build/{build}.log').read_text()
            self.assertIn('SM90_TMA_LOAD_OP, cute::C<131072>', log)
            self.assertIn('SM90_TMA_LOAD_OP, cute::C<65536>', log)

    def test_modules_can_be_replaced_separately(self):
        row = next(r for r in self.rows if r['id'] == 'v09d_cfg_b_G3_base')
        program = self.build(row)
        original = copy.deepcopy((program, self.calibration))
        baseline = composition.predict(program, self.calibration)

        class SlowerMainloop(composition.V09Mainloop):
            def intervals(self, frequency_ghz):
                loops, supported = super().intervals(frequency_ghz)
                return [[v + 1000 for v in cta] for cta in loops], supported

        class SlowerOutput(composition.V09Events):
            def output_end(self, params, start_ns, next_mainloop):
                return super().output_end(params, start_ns, next_mainloop) + 1000

        class FixedFrequency(composition.V09Frequency):
            def solve(self, envelope_us):
                return dict(window_us=envelope_us(1.6), frequency_ghz=1.6, closure=0.0)

        class ShiftPlain(composition.V09Observation):
            def project(self, solved):
                result = super().project(solved)
                result['plain_us'] += 1
                return result

        for name, service in [('mainloop', SlowerMainloop(program, self.calibration)),
                              ('events', SlowerOutput(program, self.calibration)),
                              ('frequency', FixedFrequency(program, self.calibration))]:
            with self.subTest(service=name):
                changed = composition.predict(program, self.calibration, {name: service})
                self.assertNotEqual(changed['plain_us'], baseline['plain_us'])
                self.assertNotEqual(changed['cta_events'], baseline['cta_events'])
                self.assertEqual(changed['critical_work'], baseline['critical_work'])
        changed = composition.predict(program, self.calibration,
            dict(observation=ShiftPlain(self.calibration)))
        self.assertEqual(changed['plain_us'], baseline['plain_us'] + 1)
        self.assertEqual(changed['cta_events'], baseline['cta_events'])
        self.assertEqual(changed['frequency_ghz'], baseline['frequency_ghz'])
        self.assertEqual((program, self.calibration), original)


if __name__ == '__main__':
    unittest.main()
