"""实际输入与声明服务贯通；新增报告和零价状态核对保持既有时序。"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cfg_b_composition as aggregate
import cfg_b_system as system


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = aggregate.DEFAULT_ARCHIVE.parents[5]
        cls.profile = system.load_profile(cls.repo)

    def item(self, ktiles=4, stages=4):
        return dict(case=dict(config='cfg_b',m=768,n=256,k=64*ktiles,stages=stages,input_mode='dyadic'),
            setup=dict(grid=[4,1,1],stages=stages),cta_groups=[0,1,2,3],placement='CPU接线例子',
            addresses=dict(A=0,B=2*1024**2,D=4*1024**2),
            allocation_bytes=dict(A=768*64*ktiles*2,B=256*64*ktiles*2,D=768*256*4))

    def test_six_graphs_export_combined_reports_without_changing_time(self):
        protocol = dict(calls=2,inter_call_gap_ns=100000)
        for ktiles in (4,13,32):
            for stages in (4,6):
                with self.subTest(ktiles=ktiles,stages=stages):
                    item = self.item(ktiles,stages)
                    before = system.predict(item,self.profile,protocol,report_demands=False)
                    after = system.predict(item,self.profile,protocol,report_demands=True)
                    for key in ('plain_us','consumer_span_ns','program_end_ns','history','accounting','source_state'):
                        self.assertEqual(before[key],after[key])
                    self.assertEqual(after['accounting']['input_delivery_count'],12*ktiles*3)
                    self.assertEqual(after['modules']['sm_activity']['capacity']['capacity_ctas_per_sm'],1)
                    self.assertEqual(after['source_state']['output_bytes'],12*128*128*4)
                    self.assertEqual(len(after['details'][0]['register_quota']['warps']),12)
                    json.dumps(after,allow_nan=False)

    def test_saved_three_cases_rederive_the_original_static_frequency_inputs(self):
        plan = system.read(self.repo/'results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-full-matrix-001/plan.json')
        names = {'v09d_cfg_b_G3_base','v09d_cfg_b_G3_stage4','v09d_cfg_b_G3_A16'}
        for item in plan['rows']:
            if item['case']['id'] in names:
                program,source,parameters,clocks = system.prepare(item,self.profile)
                self.assertEqual(clocks,{key:item[key] for key in clocks})
                self.assertEqual(sum(map(len,program['work']))*program['ktiles']*3,50688)
                self.assertEqual(parameters['reuse_delay_ns'],220/item['frequency_ghz'])

    def local_profile(self):
        return dict(self.profile,local_chain_transfer_cycles=dict(compute_cycles=512.,
            commit_lead_cycles=256.,completion_lag_cycles=95.,reuse_delay_cycles=0.,
            consumer_advance_cycles=1467/11,release_visibility_cycles=275.))

    def test_local_bundle_saved_profile_sets_each_phase_without_reuse_double_charge(self):
        with patch.object(system.frequency,'predict',return_value=2.), \
             patch.object(system.phase,'predict_phases',return_value=[2.,1.,4.]):
            _,_,costs,_ = system.prepare(self.item(),self.local_profile())
        self.assertEqual(costs['reuse_delay_ns'],0.)
        self.assertEqual(costs['input_release_delay_ns'],137.5)
        for price,f in zip(costs['compute_phases_ns'],[2.,1.,4.]):
            self.assertEqual(price['compute_ns'],512/f)
            self.assertEqual(price['completion_lag_ns'],95/f)
            self.assertEqual(price['loop_advance_ns'],(1467/11)/f)
            self.assertEqual(price['input_release_delay_ns'],275/f)

    def test_local_bundle_release_completion_uses_releasing_output_frequency(self):
        with patch.object(system.frequency,'predict',return_value=2.), \
             patch.object(system.phase,'predict_phases',return_value=[2.,1.,4.]):
            report=system.predict(self.item(13,4),self.local_profile(),
                dict(calls=1,inter_call_gap_ns=100000),detailed_ctas=(0,),report_demands=False)
        detail=report['details'][0]
        self.assertEqual(len(detail['observations']),3)
        nodes=detail['node_times_ns']
        for i in range(39):
            f=[2.,1.,4.][i//13]
            self.assertAlmostEqual(nodes[f'g{i}.release']-nodes[f'g{i}.release_emit'],275/f)
            if i%13!=12:
                self.assertAlmostEqual(nodes[f'g{i}.release_emit'],
                    max(nodes[f'g{i+1}.commit'],nodes[f'g{i}.compute_complete']))

    def test_random_keeps_declared_v09_frequency_substitution(self):
        item = self.item()
        item['case']['input_mode'] = 'random'
        item['case']['seed'] = 29
        _,_,_,clocks = system.prepare(item,self.profile)
        self.assertEqual(set(clocks['phase_frequency_ghz']),{clocks['frequency_ghz']})
        self.assertEqual(clocks['frequency_mode'],'经验替代：V09调用频率用于所有阶段')

    def test_declared_a_address_is_used_for_both_initialization_and_requests(self):
        item = self.item()
        item['addresses'] = {key:value+2*1024**2 for key,value in item['addresses'].items()}
        _,source,_,_ = system.prepare(item,self.profile)
        source.begin(0,100000.)
        source.input(0.,0,0,0,1.)
        self.assertEqual(source.line_sets['A',0,0][0],item['addresses']['A']//128)

    def test_private_warmup_uses_its_declared_source_branch(self):
        original = system.ProgramGemmSource.use_program
        for warm in (None, 0):
            item = self.item()
            item['setup'].update(private_source=1, store_target_ctas=1)
            if warm is not None:
                item['setup']['warmup_private_source'] = warm
            branches = []
            def observe(source, program):
                branches.append(program.get('runtime_intervention', {}).get('private_source', 0))
                return original(source, program)
            with patch.object(system.ProgramGemmSource, 'use_program', observe):
                system.predict(item, self.profile, dict(calls=2, inter_call_gap_ns=100000),
                               detailed_ctas=(0,), report_demands=False)
            self.assertEqual(branches[-2:], [1 if warm is None else warm, 1])

    def test_explicit_completion_lag_uses_call_and_each_output_frequency(self):
        self.assertNotIn('compute_completion_lag_cycles',self.profile)
        profile = dict(self.profile,compute_completion_lag_cycles=95.)
        with patch.object(system.frequency,'predict',return_value=2.), \
             patch.object(system.phase,'predict_phases',return_value=[2.,1.,4.]):
            _,_,baseline,_ = system.prepare(self.item(),self.profile)
            _,_,parameters,_ = system.prepare(self.item(),profile)
        self.assertEqual(parameters['completion_lag_ns'],47.5)
        self.assertEqual([cost['completion_lag_ns'] for cost in parameters['compute_phases_ns']],
                         [47.5,95.,23.75])
        self.assertTrue(all('completion_lag_ns' not in cost for cost in baseline['compute_phases_ns']))
        restored = dict(parameters,
            compute_phases_ns=[{key:value for key,value in cost.items() if key!='completion_lag_ns'}
                              for cost in parameters['compute_phases_ns']])
        restored.pop('completion_lag_ns')
        self.assertEqual(restored,baseline)


if __name__ == '__main__':
    unittest.main()
