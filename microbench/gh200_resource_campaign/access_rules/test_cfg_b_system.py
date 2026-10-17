"""实际输入与声明服务贯通；新增报告和零价状态核对保持既有时序。"""
import json
from pathlib import Path
import unittest

import cfg_b_composition as aggregate
import cfg_b_system as system


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = aggregate.DEFAULT_ARCHIVE.parents[3]
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
        plan = system.read(self.repo/'results/gh200_resource_campaign/access_rules/20261015-cfg-b-full-matrix-001/plan.json')
        names = {'v09d_cfg_b_G3_base','v09d_cfg_b_G3_stage4','v09d_cfg_b_G3_A16'}
        for item in plan['rows']:
            if item['case']['id'] in names:
                program,source,parameters,clocks = system.prepare(item,self.profile)
                self.assertEqual(clocks,{key:item[key] for key in clocks})
                self.assertEqual(sum(map(len,program['work']))*program['ktiles']*3,50688)
                self.assertEqual(parameters['reuse_delay_ns'],220/item['frequency_ghz'])

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


if __name__ == '__main__':
    unittest.main()
