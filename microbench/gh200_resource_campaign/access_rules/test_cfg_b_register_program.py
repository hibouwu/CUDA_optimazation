"""实际寄存器需求、启动依赖与固定时间服务的六图回归。"""
import importlib.util
from pathlib import Path
import unittest

import cfg_b_composition as aggregate
import cfg_b_program
import cfg_b_request_model as model
from test_cfg_b_request_model import without_full_query_graph, without_full_query_report


class RegisterProgramTests(unittest.TestCase):
    def program(self, ktiles=4, stages=4):
        return cfg_b_program.build_program(
            dict(config='cfg_b',m=768,n=256,k=64*ktiles,stages=stages),
            dict(grid=[4,1,1],stages=stages))

    def test_frozen_build_demand_and_distinct_coverage(self):
        demand = cfg_b_program.register_demands(self.program())
        self.assertEqual(demand['cta_reserved_registers'],64512)
        self.assertEqual(demand['steady_allocated_registers'],64512)
        self.assertEqual(demand['accumulator_fp32_per_consumer_thread'],128)
        self.assertEqual(demand['transfers_per_cta'],dict(producer_release=16384,consumer_acquire=16384))
        self.assertEqual([r['target_registers_per_thread'] for r in demand['roles']],[40,232,232])
        self.assertEqual(demand['coverage']['accumulator_update'],'compute_ns')
        self.assertEqual(demand['coverage']['accumulator_consumer'],'epilogue_prepare_ns')

    def test_software_edges_keep_warps_and_one_time_reconfiguration(self):
        graph = cfg_b_program.expand_cta(self.program(32,6),0,register_quota=True)
        registers = graph['service_nodes']['register_quota']
        self.assertEqual(len(registers),12)
        self.assertEqual(graph['nodes']['producer_first_work'],['register_dec.w0'])
        for node in registers:
            self.assertEqual(graph['nodes'][node],['entry'])
        for output in (0,1):
            permit=graph['nodes'][f'j{output}.math_permit']
            self.assertTrue(all(f'register_inc.w{warp}' in permit for warp in range(4+4*output,8+4*output)))
        self.assertNotIn('register_inc.w4',graph['nodes']['j2.math_permit'])
        self.assertTrue(set(registers)<=set(graph['nodes']['program_end']))

    def test_six_programs_preserve_prior_default_and_all_original_endpoints(self):
        previous=aggregate.DEFAULT_ARCHIVE.parents[2]/'Q07-smem/SMEM-01-replay/20261017-SMEM-01-replay-001/source'
        spec=importlib.util.spec_from_file_location('smem_stage_request_model',previous/'cfg_b_request_model.py')
        prior=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prior)
        program_spec=importlib.util.spec_from_file_location('smem_stage_program',previous/'cfg_b_program.py')
        prior_program=importlib.util.module_from_spec(program_spec)
        program_spec.loader.exec_module(prior_program)
        prior.cfg_b_program=prior_program
        parameters=dict(service_ns=2.,read_latency_ns=10.,compute_ns=7.,commit_lead_ns=2.,
            output_latency_ns=19.,epilogue_prepare_ns=3.,epilogue_write_ns=4.)
        for ktiles in (4,13,32):
            for stages in (4,6):
                with self.subTest(ktiles=ktiles,stages=stages):
                    program=self.program(ktiles,stages)
                    self.assertEqual(without_full_query_graph(cfg_b_program.expand_cta(program,0)),
                                     prior_program.expand_cta(program,0))
                    baseline=model.simulate(program,parameters,producer_prefix_ns=50.,detailed_ctas=range(4))
                    self.assertEqual(without_full_query_report(baseline), without_full_query_report(
                        prior.simulate(program,parameters,producer_prefix_ns=50.,detailed_ctas=range(4))))
                    expanded=model.simulate(program,parameters,producer_prefix_ns=50.,
                                            detailed_ctas=range(4),register_quota=True)
                    for key in ('ctas','accounting','consumer_span_ns','program_end_ns','critical_candidates'):
                        self.assertEqual(baseline[key],expanded[key])
                    for old,new in zip(baseline['details'],expanded['details']):
                        self.assertTrue(new['register_quota']['startup_complete'])
                        self.assertEqual(new['register_quota']['allocated_registers'],64512)
                        self.assertEqual(len(new['register_quota']['events']),12)
                        self.assertEqual(old['requests'],new['requests'])
                        self.assertEqual(old['output_service'],new['output_service'])
                        self.assertEqual(old['observations'],new['observations'])
                        self.assertEqual(old['node_times_ns'],{n:t for n,t in new['node_times_ns'].items()
                                                              if n in old['node_times_ns']})
                        self.assertEqual(len(new['node_times_ns'])-len(old['node_times_ns']),12)


if __name__=='__main__':
    unittest.main()
