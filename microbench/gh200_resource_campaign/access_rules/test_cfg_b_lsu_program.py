"""冻结cfg_b的普通LSU需求与六个组合图的零请求回归。"""
import importlib.util
import unittest

import cfg_b_composition as aggregate
import cfg_b_program
import cfg_b_request_model as model
from test_cfg_b_request_model import without_full_query_graph, without_full_query_report


class LsuProgramTests(unittest.TestCase):
    def program(self, ktiles=4, stages=4):
        return cfg_b_program.build_program(
            dict(config='cfg_b', m=768, n=256, k=64*ktiles, stages=stages),
            dict(grid=[4,1,1], stages=stages))

    def test_active_scalar_and_local_demand_is_zero(self):
        demand = cfg_b_program.ordinary_lsu_demands(self.program())
        self.assertEqual(demand['scope'], dict(alpha=1., beta=0., alpha_ptr=None, alpha_batch_stride=0))
        self.assertEqual(demand['ordinary_global']['static_scalar_ldg_sites'], 2)
        self.assertEqual(demand['ordinary_global']['active_loads'], 0)
        self.assertEqual(demand['ordinary_global']['active_stores'], 0)
        self.assertEqual(sum(demand['local'].values()), 0)
        self.assertEqual(demand['matrix_payload']['A_B'], 'TMA input source/destination')

    def test_constants_remain_static_program_demands(self):
        for stages in (4,6):
            demand = cfg_b_program.ordinary_lsu_demands(self.program(stages=stages))
            self.assertEqual(sum(demand['constants']['static_opcodes'].values()), 79)
            self.assertEqual(sum(v for k,v in demand['constants']['static_opcodes'].items() if '.64' in k), 22)
            self.assertIn('control', demand['constants']['coverage'])

    def test_six_programs_preserve_prior_results_and_only_add_demand_report(self):
        source = aggregate.DEFAULT_ARCHIVE.parents[2]/'Q08-register-residency/REG-01-SM-01-replay/20261017-REG-01-replay-001/source'
        spec = importlib.util.spec_from_file_location('register_stage_request_model', source/'cfg_b_request_model.py')
        prior = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prior)
        graph_spec = importlib.util.spec_from_file_location('register_stage_program', source/'cfg_b_program.py')
        prior_program = importlib.util.module_from_spec(graph_spec)
        graph_spec.loader.exec_module(prior_program)
        prior.cfg_b_program = prior_program
        parameters = dict(service_ns=2., read_latency_ns=10., compute_ns=7., commit_lead_ns=2.,
            output_latency_ns=19., epilogue_prepare_ns=3., epilogue_write_ns=4.)
        for ktiles in (4,13,32):
            for stages in (4,6):
                with self.subTest(ktiles=ktiles, stages=stages):
                    program = self.program(ktiles,stages)
                    self.assertEqual(without_full_query_graph(cfg_b_program.expand_cta(program,0)),
                                     prior_program.expand_cta(program,0))
                    baseline = model.simulate(program,parameters,producer_prefix_ns=50.,detailed_ctas=range(4))
                    self.assertEqual(without_full_query_report(baseline), without_full_query_report(
                        prior.simulate(program,parameters,producer_prefix_ns=50.,detailed_ctas=range(4))))
                    reported = model.simulate(program,parameters,producer_prefix_ns=50.,
                                              detailed_ctas=range(4),ordinary_lsu=True)
                    demand = reported['modules'].pop('ordinary_lsu')['demands']
                    self.assertEqual(demand, cfg_b_program.ordinary_lsu_demands(program))
                    self.assertEqual(baseline,reported)


if __name__ == '__main__':
    unittest.main()
