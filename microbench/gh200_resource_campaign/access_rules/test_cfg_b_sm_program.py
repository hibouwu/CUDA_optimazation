"""容量与活动报告的接线保留此前六图时间线与原价格。"""
import importlib.util
import unittest

import cfg_b_composition as aggregate
import cfg_b_program
import cfg_b_request_model as model
from test_cfg_b_request_model import without_full_query_report


class SmProgramTests(unittest.TestCase):
    def test_detail_iterator_is_consumed_once(self):
        program = cfg_b_program.build_program(
            dict(config='cfg_b',m=768,n=256,k=256,stages=4),dict(grid=[4,1,1],stages=4))
        parameters = dict(service_ns=2.,read_latency_ns=10.,compute_ns=7.,commit_lead_ns=2.,
            output_latency_ns=19.,epilogue_prepare_ns=3.,epilogue_write_ns=4.)
        expected = model.simulate(program,parameters,sm_activity=True,detailed_ctas=range(4))
        actual = model.simulate(program,parameters,sm_activity=True,detailed_ctas=iter(range(4)))
        self.assertEqual(actual,expected)

    def test_six_programs_preserve_previous_complete_results(self):
        source = aggregate.DEFAULT_ARCHIVE.parents[2]/'Q13-l1-lsu/L1-01-minimal/20261017-L1-01-replay-001/source'
        spec = importlib.util.spec_from_file_location('l1_stage_request_model',source/'cfg_b_request_model.py')
        prior = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prior)
        graph_spec = importlib.util.spec_from_file_location('l1_stage_program',source/'cfg_b_program.py')
        prior_program = importlib.util.module_from_spec(graph_spec)
        graph_spec.loader.exec_module(prior_program)
        prior.cfg_b_program = prior_program
        parameters = dict(service_ns=2.,read_latency_ns=10.,compute_ns=7.,commit_lead_ns=2.,
            output_latency_ns=19.,epilogue_prepare_ns=3.,epilogue_write_ns=4.)
        for ktiles in (4,13,32):
            for stages in (4,6):
                with self.subTest(ktiles=ktiles,stages=stages):
                    program = cfg_b_program.build_program(
                        dict(config='cfg_b',m=768,n=256,k=64*ktiles,stages=stages),
                        dict(grid=[4,1,1],stages=stages))
                    kwargs = dict(producer_prefix_ns=50.,detailed_ctas=range(4))
                    baseline = model.simulate(program,parameters,**kwargs)
                    self.assertEqual(without_full_query_report(baseline),
                                     without_full_query_report(prior.simulate(program,parameters,**kwargs)))
                    reported = model.simulate(program,parameters,sm_activity=True,**kwargs)
                    capacity = reported['modules'].pop('sm_activity')['capacity']
                    self.assertEqual(capacity['capacity_ctas_per_sm'],1)
                    self.assertEqual(set(capacity['limiting_resources']),{'registers','smem'})
                    for actor in reported['ctas']:
                        report = actor.pop('sm_activity')
                        self.assertGreater(report['composite_compute_service_ns'],0.)
                        self.assertGreaterEqual(report['compute_write_overlap_ns'],0.)
                        self.assertGreaterEqual(report['input_pending_peak'],3)
                    for actor in reported['details']:
                        windows = actor.pop('sm_activity')['windows_ns']
                        self.assertEqual(len(windows['composite_compute_service']),len(actor['groups']))
                        self.assertEqual(len(windows['input_pending']),len(actor['requests']))
                    self.assertEqual(baseline,reported)


if __name__ == '__main__':
    unittest.main()
