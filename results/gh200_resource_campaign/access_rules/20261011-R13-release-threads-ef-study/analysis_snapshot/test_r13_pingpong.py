"""Variable-output pingpong must retain both input and accumulator lifetimes."""
import hashlib
import json
import unittest

from supply_model import cfg_b_source_program, shared_stage_events
from analyze_r10_supply_events import source_costs


class VariablePingpongTests(unittest.TestCase):
    def parameters(self):
        return dict(service_ns=2., read_latency_ns=17., compute_ns=64.,
            producer_gap_ns=3.,operand_gap_ns=1.,reuse_delay_ns=4.,
            commit_lead_ns=40.,completion_lag_ns=8.,issue_floor_ns=5.,
            wait_overhead_ns=4.,loop_advance_ns=6.,return_overhead_ns=3.,
            handoff_delay_ns=2.,output_latency_ns=31.,output_weight=3.,
            epilogue_prepare_ns=71.,epilogue_write_ns=53.,
            epilogue_control_ns=2.,epilogue_tail_ns=3.)

    def test_explicit_two_output_option_preserves_all_legacy_fields(self):
        params=self.parameters()
        legacy=shared_stage_events(8,4,3,trace_requests=True,**params)
        new=shared_stage_events(8,4,3,trace_requests=True,output_counts=[2,2,2],**params)
        self.assertEqual(new,legacy)

    def test_third_math_waits_for_same_WG_accumulator_but_second_can_overlap(self):
        model=shared_stage_events(8,4,2,output_counts=[3,4],trace_requests=True,**self.parameters())
        self.assertEqual(model['input_delivery_count'],7*8*3)
        self.assertEqual(model['input_delivery_work'],7*8*2)
        self.assertEqual(len([r for r in model['requests'] if r['kind']=='store']),7*8)
        for actor in model['ctas']:
            self.assertLess(actor['starts'][8],actor['outputs'][0]['end'])
            for j in range(2,len(actor['outputs'])):
                self.assertGreaterEqual(actor['starts'][j*8],actor['outputs'][j-2]['end'])
                self.assertGreaterEqual(actor['starts'][j*8],actor['mma_returns'][j-1])
            for i,issued in enumerate(actor['issued_a']):
                self.assertGreaterEqual(actor['released'][i],actor['completed'][i])
                if i>=4:self.assertGreaterEqual(issued,actor['released'][i-4]+4)

    def test_one_output_and_invalid_treatment_shapes(self):
        model=shared_stage_events(8,4,2,output_counts=[1,1],**self.parameters())
        self.assertTrue(all(len(a['mma_returns'])==1 for a in model['ctas']))
        self.assertEqual(model['input_demand_count'],2*8*3)
        for counts in ([0,1],[1],[1.0,2],[True,2]):
            with self.assertRaisesRegex(ValueError,'counts'):
                shared_stage_events(8,4,2,output_counts=counts,**self.parameters())
        with self.assertRaisesRegex(ValueError,'second-output'):
            shared_stage_events(8,4,2,output_counts=[1,1],epilogue_defer=2,**self.parameters())
        mixed=shared_stage_events(8,4,2,output_counts=[2,1],epilogue_defer=1,**self.parameters())
        self.assertEqual(len(mixed['ctas'][1]['outputs']),1)

    def test_explicit_source_costs_preserve_payload_and_charge_only_source_generations(self):
        costs=[[[2.,1.,1.] for _ in range(8)] for _ in range(2)]
        params=self.parameters()
        baseline=shared_stage_events(4,4,2,trace_requests=True,**params)
        explicit=shared_stage_events(4,4,2,trace_requests=True,source_service_costs_ns=costs,**params)
        self.assertEqual(explicit,baseline)
        changed=[[[9.,4.,4.] for _ in range(8)] for _ in range(2)]
        more=shared_stage_events(4,4,2,trace_requests=True,source_service_costs_ns=changed,**params)
        self.assertEqual(more['input_delivery_work'],baseline['input_delivery_work'])
        self.assertEqual(more['input_demand_count'],baseline['input_demand_count'])
        self.assertEqual(more['source_busy_ns'],2*8*17)
        self.assertEqual(len([r for r in more['requests'] if r['kind']=='store']),32)
        keys=[[[('A',i),('B0',i),('B1',i)] for i in range(8)] for _ in range(2)]
        changed[1][0][0]=10.
        with self.assertRaisesRegex(ValueError,'inconsistent service cost'):
            shared_stage_events(4,4,2,share_inflight_sources=True,source_keys=keys,
                source_service_costs_ns=changed,**params)
        costs[0][0][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite nonnegative'):
            shared_stage_events(4,4,2,source_service_costs_ns=costs,**params)

    def test_original_source_identity_hash_is_unchanged(self):
        row=dict(config='cfg_b',m=2816,n=1536,k=4096,swizzle=1,lda=4096,ldb=1536)
        setup=dict(row,grid=[1,132,1],tile=[128,128,64],cluster=[1,1],stages=4)
        program=cfg_b_source_program(row,setup)
        digest=hashlib.sha256(json.dumps(program['source_keys'],separators=(',',':')).encode()).hexdigest()
        self.assertEqual(digest,'6c7dea8ca44f17d91c4889eb4bcfbc42a1f104a5823be8e872af0549511a4b31')
        self.assertEqual(program['output_counts'],[2]*132)

    def test_repeated_A_source_across_outputs_is_kept_and_partial_output_refused(self):
        row=dict(config='cfg_b',m=128,n=256,k=128,swizzle=1)
        setup=dict(row,grid=[1,1,1],stages=4)
        program=cfg_b_source_program(row,setup)
        self.assertEqual(program['output_counts'],[2])
        self.assertEqual(program['source_keys'][0][0][0],program['source_keys'][0][2][0])
        self.assertNotEqual(program['source_keys'][0][0][1],program['source_keys'][0][2][1])
        with self.assertRaisesRegex(ValueError,'complete logical output'):
            cfg_b_source_program(dict(row,n=248),dict(setup,n=248))

    def test_B_coverage_price_uses_static_boxes_and_ignores_target_measurements(self):
        row=dict(config='cfg_b',m=128,n=128,k=128,ldb=144,swizzle=1)
        setup=dict(row,grid=[1,1,1],stages=6)
        costs=source_costs(row,setup,2.,3.)
        self.assertEqual(costs,[[[2.,19.,19.],[2.,19.,19.]]])
        self.assertEqual(source_costs(dict(row,L_ns=-99999,frequency_ghz=.001),setup,2.,3.),costs)
        with self.assertRaisesRegex(ValueError,'does not price A'):
            source_costs(dict(row,lda=136),dict(setup,lda=136),2.,3.)


if __name__=='__main__':
    unittest.main()
