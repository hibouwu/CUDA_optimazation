"""Shared source reuse preserves every independent request-path service."""
import unittest

from supply_model import shared_stage_events


def model(**extra):
    source_keys=[[[(atom,i) for atom in ('A','B0','B1')] for i in range(4)] for _ in range(3)]
    args=dict(service_ns=10.,read_latency_ns=20.,compute_ns=64.,operand_gap_ns=1000.,
        share_inflight_sources=True,source_keys=source_keys,trace_requests=True,
        diagnostic_producer_offsets_ns=[0.,5.,35.])
    args.update(extra)
    return shared_stage_events(2,2,3,**args)


class InputPathTests(unittest.TestCase):
    def test_zero_preserves_complete_existing_results(self):
        for floor in (0.,20.):
            self.assertEqual(model(input_min_latency_ns=floor),
                model(input_min_latency_ns=floor,input_path_service_ns=0.))

    def test_every_demand_pays_independently_of_source_merging(self):
        result=model(input_path_service_ns=100.)
        self.assertGreater(result['input_merged_count'],0)
        path=result['input_path_service']
        self.assertEqual(path['demands'],result['input_demand_count'])
        self.assertEqual(path['work'],result['input_demand_work'])
        self.assertEqual(path['busy_ns'],100.*result['input_demand_work'])
        self.assertEqual(result['input_delivery_work'],result['input_demand_work'])
        for demand in result['input_demands']:
            self.assertEqual(demand['input_path_end_ns']-demand['input_path_start_ns'],100.*demand['work'])

    def test_source_and_request_path_are_parallel(self):
        args=dict(service_ns=0.,read_latency_ns=100.,compute_ns=64.,input_min_latency_ns=7.,trace_requests=True)
        source_limited=shared_stage_events(2,2,1,input_path_service_ns=10.,**args)
        path_limited=shared_stage_events(2,2,1,input_path_service_ns=100.,**args)
        self.assertEqual(source_limited['ctas'][0]['starts'][0],100.)
        self.assertEqual(path_limited['ctas'][0]['starts'][0],207.)

    def test_source_expires_while_request_path_is_still_busy(self):
        result=model(input_path_service_ns=100.)
        first=sorted((d for d in result['input_demands'] if d['kind']=='issue_a' and d['index']==0),
            key=lambda d:d['cta'])
        self.assertEqual([d['merged'] for d in first],[False,True,False])
        self.assertEqual([d['source_ready_ns'] for d in first],[30.,30.,65.])
        self.assertEqual([d['input_path_end_ns'] for d in first],[100.,200.,300.])

    def test_independent_fifo_does_not_charge_output_atoms(self):
        result=shared_stage_events(4,2,2,service_ns=0.,read_latency_ns=0.,compute_ns=64.,
            input_path_service_ns=4.,output_counts=[1,3],trace_requests=True)
        self.assertEqual(result['input_path_service']['demands'],48)
        self.assertEqual(result['input_path_service']['work'],32.)
        self.assertEqual(result['input_path_service']['busy_ns'],128.)
        self.assertEqual(sum(len(a['outputs'])*8 for a in result['ctas']),32)
        for actor in result['ctas']:
            for group in range(2,len(actor['issued_a'])):
                self.assertGreaterEqual(actor['issued_a'][group],actor['released'][group-2])

    def test_destination_admission_waits_for_both_branches(self):
        result=model(input_path_service_ns=100.,input_min_latency_ns=17.,destination_latency_ns=7.)
        for demand in result['input_demands']:
            self.assertEqual(demand['destination_arrival_ns'],max(demand['source_ready_ns']+7.,
                demand['input_path_end_ns']+17.))

    def test_invalid_shared_path_service_is_rejected(self):
        for value in (-1.,float('nan'),float('inf'),True):
            with self.assertRaisesRegex(ValueError,'input path service'):model(input_path_service_ns=value)


if __name__=='__main__':
    unittest.main()
