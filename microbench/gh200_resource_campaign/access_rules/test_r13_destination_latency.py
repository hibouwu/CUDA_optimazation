"""Per-demand post-source latency and its known cutoff equivalence."""
import unittest

from supply_model import shared_stage_events


def keys(kt=4,ctas=3):
    return [[[(atom,i) for atom in ('A','B0','B1')] for i in range(2*kt)] for _ in range(ctas)]


def model(**extra):
    args=dict(service_ns=10.,read_latency_ns=17.,compute_ns=64.,operand_gap_ns=3.,producer_gap_ns=2.,
        destination_service_ns=5.,output_weight=.5,trace_requests=True,share_inflight_sources=True,source_keys=keys())
    args.update(extra)
    return shared_stage_events(4,2,3,**args)


class DestinationLatencyTests(unittest.TestCase):
    def test_explicit_zero_preserves_complete_legacy_result(self):
        self.assertEqual(model(),model(destination_latency_ns=0.))

    def test_per_demand_delay_precedes_destination_fifo(self):
        result=model(destination_latency_ns=100.)
        self.assertEqual(result['input_delivery_work'],result['input_demand_work'])
        for d in result['deliveries']:
            self.assertEqual(d['destination_arrival_ns'],d['source_ready_ns']+100.)
            self.assertGreaterEqual(d['start_ns'],d['destination_arrival_ns'])
            self.assertEqual(d['service_end_ns']-d['start_ns'],5.*d['work'])

    def test_destination_transit_does_not_keep_source_subscription_open(self):
        result=model(read_latency_ns=0.,destination_latency_ns=100.,operand_gap_ns=1000.,
            diagnostic_producer_offsets_ns=[0.,5.,15.])
        first=sorted((d for d in result['input_demands'] if d['kind']=='issue_a' and d['index']==0),key=lambda d:d['cta'])
        self.assertEqual([d['merged'] for d in first],[False,True,False])
        self.assertEqual(first[0]['source_ready_ns'],10.)
        self.assertEqual(first[0]['destination_arrival_ns'],110.)
        self.assertEqual(first[2]['source_ready_ns'],25.)

    def test_full_relocation_reproduces_existing_service_end_cutoff(self):
        old=model(source_join_boundary='service_end',diagnostic_producer_offsets_ns=[0.,5.,15.])
        moved=model(read_latency_ns=0.,destination_latency_ns=17.,diagnostic_producer_offsets_ns=[0.,5.,15.])
        for field in ('ctas','makespan_ns','shared_busy_ns','last_service_end_ns','request_count',
                'input_source_count','input_merged_count','input_demand_work','input_delivery_work',
                'source_busy_ns','destination_busy_by_cta_ns'):
            self.assertEqual(old[field],moved[field])
        for a,b in zip(old['input_demands'],moved['input_demands']):
            self.assertEqual(a['source_id'],b['source_id'])
            self.assertEqual(a['source_ready_ns'],b['destination_arrival_ns'])
        self.assertTrue(any(a['source_ready_ns']!=b['source_ready_ns']
            for a,b in zip(old['input_demands'],moved['input_demands'])))

    def test_nonshared_zero_service_destination_still_has_transit(self):
        result=shared_stage_events(2,2,1,service_ns=0.,read_latency_ns=0.,compute_ns=8.,
            destination_latency_ns=31.,trace_requests=True)
        self.assertEqual(result['ctas'][0]['starts'][0],31.)
        self.assertEqual(result['input_delivery_work'],8.)
        self.assertTrue(all(d['ready_ns']==d['arrival_ns']+31. for d in result['deliveries']))

    def test_partial_address_units_expire_before_destination_transit(self):
        unique=[[[(atom,cta,i) for atom in ('A','B0','B1')] for i in range(8)] for cta in range(3)]
        units={k:(k,) for groups in unique for group in groups for k in group}
        for cta in range(3):units[unique[cta][0][0]]=(('shared-A',0),)
        result=model(source_keys=unique,source_address_units=units,source_unit_service_ns=10.,
            read_latency_ns=0.,destination_latency_ns=100.,operand_gap_ns=1000.,
            diagnostic_producer_offsets_ns=[0.,5.,15.])
        first=sorted((d for d in result['input_demands'] if d['kind']=='issue_a' and d['index']==0),key=lambda d:d['cta'])
        self.assertEqual([d['source_ready_ns'] for d in first],[10.,10.,25.])
        self.assertEqual([d['destination_arrival_ns'] for d in first],[110.,110.,125.])
        self.assertEqual(len({d['source_id'] for d in first}),3)

    def test_invalid_transit_is_rejected(self):
        for value in (-1.,float('nan'),float('inf'),True):
            with self.assertRaisesRegex(ValueError,'destination latency'):model(destination_latency_ns=value)


if __name__=='__main__':
    unittest.main()
