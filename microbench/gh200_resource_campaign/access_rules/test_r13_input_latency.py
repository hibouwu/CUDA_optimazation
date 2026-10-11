"""Each subscriber keeps its own path without holding the shared source open."""
import unittest

from supply_model import shared_stage_events


def model(**extra):
    source_keys=[[[(atom,i) for atom in ('A','B0','B1')] for i in range(4)] for _ in range(3)]
    args=dict(service_ns=10.,read_latency_ns=20.,compute_ns=64.,operand_gap_ns=1000.,
        share_inflight_sources=True,source_keys=source_keys,trace_requests=True,
        diagnostic_producer_offsets_ns=[0.,25.,35.])
    args.update(extra)
    return shared_stage_events(2,2,3,**args)


def first_A(result):
    return sorted((d for d in result['input_demands'] if d['kind']=='issue_a' and d['index']==0),
        key=lambda d:d['cta'])


class InputLatencyTests(unittest.TestCase):
    def test_zero_preserves_the_complete_existing_result(self):
        self.assertEqual(model(),model(input_min_latency_ns=0.))
        self.assertEqual(model(destination_latency_ns=7.),
            model(destination_latency_ns=7.,input_min_latency_ns=0.))

    def test_late_subscriber_waits_for_its_own_path(self):
        result=model(input_min_latency_ns=20.)
        demands=first_A(result)
        self.assertEqual([d['merged'] for d in demands],[False,True,False])
        self.assertEqual([d['source_ready_ns'] for d in demands],[30.,30.,65.])
        self.assertEqual([d['destination_arrival_ns'] for d in demands],[30.,45.,65.])
        self.assertEqual(demands[0]['source_id'],demands[1]['source_id'])
        self.assertNotEqual(demands[1]['source_id'],demands[2]['source_id'])
        self.assertEqual(result['input_delivery_work'],result['input_demand_work'])

    def test_source_transit_and_own_path_are_parallel_predecessors(self):
        demands=first_A(model(input_min_latency_ns=20.,destination_latency_ns=7.))
        self.assertEqual([d['destination_arrival_ns'] for d in demands],[37.,45.,72.])
        self.assertEqual([d['input_earliest_destination_ns'] for d in demands],[20.,45.,55.])

    def test_cannot_replace_the_floor_with_one_post_source_delay(self):
        floor=first_A(model(input_min_latency_ns=20.))
        moved=first_A(model(destination_latency_ns=15.))
        self.assertEqual([d['source_id'] for d in floor],[d['source_id'] for d in moved])
        self.assertEqual([d['destination_arrival_ns'] for d in floor],[30.,45.,65.])
        self.assertEqual([d['destination_arrival_ns'] for d in moved],[45.,45.,80.])

    def test_address_units_expire_before_a_subscriber_path_finishes(self):
        source_keys=[[[(atom,cta,i) for atom in ('A','B0','B1')] for i in range(4)] for cta in range(3)]
        units={key:(key,) for groups in source_keys for group in groups for key in group}
        for cta in range(3):units[source_keys[cta][0][0]]=(('shared_A',0),)
        demands=first_A(model(source_keys=source_keys,source_address_units=units,
            source_unit_service_ns=10.,input_min_latency_ns=20.))
        self.assertEqual(len({d['source_id'] for d in demands}),3)
        self.assertEqual([d['source_ready_ns'] for d in demands],[30.,30.,65.])
        self.assertEqual([d['destination_arrival_ns'] for d in demands],[30.,45.,65.])

    def test_nonshared_demand_also_has_its_own_minimum_path(self):
        result=shared_stage_events(2,2,1,service_ns=0.,read_latency_ns=0.,compute_ns=8.,
            input_min_latency_ns=31.,trace_requests=True)
        self.assertEqual(result['ctas'][0]['starts'][0],31.)
        self.assertTrue(all(d['ready_ns']==d['arrival_ns']+31. for d in result['deliveries']))

    def test_invalid_minimum_path_is_rejected(self):
        for value in (-1.,float('nan'),float('inf'),True):
            with self.assertRaisesRegex(ValueError,'input minimum latency'):
                model(input_min_latency_ns=value)


if __name__=='__main__':
    unittest.main()
