"""Address-sharing feedback preserves whole requests, stages and destinations."""
import unittest

from supply_model import shared_stage_events


def program(kt=4,ctas=3):
    keys=[[[(cta,i,atom) for atom in range(3)] for i in range(2*kt)] for cta in range(ctas)]
    units={key:tuple((key,n) for n in range(2 if atom==0 else 1))
        for groups in keys for group in groups for atom,key in enumerate(group)}
    return keys,units


def simulate(keys,units=None,**overrides):
    params=dict(service_ns=20.,read_latency_ns=17.,compute_ns=8.,
        operand_gap_ns=1.,producer_gap_ns=1.,destination_service_ns=3.,output_weight=.5)
    params.update(overrides)
    return shared_stage_events(len(keys[0])//2,2,len(keys),source_keys=keys,
        share_inflight_sources=True,source_address_units=units,
        source_unit_service_ns=10. if units is not None else None,**params)


class AddressUnitTests(unittest.TestCase):
    def test_disjoint_units_reproduce_legacy_box_service(self):
        keys,units=program()
        old=simulate(keys)
        new=simulate(keys,units)
        new.pop('address_unit_service')
        self.assertEqual(old,new)

    def test_partial_and_fully_joined_boxes_keep_separate_deliveries(self):
        keys,units=program(kt=2)
        for cta,footprint in enumerate(((0,1),(1,2),(0,2))):
            units[keys[cta][0][0]]=tuple(('A',n) for n in footprint)
        model=simulate(keys,units,trace_requests=True,operand_gap_ns=1000.)
        sources=[s for s in model['sources'] if s['kind']=='issue_a' and s['index']==0]
        self.assertEqual([s['new_address_units'] for s in sources],[2,1,0])
        self.assertEqual([s['joined_address_units'] for s in sources],[0,1,2])
        self.assertEqual([s['source_service_ns'] for s in sources],[20.,10.,0.])
        self.assertEqual([s['ready_ns'] for s in sources],[37.,47.,47.])
        self.assertEqual([s['joined_owner_ids'] for s in sources],[[],[0],[0,1]])
        demands=[d for d in model['input_demands'] if d['kind']=='issue_a' and d['index']==0]
        self.assertEqual(len({d['source_id'] for d in demands}),3)
        self.assertTrue(all(not d['merged'] for d in demands))
        self.assertEqual(len([d for d in model['deliveries'] if d['kind']=='issue_a' and d['index']==0]),3)
        self.assertEqual(model['input_delivery_work'],model['input_demand_work'])
        self.assertEqual(model['source_busy_ns'],model['address_unit_service']['generations']*10.)
        self.assertEqual(model['request_count'],model['input_source_count']-1+3*16)

    def test_whole_box_join_does_not_create_a_second_unit_batch(self):
        keys,units=program(kt=2,ctas=2)
        keys[1][0][0]=keys[0][0][0]
        model=simulate(keys,units,trace_requests=True,operand_gap_ns=1000.)
        demands=[d for d in model['input_demands'] if d['kind']=='issue_a' and d['index']==0]
        self.assertEqual([d['merged'] for d in demands],[False,True])
        self.assertEqual(demands[0]['source_id'],demands[1]['source_id'])
        source=next(s for s in model['sources'] if s['source_id']==demands[0]['source_id'])
        self.assertEqual(source['new_address_units'],2)
        self.assertEqual(source['joined_address_units'],0)

    def test_expiry_is_strict_and_old_cleanup_preserves_new_unit_owner(self):
        keys,units=program(kt=2)
        for cta in range(3):units[keys[cta][0][0]]=(('A',0),)
        model=simulate(keys,units,trace_requests=True,operand_gap_ns=1000.,read_latency_ns=0.,
            diagnostic_producer_offsets_ns=[0.,10.,15.])
        sources=[s for s in model['sources'] if s['kind']=='issue_a' and s['index']==0]
        self.assertEqual([s['new_address_units'] for s in sources],[1,1,0])
        self.assertEqual([s['joined_address_units'] for s in sources],[0,0,1])
        self.assertEqual([s['ready_ns'] for s in sources],[10.,20.,20.])

    def test_service_changes_feed_back_to_slot_reuse(self):
        keys,units=program()
        for cta in range(3):units[keys[cta][0][0]]=(('A',cta),('A',cta+1))
        old=simulate(keys,read_latency_ns=0.)
        new=simulate(keys,units,read_latency_ns=0.,trace_requests=True)
        self.assertLess(new['source_busy_ns'],old['source_busy_ns'])
        self.assertTrue(any(a['issued_a'][i]!=b['issued_a'][i]
            for a,b in zip(old['ctas'],new['ctas']) for i in range(2,8)))
        for actor in new['ctas']:
            self.assertTrue(all(actor['issued_a'][i]>=actor['released'][i-2] for i in range(2,8)))
        sources={s['source_id']:s for s in new['sources']}
        self.assertTrue(all(d['start_ns']>=sources[d['source_id']]['ready_ns'] for d in new['deliveries']))
        self.assertEqual(new['input_demand_work'],old['input_demand_work'])
        self.assertEqual(new['input_delivery_work'],old['input_delivery_work'])
        self.assertEqual(len([r for r in new['requests'] if r['kind']=='store']),3*16)

    def test_invalid_service_contract_is_rejected(self):
        keys,units=program(kt=2,ctas=1)
        for value in (-1.,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):
                shared_stage_events(2,2,1,source_keys=keys,share_inflight_sources=True,
                    source_address_units=units,source_unit_service_ns=value,
                    service_ns=20.,read_latency_ns=0.,compute_ns=8.)
        with self.assertRaises(ValueError):simulate(keys,units,source_join_boundary='service_end')
        for bad in ((),(('A',0),('A',0)),[('A',0)]):
            units[keys[0][0][0]]=bad
            with self.assertRaises(ValueError):simulate(keys,units)

    def test_time_unit_scaling_preserves_events_and_subscriptions(self):
        keys,units=program()
        for cta in range(3):units[keys[cta][0][0]]=(('A',cta),('A',cta+1))
        params=dict(service_ns=20.,read_latency_ns=17.,compute_ns=8.,operand_gap_ns=1.,
            producer_gap_ns=1.,destination_service_ns=3.,output_weight=.5)
        results=[]
        for factor in (1.,2.):
            scaled={k:v*factor if k.endswith('_ns') else v for k,v in params.items()}
            results.append(shared_stage_events(4,2,3,source_keys=keys,share_inflight_sources=True,
                source_address_units=units,source_unit_service_ns=10.*factor,trace_requests=True,**scaled))
        a,b=results
        for field in ('request_count','input_source_count','input_merged_count','input_demand_work','input_delivery_work'):
            self.assertEqual(a[field],b[field])
        for field in ('generations','joins_between_box_generations'):
            self.assertEqual(a['address_unit_service'][field],b['address_unit_service'][field])
        for field in ('makespan_ns','shared_busy_ns','source_busy_ns','destination_busy_ns'):
            self.assertEqual(a[field]*2,b[field])
        for before,after in zip(a['ctas'],b['ctas']):
            for field in ('issued_a','issued_b0','issued_b1','ready','starts','service_frontier',
                    'commits','completed','wait1_return','released','mma_returns','main_ends'):
                self.assertEqual([t*2 for t in before[field]],after[field])
        for collection in ('sources','input_demands','deliveries','requests'):
            for before,after in zip(a[collection],b[collection]):
                for field,value in before.items():
                    if field.endswith('_ns') and value is not None:self.assertEqual(value*2,after[field])


if __name__=='__main__':
    unittest.main()
