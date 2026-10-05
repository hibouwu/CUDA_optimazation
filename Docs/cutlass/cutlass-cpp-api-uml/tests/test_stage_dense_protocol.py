import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from stage_dense_tmem_protocol import assemble


class DenseProtocolStagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads((ROOT/'data/module-drafts/dense_protocols/tmem_lifetime.json').read_text())
        cls.scope=json.loads((ROOT/'data/scope.json').read_text());cls.nodes={};cls.edges={}
        for name in ['host.json','type_path.json','contracts.json','gemm_dispatch.json']:
            part=json.loads((ROOT/'data/modules/dense_fp16'/name).read_text())
            cls.nodes.update((n['id'],n)for n in part.get('nodes',[]));cls.edges.update((e['id'],e)for e in part.get('edges',[]))
        cls.result=assemble(cls.protocol,cls.nodes,cls.edges,cls.scope)

    def test_no_event_order_or_state_transition_is_manufactured(self):
        staged=self.result['protocols'][0]
        for field in ['partial_order','state_transitions','resources','actors']:
            self.assertEqual(staged[field],self.protocol[field])
        self.assertEqual([e['id']for e in staged['events']],[e['id']for e in self.protocol['events']])

    def test_each_api_event_has_a_canonical_api_and_call(self):
        for event in self.result['protocols'][0]['events']:
            if event['kind']in {'api_call','builtin_call'}:
                self.assertTrue(event.get('api_node_ref'),event['id']);self.assertTrue(event.get('call_edge_ref'),event['id'])

    def test_same_physical_call_reuses_edge_but_cta_events_stay_separate(self):
        events={e['id']:e for e in self.result['protocols'][0]['events']}
        calls=[events['tmem.cta'+str(cta)+'.init_dealloc']['call_edge_ref']for cta in range(4)]
        self.assertEqual(len(set(calls)),1)
        self.assertEqual(len({events['tmem.cta'+str(cta)+'.init_dealloc']['actor_id']for cta in range(4)}),4)

    def test_wrong_supplement_expression_is_rejected(self):
        broken=copy.deepcopy(self.protocol);event=next(e for e in broken['events']if e['id']=='tmem.cta0.init_dealloc')
        event['callsite']['source_expression']='tmem_deallocation_result_barrier.init(1);'
        with self.assertRaisesRegex(ValueError,'callsite failed'):assemble(broken,self.nodes,self.edges,self.scope)


if __name__=='__main__':unittest.main()
