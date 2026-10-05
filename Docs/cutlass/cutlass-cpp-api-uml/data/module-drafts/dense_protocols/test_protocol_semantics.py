"""D01/D02/Q01 bounded counterexamples; no CUDA/GPU execution."""
import hashlib
import json
from pathlib import Path
import unittest
from protocol_semantics import acyclic,derive_state_facts,expand_graph,reaches,rule_satisfied

HERE=Path(__file__).resolve().parent

class TmemSemanticsRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=json.loads((HERE/'tmem_lifetime.json').read_text())
        cls.events={e['id']:e for e in cls.d['events']};cls.graph=expand_graph(cls.d)

    def test_v1_is_preserved_with_original_findings(self):
        raw=(HERE/'history/v1-3af156cc/tmem_lifetime.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'3af156ccb074cc3f0e4d819069d1281fcfded5e3635e9d217f56f8a7b1bbecb6')
        old=json.loads(raw)
        edges={tuple(o['before'].values()):[] for o in old['partial_order']}
        for o in old['partial_order']:edges.setdefault(tuple(o['before'].values()),[]).append(tuple(o['after'].values()))
        self.assertEqual(edges.get(('tmem.cta0.allocate','effect'),[]),[])

    def test_d01_same_address_read_before_write_is_rejected(self):
        for c in range(4):
            write=(f'tmem.cta{c}.allocate','effect','operation')
            for role in ('mma','epi'):
                read=(f'tmem.cta{c}.{role}_read_base','return',0)
                self.assertTrue(reaches(self.graph,write,read))
                self.assertFalse(acyclic(self.graph,[(read,write)]))

    def test_d01_does_not_equate_store_effect_with_api_return(self):
        for c in range(4):
            self.assertFalse(reaches(self.graph,(f'tmem.cta{c}.allocate','effect','operation'),(f'tmem.cta{c}.allocate','return',0)))

    def test_d02_local_instruction_is_not_paired_completion(self):
        self.assertEqual(self.events['tmem.cta0.free']['effect']['kind'],'local_dealloc_instruction_issued')
        self.assertNotIn('tmem.cta0.allocation.deallocated',{s['id'] for r in self.d['resources'] for s in r['states']})
        self.assertTrue(acyclic(self.graph,[(('tmem.cta0.free','effect','operation'),('tmem.cta1.free','enter',0))]))
        rule=self.d['validity_rules'][0];pre={'uniform_512_columns_same_logical_warp':True}
        own={('tmem.cta0.free','effect','operation')}
        self.assertFalse(rule_satisfied(rule,own,pre))
        own.add(('tmem.cta1.free','effect','operation'))
        self.assertFalse(rule_satisfied(rule,own,pre))
        own|={(f'tmem.cta{c}.free','enter',t) for c in (0,1) for t in range(32)}
        self.assertTrue(rule_satisfied(rule,own,pre));self.assertFalse(rule_satisfied(rule,own,{}))

    def test_d02_peer_enter_is_not_a_local_return_predecessor(self):
        self.assertFalse(reaches(self.graph,('tmem.cta1.free','enter',0),('tmem.cta0.free','return',0)))
        a=('tmem.cta0.free','enter',0);b=('tmem.cta1.free','enter',0)
        self.assertTrue(acyclic(self.graph,[(a,b)]));self.assertTrue(acyclic(self.graph,[(b,a)]))

    def test_d03_v2_counterexample_is_preserved(self):
        raw=(HERE/'history/v2-253091a3/tmem_lifetime.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'253091a30d2367bb268b230831a195adb2a6bf8add419a61ea1841abee77e4e0')
        old=expand_graph(json.loads(raw))
        self.assertEqual(self.d['partial_order'][:248],json.loads(raw)['partial_order'])
        for c in range(4):
            self.assertTrue(acyclic(old,[((f'tmem.cta{c}.free','return',0),(f'tmem.cta{c}.free','effect','operation'))]))

    def test_d03_return_before_local_issue_is_rejected_for_all_128_threads(self):
        edges=[o for o in self.d['partial_order'] if o['relation']=='local_operation_before_local_return']
        self.assertEqual(len(edges),4)
        for o in edges:
            self.assertEqual(o['before']['event_id'],o['after']['event_id'])
            self.assertEqual(o['quantifier']['kind'],'operation_to_each_target')
        for c in range(4):
            effect=(f'tmem.cta{c}.free','effect','operation')
            for thread in range(32):
                returned=(f'tmem.cta{c}.free','return',thread)
                self.assertTrue(reaches(self.graph,effect,returned))
                self.assertFalse(acyclic(self.graph,[(returned,effect)]))

    def test_q01_same_thread_order_does_not_create_family_barrier(self):
        self.assertTrue(reaches(self.graph,('tmem.cta0.arrive795','return',0),('tmem.cta0.free','enter',0)))
        self.assertFalse(reaches(self.graph,('tmem.cta0.arrive795','return',1),('tmem.cta0.free','enter',0)))
        self.assertTrue(acyclic(self.graph,[(('tmem.cta0.free','enter',0),('tmem.cta0.arrive795','return',1))]))
        for o in self.d['partial_order']:
            if o['relation']=='program_order':self.assertEqual(o['quantifier']['kind'],'pointwise_same_thread')

    def test_q01_named_barrier_needs_whole_families(self):
        epi={('tmem.cta0.alloc_wait','effect',t) for t in range(128)}
        mma={('tmem.cta0.alloc_arrive','effect',t) for t in range(31)}
        states=derive_state_facts(self.d,epi|mma)
        self.assertIn('tmem.cta0.allocation_barrier.epi_only',states)
        self.assertNotIn('tmem.cta0.allocation_barrier.both',states)
        mma.add(('tmem.cta0.alloc_arrive','effect',31))
        self.assertIn('tmem.cta0.allocation_barrier.both',derive_state_facts(self.d,epi|mma))
        self.assertEqual(self.events['tmem.cta0.allocate']['anchor_quantifiers']['effect']['kind'],'local_warp_operation')
        self.assertEqual(self.events['tmem.cta0.allocate']['anchor_quantifiers']['return']['kind'],'per_thread_instance')

    def test_q01_history_rules_do_not_consume_or_time_order_facts(self):
        for t in self.d['state_transitions']:
            self.assertIsInstance(t['trigger_quantifier'],dict)
            self.assertFalse(t['derivation']['adds_temporal_order'])
            self.assertFalse(t['derivation']['consumes_from_fact'])
        initial=derive_state_facts(self.d,set())
        all_named={('tmem.cta0.alloc_arrive','effect',t) for t in range(32)}|{('tmem.cta0.alloc_wait','effect',t) for t in range(128)}
        self.assertLess(initial,derive_state_facts(self.d,all_named))

if __name__=='__main__':unittest.main()
