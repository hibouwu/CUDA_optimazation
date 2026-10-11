"""Lifecycle boundary checks with source-port fixtures from B commit 5c987464."""
import copy
import gzip
import json
from pathlib import Path
import unittest

from output_model import cfg_c_rank_dependencies, cfg_c_empty_arrival_routes, dependency_bounds, dependency_order


def source_fixture(k=3,t=2):
    path=Path(__file__).with_name('fixtures')/f'cfg_c_source_ports_s4_k{k}_t{t}.json.gz'
    with gzip.open(path,'rt') as stream:data=json.load(stream)
    assert data['source_commit']=='5c987464e0c370dc4303a961a9c5f68a88f5b414'
    return data['graph']


class CfgCRankDependenciesTests(unittest.TestCase):
    def setUp(self):
        self.source=source_fixture()
        self.graph=cfg_c_rank_dependencies(self.source)
        self.nodes=self.graph['nodes']

    def test_physical_sources_full_and_arrival_mask(self):
        self.assertEqual(self.graph['tile'],[256,128,64])
        self.assertEqual(self.graph['cluster'],[1,2])
        self.assertEqual(self.graph['local_output_layout']['subtiles_per_output'],8)
        for rank in (0,1):
            incoming=[d for d in self.graph['input_deliveries'] if d['group']==0 and d['destination_rank']==rank]
            self.assertEqual(sum(d['bytes'] for d in incoming),49152)
            self.assertEqual({(d['source_rank'],d['atom']) for d in incoming},
                             {(0,'A'),(1,'A'),(rank,'B0'),(rank,'B1')})
        for source in (0,1):
            routes=[r for r in cfg_c_empty_arrival_routes() if r['source_rank']==source]
            self.assertEqual([(r['physical_thread'],r['destination_rank']) for r in routes],
                             [(128,0),(168,1),(256,0),(296,1)])
            self.assertTrue(all(r['source_coord']==[0,source] for r in routes))
            self.assertTrue(all(r['destination_coord']==[0,r['destination_rank']] for r in routes))

    def test_open_empty_lists_do_not_erase_acquire_predecessors(self):
        original=copy.deepcopy(self.source)
        graph=cfg_c_rank_dependencies(self.source)
        self.assertEqual(original,self.source)
        for name,parents in self.source['nodes'].items():
            self.assertTrue(set(parents)<=set(graph['nodes'][name]))
        for rank in (0,1):
            self.assertEqual(self.nodes[f'r{rank}.g4.arrive_expect_tx'],[f'r{rank}.g4.acquire'])
            self.assertEqual(self.nodes[f'r{rank}.g4.acquire_return'],[f'r{rank}.g4.arrive_expect_tx'])
            self.assertIn(f'r{rank}.g3.B_issue',self.nodes[f'r{rank}.g4.acquire'])
            self.assertEqual({p for p in self.nodes[f'r{rank}.g4.acquire'] if 'empty_to' in p},
                             {f'r{s}.w{w}.g0.empty_to_r{rank}' for s in (0,1) for w in (0,1)})
        self.assertFalse(any('empty_to' in p for p in self.nodes['r0.g3.acquire']))
        self.assertEqual({n for n,p in self.nodes.items() if not p},{'cluster_initialized'})
        self.assertEqual(self.graph['external_producer_ports'],[])

    def test_cross_output_state_and_both_warpgroup_release(self):
        self.assertEqual([(g['slot'],g['generation'],g['empty_wait_phase'],g['full_wait_phase'])
                          for g in self.graph['groups'][3:5]],[(3,0,1,0),(0,1,0,1)])
        for rank in (0,1):
            for wg in (0,1):
                prefix=f'r{rank}.w{wg}.'
                self.assertIn(prefix+'t0.DONE',self.nodes[prefix+'g3.wait_return'])
                self.assertEqual(self.nodes[prefix+'g0.safe_release'],[prefix+'g1.commit',prefix+'g0.retired'])
                self.assertEqual(self.nodes[prefix+'g2.safe_release'],[prefix+'g2.tail_wait0'])
                self.assertIn(prefix+'g2.retired',self.nodes[prefix+'g2.tail_wait0'])
                self.assertEqual(self.nodes[prefix+'g2.mma_return'],[prefix+'g1.release_issue'])
                self.assertEqual(self.nodes[prefix+'g2.empty_to_r1'],[prefix+'g2.empty_emit_to_r1'])
                self.assertNotIn(prefix+'g2.empty_to_r1',self.nodes[prefix+'t0.MAIN'])

    def test_short_work_tail_checks_unused_initial_slots(self):
        graph=cfg_c_rank_dependencies(source_fixture(2,1))
        waits=[w for w in graph['producer_tail_waits'] if w['rank']==0]
        self.assertEqual([(w['slot'],w['release_group'],w['initial_empty']) for w in waits],
                         [(2,None,True),(3,None,True),(0,0,False),(1,1,False)])
        self.assertEqual([w['empty_wait_phase'] for w in waits],[1,1,0,0])
        self.assertIn('cluster_initialized',graph['nodes'][waits[0]['event']])
        for wait in waits[2:]:
            self.assertEqual(len([n for n in graph['nodes'][wait['event']] if 'empty_to' in n]),4)
        dependency_order(graph)

    def test_eight_subtiles_and_complete_keep_source_and_visibility_separate(self):
        for rank in (0,1):
            prefix=f'r{rank}.'
            self.assertIn(prefix+'d7.acquire_after',self.nodes[prefix+'w0.t0.DONE'])
            self.assertIn(prefix+'d7.read_retired',self.nodes[prefix+'d8.acquire_after'])
            self.assertIn(prefix+'d8.acquire_after',self.nodes[prefix+'d9.r2s'])
            self.assertIn(prefix+'d15.read_retired',self.nodes[prefix+'store_tail_read0'])
            self.assertNotIn(prefix+'store_tail_read0',self.nodes[prefix+'w0.FINAL'])
            self.assertIn(prefix+'store_tail_read0',self.nodes[prefix+'w1.FINAL'])
        self.assertEqual(set(self.nodes['source_safe_exit']),{f'r{r}.producer_tail' for r in (0,1)}|
                         {f'r{r}.w{w}.FINAL' for r in (0,1) for w in (0,1)})
        self.assertEqual(self.nodes['cuda_complete'],['source_safe_exit','r0.D_visible','r1.D_visible'])
        self.assertNotIn('r0.D_visible',self.nodes['source_safe_exit'])
        for guarantee in self.graph['source_guarantees']:
            self.assertIn(guarantee['source_completion'],self.nodes)
            self.assertIn(guarantee['guarantee_token'],self.nodes)

    def test_local_b_delay_reaches_partner_through_finite_empty_reuse(self):
        # Deliberately synthetic positive lags; this is a dependency witness only.
        lag={n:1 for n in self.nodes}
        baseline=dependency_bounds(self.graph,latencies_ns=lag)
        lag['r1.g0.B0_source_ready']+=1000
        delayed=dependency_bounds(self.graph,latencies_ns=lag)
        self.assertEqual(delayed['r0.g0.full'],baseline['r0.g0.full'])
        self.assertGreater(delayed['r1.g0.full'][0],baseline['r1.g0.full'][0])
        self.assertGreater(delayed['r0.g4.acquire'][0],baseline['r0.g4.acquire'][0])
        anchored=dependency_bounds(self.graph,{'cluster_initialized':(100,100)})
        self.assertEqual(anchored['cuda_complete'],[100,None])

    def test_reject_wrong_geometry_partial_delivery_and_unknown_roots(self):
        for key,value in (('schema','cfg_a'),('stages',6),('stages',True),('k_tiles',1),('output_tiles',0)):
            changed=copy.deepcopy(self.source);changed[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):cfg_c_rank_dependencies(changed)
        changed=copy.deepcopy(self.source);changed['input_deliveries'][0]['bytes']=8192
        with self.assertRaises(ValueError):cfg_c_rank_dependencies(changed)
        changed=copy.deepcopy(self.source);changed['groups'][4]['slot']=1
        with self.assertRaises(ValueError):cfg_c_rank_dependencies(changed)
        changed=copy.deepcopy(self.source);changed['nodes']['r0.g0.full'].remove('r1.g0.A_delivered_r0')
        with self.assertRaises(ValueError):cfg_c_rank_dependencies(changed)
        changed=copy.deepcopy(self.source);changed['nodes']['unresolved_other_root']=[]
        with self.assertRaises(ValueError):cfg_c_rank_dependencies(changed)
        changed=copy.deepcopy(self.source);changed['nodes']['r0.g0.A_source_ready'].append('cuda_complete')
        with self.assertRaisesRegex(ValueError,'cycle'):cfg_c_rank_dependencies(changed)


if __name__=='__main__':unittest.main()
