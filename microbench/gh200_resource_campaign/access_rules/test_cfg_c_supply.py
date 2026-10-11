"""cfg_c source ownership, physical partners and open full prerequisites."""
import unittest

from supply_model import cfg_c_source_bindings,cfg_c_source_port_graph


class CfgCSupplyTests(unittest.TestCase):
    def binding(self,m=256,n=256,k=1024,**extra):
        row=dict(config='cfg_c',m=m,n=n,k=k,swizzle=1)
        row.update(extra)
        return cfg_c_source_bindings(row,dict(grid=[1,2,1],cluster=[1,2],stages=6))

    def test_A_multicast_and_local_B_ownership(self):
        binding=self.binding();cluster=binding['clusters'][0]
        self.assertEqual(cluster['ctas'],[0,1])
        for rank in range(2):
            output=cluster['rank_outputs'][rank][0]
            self.assertEqual([s['payload_KiB'] for s in output['sources']],[16,8,8])
            self.assertEqual(output['sources'][0]['origin'],[rank*128,0])
            self.assertEqual([s['destination_ranks'] for s in output['sources']],[[0,1],[rank],[rank]])
            self.assertEqual([len(output['destination'][op]['source_atoms']) for op in ('A','B')],[2,2])
        self.assertEqual(binding['totals']['source_payload_KiB'],64*16)
        self.assertEqual(binding['totals']['destination_valid_KiB'],96*16)

    def test_output_OOB_rank_still_supplies_partner_A(self):
        binding=self.binding(n=128)
        output=binding['clusters'][0]['rank_outputs'][1][0]
        self.assertEqual(output['output']['valid_KiB'],0)
        self.assertEqual(output['sources'][0]['valid_KiB'],16)
        self.assertEqual(output['sources'][0]['destination_ranks'],[0,1])
        self.assertEqual(output['destination']['B']['fill_KiB'],16)

    def test_A_partial_and_address_zero_keep_full48KiB(self):
        oob=self.binding(m=192,n=128,input_map_m=192,storage_m=256)
        zero=self.binding(m=192,n=128,input_map_m=256,storage_m=256)
        for binding,valid,fill in ((oob,24,8),(zero,32,0)):
            for rank in range(2):
                output=binding['clusters'][0]['rank_outputs'][rank][0]
                self.assertEqual((output['destination']['A']['valid_KiB'],output['destination']['A']['fill_KiB']),(valid,fill))
                self.assertEqual(sum(output['destination'][op]['valid_KiB']+output['destination'][op]['fill_KiB'] for op in ('A','B')),48)
        self.assertEqual(oob['totals']['output_valid_KiB'],zero['totals']['output_valid_KiB'])

    def test_ring_continues_across_outputs(self):
        binding=self.binding(m=512)
        for rank in range(2):
            output=binding['clusters'][0]['rank_outputs'][rank][1]
            state=output['first_input_state']
            self.assertEqual((state['group'],state['slot'],state['generation']),(16,4,2))

    def test_physical_rank_mapping_and_rejected_cross_cluster_scheduler(self):
        for swizzle in (1,8):
            for raster,grid in (('along_m',[2,4,1]),('along_n',[1,8,1])):
                row=dict(config='cfg_c',m=2048,n=1536,k=128,swizzle=swizzle,raster_order=raster)
                binding=cfg_c_source_bindings(row,dict(grid=grid,stages=4))
                self.assertEqual(binding['totals']['output_valid_KiB'],2048*1536*4/1024)
                for cluster in binding['clusters']:
                    self.assertEqual(cluster['ctas'][1]-cluster['ctas'][0],grid[0])
        row=dict(config='cfg_c',m=2048,n=1536,k=128,swizzle=1,raster_order='along_n')
        with self.assertRaisesRegex(ValueError,'physical cluster'):
            cfg_c_source_bindings(row,dict(grid=[2,4,1],stages=4))

    def test_open_source_graph_requires_both_A_sources_and_two_local_Bs(self):
        graph=cfg_c_source_port_graph(6,16,2);nodes=graph['nodes']
        self.assertEqual(graph['totals']['source_payload_bytes'],32*64*1024)
        self.assertEqual(graph['totals']['destination_payload_bytes'],32*96*1024)
        self.assertEqual(nodes['r1.g0.full'],['r1.g0.arrive_expect_tx','r0.g0.A_delivered_r1',
            'r1.g0.A_delivered_r1','r1.g0.B0_delivered_r1','r1.g0.B1_delivered_r1'])
        self.assertEqual(nodes['r0.g6.acquire_return'],[])
        self.assertIn('r0.g6.acquire_return',graph['external_producer_ports'])
        self.assertFalse(any('empty' in n or 'release' in n for n in nodes))
        self.assertEqual(graph['groups'][16]['slot'],4)
        ancestors=set();pending=['r0.g0.B0_delivered_r0']
        while pending:
            name=pending.pop()
            if name not in ancestors:ancestors.add(name);pending.extend(nodes[name])
        self.assertNotIn('r0.g0.B1_issue',ancestors)

    def test_scope_and_observation_rejection_boundary(self):
        self.assertEqual(self.binding(),self.binding(L_ns=-1e30,measured_frequency=.001))
        with self.assertRaisesRegex(ValueError,'cfg_c'):
            cfg_c_source_bindings(dict(config='cfg_a'),{})
        for dimensions in ((True,16,2),(4,1,2),(4,16,0)):
            with self.assertRaises(ValueError):cfg_c_source_port_graph(*dimensions)


if __name__=='__main__':
    unittest.main()
