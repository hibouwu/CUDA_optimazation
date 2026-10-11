"""A source-guarantee stamp must not become an exact source-completion time."""
import copy
import unittest

from analyze_r15_lifecycle import output_source_windows, output_cycle_bounds
from output_model import (cfg_a_output_source_guarantees, cooperative_output_source_layout,
                          cooperative_output_source_dependencies, cooperative_output_valid_subtiles,
                          dependency_bounds, dependency_order)


class OutputLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for role in (1,2):
            for tile,(main,permit,release) in enumerate([(10,12,110),(100,102,150)]):
                self.rows.append(dict(role=role,seq=tile,main_end_cycle=main,
                                      epi_permit_cycle=permit,source_release_cycle=release if role==2 else None))

    def test_middle_interval_contains_next_output_and_terminal_does_not(self):
        windows=output_source_windows(self.rows,2)
        self.assertEqual(windows[0]['permit_to_release_cycles'],98)
        self.assertEqual(windows[0]['carried_to_next_permit_cycles'],90)
        self.assertEqual(windows[0]['after_next_permit_cycles'],8)
        self.assertEqual(windows[0]['guarantee_token'],'r0.d4.acquire_after')
        self.assertEqual(windows[1]['terminal_service_upper_cycles'],48)
        self.assertEqual(windows[1]['carried_to_next_permit_cycles'],0)
        self.assertEqual(windows[1]['guarantee_token'],'r0.store_tail_read0')

    def test_guarantee_does_not_force_actual_source_read_to_wait_for_next_output(self):
        windows=output_source_windows(self.rows,2)
        setup=dict(stages=6,k=128,tiles_per_cta=2)
        intervals=output_cycle_bounds(setup,self.rows,windows)
        self.assertLess(intervals[0]['lower_cycles'],windows[0]['guarantee_lower_cycle']-10)
        self.assertLessEqual(intervals[0]['upper_cycles'],100)
        self.assertEqual(intervals[0]['observed_guarantee_upper_cycles'],100)

    def test_reject_middle_guarantee_before_next_permit_and_missing_issuer(self):
        rows=copy.deepcopy(self.rows)
        rows[2]['source_release_cycle']=101
        with self.assertRaisesRegex(ValueError,'next output'):
            output_source_windows(rows,2)
        with self.assertRaisesRegex(ValueError,'issuer tile'):
            output_source_windows(self.rows[:-1],2)

    def test_cfg_a_layout_reproduces_accepted_guarantee_metadata(self):
        for count in (1,2,8):
            expected=[entry for entry in cfg_a_output_source_guarantees(count) if entry['rank']==0]
            self.assertEqual(cooperative_output_source_layout('cfg_a',count)['source_guarantees'],expected)

    def test_cfg_c_and_legacy_baseline_keep_distinct_subtile_coordinates(self):
        target=cooperative_output_source_layout('cfg_c',2)
        legacy=cooperative_output_source_layout('r14_baseline',2)
        self.assertEqual(target['subtiles_per_output'],8)
        self.assertEqual(target['source_guarantees'],legacy['source_guarantees'])
        self.assertEqual([item['origin'] for item in target['subtiles'][:4]],[[0,0],[128,0],[0,32],[128,32]])
        self.assertEqual([item['origin'] for item in legacy['subtiles'][:4]],[[0,0],[0,32],[0,64],[0,96]])
        self.assertEqual(target['subtiles'][8]['buffer'],0)
        self.assertEqual(target['subtiles'][8]['generation'],4)
        self.assertEqual(target['source_guarantees'][0]['guarantee_token'],'r0.d8.acquire_after')
        self.assertEqual(target['source_guarantees'][0]['reuse_subtile'],9)

    def test_legacy_observation_binds_eight_chunks_without_claiming_cfg_c_data(self):
        windows=output_source_windows(self.rows,2,'baseline')
        self.assertEqual(windows[0]['guarantee_token'],'r0.d8.acquire_after')
        self.assertEqual(windows[0]['source_generation'],3)
        self.assertEqual(windows[1]['source_generation'],7)
        intervals=output_cycle_bounds(dict(tiles_per_cta=2),self.rows,windows,'baseline')
        self.assertLess(intervals[0]['lower_cycles'],windows[0]['guarantee_lower_cycle']-10)
        self.assertEqual(intervals[0]['upper_cycles'],100)
        with self.assertRaisesRegex(ValueError,'not cfg_c'):
            output_source_windows(self.rows,2,'cfg_c')

    def test_local_source_graph_protects_reuse_but_does_not_bound_d_visibility(self):
        for config in ('cfg_a','cfg_c','r14_baseline'):
            graph=cooperative_output_source_dependencies(config,2)
            dependency_order(graph)
            bounds=dependency_bounds(graph,{'r0.d0.read_retired':(100,100)})
            self.assertEqual(bounds['r0.d1.acquire_after'][0],100)
            self.assertEqual(bounds['r0.d2.r2s'][0],100)
            self.assertIsNone(bounds['r0.D_visible'][1])
            self.assertNotIn('cuda_complete',graph['nodes'])
            self.assertNotIn('source_safe_exit',graph['nodes'])

    def test_output_layout_rejects_unbound_shapes_and_counts(self):
        for count in (0,-1,True,1.5):
            with self.assertRaises(ValueError):cooperative_output_source_layout('cfg_c',count)
        with self.assertRaises(ValueError):cooperative_output_source_layout('cfg_b',1)

    def test_equal_area_cfg_c_boundaries_keep_different_buffer_coverage(self):
        along_n=cooperative_output_valid_subtiles('cfg_c',256,64,[0,0])
        along_m=cooperative_output_valid_subtiles('cfg_c',128,128,[0,0])
        self.assertEqual(along_n['valid_bytes'],along_m['valid_bytes'])
        self.assertEqual([item['valid_bytes'] for item in along_n['subtiles']],[16384]*4+[0]*4)
        self.assertEqual([item['valid_bytes'] for item in along_m['subtiles']],[16384,0]*4)
        self.assertEqual(along_n['buffer_valid_bytes'],[32768,32768])
        self.assertEqual(along_m['buffer_valid_bytes'],[65536,0])
        self.assertEqual(along_n['last_valid_subtile'],3)
        self.assertEqual(along_m['last_valid_subtile'],6)

    def test_output_oob_keeps_all_groups_and_global_generations(self):
        oob=cooperative_output_valid_subtiles('cfg_c',256,128,[0,1],3)
        self.assertEqual(oob['valid_bytes'],0)
        self.assertEqual(len(oob['subtiles']),8)
        self.assertIsNone(oob['last_valid_subtile'])
        self.assertEqual(oob['subtiles'][0]['subtile'],24)
        self.assertEqual(oob['subtiles'][0]['generation'],12)
        a=cooperative_output_valid_subtiles('cfg_a',64,128,[0,0])
        self.assertEqual([item['valid_bytes'] for item in a['subtiles']],[8192]*4)


if __name__=='__main__':
    unittest.main()
