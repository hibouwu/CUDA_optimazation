"""Source-backed handoff checks, not Scheduler runtime/CLC coverage proof."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_atlas import source_selector_matches


class SchedulerHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas=json.loads((ROOT/'data/atlas.json').read_text())
        cls.part=json.loads((ROOT/'data/modules/dense_fp16/scheduler_handoff.json').read_text())
        cls.nodes={n['id']:n for n in cls.atlas['nodes']}
        cls.edges={e['id']:e for e in cls.atlas['edges']}
        cls.overviews=json.loads((ROOT/'data/overviews.json').read_text())
        cls.kernel=(ROOT/'snapshot/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp').read_text().splitlines()
        cls.scheduler=(ROOT/'snapshot/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp').read_text().splitlines()

    def test_all_new_interfaces_are_linked_to_canonical_declarations(self):
        self.assertEqual(len(self.part['nodes']),13)
        for authored in self.part['nodes']:
            node=self.nodes[authored['id']]
            self.assertEqual(node['declaration_status'],'linked_to_source_ledger',node['id'])
            occurrence={'path':node['path'],'kind':node['entity_kind'],'qualified_name':node['full_name'],
                        'parse_status':node['parse_status'],'signature_range':node['signature_range'],'raw_signature':node['signature']}
            self.assertTrue(source_selector_matches(authored['source_selector'],occurrence),node['id'])
            raw=(ROOT/'snapshot'/node['path']).read_bytes()
            span=node['signature_range']
            self.assertEqual(raw[span['start_byte']:span['end_byte']].decode().rstrip(),node['signature'])

    def test_selector_cannot_trim_range_to_hide_a_mismatch(self):
        node=self.nodes['scheduler.initial']
        selector=copy.deepcopy(node['source_selector']); selector['signature_range']['end_byte']-=1
        occurrence={'path':node['path'],'kind':node['entity_kind'],'qualified_name':node['full_name'],
                    'parse_status':'parsed','signature_range':node['signature_range'],'raw_signature':node['signature']}
        self.assertFalse(source_selector_matches(selector,occurrence))

    def test_same_named_overloads_have_distinct_identity_and_parameter_lists(self):
        for left,right,count_left,count_right in [('scheduler.fetch','scheduler.fetch_compat',3,1),
                                                 ('scheduler.epilogue','scheduler.epilogue_compat',1,2)]:
            a,b=self.nodes[left],self.nodes[right]
            self.assertNotEqual(a['entity_id'],b['entity_id'])
            self.assertEqual(len(a['parameters']),count_left)
            self.assertEqual(len(b['parameters']),count_right)

    def test_all_five_fetch_sites_are_independent_and_choose_three_parameter_overload(self):
        expected=[664,701,741,819,877]
        calls=[e for e in self.edges.values() if e['target']=='scheduler.fetch' and e['relation']=='calls']
        self.assertEqual(sorted(e['callsite']['start_line'] for e in calls),expected)
        self.assertEqual(len({e['callsite']['callsite_id'] for e in calls}),5)
        self.assertFalse(any(e['target']=='scheduler.fetch_compat' and e['relation']=='calls' for e in self.edges.values()))

    def test_both_epilogue_overloads_are_actually_used_in_different_roles(self):
        self.assertEqual(self.edges['scheduler.call.epilogue_compat.816']['target'],'scheduler.epilogue_compat')
        self.assertEqual(self.edges['scheduler.call.epilogue.910']['target'],'scheduler.epilogue')
        self.assertIn('is_participant.epi_load',self.edges['scheduler.call.epilogue_compat.816']['condition'])
        self.assertIn('is_participant.epilogue',self.edges['scheduler.call.epilogue.910']['condition'])

    def test_fixup_closure_reuses_original_callsite_not_a_duplicate(self):
        edge=self.edges['contract.edge.kernel_fixup']
        self.assertEqual(edge['target'],'scheduler.fixup_state')
        self.assertEqual(edge['resolution'],'configured')
        same=[e for e in self.edges.values() if e.get('callsite',{}).get('path')==edge['callsite']['path'] and e.get('callsite',{}).get('start_line')==898]
        self.assertEqual([e['id'] for e in same],['contract.edge.kernel_fixup'])
        self.assertIn('contract.external.scheduler_fixup',self.nodes) # history retained

    def test_current_work_finishes_before_next_validity_controls_mma_and_store_loops(self):
        for first,last,operation in [(737,775,'collective_mainloop.mma('),(876,934,'collective_epilogue.template store<IsOverlappingAccum>(')]:
            block='\n'.join(self.kernel[first-1:last])
            self.assertLess(block.index('scheduler.fetch_next_work('),block.index(operation))
            self.assertLess(block.index(operation),block.index('work_tile_info = next_work_tile_info;'))
            self.assertLess(block.index('work_tile_info = next_work_tile_info;'),block.index('work_tile_info.is_valid()'))

    def test_input_epilogue_keeps_old_coordinates_after_early_work_info_update(self):
        block='\n'.join(self.kernel[814:855])
        assignment=block.index('work_tile_info = next_work_tile_info;')
        load=block.index('collective_epilogue.template load<IsOverlappingAccum>(')
        coordinate=block.index('cta_coord_mnkl = scheduler.work_tile_to_cta_coord')
        self.assertLess(assignment,load);self.assertLess(load,coordinate)
        self.assertIn('旧坐标',self.edges['scheduler.call.fetch.819']['condition'])

    def test_selected_full_k_policy_does_not_read_work_info_to_split_reduction(self):
        count_body='\n'.join(self.scheduler[502:506])
        self.assertIn('return cute::size(cute::ceil_div',count_body)
        self.assertNotIn('work_tile_info',count_body)
        self.assertEqual(self.scheduler[511].strip(),'// All work units returned by this scheduler start from K tile 0')
        self.assertIn('return 0u;','\n'.join(self.scheduler[509:514]))

    def test_fixup_is_not_a_hidden_wait_or_reduction(self):
        self.assertEqual(self.scheduler[562].strip(),'return acc_pipe_consumer_state;')
        self.assertEqual(self.scheduler[563].strip(),'}')
        self.assertIn('return true;','\n'.join(self.scheduler[524:528]))
        self.assertIn('return false;','\n'.join(self.scheduler[532:537]))

    def test_new_processes_have_whole_views_and_do_not_lose_any_staged_edge(self):
        graphs={d['id']:d for d in self.overviews['diagrams']}
        seen=set()
        for view in self.part['views']:
            graph=graphs['overview-'+view['id']]
            self.assertEqual(graph['edge_ids'],view['edge_ids'])
            seen.update(view['edge_ids'])
        self.assertEqual(seen,{e['id'] for e in self.part['edges']}|{'contract.edge.kernel_fixup'})

    def test_new_parameters_have_per_interface_explanations(self):
        docs=json.loads((ROOT/'data/interface-docs.json').read_text())['nodes']
        for node in self.part['nodes']:
            entry=docs[node['id']]
            self.assertTrue(entry['summary']);self.assertTrue(entry['execution'])
            self.assertEqual(len(entry['parameters']['arguments']),len(self.nodes[node['id']]['parameters']))


if __name__=='__main__':unittest.main()
