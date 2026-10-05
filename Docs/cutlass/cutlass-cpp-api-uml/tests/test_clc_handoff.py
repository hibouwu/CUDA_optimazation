"""Checks the selected CLC source handoff, not hardware execution correctness."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_atlas import source_selector_matches


class CLCHandoffTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.atlas=json.loads((ROOT/'data/atlas.json').read_text())
  cls.part=json.loads((ROOT/'data/modules/dense_fp16/clc_handoff.json').read_text())
  cls.nodes={n['id']:n for n in cls.atlas['nodes']};cls.edges={e['id']:e for e in cls.atlas['edges']}
  cls.pipe=(ROOT/'snapshot/include/cutlass/pipeline/sm100_pipeline.hpp').read_text().splitlines()
  cls.sched=(ROOT/'snapshot/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp').read_text().splitlines()
  cls.kernel=(ROOT/'snapshot/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp').read_text().splitlines()

 def test_all_sixteen_cpp_declarations_use_existing_canonical_identity(self):
  declarations=[n for n in self.part['nodes'] if n['kind'] in {'api','type'}]
  self.assertEqual(len(declarations),16)
  for authored in declarations:
   node=self.nodes[authored['id']]
   self.assertEqual(node['declaration_status'],'linked_to_source_ledger')
   occurrence={'path':node['path'],'kind':node['entity_kind'],'qualified_name':node['full_name'],
               'parse_status':node['parse_status'],'raw_signature':node['signature'],'signature_range':node['signature_range']}
   self.assertTrue(source_selector_matches(authored['source_selector'],occurrence),node['id'])
   span=node['signature_range'];raw=(ROOT/'snapshot'/node['path']).read_bytes()
   self.assertEqual(raw[span['start_byte']:span['end_byte']].decode().rstrip(),node['signature'])

 def test_default_token_braces_are_not_mistaken_for_method_body(self):
  for identifier in ('clc.acquire','clc.wait'):
   node=self.nodes[identifier]
   self.assertIn('{BarrierStatus::WaitAgain}',node['signature'])
   self.assertEqual(node['parameters'][1]['default'],'{BarrierStatus::WaitAgain}')

 def test_public_and_private_overloads_are_independent(self):
  for a,b,left,right in [('clc.acquire','clc.acquire_stage',2,3),('clc.wait','clc.wait_stage',2,3),('clc.release','clc.release_stage',1,1)]:
   self.assertNotEqual(self.nodes[a]['entity_id'],self.nodes[b]['entity_id'])
   self.assertEqual(len(self.nodes[a]['parameters']),left);self.assertEqual(len(self.nodes[b]['parameters']),right)
   self.assertEqual(self.nodes[a]['access'],'public');self.assertEqual(self.nodes[b]['access'],'private')

 def test_query_is_issued_after_acquire_but_no_manual_commit_is_inserted(self):
  block='\n'.join(self.sched[437:451])
  self.assertLess(block.index('producer_acquire('),block.index('issue_clc_query('))
  self.assertIn('if (cute::elect_one_sync())',block)
  self.assertNotIn('producer_commit',block)
  self.assertFalse(any(e['source'] in {'clc.advance','clc.issue','host.kernel.operator'} and e['target']=='clc.commit' for e in self.edges.values()))

 def test_invalid_response_is_released_before_returning_work(self):
  block='\n'.join(self.sched[453:473])
  wait,read,release=[block.index(x) for x in ('consumer_wait(','work_tile_info_from_clc_response(','consumer_release(')]
  self.assertLess(wait,read);self.assertLess(read,release)
  self.assertLess(release,block.index('work_tile.is_valid()'))
  self.assertNotIn('if (work_tile.is_valid()',block[:release])

 def test_empty_wait_and_expected_bytes_are_distinct_from_full_wait(self):
  acquire='\n'.join(self.pipe[1093:1104]);consumer='\n'.join(self.pipe[1122:1129])
  self.assertLess(acquire.index('empty_barrier_ptr_'),acquire.index('arrive_and_expect_tx'))
  self.assertIn('lane_idx_ < cluster_size_',acquire)
  self.assertIn('full_barrier_ptr_[stage].wait(phase)',consumer)
  self.assertEqual(self.edges['clc.call.clc_acquire_stage.1103']['target'],'clc.expect_remote')
  self.assertNotEqual(self.nodes['clc.expect_remote']['entity_id'],self.nodes['contract.api.expect_tx']['entity_id'])

 def test_thread_roles_do_not_overcount_inactive_scheduler_warps(self):
  source='\n'.join(self.kernel[448:517])
  self.assertIn('(warp_category == WarpCategory::Sched) && is_first_cta_in_cluster',source)
  self.assertIn('NumSchedThreads + cluster_size *',source)
  self.assertIn('if (is_epi_load_needed)',source)
  for needs_epi,expected in ((False,800),(True,928)):
   participants={(0,'sched',lane) for lane in range(32)}
   for cta in range(4):
    for role,count in [('main',32),('mma',32),('epilogue',128),('epi_load',32 if needs_epi else 0)]:
     participants.update((cta,role,lane) for lane in range(count))
   self.assertEqual(len(participants),expected)
   self.assertEqual(len(participants),32+4*(32+128+32)+(4*32 if needs_epi else 0))
   self.assertNotEqual(len(participants),4*(32+32+128+32)+(4*32 if needs_epi else 0))

 def test_release_goes_to_producer_cta_and_tail_does_not_reset_bytes(self):
  release='\n'.join(self.pipe[1130:1135]);tail='\n'.join(self.pipe[1039:1050])
  self.assertIn('arrive(params_.producer_blockid)',release)
  self.assertIn('count < Stages',tail);self.assertIn('empty_barrier_ptr_',tail)
  self.assertNotIn('expect_tx',tail);self.assertNotIn('transaction_bytes',tail)

 def test_hardware_effect_is_not_a_synchronous_api_return(self):
  self.assertEqual(self.edges['clc.effect.issue']['relation'],'issues')
  self.assertEqual(self.nodes['clc.hw.query']['kind'],'hardware_event')
  self.assertIn('异步',self.edges['clc.effect.write_response']['condition'])
  self.assertIn('complete_tx硬件效果',self.edges['clc.effect.notify_full']['condition'])
  self.assertIn('NOT_IMPLEMENTED',self.edges['clc.effect.issue']['condition'])

 def test_each_new_callsite_has_exact_source_bytes_and_identity(self):
  calls=[self.edges[e['id']] for e in self.part['edges'] if e['relation']=='calls']
  self.assertEqual(len(calls),18);self.assertEqual(len({e['callsite']['callsite_id'] for e in calls}),18)
  for edge in calls:
   c=edge['callsite'];raw=(ROOT/'snapshot'/c['path']).read_bytes()
   self.assertEqual(raw[c['start_byte']:c['end_byte']].decode(),c['raw_source_expression'])

 def test_manual_completion_stays_out_of_normal_publish_graph(self):
  views={v['id']:v for v in self.part['views']}
  publish=set(views['clc.handoff.publish']['edge_ids']);manual=set(views['clc.handoff.manual']['edge_ids'])
  self.assertTrue(manual);self.assertFalse(manual&publish)
  self.assertEqual(set(e['id'] for e in self.part['edges']),set().union(*(set(v['edge_ids']) for v in views.values())))

 def test_all_new_callable_parameters_have_explanations(self):
  docs=json.loads((ROOT/'data/interface-docs.json').read_text())['nodes']
  for authored in self.part['nodes']:
   if authored['kind'] not in {'api','type'}:continue
   d=docs[authored['id']]
   self.assertTrue(d['summary']);self.assertTrue(d['execution'])
   self.assertEqual(len(d['parameters']['arguments']),len(self.nodes[authored['id']]['parameters']))

if __name__=='__main__':unittest.main()
