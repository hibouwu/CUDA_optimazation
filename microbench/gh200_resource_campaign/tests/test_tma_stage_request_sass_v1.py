"""Actual18 B2 full-code/resource rejection tests; no GPU or compiler execution."""
import copy,json,re,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auditors.tma_stage_request_sass_v1 import audit_sass,audit_target,target_rows,audit_compile_resources
from auditors.tma_stage_request_sass_baseline_v1 import BASELINE
REPO=ROOT.parents[1]
BASE=REPO/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/target-compile-s16-short-r2/diagnostics/tma_stage_request'
CONTRACT=json.loads((ROOT/'contracts/tma_stage_request_short_v1.draft.json').read_text())
class SassTests(unittest.TestCase):
 def test_actual18_full_identity_lifecycle_and_resources(self):
  result=audit_sass((BASE/'sass.txt').read_text(),CONTRACT);resources=audit_compile_resources((BASE/'compile.stderr').read_text());self.assertEqual(len(result),18);self.assertEqual(len(resources),18)
  for row in result:
   self.assertTrue(row['static_site_count_is_not_dynamic_count']);self.assertEqual(row['dynamic_transport_requests'],'I*R per CTA');self.assertFalse(row['GPU_qualification'])
 def test_critical_actual_instruction_predicate_operand_encoding_negatives(self):
  text=(BASE/'sass.txt').read_text();count=0
  for symbol in BASELINE:
   rows=target_rows(text,symbol);record=audit_target(rows,symbol);pcs=set(record['static_request_site_pcs']+record['primary_clock_pcs']+list(record['steady_CTA_roles'].values())+record['item_loop_backedge']+list(record['tail_CTA_roles'].values()))
   for key in ('expect_tx_pc','arrive_token_pc','commit_pc','steady_full_wait_pc','tail_full_wait_pc'):
    if key in record:pcs.add(record[key])
   for key in ('mbarrier_init_pcs','mbarrier_invalidate_pcs','steady_acquire_pcs','tail_acquire_pcs','full_visibility_pcs','global_proxy_pcs'):
    pcs.update(record.get(key,[]))
   for index,row in enumerate(rows):
    if row['pc'] not in pcs:continue
    for mutation in ('instruction','predicate','encoding'):
     altered=copy.deepcopy(rows)
     if mutation=='instruction':altered[index]['instruction']+=' RZ'
     elif mutation=='predicate':altered[index]['instruction']='@P7 '+altered[index]['instruction']
     else:altered[index]['encoding'][1]='0x0000000000000000'
     with self.assertRaises(ValueError):audit_target(altered,symbol)
     count+=1
  self.assertGreater(count,900);print('S16 actual instruction mutations rejected:',count)
 def test_target_set_incomplete_encoding_contract_and_resource_negatives(self):
  text=(BASE/'sass.txt').read_text();symbol=next(iter(BASELINE))
  with self.assertRaises(ValueError):audit_sass(text.replace('Function : '+symbol,'Function : '+symbol+'_wrong',1),CONTRACT)
  with self.assertRaises(ValueError):audit_sass(text+text,CONTRACT)
  with self.assertRaises(ValueError):audit_sass(text+ '\nFunction : injected\n',CONTRACT)
  with self.assertRaises(ValueError):audit_sass(text.replace('/* 0x','/* broken0x',1),CONTRACT)
  contract=copy.deepcopy(CONTRACT);contract['cases'].pop()
  with self.assertRaises(ValueError):audit_sass(text,contract)
  compile_text=(BASE/'compile.stderr').read_text()
  for old,new in [('Used 32 registers','Used 31 registers'),('used 1 barriers','used 2 barriers'),('0 bytes stack frame','8 bytes stack frame'),('0 bytes spill stores','4 bytes spill stores'),('0 bytes spill loads','4 bytes spill loads')]:
   with self.assertRaises(ValueError):audit_compile_resources(compile_text.replace(old,new,1))
if __name__=='__main__':unittest.main()
