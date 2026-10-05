"""Actual21 target evidence and failclosed full-code/role/resource mutations."""
from pathlib import Path
import unittest,json,copy,re
from auditors import cluster_dsm_sass_v2 as audit
ROOT=Path(__file__).resolve().parents[3]
ACTUAL=ROOT/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/target-compile-s17-short-r2/diagnostics/cluster_dsm_multicast'
CONTRACT=ROOT/'microbench/gh200_resource_campaign/contracts/cluster_dsm_multicast_short_v1.draft.json'

class ClusterSassTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.text=(ACTUAL/'sass.txt').read_text();cls.compiler=(ACTUAL/'compile.stderr').read_text();cls.contract=json.loads(CONTRACT.read_text())
 def test_actual21_complete_identity_and_resource(self):
  rows=audit.audit_sass(self.text,self.contract);resources=audit.audit_compile_resources(self.compiler);self.assertEqual(len(rows),21);self.assertEqual(len(resources),21)
  for r in rows:
   if 'bulk_' in r['symbol']:self.assertEqual(len(r['receiver_wait_pcs']),4);self.assertNotEqual(r['item_loop_backedge'],r['issuer_election_retry'])
   if 'write_' in r['symbol']:self.assertEqual(len(r['timed_cluster_gates']),2);self.assertEqual(len(r['target_consumer_read_pcs']),8)
   if 'read_' in r['symbol']:self.assertEqual(len(r['timed_cluster_gates']),1)
 def test_each_critical_role_predicate_operand_and_128bit_rejects(self):
  mutations=0
  for symbol in audit.BASELINE:
   rows=audit.target_rows(self.text,symbol)
   selected=[n for n,x in enumerate(rows) if any(t in x['instruction'] for t in ('UCGABAR','BAR.SYNC','LD.E.STRONG','ST.E.STRONG','UBLKCP','SYNCS.','FENCE.','STG.E','SR_CLOCK','BRA','EXIT'))]
   self.assertTrue(selected)
   for n in selected:
    for kind in ('instruction','encoding'):
     bad=copy.deepcopy(rows)
     if kind=='instruction':bad[n]['instruction']='NOP'
     else:bad[n]['encoding'][1]='0x0000000000000000'
     with self.assertRaises(ValueError):audit.audit_target(bad,symbol)
     mutations+=1
   # Removing a post-EXIT row also changes complete-code identity.
   bad=copy.deepcopy(rows);bad.pop()
   with self.assertRaises(ValueError):audit.audit_target(bad,symbol)
  self.assertGreater(mutations,1000)
  print('S17 critical full-code mutations rejected:',mutations)
 def test_missing_duplicate_incomplete_and_changed_contract_rejects(self):
  symbol=next(iter(audit.BASELINE));part=next(p for p in re.split(r'Function\s*:\s*',self.text)[1:] if p.splitlines()[0].strip()==symbol)
  for text in (self.text.replace('Function : '+symbol,'Function : renamed_'+symbol),self.text+'\nFunction : '+part,self.text.replace('arch = sm_90a','arch = sm_90')):
   with self.assertRaises(ValueError):audit.audit_sass(text,self.contract)
  c=copy.deepcopy(self.contract);c['cases'].pop()
  with self.assertRaises(ValueError):audit.audit_sass(self.text,c)
  row=audit.target_rows(self.text,symbol)[0];bad=self.text.replace(row['encoding'][1],'0x0',1)
  with self.assertRaises(ValueError):audit.audit_sass(bad,self.contract)
 def test_resource_drift_stack_spill_missing_and_duplicate_rejects(self):
  for text in (self.compiler.replace('Used 32 registers','Used 33 registers',1),self.compiler.replace('16424 bytes smem','16432 bytes smem',1),self.compiler.replace('0 bytes stack frame','8 bytes stack frame',1),self.compiler.replace('0 bytes spill stores','4 bytes spill stores',1),self.compiler.replace("Compiling entry function 's17_bulk_all_c8'","Compiling entry function 'wrong_target'",1),self.compiler+self.compiler):
   with self.assertRaises(ValueError):audit.audit_compile_resources(text)
if __name__=='__main__':unittest.main()
