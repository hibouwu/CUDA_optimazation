"""Read-only ABI simulations; real signed binary is never executed by CPU tests."""
import unittest,tempfile,json,copy,shutil,contextlib
from pathlib import Path
from unittest.mock import patch
from common.suite_io import read_json,sha
from runners import s17_query_v1 as query_module,environment
from test_s17_short_plan_v1 import DEVICE,rows,ROOT
REPO=ROOT.parents[1]
BINARY=REPO/query_module.SUITE/'implementation/target-compile-s17-short-r2/diagnostics/cluster_dsm_multicast/probe'
REVIEW=REPO/query_module.SUITE/query_module.SOURCE_REVIEW

class QueryTests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.repo=Path(self.temp.name);dest=self.repo/query_module.SUITE/query_module.SOURCE_REVIEW;dest.parent.mkdir(parents=True);shutil.copy2(REVIEW,dest);self.binary=self.repo/query_module.SUITE/'implementation/target-compile-s17-short-r2/diagnostics/cluster_dsm_multicast/probe';self.binary.parent.mkdir(parents=True);shutil.copy2(BINARY,self.binary);self.output=self.repo/'deployment_receipts/query-CPUfixture/cluster-query.json';self.env={'job':'CPUfixture','uuid':DEVICE['uuid'],'name':'GH200 CPUfixture','driver':'CPUfixture','compiler':{},'tools':{},'execution_uid':0};self.payload=[DEVICE,*rows()];self.failed=False
 def tearDown(self):self.temp.cleanup()
 def bounded(self,argv,cwd,stdout,stderr,timeout,uuid):
  self.assertEqual(argv,[str(self.binary),'cluster-device']);self.assertEqual(timeout,30);stdout.write_bytes(('\n'.join(json.dumps(x) for x in self.payload)+'\n').encode());stderr.write_bytes(b'')
  return {'returncode':2 if self.failed else 0,'timed_out':False,'cleanup_confirmed':True,'argv':argv,'timeout_seconds':30,'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr),'CPU_fixture':True}
 def call(self):
  with patch.object(query_module,'validate_gate',return_value=read_json(REVIEW)),patch.object(environment,'inspect_allocation',return_value=self.env),patch.object(environment,'budget'),patch.object(query_module,'gpu_lock',side_effect=lambda x:contextlib.nullcontext()),patch.object(query_module,'bounded',side_effect=self.bounded):return query_module.query(self.repo,self.binary,self.output)
 def test_two_refreshes_preserve_immutable_original_outputs(self):
  first=self.call();second=self.call();self.assertNotEqual(first['stdout']['path'],second['stdout']['path']);self.assertTrue((self.repo/first['stdout']['path']).is_file());self.assertEqual(second['GPU_target_launches'],0);self.assertIs(second['GPU_numerical_qualified'],False)
 def test_process_failure_preserves_old_alias_and_error_evidence(self):
  first=self.call();before=self.output.read_bytes();self.failed=True
  with self.assertRaises(ValueError):self.call()
  self.assertEqual(self.output.read_bytes(),before);self.assertEqual(len(list(self.output.parent.rglob('process.json'))),2)
 def test_actual_query_errors_not_hidden_or_numeric_pass(self):
  self.payload[1]['cuda_error']=801;result=self.call();self.assertEqual(result['capability_statuses']['s17_local_read_c2']['status'],'query_error_before_target');self.assertIs(result['GPU_numerical_qualified'],False)
 def test_namespace_and_binary_or_UUID_drift_refused(self):
  for output in (self.repo/'..'/'outside.json',self.repo/'source.py'):
   with self.assertRaises(ValueError):query_module.query(self.repo,BINARY,output)
  self.payload[0]=dict(DEVICE,uuid='GPU-11111111-1111-1111-1111-111111111111')
  with self.assertRaises(ValueError):self.call()
  self.assertFalse(self.output.exists())
if __name__=='__main__':unittest.main()
