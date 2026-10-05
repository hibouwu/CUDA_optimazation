"""Immutable quota refresh regression; no actual GPFS/GPU calls on CPU."""
import hashlib,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from common import s16_quota_v1 as quota
HEADER='mmlsquota:user:HEADER:version:reserved:reserved:filesystemName:quotaType:id:name:blockUsage:blockQuota:blockLimit:blockInDoubt:blockGrace:filesUsage:filesQuota:filesLimit:filesInDoubt:filesGrace:remarks:fid:filesetname:\n'
def row(used):return 'mmlsquota:user:0:1:::gpfs:USR:'+str(os.geteuid())+':fixture:'+str(used)+':150:20971520:7:7 days:100:1000000:1500000:0:none:fixture:2:scratch:\n'
class QuotaTests(unittest.TestCase):
 def test_two_refreshes_same_alias_preserve_old_and_budget_each_config(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);output=repo/'current_quota.json';count=[0]
   def fake(argv,cwd,stdout,stderr,timeout):
    count[0]+=1
    with stdout.open('x') as stream:stream.write(HEADER+row(100*count[0]))
    stderr.open('x').close();return {'returncode':0,'cleanup_confirmed':True,'timed_out':False}
   with patch.object(quota,'quota_scope',lambda r,o:None),patch.object(quota,'bounded',fake):
    first=quota.query_quota(repo,output);first_stdout=repo/first['query_stdout']['path'];old=first_stdout.read_bytes();first_receipt=first_stdout.parent/'quota.json';old_receipt=first_receipt.read_bytes();second=quota.query_quota(repo,output);third=quota.query_quota(repo,output)
   self.assertEqual(count[0],3);self.assertNotEqual(first['query_stdout']['path'],second['query_stdout']['path']);self.assertNotEqual(second['query_stdout']['path'],third['query_stdout']['path']);self.assertEqual(first_stdout.read_bytes(),old);self.assertEqual(first_receipt.read_bytes(),old_receipt);self.assertEqual(hashlib.sha256(old).hexdigest(),first['query_stdout']['sha256']);self.assertEqual(quota.read_json(output),third);self.assertEqual(len(list((repo/'current_quota.queries').iterdir())),3)
   for receipt in (first,second,third):self.assertGreater(receipt['hard_bytes']-receipt['used_bytes']-receipt['reserve_bytes'],4490994944*2)
 def test_parser_units_uid_fileset_and_in_doubt(self):
  result=quota.quota_rows(HEADER+row(100),os.geteuid(),'scratch');self.assertEqual(result['used_bytes'],107*1024)
  for text,uid,fileset in [(row(100),os.geteuid(),'scratch'),(HEADER+row(100),-1,'scratch'),(HEADER+row(100),os.geteuid(),'home'),(HEADER+row(100)+row(100),os.geteuid(),'scratch')]:
   with self.assertRaises(ValueError):quota.quota_rows(text,uid,fileset)
 def test_failure_does_not_publish_current_alias(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);output=repo/'current.json'
   def fake(argv,cwd,stdout,stderr,timeout):
    stdout.open('x').close()
    with stderr.open('x') as stream:stream.write('failed')
    return {'returncode':1,'cleanup_confirmed':True,'timed_out':False}
   with patch.object(quota,'quota_scope',lambda r,o:None),patch.object(quota,'bounded',fake):
    with self.assertRaises(ValueError):quota.query_quota(repo,output)
   self.assertFalse(output.exists());self.assertTrue(list((repo/'current.queries').rglob('process.json')));results=list((repo/'current.queries').rglob('quota.json'));self.assertEqual(len(results),1);self.assertEqual(quota.read_json(results[0])['status'],'quota_query_failed')
if __name__=='__main__':unittest.main()
