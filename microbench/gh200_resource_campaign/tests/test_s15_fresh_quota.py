"""CPU-only real bounded quota IO; GPFS mount and provider are labeled fixtures."""
from pathlib import Path
import hashlib,json,os,sys,tempfile,time,unittest
from unittest.mock import patch
from common.suite_io import bounded,sha,read_json
from runners.s15_storage import query_quota
from runners.s15_short_controller import quota_headroom
HEADER='mmlsquota:user:HEADER:version:reserved:reserved:filesystemName:quotaType:id:name:blockUsage:blockQuota:blockLimit:blockInDoubt:blockGrace:filesUsage:filesQuota:filesLimit:filesInDoubt:filesGrace:remarks:fid:filesetname:\n'
def provider(uid,grace='none'):
 return HEADER+f'mmlsquota:user:0:1:::gpfs:USR:{uid}:CPU-fixture:100:150:20971520:7:7 days:100:1000000:1500000:0:{grace}:CPU-fixture:2:scratch:\n'
class FreshQuotaTests(unittest.TestCase):
 def fixture(self,repo,text=None,exit_code=0):
  original=Path.is_relative_to
  def mount(path,parent):
   if path==repo and parent==Path('/gpfs/scratch'):return True
   return original(path,parent)
  def execute(argv,cwd,stdout,stderr,timeout,**kwargs):
   self.assertEqual(argv,['/usr/lpp/mmfs/bin/mmlsquota','-e','-u',str(os.geteuid()),'-Y','gpfs']);self.assertEqual(timeout,30)
   script='import sys;sys.stdout.write('+repr(provider(os.geteuid()) if text is None else text)+');sys.stderr.write("CPU quota fixture\\n");sys.exit('+str(exit_code)+')'
   return bounded([sys.executable,'-c',script],cwd,stdout,stderr,timeout,**kwargs)
  return patch('pathlib.Path.is_relative_to',new=mount),patch('runners.s15_storage.bounded',side_effect=execute)
 def capture(self,folder):return {p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}
 def test_same_alias_two_queries_preserve_complete_first_receipt(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);alias=repo/'quota.json';mount,execute=self.fixture(repo)
   with mount,execute:
    first=query_quota(repo,alias);folder=repo/first['query_stdout']['path'];folder=folder.parent;before=self.capture(folder)
    second=query_quota(repo,alias);self.assertNotEqual(first['query_stdout']['path'],second['query_stdout']['path']);self.assertEqual(before,self.capture(folder));self.assertEqual(read_json(alias),second)
    for result in (first,second):
     query=(repo/result['query_stdout']['path']).parent;self.assertTrue(all((query/n).is_file() for n in ('stdout','stderr','process.json','result.json')));self.assertEqual(read_json(query/'result.json'),result);self.assertEqual(read_json(query/'process.json')['returncode'],0)
    self.assertGreater(quota_headroom(repo,alias,{'execution_uid':os.geteuid()}),0)
 def test_failed_process_and_bad_parse_do_not_replace_current_alias(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);alias=repo/'quota.json';mount,execute=self.fixture(repo)
   with mount,execute:query_quota(repo,alias)
   before=sha(alias);queries=repo/'quota.json.queries';successful=self.capture(queries)
   for text,code in [(None,13),(provider(os.geteuid()+1),0),(provider(os.geteuid(),'expired'),0)]:
    mount,execute=self.fixture(repo,text,code)
    with mount,execute:
     with self.assertRaises(ValueError):query_quota(repo,alias)
    self.assertEqual(sha(alias),before)
   for name,digest in successful.items():self.assertEqual(sha(queries/name),digest)
   rejected=[p for p in queries.glob('*/result.json') if read_json(p)['status']=='quota_query_rejected'];self.assertEqual(len(rejected),3)
   self.assertTrue(all((p.parent/'process.json').is_file() and (p.parent/'stdout').is_file() and (p.parent/'stderr').is_file() for p in rejected))
 def test_forced_identity_collision_never_overwrites_old_query(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);alias=repo/'quota.json';mount,execute=self.fixture(repo);now=time.time_ns()
   with mount,execute,patch('runners.s15_storage.time.time_ns',return_value=now),patch('runners.s15_storage.uuid.uuid4',return_value=type('Nonce',(),{'hex':'fixedCPUfixture'})()):
    query_quota(repo,alias);before=self.capture(repo)
    with self.assertRaises(FileExistsError):query_quota(repo,alias)
    self.assertEqual(before,self.capture(repo))
 def test_stale_alias_refused_then_next_fresh_query_recovers(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);alias=repo/'quota.json';mount,execute=self.fixture(repo)
   with mount,execute:
    first=query_quota(repo,alias);old=dict(first,queried_unix_ns=time.time_ns()-61_000_000_000);alias.write_text(json.dumps(old))
    with self.assertRaisesRegex(ValueError,'freshness'):quota_headroom(repo,alias,{'execution_uid':os.geteuid()})
    new=query_quota(repo,alias);self.assertNotEqual(first['query_stdout']['path'],new['query_stdout']['path']);self.assertGreater(quota_headroom(repo,alias,{'execution_uid':os.geteuid()}),0)
 def test_consecutive_configuration_refreshes_share_alias_without_shared_raw(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);alias=repo/'quota.json';mount,execute=self.fixture(repo);paths=[]
   with mount,execute:
    for coordinate in (101,0,1,2):
     result=query_quota(repo,alias);self.assertGreater(quota_headroom(repo,alias,{'execution_uid':os.geteuid()}),0);paths.append(result['query_stdout']['path'])
   self.assertEqual(len(paths),len(set(paths)));self.assertEqual(len(list((repo/'quota.json.queries').glob('*/process.json'))),4)
 def test_query_archive_symlink_rejected_before_provider(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);repo=base/'repo';repo.mkdir();outside=base/'outside';outside.mkdir();(repo/'quota.json.queries').symlink_to(outside,target_is_directory=True);mount,execute=self.fixture(repo)
   with mount,execute as process:
    with self.assertRaisesRegex(ValueError,'archive'):query_quota(repo,repo/'quota.json')
    self.assertEqual(process.call_count,0)
   self.assertEqual(list(outside.iterdir()),[])
if __name__=='__main__':unittest.main()
