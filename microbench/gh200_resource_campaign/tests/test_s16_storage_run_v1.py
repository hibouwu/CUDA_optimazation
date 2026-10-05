"""Actual small CPU writer/host/tar tests; GPU identities below are fixtures."""
import unittest,tempfile,json,hashlib,copy
from pathlib import Path
from unittest.mock import patch
from common.suite_io import atomic_json,digest,sha
from runners import s16_storage_v1 as storage
from runners import s16_run_v1 as runner
from runners import s16_short_controller as rules
from s16_controller_archive_fixture import payload

class StorageRunTests(unittest.TestCase):
 def test_real_cpu_seal_and_complete_decode(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);point,run,data=payload(repo);identity={'fixture':'synthetic GPU registry; actual CPU host/tar'}
   manifest={n:hashlib.sha256(b).hexdigest() for n,b in data.items() if n!='validation_state.json'}
   data['validation_manifest.json']=json.dumps(manifest).encode()
   for n,b in data.items():p=repo/run/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
   entry={'coordinate':point,'state':'resident_passed','family_identity_sha256':digest(identity),'manifest_sha256':sha(repo/run/'validation_manifest.json')}
   suite=repo/rules.SUITE;atomic_json(suite/'tma_stage_request-short-v1-ledger.json',{'identity':identity,'entries':{str(point['index']):entry}})
   with patch.object(storage,'coordinate',return_value=point):
    out=repo/storage.PACK_NAMESPACE/'CPU-fixture';packed=storage.seal(repo,point['index'],out)
    self.assertEqual(packed['state'],'archive_remote_only');self.assertTrue((out/'remote.json').is_file())
    storage.verify_pack(repo,packed,'offhost',out/'fixture-offhost.json')
    origin=out/'fixture-offhost.origin.json';obj=json.loads(origin.read_text());obj['verifier']['host']='synthetic-distinct-CPU-fixture';atomic_json(origin,obj)
    admitted=storage.admit(repo,out/'entry.json',out/'fixture-offhost.json',origin)
    self.assertEqual(admitted['state'],'archived_verified')
    self.assertEqual(rules.next_action(repo/run,admitted,repo,point,identity),'archive_audit')
    with self.assertRaises(ValueError):storage.retire(repo,point['index'])
    with self.assertRaises(ValueError):storage.seal(repo,point['index'],out)
    with self.assertRaises(ValueError):storage.seal(repo,point['index'],repo/'wrong-family')
   (repo/run/'diagnostic_summary.json').write_text('{}')
   with self.assertRaises(ValueError):storage.inventory(repo,run)
 def test_namespace_symlink_and_finite_index(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);(repo/'escape').mkdir();base=repo/storage.PACK_NAMESPACE;base.parent.mkdir(parents=True);base.symlink_to(repo/'escape',target_is_directory=True)
   with self.assertRaises(ValueError):storage.pack_destination(repo,storage.PACK_NAMESPACE+'/x',new=True)
   with self.assertRaises(ValueError):storage.local_destination(repo,'../escape')
 def test_run_disabled_without_own_gate(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td)
   with self.assertRaises((ValueError,FileNotFoundError)):runner.execute(repo,0,1,repo/'device',repo/'quota',repo/'writer')
   for start,stop in ((0,5),(1,1),(35,37),(-1,1)):
    with self.assertRaises(ValueError):runner.execute(repo,start,stop,repo/'device',repo/'quota',repo/'writer')
 def test_max_first_and_no_failed_retry(self):
  with self.assertRaises(ValueError):runner.max_first_ready({'entries':{}})
  point={'case_id':rules.MAX_FIRST};entry={'coordinate':point,'state':'resident_passed'};runner.max_first_ready({'entries':{'0':entry}})
  entry['state']='failed'
  with self.assertRaises(ValueError):runner.max_first_ready({'entries':{'0':entry}})
 def test_invoke_failure_stops_and_fixed_cli(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);code=repo/rules.CORE_CAMPAIGN;suite=repo/rules.SUITE;point={'index':0,'case_id':rules.MAX_FIRST,'profile_id':'stage_short_1_2_5_33_v1','seed':3};root=repo/storage.run_name(point);ledger={'entries':{}};path=suite/'ledger.json'
   with patch.object(runner.subprocess,'run') as invoke:
    invoke.return_value.returncode=9
    self.assertEqual(runner.invoke(code,suite,root,point,'run',ledger,path,'hash'),9)
    argv=invoke.call_args.args[0];self.assertEqual(argv[-2:],['--seed','3']);self.assertIn('--adapter-manifest',argv)
   self.assertEqual(ledger['entries']['0']['state'],'failed')
   with self.assertRaises(ValueError):rules.next_action(root,ledger['entries']['0'],repo,point,{'x':1})
 def test_resident_bytes_no_symlink(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);p={'case_id':'fixture','profile_id':'fixed'};root=repo/storage.run_name(p);root.mkdir(parents=True);(root/'x').write_bytes(b'abc');self.assertEqual(runner.resident_bytes(repo,[p]),3);(root/'bad').symlink_to(root/'x')
   with self.assertRaises(ValueError):runner.resident_bytes(repo,[p])
if __name__=='__main__':unittest.main()
