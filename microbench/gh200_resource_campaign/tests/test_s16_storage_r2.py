"""Regression for independently reproduced budget provenance and pre-IO escape."""
import unittest,tempfile,json,copy
from pathlib import Path
from unittest.mock import patch
from common.suite_io import sha
from common import s16_quota_v2 as quota
from runners import s16_storage_v1 as storage
from runners import s16_run_v1 as run
from runners.s16_short_controller import SUITE
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]

class StorageR2Tests(unittest.TestCase):
 def test_parent_escape_and_symlink_rejected_before_any_query(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td)/'repo';repo.mkdir();out=repo/'..'/'outside'/'quota.json'
   with patch.object(quota,'original_query_quota') as producer:
    with self.assertRaises(ValueError):quota.query_quota(repo,out)
    producer.assert_not_called()
   with patch.object(storage,'controller_gates') as gates:
    with self.assertRaises(ValueError):storage.query_device(repo,repo/'binary',out)
    gates.assert_not_called()
   self.assertFalse((repo.parent/'outside').exists())
   (repo/'link').symlink_to(repo.parent,target_is_directory=True)
   with self.assertRaises(ValueError):quota.query_destination(repo,repo/'link'/'x')
   self.assertFalse((repo.parent/'x').exists())
 def test_two_valid_calls_delegate_distinct_refresh_without_path_escape(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);out=repo/'queries'/'current.json'
   with patch.object(quota,'original_query_quota',return_value={'fixture':'CPU wrapper only'}) as provider:
    quota.query_quota(repo,out);quota.query_quota(repo,out)
    self.assertEqual(provider.call_count,2);self.assertEqual(provider.call_args.args,(repo.resolve(),out))
 def test_arbitrary_selfreported_package_rejected_even_with_actual_gate(self):
  with tempfile.TemporaryDirectory() as td:
   package=Path(td);(package/'source_manifest.json').write_text('{}')
   grant={'evidence_root':SUITE+'/implementation/s16-arm-writer-package-a/actual-writer-evidence','producer_root':'/gpfs/scratch/actual','source_manifest_sha256':'0'*64,'collection':{'path':'x','sha256':'0'*64},'raw_archive':{'path':'y','sha256':'0'*64}}
   with patch.object(run,'validate_gate',return_value={'writer_budget_authorization':grant,'gate_files':{}}):
    with self.assertRaisesRegex(ValueError,'arbitrary CLI package rejected'):run.writer_budget(package,ROOT,1)
   with self.assertRaises((ValueError,FileNotFoundError)):run.writer_budget(package,ROOT,1)
 def test_actual_collected_budget_relocation_and_missing_bound_metadata(self):
  suite=REPO/SUITE;package=suite/'implementation/s16-arm-writer-package-a/actual-writer-evidence'
  if not package.is_dir():self.skipTest('actual ARM evidence not present')
  base=suite/'implementation/s16-arm-writer-package-a';collection=base/'actual-writer-collection.json';archive=base/'s16-arm-writer-731153-complete.delta4.tar.xz'
  grant={'evidence_root':str(package.relative_to(REPO)),'producer_root':'/gpfs/scratch/hibouwu/gh200_resource_campaign/20261002-v2-s16-arm-writer-a','source_manifest_sha256':sha(package/'source_manifest.json'),'collection':{'path':str(collection.relative_to(REPO)),'sha256':sha(collection)},'raw_archive':{'path':str(archive.relative_to(REPO)),'sha256':sha(archive)}}
  files={str(p.relative_to(REPO)):sha(p) for p in package.rglob('*') if p.is_file()};files.update({str(collection.relative_to(REPO)):sha(collection),str(archive.relative_to(REPO)):sha(archive)})
  fake_gate={'writer_budget_authorization':grant,'gate_files':files}
  # Only the gate object is mocked to test its consumer prior to independent sign;
  # metadata/stdout/binary/raw archive refer to actual collected731153, no execution.
  with patch.object(run,'validate_gate',return_value=fake_gate),patch.object(run,'sha',side_effect=lambda p:'mock-review' if str(p).endswith('S16-ARM-writer-result-B-review.json') else sha(p)):
   result=run.writer_budget(package,ROOT,4490994944);self.assertEqual(result['covered_artifact_bytes'],4490994944)
   bad=copy.deepcopy(fake_gate);bad['gate_files'].pop(str((package/'writer_budget/writer').relative_to(REPO)))
   with patch.object(run,'validate_gate',return_value=bad):
    with self.assertRaises(ValueError):run.writer_budget(package,ROOT,1)

 def test_changed_closure_cannot_publish_bound_success_receipt(self):
  from s16_controller_archive_fixture import payload,pack
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);point,namespace,data=payload(repo)
   (repo/storage.PACK_NAMESPACE).mkdir(parents=True)
   entry=pack(repo,point,namespace,data,{'fixture':'CPU'},storage.PACK_NAMESPACE+'/fixture')
   closure=repo/entry['closure']['path'];obj=json.loads(closure.read_text());obj['run_path']='unrelated-run';closure.write_text(json.dumps(obj))
   output=repo/storage.PACK_NAMESPACE/'unpublished.json'
   with self.assertRaises(ValueError):storage.verify_pack(repo,entry,'remote',output)
   self.assertFalse(output.exists());self.assertFalse(output.with_suffix('.origin.json').exists())

 def test_verify_outputs_outside_symlink_archive_and_existing_rejected(self):
  from s16_controller_archive_fixture import payload,pack
  with tempfile.TemporaryDirectory() as td:
   parent=Path(td);repo=parent/'repo';repo.mkdir();point,namespace,data=payload(repo)
   (repo/storage.PACK_NAMESPACE).mkdir(parents=True)
   entry=pack(repo,point,namespace,data,{'fixture':'CPU'},storage.PACK_NAMESPACE+'/fixture')
   archive=repo/entry['archive']['path'];before=sha(archive)
   outside=parent/'outside';outside.mkdir();link=repo/storage.PACK_NAMESPACE/'link';link.symlink_to(outside,target_is_directory=True)
   used=repo/storage.PACK_NAMESPACE/'existing.json';used.write_text('old immutable receipt')
   originonly=repo/storage.PACK_NAMESPACE/'origin-only.json';originonly.with_suffix('.origin.json').write_text('old origin')
   paths=[outside/'receipt.json',repo/'..'/'outside'/'parent.json',link/'linked.json',archive,used,originonly]
   for output in paths:
    with self.assertRaises(ValueError):storage.verify_pack(repo,entry,'remote',output)
   self.assertEqual(sha(archive),before);self.assertEqual(used.read_text(),'old immutable receipt');self.assertFalse(originonly.exists());self.assertEqual(list(outside.iterdir()),[])
