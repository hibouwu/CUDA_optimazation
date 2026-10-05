import unittest,tempfile,json,shutil,hashlib,subprocess
from unittest.mock import patch
from pathlib import Path
from runners.s15_storage import quota_rows,inventory,seal,verify_pack,retire,run_name,coordinate,collect,local_destination
from runners.s15_short_controller import SUITE,CORE_CAMPAIGN,CONTRACT,PROFILES

CODE=Path(__file__).resolve().parents[1]
HEADER='mmlsquota:user:HEADER:version:reserved:reserved:filesystemName:quotaType:id:name:blockUsage:blockQuota:blockLimit:blockInDoubt:blockGrace:filesUsage:filesQuota:filesLimit:filesInDoubt:filesGrace:remarks:fid:filesetname:\n'
ROW='mmlsquota:user:0:1:::gpfs:USR:100800:hibouwu:100:150:200:7:7 days:100:1000000:1500000:0:none:romeo:2:scratch:\n'

class StorageTests(unittest.TestCase):
    def test_quota_header_uid_fileset_and_indoubt(self):
        self.assertEqual(quota_rows(HEADER+ROW,100800,'scratch'),{'hard_bytes':200*1024,'used_bytes':107*1024})
        for text,uid,fileset in [(ROW,100800,'scratch'),(HEADER+ROW,1,'scratch'),(HEADER+ROW,100800,'home'),(HEADER+ROW+ROW,100800,'scratch'),(HEADER+ROW.replace(':200:',':0:'),100800,'scratch')]:
            with self.assertRaises(ValueError):quota_rows(text,uid,fileset)
    def fixture(self,repo):
        code=repo/CORE_CAMPAIGN
        for file in (CONTRACT,PROFILES):
            (code/file).parent.mkdir(parents=True,exist_ok=True)
            if (CODE/file).resolve()!=(code/file).resolve():shutil.copyfile(CODE/file,code/file)
        c=coordinate(repo,0);root=repo/run_name(c);root.mkdir(parents=True)
        data={'validation_spec.json':{'stage':'S15','family':'tma_tensor_2d','kind':'case_diagnostic',**{k:c[k] for k in ('case_id','profile_id','seed')}},'diagnostic_summary.json':{'status':'case_diagnostic_passed'},'attempts/attempt_00/receipt.json':{'cleanup_confirmed':True}}
        for name,obj in data.items():p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))
        (root/'attempts/attempt_00/raw.jsonl').write_text(json.dumps({'type':'validation',**{k:c[k] for k in ('case_id','profile_id','seed')}})+'\n')
        (root/'payload.u16le').write_bytes(bytes(range(256))*4096)
        manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
        (root/'validation_manifest.json').write_text(json.dumps(manifest));(root/'validation_state.json').write_text('{"status":"case_diagnostic_passed"}')
        return c,root
    def test_actual_seal_full_decode_twice_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);c,root=self.fixture(repo);before=inventory(repo,run_name(c));entry=seal(repo,0,repo/'sealed')
            result=verify_pack(repo,entry,'offhost',repo/'offhost.json');self.assertEqual(result['member_count'],len(before));self.assertEqual(result['members'],before);self.assertEqual(inventory(repo,run_name(c)),before)
            self.assertFalse(entry['state']=='archived_verified');self.assertTrue((root/'payload.u16le').exists())
            with self.assertRaises(ValueError):seal(repo,0,repo/'sealed')
    def test_inventory_rejects_extra_changed_link_and_unclean(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);c,root=self.fixture(repo);(root/'extra').write_text('oops')
            with self.assertRaises(ValueError):inventory(repo,run_name(c))
            (root/'extra').unlink();(root/'link').symlink_to(root/'payload.u16le')
            with self.assertRaises(ValueError):inventory(repo,run_name(c))
            (root/'link').unlink();(root/'payload.u16le').write_bytes(b'changed')
            with self.assertRaises(ValueError):inventory(repo,run_name(c))
    def test_local_collection_symlink_zero_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);repo=base/'source';repo.mkdir();c,root=self.fixture(repo);entry=seal(repo,0,repo/'cpu-sealed')
            names=['cpu-sealed/entry.json']+[entry[k]['path'] for k in ['archive','index','closure','archive_verification']]+['cpu-sealed/remote.origin.json']
            records={n:{'bytes':(repo/n).stat().st_size,'sha256':hashlib.sha256((repo/n).read_bytes()).hexdigest()} for n in names}
            destination=base/'destination';destination.mkdir();outside=base/'outside';outside.mkdir();(destination/'cpu-sealed').symlink_to(outside,target_is_directory=True)
            response=subprocess.CompletedProcess([],0,json.dumps({'files':records,'entry':entry}),'')
            with patch('runners.s15_storage.subprocess.run',return_value=response) as network:
                with self.assertRaisesRegex(ValueError,'symlink'):collect(Path('/gpfs/scratch/hibouwu/source'),Path('cpu-sealed/entry.json'),destination)
                self.assertEqual(network.call_count,1)
            self.assertEqual(list(outside.iterdir()),[])
    def test_local_destination_rejects_escape_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);(repo/'present').write_text('old')
            for name in ('../escape','/absolute','present'):
                with self.assertRaises(ValueError):local_destination(repo,name,new=True)
            parent=repo/'regular';parent.write_text('file')
            with self.assertRaises(ValueError):local_destination(repo,'regular/child',new=True)
    def test_expired_inode_grace_is_rejected(self):
        row=ROW.replace(':100:1000000:1500000:0:none:',':1100000:1000000:1500000:0:expired:')
        with self.assertRaisesRegex(ValueError,'inode grace'):quota_rows(HEADER+row,100800,'scratch')
    def test_retire_canonical_local_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):retire(Path(tmp),0)
if __name__=='__main__':unittest.main()
