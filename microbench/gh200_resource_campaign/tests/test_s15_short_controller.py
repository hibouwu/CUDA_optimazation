import unittest,json,tempfile,tarfile,io,hashlib,copy
from pathlib import Path
from runners.s15_short_controller import coordinates,artifact_bytes,next_action,validate_archived,SUITE

ROOT=Path(__file__).resolve().parents[1]

class S15ControllerTests(unittest.TestCase):
    def setUp(self):
        self.contract=json.loads((ROOT/'contracts/tma_tensor_2d_short_v1.draft.json').read_text());self.profiles=json.loads((ROOT/'contracts/tma_tensor_2d_validation_profiles_v1.draft.json').read_text());self.device={'sms':132,'smem_per_sm_bytes':233472,'registers_per_sm':65536}
    def test204_paired_no_extra_seed_scan(self):
        points=coordinates(self.contract,self.profiles);self.assertEqual(len(points),204);self.assertEqual([(r['seed'],r['profile_id']) for r in points[:3]],[(p['required_seed'],p['id']) for p in self.profiles['profiles']])
    def test_artifact_sum_includes_both_capture_and_padding(self):
        c=next(c for c in self.contract['cases'] if c['id']=='gmem_to_smem_64kib_padding_none_all_gpu');r=artifact_bytes(c,self.profiles['profiles'][2],self.device);self.assertEqual(r['blocks_upper_bound'],396);self.assertEqual(r['logical_bytes'],396*33*65536);self.assertEqual(r['physical_bytes'],r['logical_bytes']);self.assertEqual(r['global_ring_padding_guard_bytes'],396*32*256*272+256);self.assertEqual(r['artifact_bytes_upper_bound'],2595257664)
    def test_no_assumed_actual_occupancy(self):
        c=next(c for c in self.contract['cases'] if c['id']=='gmem_to_smem_32kib_continuous_sw128_all_gpu');r=artifact_bytes(c,self.profiles['profiles'][2],self.device);self.assertFalse(r['actual_occupancy_claim']);self.assertLessEqual(r['blocks_upper_bound'],528)
    def test_source_directory_missing_with_passed_ledger_never_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'absent';coordinate=coordinates(self.contract,self.profiles)[0]
            with self.assertRaises(ValueError):next_action(root,{'coordinate':coordinate,'state':'resident_passed'},Path(tmp),coordinate)
    def test_pending_without_attempt_can_resume_but_started_cannot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();(root/'attempts').mkdir();(root/'validation_state.json').write_text('{"status":"checkpoint"}');c=coordinates(self.contract,self.profiles)[0]
            self.assertEqual(next_action(root,{'coordinate':c,'state':'started_unknown'},Path(tmp),c),'resume')
            (root/'attempts/attempt_00').mkdir()
            with self.assertRaises(ValueError):next_action(root,{'coordinate':c,'state':'started_unknown'},Path(tmp),c)
    def test_partial_initialization_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();c=coordinates(self.contract,self.profiles)[0]
            with self.assertRaises(ValueError):next_action(root,None,Path(tmp),c)

    def test_archive_run_identity_and_receipt_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);c=coordinates(self.contract,self.profiles)[0]
            run=SUITE+'/tma_tensor_2d/short-v1-'+c['case_id']+'--'+c['profile_id']
            payload={'validation_spec.json':json.dumps({'stage':'S15','family':'tma_tensor_2d','kind':'case_diagnostic',**{k:c[k] for k in ('case_id','profile_id','seed')}}).encode(),'diagnostic_summary.json':b'{"status":"case_diagnostic_passed"}','attempts/attempt_00/raw.jsonl':json.dumps({'type':'validation',**{k:c[k] for k in ('case_id','profile_id','seed')}}).encode()}
            digest=lambda data:hashlib.sha256(data).hexdigest()
            manifest={k:digest(v) for k,v in payload.items()};payload['validation_manifest.json']=json.dumps(manifest).encode();payload['validation_state.json']=b'{"status":"case_diagnostic_passed"}'
            members={run+'/'+k:{'bytes':len(v),'sha256':digest(v)} for k,v in payload.items()}
            with tarfile.open(repo/'pack.tar.xz','w:xz') as out:
                for name,data in payload.items():
                    info=tarfile.TarInfo(run+'/'+name);info.size=len(data);out.addfile(info,io.BytesIO(data))
            ah=digest((repo/'pack.tar.xz').read_bytes());mh=digest(payload['validation_manifest.json'])
            index={'schema_version':1,'namespace':'repository_relative_v1','archive_sha256':ah,'archive_bytes':(repo/'pack.tar.xz').stat().st_size,'members':members,'roots':{'run':run+'/validation_manifest.json'}}
            closure={'schema_version':1,'run_path':run,'manifest_sha256':mh,'members':members}
            entry={'coordinate':c,'state':'archived_verified','run_path':run,'manifest_sha256':mh}
            for key,name,obj in [('index','index.json',index),('closure','closure.json',closure)]:
                (repo/name).write_text(json.dumps(obj));entry[key]={'path':name,'sha256':digest((repo/name).read_bytes())}
            entry['archive']={'path':'pack.tar.xz','sha256':ah}
            for key,location in [('archive_verification','remote'),('offhost_receipt','offhost')]:
                obj={'schema_version':1,'status':'full_members_verified','location':location,'archive_sha256':ah,'index_sha256':entry['index']['sha256'],'closure_sha256':entry['closure']['sha256'],'run_path':run,'manifest_sha256':mh,'members':members,'member_count':len(members),'uncompressed_bytes':sum(m['bytes'] for m in members.values()),'verification':'complete_decode_all_members_sha256'}
                name=key+'.json';(repo/name).write_text(json.dumps(obj));entry[key]={'path':name,'sha256':digest((repo/name).read_bytes())}
            self.assertEqual(validate_archived(entry,repo,c),'archived_verified')
            for field,value in [('manifest_sha256','0'*64),('run_path',run+'wrong')]:
                bad=copy.deepcopy(entry);bad[field]=value
                with self.assertRaises(ValueError):validate_archived(bad,repo,c)
            for field,value in [('members',{}),('member_count',1),('index_sha256','0'*64),('verification','placeholder'),('run_path',run+'wrong')]:
                original=json.loads((repo/'offhost_receipt.json').read_text());bad_receipt=dict(original);bad_receipt[field]=value
                (repo/'bad_receipt.json').write_text(json.dumps(bad_receipt));bad=copy.deepcopy(entry);bad['offhost_receipt']={'path':'bad_receipt.json','sha256':digest((repo/'bad_receipt.json').read_bytes())}
                with self.assertRaises(ValueError):validate_archived(bad,repo,c)

if __name__=='__main__':unittest.main()
