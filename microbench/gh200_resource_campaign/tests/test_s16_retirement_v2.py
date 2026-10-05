"""Real small CPU pack/retirement simulations. No producer-node/GPU proof."""
from pathlib import Path
import json,os,shutil,socket,tarfile,tempfile,unittest
from unittest.mock import patch
from runners import s16_retirement_v2 as tool
from common.suite_io import sha,digest
from s16_controller_archive_fixture import payload,pack
from runners.s16_short_controller import CONTRACT,PROFILES,family_identity
ROOT=Path(__file__).resolve().parents[1]
REAL_SCRATCH_REPO=tool.scratch_repo
REAL_CPU_ALLOCATION=tool.cpu_allocation

class RetirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=tempfile.TemporaryDirectory();cls.point,cls.run_path,cls.data=payload(Path(cls.original.name))
    @classmethod
    def tearDownClass(cls):cls.original.cleanup()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.repo=Path(self.tmp.name)/'repo';self.repo.mkdir()
        for name in (CONTRACT,PROFILES):
            dest=self.repo/'microbench/gh200_resource_campaign'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        self.device=json.loads(self.data['environment/device.json']);initial=json.loads(self.data['environment/initial.json'])
        environment={'uuid':self.device['uuid'],'execution_uid':initial['execution_uid'],'name':self.device['name'],'driver':'CPU fixture','compiler':'CPU fixture','tools':{}}
        identity=family_identity(environment,__import__('hashlib').sha256(self.data['environment/device.json']).hexdigest(),{'CPUfixtureSource':'0'*64},{'CPUfixtureController':'0'*64})
        tag=tool.SUITE+'/tma_stage_request_packed/v1/cpu-fixture';(self.repo/tag).parent.mkdir(parents=True)
        self.entry=pack(self.repo,self.point,self.run_path,self.data,identity,tag)
        with tarfile.open(self.repo/self.entry['archive']['path']) as t:t.extractall(self.repo,filter='data')
        self.entry['verification_origins']={}
        for role,key in [('remote','archive_verification'),('offhost','offhost_receipt')]:
            path=self.repo/tag/(role+'.origin.json');path.write_text(json.dumps({'schema_version':1,'receipt_sha256':self.entry[key]['sha256'],'archive_sha256':self.entry['archive']['sha256'],'verifier':{'host':'CPU-'+role}}));self.entry['verification_origins'][role]={'path':str(path.relative_to(self.repo)),'sha256':sha(path)}
        self.ledger=self.repo/tool.SUITE/'tma_stage_request-short-v1-ledger.json';self.ledger.write_text(json.dumps({'identity':identity,'entries':{str(self.point['index']):self.entry}}))
        self.uuidlock=Path(self.tmp.name)/'uuid.lock';self.quarantine=Path(self.tmp.name)/'quarantine';self.registry=Path(self.tmp.name)/'registry';self.registry.mkdir(mode=0o700);(self.registry/'active').mkdir(mode=0o700)
        protected=self.repo/'protected-original';protected.write_text('immutable SOURCE fixture')
        source=self.repo.parent/'source-manifest.json';source.write_text(json.dumps({'protected-original':sha(protected)}))
        driver=self.repo.parent/'driver.sh';driver.write_text('CPU fixture never executed')
        (self.repo.parent/'launch-manifest.json').write_text(json.dumps({'source-manifest.json':sha(source),'driver.sh':sha(driver)}))
        self.patchers=[patch.object(tool,'scratch_repo',return_value=self.repo),patch.object(tool,'controller_gates'),patch.object(tool,'cpu_allocation',return_value={'CPU_fixture_allocation_only':True}),patch.object(tool.socket,'gethostname',return_value='CPU-fixture-host'),patch.object(tool,'gpu_paths',return_value=(self.uuidlock,self.quarantine)),patch.object(tool,'registry_root',return_value=self.registry)]
        for item in self.patchers:item.start()
    def tearDown(self):
        for item in reversed(self.patchers):item.stop()
        self.tmp.cleanup()
    def invoke(self):return tool.retire(self.repo,self.point['index'],self.run_path)
    def test_consumes_actual_archived_verified_ledger_and_query_metadata(self):
        import copy
        repo=ROOT.parents[1]
        evidence=repo/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-retirement-v2/actual-ledger-r3'
        ledger=json.loads((evidence/'ledger.json').read_text());entry=ledger['entries']['0']
        self.assertEqual(entry['state'],'archived_verified')
        self.assertEqual(entry['family_identity_sha256'],digest(ledger['identity']))
        self.assertEqual(sha(evidence/'resident-device.json'),sha(evidence/'original-query-device.json'))
        initial=json.loads((evidence/'resident-initial.json').read_text())
        device=tool.family_device(ledger['identity'],evidence/'resident-device.json',initial)
        self.assertEqual(device['uuid'],ledger['identity']['environment']['uuid'])
        for change in (lambda identity:identity.update(device_sha256='0'*64),lambda identity:identity['environment'].update(uuid='GPU-00000000-0000-0000-0000-000000000000'),lambda identity:identity.update(device={'uuid':device['uuid']})):
            identity=copy.deepcopy(ledger['identity']);change(identity)
            with self.assertRaises(ValueError):tool.family_device(identity,evidence/'resident-device.json',initial)
    def test_actual_CPU_pack_complete_retire_preserves_all_sealed_bytes(self):
        refs=[self.entry[k] for k in ('archive','index','closure','archive_verification','offhost_receipt')]+list(self.entry['verification_origins'].values())
        before={r['path']:sha(self.repo/r['path']) for r in refs};ledger_before=sha(self.ledger)
        result=self.invoke();self.assertEqual(result['status'],'resident_retired_archived_evidence_retained')
        self.assertFalse((self.repo/self.run_path).exists());self.assertEqual(sha(self.ledger),ledger_before)
        self.assertEqual(before,{r['path']:sha(self.repo/r['path']) for r in refs})
        receipt=list((self.repo/tool.SUITE/'implementation/s16-retire-v2').rglob('complete.json'));self.assertEqual(len(receipt),1)
    def test_wrong_producer_and_explicit_run_reject_before_delete(self):
        with patch.object(tool.socket,'gethostname',return_value='CPU-login-host'):
            with self.assertRaises(ValueError):self.invoke()
        with self.assertRaises(ValueError):tool.retire(self.repo,self.point['index'],self.run_path+'/../unrelated')
        self.assertTrue((self.repo/self.run_path).is_dir());self.assertFalse(self.uuidlock.exists())
    def test_quarantine_and_active_any_member_rejected(self):
        self.quarantine.write_text('preserved')
        with self.assertRaises(ValueError):self.invoke()
        self.quarantine.unlink();active=self.registry/'active';(active/'unrecognized').write_text('active')
        with self.assertRaises(ValueError):self.invoke()
        self.assertTrue((self.repo/self.run_path).is_dir());self.assertFalse((self.repo/tool.SUITE/'implementation/s16-retire-v2').exists())
    def test_resident_and_pack_and_origin_drift_rejected(self):
        root=self.repo/self.run_path;file=root/'binary/probe';original=file.read_bytes();file.write_bytes(original+b'drift')
        with self.assertRaises(ValueError):self.invoke()
        file.write_bytes(original);ref=self.entry['verification_origins']['offhost'];file=self.repo/ref['path'];original=file.read_bytes();file.write_bytes(original+b' ')
        with self.assertRaises(ValueError):self.invoke()
        file.write_bytes(original);file=self.repo/self.entry['closure']['path'];file.write_text('{}')
        with self.assertRaises(ValueError):self.invoke()
        self.assertTrue(root.exists())
    def test_unknown_partial_delete_records_failure_and_no_reentry(self):
        def partial(root):
            (root/'binary/probe').unlink();raise OSError('CPU simulated partial removal')
        with patch.object(tool.shutil,'rmtree',side_effect=partial) as deletion:
            with self.assertRaises(OSError):self.invoke()
            self.assertEqual(deletion.call_count,1)
            with self.assertRaises(ValueError):self.invoke()
            self.assertEqual(deletion.call_count,1)
        folder=self.repo/tool.SUITE/'implementation/s16-retire-v2';self.assertEqual(len(list(folder.rglob('started.json'))),1);self.assertEqual(len(list(folder.rglob('failed.json'))),1)
    def test_existing_receipt_or_symlink_never_overwritten(self):
        folder=self.repo/tool.SUITE/'implementation/s16-retire-v2';folder.mkdir(parents=True)
        output=folder/('index-'+str(self.point['index'])+'-'+self.entry['manifest_sha256']);output.mkdir();(output/'started.json').write_text('immutable old evidence')
        with self.assertRaises(ValueError):self.invoke()
        self.assertEqual((output/'started.json').read_text(),'immutable old evidence')
        shutil.rmtree(output);outside=Path(self.tmp.name)/'outside';outside.mkdir();output.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):self.invoke()
        self.assertEqual(list(outside.iterdir()),[])
    def test_required_private_registry_and_source_closure(self):
        active=self.registry/'active';active.rmdir()
        with self.assertRaises(ValueError):self.invoke()
        active.mkdir(mode=0o700);active.chmod(0o777)
        with self.assertRaises(ValueError):self.invoke()
        active.chmod(0o700);(self.repo/'protected-original').write_text('SOURCE drift')
        with self.assertRaises(ValueError):self.invoke()
        self.assertTrue((self.repo/self.run_path).exists())
    def test_source_changed_during_delete_never_completes(self):
        real=shutil.rmtree
        def modified(root):
            (self.repo/'protected-original').write_text('CPU simulated source drift');real(root)
        with patch.object(tool.shutil,'rmtree',side_effect=modified):
            with self.assertRaises(ValueError):self.invoke()
        folder=self.repo/tool.SUITE/'implementation/s16-retire-v2'
        self.assertEqual(len(list(folder.rglob('failed.json'))),1)
        self.assertEqual(len(list(folder.rglob('complete.json'))),0)
    def test_actual_Slurm_CPU_predicates_positive_and_negative(self):
        from types import SimpleNamespace
        info='JobId=123 JobState=RUNNING UserId=fixture('+str(os.geteuid())+') NumNodes=1 NumCPUs=1 TimeLimit=00:10:00 NodeList=CPU-fixture-host AllocTRES=cpu=1,mem=4G,node=1 TresPerNode=(null)'
        def runcheck(value,hosts='CPU-fixture-host'):
            def respond(argv,**kw):
                out='ClusterName = romeo' if argv[-1]=='config' else hosts if 'hostnames' in argv else value
                return SimpleNamespace(returncode=0,stdout=out,stderr='')
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch.object(tool.subprocess,'run',side_effect=respond):return REAL_CPU_ALLOCATION('CPU-fixture-host')
        self.assertFalse(runcheck(info)['GPU_allocation'])
        for value in (info.replace('RUNNING','PENDING'),info.replace('NumNodes=1','NumNodes=2'),info.replace('NumCPUs=1','NumCPUs=0'),info.replace('00:10:00','01:00:00'),info+' ReqTRES=gres/gpu=1',info.replace('AllocTRES=cpu=1,mem=4G,node=1','AllocTRES=mem=4G,node=1'),info.replace('UserId=fixture('+str(os.geteuid())+')','UserId=other(999999)')):
            with self.subTest(value=value),self.assertRaises(ValueError):runcheck(value)
        with self.assertRaises(ValueError):runcheck(info,'other-node')
    def test_production_scope_no_gpu_cleanup_calls(self):
        with self.assertRaises(ValueError):REAL_SCRATCH_REPO(self.repo)
        text=(ROOT/'runners/s16_retirement_v2.py').read_text()
        import ast
        tree=ast.parse(text);calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertTrue(calls.isdisjoint({'gpu_lock','reconcile','stop_group','kill','bounded','subprocess'}))

if __name__=='__main__':unittest.main()
