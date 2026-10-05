"""Synthetic pilot/identity fixtures only; no GPU and no performance qualification."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(Path(__file__).parent))
from test_suite_core import DEVICE,PROTOCOL
from common.suite_io import atomic_json,read_json,sha
from common.calibration import calibrated_cases,pilot_case,seal_resolution,resolved_contract,require_B_bound_artifacts,import_resolved_binary,verify_import_binding
from auditors.legacy_compute import arithmetic,TIMING
from runners.suite_runner import initialize
CONTRACT=read_json(HERE/'contracts/legacy_fma.json')


def pilot_fixture(root,spec,case,number):
    folder=root/'calibration'/case['id']/'batch_00/trial_00/attempt_00';folder.mkdir(parents=True)
    pc=pilot_case(case);blocks=1 if case['scope']=='one_cta' else DEVICE['sms']*4
    expected=arithmetic(pc,blocks);checked=expected['checked']
    row={'schema_version':2,'type':'trial','case_id':case['id'],'iterations':8192,'seed':3,'threads':case['threads'],'blocks':blocks,'scope':case['scope'],'errors':0,
         'work_unit':'FLOP','work_count':expected['work'],'read_payload_bytes':0,'write_payload_bytes':0,'start_ns':100,'stop_ns':1000100,'event_ms':10,
         'warmup_samples_ns':[1000000]*8,'warmup_converged':True,'cache_residency_proven':False,'physical_hbm_bytes_proven':False,
         'blocks_detail':[{'block_id':i,'smid':i%DEVICE['sms'],'start_ns':100,'stop_ns':1000100,'start_cycle':200,'stop_cycle':2000200} for i in range(blocks)],
         'correctness':{'method':'uniform_and_nonuniform_full_output_v1','checked_elements':checked,'input_conditions':'legacy uniform constants; separate seeded nonuniform validation; D reset each launch'},
         'occupancy_limit_ctas_per_sm':4,'registers_per_thread':32,'static_smem_bytes':0,'local_size_bytes':0,'output_elements_checked':checked,'max_abs_error':0,
         'timing_model':TIMING,'phase':case['parameters']['phase'],
         'nonuniform_validation':{'iterations':[1,2],'checked_elements':checked*2,'input_seed':3,'errors':0,'reference_model':'integer_dyadic_rne_fma_v1'}}
    (folder/'raw.jsonl').write_text(json.dumps(DEVICE)+'\n'+json.dumps(row)+'\n');(folder/'stderr').write_text('')
    atomic_json(folder/'receipt.json',{'status':'valid','case_id':case['id'],'batch':0,'trial':0,'seed':3,'relative_command':['binary/probe',case['id'],'8192','3'],
                'binary_sha256':spec['binary_sha256'],'raw_sha256':sha(folder/'raw.jsonl'),'stderr_sha256':sha(folder/'stderr'),
                'pid':1000+number,'pgid':1000+number,'host_start_ns':number*10000+1,'host_stop_ns':number*10000+100,'returncode':0,'timed_out':False,'cleanup_confirmed':True})


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        atomic_json(self.root/'contract.json',CONTRACT);self.spec={'schema_version':2,'fixture':True,'kind':'preflight','contract_path':'contract.json','binary_sha256':'a'*64}
        for n,c in enumerate(calibrated_cases(CONTRACT)):pilot_fixture(self.root,self.spec,c,n)
    def tearDown(self):self.temp.cleanup()

    def test_complete_pilot_set_seals_and_deterministically_resolves(self):
        value=seal_resolution(self.root,self.spec,CONTRACT,DEVICE,PROTOCOL)
        self.assertEqual(len(value['cases']),36)
        effective=resolved_contract(self.root,self.spec,CONTRACT,DEVICE,PROTOCOL)
        for old,new in zip(CONTRACT['cases'],effective['cases']):
            self.assertEqual(new['iterations'],81920 if old['iteration_policy']['kind']=='calibrated' else old['iterations'])
        before=sha(self.root/'resolved_cases.json');seal_resolution(self.root,self.spec,CONTRACT,DEVICE,PROTOCOL)
        self.assertEqual(before,sha(self.root/'resolved_cases.json'))

    def test_missing_duplicate_wrong_resolution_and_raw_tamper_rejected(self):
        seal_resolution(self.root,self.spec,CONTRACT,DEVICE,PROTOCOL);original=read_json(self.root/'resolved_cases.json')
        for mutation in ('missing','duplicate','wrong_iterations','wrong_binary','wrong_device'):
            value=copy.deepcopy(original)
            if mutation=='missing':value['cases'].pop()
            if mutation=='duplicate':value['cases'][1]=value['cases'][0]
            if mutation=='wrong_iterations':value['cases'][0]['resolved_iterations']=8192
            if mutation=='wrong_binary':value['cases'][0]['binary_sha256']='b'*64
            if mutation=='wrong_device':value['cases'][0]['device_identity']['uuid']='GPU-wrong'
            atomic_json(self.root/'resolved_cases.json',value);self.spec['resolved_cases']['sha256']=sha(self.root/'resolved_cases.json')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):resolved_contract(self.root,self.spec,CONTRACT,DEVICE,PROTOCOL)
        atomic_json(self.root/'resolved_cases.json',original);self.spec['resolved_cases']['sha256']=sha(self.root/'resolved_cases.json')
        raw=self.root/original['cases'][0]['pilot_raw']['path'];raw.write_text(raw.read_text()+'\n')
        with self.assertRaises(ValueError):resolved_contract(self.root,self.spec,CONTRACT,DEVICE,PROTOCOL)

    def test_formal_cannot_use_8192_placeholder_without_explicit_resolution(self):
        formal={**self.spec,'kind':'formal'}
        with self.assertRaisesRegex(ValueError,'B-bound resolved'):resolved_contract(self.root,formal,CONTRACT,DEVICE,PROTOCOL)
        with self.assertRaisesRegex(ValueError,'--resolved-cases'):
            initialize(HERE.parents[1],self.root/'suite',self.root/'suite/legacy_fma/run',HERE/'contracts/legacy_fma.json')
        self.assertFalse((self.root/'suite').exists())

    def test_relocation_uses_only_relative_pilot_paths(self):
        seal_resolution(self.root,self.spec,CONTRACT,DEVICE,PROTOCOL)
        with tempfile.TemporaryDirectory() as temp:
            moved=Path(temp)/'copy';shutil.copytree(self.root,moved)
            result=resolved_contract(moved,self.spec,CONTRACT,DEVICE,PROTOCOL)
            self.assertEqual(len(result['cases']),93)

    def test_B_must_directly_bind_pilot_and_resolved_artifacts(self):
        seal_resolution(self.root,self.spec,CONTRACT,DEVICE,PROTOCOL)
        files=['resolved_cases.json',read_json(self.root/'resolved_cases.json')['cases'][0]['pilot_raw']['path']]
        review={'gate_files':{self.root.name+'/'+name:sha(self.root/name) for name in files}}
        self.assertEqual(len(require_B_bound_artifacts(self.root.parent,self.root,review,files)),2)
        review['gate_files'].pop(self.root.name+'/'+files[1])
        with self.assertRaisesRegex(ValueError,'directly bind'):require_B_bound_artifacts(self.root.parent,self.root,review,files)


class FrozenBinaryImportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.repo=Path(self.temp.name);self.origin=self.repo/'origin';self.origin.mkdir();self.root=self.repo/'formal';self.root.mkdir()
        self.env={'uuid':DEVICE['uuid'],'name':'GH200 fixture','driver':'fixture','compiler':'CUDA12.9 fixture','tools':{},'execution_uid':1}
        files={'binary/probe':'CPU fixture binary, must never execute','build/sass.txt':'CPU fixture SASS','build/shared_libraries.json':'{}','build/command.json':'{}','preflight/case/raw.jsonl':'CPU fixture smoke','calibration/case/raw.jsonl':'CPU fixture pilot'}
        for name,text in files.items():
            path=self.origin/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        atomic_json(self.origin/'build/dependencies.json',{'local':['repo/probe.cu'],'external_toolchain':{}})
        atomic_json(self.origin/'contract.json',CONTRACT)
        atomic_json(self.origin/'resolved_cases.json',{'schema_version':2,'family':'legacy_fma','phase':'B-calibration','cases':[]})
        atomic_json(self.origin/'preflight_summary.json',{'hardware_qualification':False})
        atomic_json(self.origin/'snapshot/manifest.json',{'repo/probe.cu':'1'*64})
        atomic_json(self.origin/'environment/initial.json',self.env);atomic_json(self.origin/'environment/device.json',DEVICE)
        self.old={'kind':'preflight','fixture':True,'contract_path':'contract.json','snapshot_manifest_sha256':sha(self.origin/'snapshot/manifest.json'),
                  'binary_sha256':sha(self.origin/'binary/probe'),'sass_sha256':sha(self.origin/'build/sass.txt'),
                  'dependencies_sha256':sha(self.origin/'build/dependencies.json'),'shared_libraries_sha256':sha(self.origin/'build/shared_libraries.json'),
                  'resolved_cases':{'path':'resolved_cases.json','sha256':sha(self.origin/'resolved_cases.json')}}
        atomic_json(self.origin/'run_spec.json',self.old)
        atomic_json(self.root/'snapshot/manifest.json',{'repo/probe.cu':'1'*64})
        self.review={'gate_files':{'origin/'+p.relative_to(self.origin).as_posix():sha(p) for p in self.origin.rglob('*') if p.is_file()}}
        atomic_json(self.root/'snapshot/suite/reviews/S05-B-review.json',self.review)
        self.spec={'kind':'formal','fixture':True,'reviews':[{'stage':'S05','phase':'B','path':'reviews/S05-B-review.json','sha256':sha(self.root/'snapshot/suite/reviews/S05-B-review.json')}],
                   'contract_path':'contract.json'}
    def tearDown(self):self.temp.cleanup()
    def do_import(self):
        # Provenance and full smoke arithmetic are covered separately; this fixture isolates exact import behavior.
        with patch('auditors.suite.load_run',return_value=(self.old,CONTRACT,PROTOCOL,DEVICE,None)),patch('runners.suite_runner.audit_preflight'),patch('runners.suite_runner.compile_probe',side_effect=AssertionError('must not compile')):
            import_resolved_binary(self.repo,self.root,self.spec,CONTRACT,self.origin/'resolved_cases.json',self.env)

    def test_exact_B_binary_is_copied_without_recompilation(self):
        self.do_import()
        self.assertEqual((self.root/'binary/probe').read_bytes(),(self.origin/'binary/probe').read_bytes())
        self.assertFalse(read_json(self.root/'calibration_origin/B_artifact_binding.json')['binary_recompiled'])
        verify_import_binding(self.root,self.spec,CONTRACT)
        (self.root/'binary/probe').write_text('rebuilt binary')
        self.spec['binary_sha256']=sha(self.root/'binary/probe')
        with self.assertRaisesRegex(ValueError,'artifact changed'):verify_import_binding(self.root,self.spec,CONTRACT)

    def test_changed_compiled_dependency_cannot_reuse_B_resolution(self):
        atomic_json(self.root/'snapshot/manifest.json',{'repo/probe.cu':'2'*64})
        with self.assertRaisesRegex(ValueError,'compiled source dependency'):self.do_import()
        self.assertFalse((self.root/'binary/probe').exists())

    def test_unbound_binary_and_changed_device_environment_rejected(self):
        self.review['gate_files'].pop('origin/binary/probe');atomic_json(self.root/'snapshot/suite/reviews/S05-B-review.json',self.review)
        with self.assertRaisesRegex(ValueError,'directly bind'):self.do_import()
        self.env={**self.env,'driver':'different driver'}
        with self.assertRaisesRegex(ValueError,'environment changed'):self.do_import()


class InterruptedImportTests(unittest.TestCase):
    def test_interrupted_formal_import_refuses_compile_or_device_query(self):
        from runners.suite_runner import execute
        with tempfile.TemporaryDirectory() as temp:
            suite=Path(temp)/'suite';root=suite/'legacy_fma/run';root.mkdir(parents=True)
            atomic_json(root/'run_spec.json',{'kind':'formal'})
            with patch('runners.suite_runner.check_snapshot',return_value=CONTRACT),patch('runners.suite_runner.inspect_allocation',side_effect=AssertionError('no device query')),patch('runners.suite_runner.compile_probe',side_effect=AssertionError('no recompile')):
                with self.assertRaisesRegex(ValueError,'never recompile'):execute(root,suite)
