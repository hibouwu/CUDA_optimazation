"""CPU-only guards for case diagnostics. Fixtures grant no hardware qualification."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess
import unittest
from unittest.mock import patch
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from common.suite_io import sha
from runners import validation_diagnostic as d


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy=json.loads((BASE/'contracts/early_validation_v1.json').read_text())
        cls.profile={'id':'short','target_iterations':[1,2],'input_profiles':['uniform','nonuniform'],
          'maximum_target_launches':4,'maximum_explicit_auxiliary_launches':0,
          'completion_requirements':['complete'], 'reference_identity':{'model':'fixture'},
          'declared_output_check':{'comparison':'exact'},'output_evidence_kind':'error_count_only'}
        cls.case={'id':'case','scope':'one_cta','threads':32}

    def row(self):
        return {'schema_version':2,'validation_schema_version':1,'type':'validation',
          'case_id':'case','profile_id':'short','seed':3,'scope':'one_cta','threads':32,'blocks':1,'errors':0,
          'target_launches':[{'launch_index':i,'iterations':1+i%2,'input_profile':'uniform' if i<2 else 'nonuniform','threads':32,'blocks':1} for i in range(4)],
          'checks':[{'launch_index':i,'reference_model':'fixture','reference_sha256':'a'*64,'comparison':'exact','tolerance_id':None,
                     'checked_elements':32,'expected_elements':32,'errors':0,'completed':True,'verified_CTA_ids':[0],'output_artifacts':[]} for i in range(4)],
          'resource_identity':{'kernel_symbol':'fixture','registers_per_thread':16,'static_smem_bytes':0,'dynamic_smem_bytes':0,'local_size_bytes':0,'occupancy_limit_ctas_per_sm':1,'extensions':{}},
          'performance_eligible':False,'warmup_executed':False,'pilot_executed':False}

    def test_profile_launch_iteration_bounds_and_missing_fields(self):
        bundle={'schema_version':1,'profiles':[self.profile]}
        self.assertEqual(d.profile_from(bundle,'short',self.policy),self.profile)
        for key,value in [('target_iterations',[1,65]),('maximum_target_launches',3),('input_profiles',[]),('target_iterations',[1,1])]:
            bad=copy.deepcopy(bundle);bad['profiles'][0][key]=value
            with self.assertRaises(ValueError):d.profile_from(bad,'short',self.policy)
        for key in self.policy['profile_required_fields']:
            bad=copy.deepcopy(bundle);del bad['profiles'][0][key]
            with self.assertRaises(ValueError):d.profile_from(bad,'short',self.policy)

    def test_core_refuses_trial_warmup_pilot_and_performance(self):
        d.validation_envelope(self.row(),self.case,self.profile,3,self.policy)
        for key,value in [('type','trial'),('warmup_executed',True),('pilot_executed',True),('performance_eligible',True),('errors',1),('schema_version',True)]:
            bad=self.row();bad[key]=value
            with self.assertRaises(ValueError):d.validation_envelope(bad,self.case,self.profile,3,self.policy)

    def test_launch_identity_full_output_and_completion_negatives(self):
        for mutate in [lambda r:r['target_launches'][0].update(iterations=8192),
                       lambda r:r['target_launches'][1].update(launch_index=0),
                       lambda r:r['checks'][0].update(checked_elements=31),
                       lambda r:r['checks'][0].update(completed=False),
                       lambda r:r['checks'][0].update(verified_CTA_ids=[0,0]),
                       lambda r:r['checks'].pop(),lambda r:r.update(blocks=0)]:
            row=self.row();mutate(row)
            with self.assertRaises(ValueError):d.validation_envelope(row,self.case,self.profile,3,self.policy)

    def test_jsonl_rejects_duplicate_fields_nan_and_partial_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'raw'
            for text in ('{"type":"device"}\n', '{"type":"device"}\n{"type":"validation","type":"trial"}\n',
                         '{"type":"device"}\n{"type":"validation","x":NaN}\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):d.strict_lines(path)
            path.write_text('{"type":"device"}\n{"type":"validation"}\n')
            self.assertEqual(d.strict_lines(path)[1]['type'],'validation')

    def test_output_artifact_shape_hash_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'out.bin').write_bytes(bytes(128))
            row=self.row();item={'path':'out.bin','sha256':sha(root/'out.bin'),'dtype':'uint32','shape':[32],'evidence_kind':'full_values'}
            row['checks'][0]['output_artifacts']=[item]
            self.assertEqual(len(d.verify_output_artifacts(root,row)),1)
            for field,value in [('path','../out.bin'),('sha256','a'*64),('shape',[31]),('dtype','unknown')]:
                bad=copy.deepcopy(row);bad['checks'][0]['output_artifacts'][0][field]=value
                with self.assertRaises(ValueError):d.verify_output_artifacts(root,bad)

    def test_failed_or_completed_diagnostic_never_launches_again(self):
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for status in ('failed','case_diagnostic_passed'):
                (root/'validation_state.json').write_text(json.dumps({'status':status}))
                with patch.object(d,'file_lock',return_value=nullcontext()),patch.object(d,'load',return_value=({'execution_root':str(root)},None,None,None,None)),patch.object(d,'inspect_allocation',side_effect=AssertionError('must not query/execute GPU')):
                    with self.assertRaises(ValueError):d.execute(root)

    def test_manifest_unknown_module_abi_and_unbound_files_rejected(self):
        repo=BASE.parents[1]
        paths=[BASE/'contracts/low_precision_lowering_v3.json',BASE/'contracts/low_precision_validation_profiles_v1.json']
        original=json.loads((BASE/'contracts/low_precision_validation_adapter_v1.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'manifest.json'
            for key,value in [('module','os'),('module','auditors../escape'),('adapter_abi_version',2),('adapter_abi_version',True),('contract_sha256','a'*64),('profiles_sha256','b'*64),('files_sha256',{})]:
                bad=copy.deepcopy(original);bad[key]=value;file.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):d.manifest_check(repo,*paths,file)

    def test_initialization_freezes_dependency_and_rejects_unreviewed_adapter(self):
        from common.suite_io import DIMENSIONS
        with tempfile.TemporaryDirectory(prefix='gh200-diagnostic-fixture-') as tmp:
            repo=Path(tmp)/'repo';code=repo/d.CAMPAIGN
            shutil.copytree(BASE,code,ignore=shutil.ignore_patterns('__pycache__'))
            suite=repo/'results/fixture';(suite/'reviews').mkdir(parents=True)
            contract_path=code/'contracts/fixture.json';profiles_path=code/'contracts/fixture_profiles.json'
            manifest_path=code/'contracts/fixture_manifest.json';module=code/'auditors/fixture_validation.py'
            module.write_text('''ADAPTER_ID="fixture_v1"
ADAPTER_ABI_VERSION=1
def validation_argv(*a): return []
def validate_validation(device,row,case,profile,seed):
    return dict(status="pass",case_id=case["id"],profile_id=profile["id"],target_launches=4,
                checked_elements=128,output_evidence_kind="error_count_only",performance_eligible=False)
def validate_prior_evidence(*a): return {}
def audit_sass(*a): return []
def validate_device(*a): return {}
''')
            source=code/'probes/fixture.cu';source.write_text('// CPU fixture only; never compiled or executed.\n')
            contract={'schema_version':2,'stage':'TEST','family':'fixture','adapter_id':'fixture_v1',
                      'source':str(source.relative_to(repo)),'dependencies':[],
                      'cases':[self.case],'review_dependencies':[{'stage':'TEST','phase':'A','path':'reviews/family-A.json'}]}
            contract_path.write_text(json.dumps(contract));profiles_path.write_text(json.dumps({'schema_version':1,'family':'fixture','profiles':[self.profile]}))
            manifest={'adapter_id':'fixture_v1','adapter_abi_version':1,'module':'auditors.fixture_validation',
                      'exports':self.policy['new_adapter_exports'],'files_sha256':{str(p.relative_to(repo)):sha(p) for p in (module,source)},
                      'contract_sha256':sha(contract_path),'profiles_sha256':sha(profiles_path),
                      'review_dependencies':[{'stage':'TEST','phase':'early-validation-source-B','path':'reviews/family-source-B.json'}]}
            manifest_path.write_text(json.dumps(manifest))
            bindings={str(p.relative_to(repo)):sha(p) for p in (module,source,contract_path,profiles_path,manifest_path)}
            bindings.update({name:sha(repo/name) for name in d.CORE_FILES})
            def gate(name,stage,phase):
                path=suite/name;path.write_text(json.dumps({'schema_version':2,'stage':stage,'phase':phase,'status':'pass',
                    'implementer':'CPU fixture author','reviewer':'CPU fixture reviewer',
                    'checks':{key:{'status':'pass'} for key in DIMENSIONS},'findings':[],
                    'gate_files':bindings,'fixture_no_hardware_qualification':True}))
            for name,stage,phase in [('reviews/S02-A-review.json','S02','A'),
                                     ('reviews/S02-preflight-A-review-r2.json','S02','preflight-A'),
                                     (d.CORE_REVIEW,'S02','early-validation-B'),
                                     ('reviews/S02-early-validation-A-review-r2.json','S02','early-validation-A'),
                                     ('reviews/family-A.json','TEST','A'),
                                     ('reviews/family-source-B.json','TEST','early-validation-source-B')]:gate(name,stage,phase)
            output=suite/'fixture'/'run'
            with patch.object(d,'inspect_allocation',return_value={'CPU_fixture':True}):
                frozen=d.initialize(repo,suite,output,contract_path,profiles_path,manifest_path,'case','short',3)
            self.assertTrue(frozen.is_file())
            spec=json.loads((output/'validation_spec.json').read_text())
            self.assertFalse(spec['family_B3_eligible']);self.assertFalse(spec['performance_eligible'])
            # This status replay only reads the frozen files and runs no device query.
            result=subprocess.run([sys.executable,'-B',str(frozen),'status',str(output)],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            # Build and process fixtures test sealing/replay mechanics, never device behavior.
            (output/'binary/probe').write_text('CPU fixture; not executable GPU code')
            (output/'build/sass.stdout').write_text('CPU fixture SASS placeholder')
            (output/'build/dependencies.json').write_text('{}')
            spec.update(binary_sha256=sha(output/'binary/probe'),sass_sha256=sha(output/'build/sass.stdout'),dependencies_sha256=sha(output/'build/dependencies.json'))
            (output/'validation_spec.json').write_text(json.dumps(spec))
            device={'type':'device','uuid':'GPU-00000000-0000-0000-0000-000000000001','CPU_fixture':True}
            (output/'environment/device.json').write_text(json.dumps(device))
            attempt=output/'attempts/attempt_00';attempt.mkdir()
            (attempt/'raw.jsonl').write_text(json.dumps(device)+'\n'+json.dumps(self.row())+'\n')
            (attempt/'stderr').write_text('')
            receipt={'argv':[str(output/'binary/probe'),'validate-only','case','short','3'],
                     'timeout_seconds':30,'pid':1001,'pgid':1001,'returncode':0,'timed_out':False,
                     'cleanup_confirmed':True,'host_start_ns':1,'host_stop_ns':2,
                     'stdout_sha256':sha(attempt/'raw.jsonl'),'stderr_sha256':sha(attempt/'stderr')}
            receipt['gpu_process_registration']={'state':'cleanup_confirmed','gpu_uuid':device['uuid'].lower(),
                'pid':1001,'pgid':1001,'argv':receipt['argv'],'cleanup_receipt':copy.deepcopy(receipt)}
            (attempt/'receipt.json').write_text(json.dumps(receipt))
            worker='''import sys;from pathlib import Path
sys.path.insert(0,sys.argv[1]);from runners import validation_diagnostic as d
root=Path(sys.argv[2]);spec,contract,profile,case,module=d.load(root)
proof=d.attempt_evidence(root,spec,profile,case,module,root/"attempts/attempt_00")
d.seal(root,spec,proof)
'''
            result=subprocess.run([sys.executable,'-B','-c',worker,str(output/'snapshot/repo'/d.CAMPAIGN),str(output)],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            moved=Path(tmp)/'moved';shutil.copytree(output,moved)
            result=subprocess.run([sys.executable,'-B',str(moved/'snapshot/repo'/d.ENTRY),'audit',str(moved)],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(json.loads(result.stdout)['family_B3_eligible'])
            failed=Path(tmp)/'failed';shutil.copytree(output,failed)
            for name in ('diagnostic_summary.json','validation_manifest.json'):(failed/name).unlink()
            (failed/'failure_summary.json').write_text(json.dumps({'status':'case_diagnostic_failed',
                'reason':'CPU fixture process failure','case_id':'case','performance_eligible':False,'family_B3_eligible':False}))
            (failed/'validation_state.json').write_text(json.dumps({'status':'failed'}))
            bad_receipt=json.loads((failed/'attempts/attempt_00/receipt.json').read_text());bad_receipt['returncode']=2
            (failed/'attempts/attempt_00/receipt.json').write_text(json.dumps(bad_receipt))
            mapping={p.relative_to(failed).as_posix():sha(p) for p in failed.rglob('*') if p.is_file() and p.name!='validation_state.json'}
            (failed/'failure_manifest.json').write_text(json.dumps(mapping))
            result=subprocess.run([sys.executable,'-B',str(failed/'snapshot/repo'/d.ENTRY),'audit',str(failed)],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'],'case_diagnostic_failed')
            self.assertFalse(json.loads(result.stdout)['family_B3_eligible'])
            # An interrupted run with a completed receipt must reuse it rather than launch.
            (output/'validation_state.json').write_text(json.dumps({'status':'running'}))
            resume_worker='''import sys;from pathlib import Path;from contextlib import nullcontext;from unittest.mock import patch
sys.path.insert(0,sys.argv[1]);from runners import validation_diagnostic as d
root=Path(sys.argv[2]);env={"uuid":"GPU-00000000-0000-0000-0000-000000000001"}
def command(root,name,argv,timeout,env,**kw):
    d.atomic_json(root/(name+".stdout"),d.read_json(root/"environment/device.json"))
with patch.object(d,"file_lock",return_value=nullcontext()),patch.object(d,"inspect_allocation",return_value=env),patch.object(d,"environment_identity",return_value={}),patch.object(d,"compile_probe"),patch.object(d,"verify_files"),patch.object(d,"gpu_lock",return_value=nullcontext([])),patch.object(d,"verify_interrupted_gpu_groups"),patch.object(d,"run_command",side_effect=command),patch.object(d,"read_json",wraps=d.read_json),patch.object(d,"bounded",side_effect=AssertionError("completed target must not execute again")):
    d.execute(root)
'''
            (output/'build/shared_libraries.json').write_text('{}')
            result=subprocess.run([sys.executable,'-B','-c',resume_worker,str(output/'snapshot/repo'/d.CAMPAIGN),str(output)],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            spec_path=output/'validation_spec.json';original_spec=spec_path.read_text()
            tampered=json.loads(original_spec);tampered['reviews']=[];spec_path.write_text(json.dumps(tampered))
            result=subprocess.run([sys.executable,'-B',str(frozen),'status',str(output)],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('required review set',result.stderr)
            spec_path.write_text(original_spec)
            (output/'snapshot/repo'/module.relative_to(repo)).write_text('tampered')
            result=subprocess.run([sys.executable,'-B',str(frozen),'status',str(output)],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0)
            source_gate=suite/'reviews/family-source-B.json';record=json.loads(source_gate.read_text())
            del record['gate_files'][str(source.relative_to(repo))];source_gate.write_text(json.dumps(record))
            with patch.object(d,'inspect_allocation',side_effect=AssertionError('unbound source must fail before environment query')):
                with self.assertRaisesRegex(ValueError,'source-B omits'):
                    d.initialize(repo,suite,suite/'fixture'/'unbound',contract_path,profiles_path,manifest_path,'case','short',3)
            (suite/'reviews/family-source-B.json').unlink()
            with patch.object(d,'inspect_allocation',side_effect=AssertionError('unreviewed code must fail before environment query')):
                with self.assertRaises(ValueError):d.initialize(repo,suite,suite/'fixture'/'another',contract_path,profiles_path,manifest_path,'case','short',3)

if __name__=='__main__':unittest.main()
