"""Offline runner safety tests. All GPU/process execution is mocked explicitly."""
import contextlib
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
from common.suite_io import atomic_json,sha
from runners import accumulation_diagnostic as runner
from runners.environment import Checkpoint


class AccumulationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.repo=Path(self.temp.name)
        self.suite=self.repo/'suite';self.root=self.suite/'low_precision'/'run'
        for name in ('environment','build','binary','attempts'):(self.root/name).mkdir(parents=True,exist_ok=True)
        self.device={'type':'device','uuid':'GPU-00000000-0000-0000-0000-000000000000'}
        self.resource={'kernel_symbol':'lp_wgmma_e4m3_g1'}
        self.env={'uuid':self.device['uuid'],'name':'GH200','driver':'same','compiler':'12.9','tools':{},'execution_uid':1000}
        self.spec={'schema_version':2,'kind':runner.KIND,'execution_root':str(self.root),'case_id':'wgmma_e4m3_g1_one_cta',
                   'binary_sha256':'test',**{k:False for k in runner.FALSE_FLAGS}}
        self.result={'status':'collected_finite_difference','output_count':8192,'difference_count':8192}
        self.module=types.SimpleNamespace(audit_values=lambda *args:copy.deepcopy(self.result))
        self.loaded=(self.spec,{}, {}, self.repo/'baseline',self.device,self.resource,self.module)
        atomic_json(self.root/'environment/initial.json',self.env)
        atomic_json(self.root/'diagnostic_state.json',{'status':'pending'})
        self.calls=[]

    def tearDown(self):self.temp.cleanup()

    def write_receipt(self,argv,stdout,stderr,timeout,*,returncode=0):
        receipt={'argv':argv,'returncode':returncode,'timed_out':False,'cleanup_confirmed':True,
                 'timeout_seconds':timeout,'host_start_ns':10,'host_stop_ns':20,'pid':100,'pgid':100,
                 'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr)}
        receipt['gpu_process_registration']={'state':'cleanup_confirmed','gpu_uuid':self.device['uuid'].lower(),
            'argv':argv,'pid':100,'pgid':100,'cleanup_receipt':copy.deepcopy(receipt)}
        return receipt

    def command(self,root,name,argv,timeout,env,**kwargs):
        self.calls.append(('command',argv[1],kwargs.get('gpu')))
        self.assertEqual(argv,[str(self.root/'binary/probe'),'device']);self.assertEqual(timeout,30)
        stdout=root/(name+'.stdout');stderr=root/(name+'.stderr')
        stdout.write_text(json.dumps(self.device)+'\n'+json.dumps({'type':'resource','resource_identity':self.resource})+'\n');stderr.write_text('')
        receipt=self.write_receipt(argv,stdout,stderr,timeout);atomic_json(root/(name+'.receipt.json'),receipt)
        return receipt

    def collect(self,argv,cwd,stdout,stderr,timeout,**kwargs):
        self.calls.append(('collect',argv[1],timeout))
        self.assertEqual(argv,[str(self.root/'binary/probe'),'collect']);self.assertEqual(timeout,120)
        self.assertEqual(kwargs,{'uuid':self.device['uuid']})
        # Full values retained in raw. The separate auditor's tests check their
        # FP32 semantics; runner tests protect storage, process and path identity.
        row={'type':runner.KIND,'outputs':[{'index':i,'value':16383.0,'f32_bits':1182792704} for i in range(8192)]}
        stdout.write_text(json.dumps(self.device)+'\n'+json.dumps(row)+'\n');stderr.write_text('')
        return self.write_receipt(argv,stdout,stderr,timeout)

    def mocks(self,stack):
        stack.enter_context(patch.object(runner,'load',side_effect=lambda p:self.loaded))
        stack.enter_context(patch.object(runner,'file_lock',side_effect=lambda p:contextlib.nullcontext()))
        stack.enter_context(patch.object(runner,'gpu_lock',side_effect=lambda u:contextlib.nullcontext([])))
        stack.enter_context(patch.object(runner,'inspect_allocation',return_value=self.env))
        stack.enter_context(patch.object(runner,'verify_build',side_effect=lambda *a,**kw:{'equal':True}))
        stack.enter_context(patch.object(runner,'verify_interrupted_gpu_groups',side_effect=lambda *a:self.calls.append(('cleanup',))))
        stack.enter_context(patch.object(runner,'budget',side_effect=lambda env,t:self.calls.append(('budget',t))))
        stack.enter_context(patch.object(runner,'run_command',side_effect=self.command))
        stack.enter_context(patch.object(runner,'bounded',side_effect=self.collect))

    def test_single_launch_retains_full_values_and_offline_relocation(self):
        with contextlib.ExitStack() as stack:
            self.mocks(stack)
            self.assertEqual(runner.execute(self.root),0)
            self.assertEqual([x for x in self.calls if x[0]=='collect'],[('collect','collect',120)])
            self.assertLess(self.calls.index(('cleanup',)),self.calls.index(('command','device',True)))
            self.assertLess(self.calls.index(('budget',120)),self.calls.index(('collect','collect',120)))
            summary=runner.audit(self.root)
            self.assertEqual(summary['status'],'diagnostic-collected')
            self.assertTrue(all(summary[k] is False for k in runner.FALSE_FLAGS))
            _,row=runner.strict_rows(self.root/'attempts/attempt_00/raw.jsonl',runner.KIND)
            self.assertEqual(len(row['outputs']),8192)
            self.assertEqual(row['outputs'][-1]['index'],8191)
            # Absolute receipt argv retains original launch root; replay only
            # reads local relative files, so a moved copy remains auditable.
            moved=self.repo/'moved-archive';shutil.copytree(self.root,moved)
            self.assertEqual(runner.audit(moved),summary)
            with self.assertRaisesRegex(ValueError,'audit-only'):runner.execute(moved)

    def test_failed_and_completed_runs_never_relaunch(self):
        for status in ('failed','diagnostic-collected'):
            atomic_json(self.root/'diagnostic_state.json',{'status':status})
            with contextlib.ExitStack() as stack:
                self.mocks(stack)
                with self.assertRaisesRegex(ValueError,'cannot be rerun'):runner.execute(self.root)
            self.assertEqual(self.calls,[])

    def test_cleanup_failure_blocks_every_gpu_command(self):
        with contextlib.ExitStack() as stack:
            self.mocks(stack)
            stack.enter_context(patch.object(runner,'verify_interrupted_gpu_groups',side_effect=ValueError('cleanup unconfirmed')))
            with self.assertRaisesRegex(ValueError,'cleanup unconfirmed'):runner.execute(self.root)
            self.assertFalse(any(c[0] in ('command','collect') for c in self.calls))
            self.assertEqual(runner.audit(self.root)['status'],'diagnostic-failed')

    def test_insufficient_budget_checkpoints_without_collection(self):
        with contextlib.ExitStack() as stack:
            self.mocks(stack)
            stack.enter_context(patch.object(runner,'budget',side_effect=Checkpoint('less than 120+20')))
            self.assertEqual(runner.execute(self.root),75)
            self.assertFalse(any(c[0]=='collect' for c in self.calls))
            self.assertEqual(json.loads((self.root/'diagnostic_state.json').read_text())['status'],'checkpoint')

    def test_resource_or_environment_drift_rejected_before_collection(self):
        with contextlib.ExitStack() as stack:
            self.mocks(stack)
            stack.enter_context(patch.object(runner,'inspect_allocation',return_value={**self.env,'driver':'changed'}))
            with self.assertRaisesRegex(ValueError,'environment drift'):runner.execute(self.root)
            self.assertEqual(self.calls,[])
        def changed_query(*args,**kwargs):
            receipt=self.command(*args,**kwargs)
            p=self.root/(args[1]+'.stdout');rows=p.read_text().splitlines();rows[1]=json.dumps({'type':'resource','resource_identity':{'changed':True}});p.write_text('\n'.join(rows)+'\n')
            return receipt
        with contextlib.ExitStack() as stack:
            self.mocks(stack);stack.enter_context(patch.object(runner,'run_command',side_effect=changed_query))
            with self.assertRaisesRegex(ValueError,'resource/device differs'):runner.execute(self.root)
            self.assertFalse(any(c[0]=='collect' for c in self.calls))

    def test_unknown_or_failed_existing_receipt_not_retried(self):
        folder=self.root/'attempts/attempt_00';folder.mkdir()
        receipt=self.collect([str(self.root/'binary/probe'),'collect'],folder,folder/'raw.jsonl',folder/'stderr',120,uuid=self.device['uuid'])
        receipt['returncode']=2;atomic_json(folder/'receipt.json',receipt)
        atomic_json(self.root/'environment/device.json',self.device);self.calls.clear()
        with contextlib.ExitStack() as stack:
            self.mocks(stack)
            with self.assertRaisesRegex(ValueError,'unsuccessful command'):runner.execute(self.root)
            self.assertFalse(any(c[0] in ('collect','command') for c in self.calls))

    def test_interrupted_attempt_preserved_and_new_attempt_after_cleanup(self):
        folder=self.root/'attempts/attempt_00';folder.mkdir();(folder/'raw.jsonl').write_text('partial')
        atomic_json(self.root/'diagnostic_state.json',{'status':'running'})
        with contextlib.ExitStack() as stack:
            self.mocks(stack);self.assertEqual(runner.execute(self.root),0)
            self.assertEqual((folder/'raw.jsonl').read_text(),'partial')
            self.assertTrue((self.root/'attempts/attempt_01/receipt.json').exists())
            self.assertLess(self.calls.index(('cleanup',)),self.calls.index(('collect','collect',120)))

    def test_partial_process_failure_and_corrupt_receipt_cannot_pass(self):
        original_collect=self.collect
        def failed(*args,**kwargs):
            receipt=original_collect(*args,**kwargs);receipt['returncode']=2
            return receipt
        with contextlib.ExitStack() as stack:
            self.mocks(stack);stack.enter_context(patch.object(runner,'bounded',side_effect=failed))
            with self.assertRaisesRegex(ValueError,'collection failed'):runner.execute(self.root)
            self.assertEqual(runner.audit(self.root)['status'],'diagnostic-failed')
            self.assertTrue((self.root/'attempts/attempt_00/raw.jsonl').exists())
            (self.root/'attempts/attempt_00/raw.jsonl').write_text('tampered')
            with self.assertRaises(ValueError):runner.audit(self.root)

    def test_mandatory_reviews_cannot_be_deleted_from_spec(self):
        base=json.loads((ROOT/'contracts/low_precision_lowering_v3.json').read_text())
        required=runner.review_requirements(base)
        self.assertIn(('S08','long-accumulation-A','reviews/S08-long-accumulation-A-review.json'),required)
        self.assertIn(('S08','long-accumulation-source-B','reviews/S08-long-accumulation-source-B-review.json'),required)
        bindings=[{'stage':s,'phase':p,'path':n,'sha256':'a'*64} for s,p,n in required]
        for i in range(len(bindings)):
            with self.assertRaisesRegex(ValueError,'mandatory review set'):
                runner.verify_authorization(self.repo,self.suite,base,bindings[:i]+bindings[i+1:])
        with self.assertRaisesRegex(ValueError,'mandatory review set'):
            runner.verify_authorization(self.repo,self.suite,base,bindings+bindings[:1])
        # No source-B file => initialize cannot reach allocation inspection.
        actual_suite=REPO/'results/gh200_resource_campaign/20261001-resource-suite-v2'
        original_validate=runner.validate_gate
        def missing_source_gate(path,root,stage,phase):
            if phase=='long-accumulation-source-B':raise ValueError('fixture missing source-B')
            return original_validate(path,root,stage,phase)
        with patch.object(runner,'validate_gate',side_effect=missing_source_gate), patch.object(runner,'inspect_allocation') as inspect:
            with self.assertRaises(ValueError):
                runner.initialize(REPO,actual_suite,actual_suite/'low_precision/test-forbidden-unreviewed')
            inspect.assert_not_called()

    def test_argv_timeout_registration_and_raw_mutations(self):
        folder=self.root/'attempts/attempt_00';folder.mkdir()
        argv=[str(self.root/'binary/probe'),'collect']
        original=self.collect(argv,folder,folder/'raw.jsonl',folder/'stderr',120,uuid=self.device['uuid'])
        mutations=[lambda r:r.update(argv=argv+['8192']),lambda r:r.update(timeout_seconds=30),
                   lambda r:r.update(returncode=1),lambda r:r.update(timed_out=True),
                   lambda r:r.update(cleanup_confirmed=False),lambda r:r.update(stdout_sha256='0'*64),
                   lambda r:r['gpu_process_registration'].update(state='running'),
                   lambda r:r['gpu_process_registration'].update(gpu_uuid='wrong'),
                   lambda r:r['gpu_process_registration'].update(pid=101),
                   lambda r:r['gpu_process_registration'].update(cleanup_receipt={})]
        for mutate in mutations:
            receipt=copy.deepcopy(original);mutate(receipt)
            with self.assertRaises(ValueError):runner.verify_receipt(receipt,folder/'raw.jsonl',folder/'stderr',argv,120,self.device['uuid'])
        for content in ('{}\n','{"type":"device","type":"device"}\n{"type":"long_accumulation_diagnostic"}\n',
                        '{"type":"device"}\n{"type":"trial"}\n','{"type":"device"}\n{"type":"long_accumulation_diagnostic","value":NaN}\n'):
            (folder/'raw.jsonl').write_text(content)
            with self.assertRaises(ValueError):runner.strict_rows(folder/'raw.jsonl',runner.KIND)

    def test_real_auditor_collection_replay_and_complete_value_tamper(self):
        from auditors import low_precision_accumulation
        from test_low_precision_accumulation import AccumulationValueTests
        # Use the reviewed wrapper row schema through the actual pure auditor.
        fixture=AccumulationValueTests();fixture.setUp()
        device,row,contract,resource=fixture.device,fixture.row,fixture.contract,fixture.resource
        device['type']='device'
        self.device=device;self.resource=resource
        self.env['uuid']=device['uuid']
        atomic_json(self.root/'environment/initial.json',self.env)
        self.loaded=(self.spec,{},contract,self.repo/'baseline',device,resource,low_precision_accumulation)
        def real_collect(argv,cwd,stdout,stderr,timeout,**kwargs):
            stdout.write_text(json.dumps(device)+'\n'+json.dumps(row)+'\n');stderr.write_text('')
            return self.write_receipt(argv,stdout,stderr,timeout)
        with contextlib.ExitStack() as stack:
            self.mocks(stack);stack.enter_context(patch.object(runner,'bounded',side_effect=real_collect))
            self.assertEqual(runner.execute(self.root),0)
            summary=runner.audit(self.root)
            self.assertEqual(summary['evidence']['result']['output_count'],8192)
            raw=self.root/'attempts/attempt_00/raw.jsonl'
            corrupted=copy.deepcopy(row);corrupted['outputs'].pop()
            raw.write_text(json.dumps(device)+'\n'+json.dumps(corrupted)+'\n')
            with self.assertRaises(ValueError):runner.audit(self.root)

    def test_actual_build_equivalence_and_identity_mutations(self):
        from auditors import low_precision_accumulation
        baseline=REPO/'results/gh200_resource_campaign/20261001-resource-suite-v2'/runner.BASELINE
        base=json.loads((ROOT/'contracts/low_precision_lowering_v3.json').read_text())
        env=json.loads((baseline/'environment/initial.json').read_text())
        atomic_json(self.root/'environment/initial.json',env)
        for src,dst in [('binary/probe','binary/probe'),('build/sass.stdout','build/sass.txt'),
                        ('build/sass_audit.json','build/sass_audit.json'),('build/dependencies.json','build/dependencies.json'),
                        ('build/shared_libraries.json','build/shared_libraries.json')]:
            shutil.copyfile(baseline/src,self.root/dst)
        shutil.copytree(baseline/'snapshot',self.root/'snapshot')
        wrapper=self.root/'snapshot/repo'/runner.WRAPPER;wrapper.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/runner.WRAPPER,wrapper)
        mapping=json.loads((self.root/'snapshot/manifest.json').read_text());mapping['repo/'+runner.WRAPPER]=sha(wrapper)
        atomic_json(self.root/'snapshot/manifest.json',mapping)
        deps=json.loads((self.root/'build/dependencies.json').read_text());deps['local'].append('repo/'+runner.WRAPPER)
        atomic_json(self.root/'build/dependencies.json',deps)
        spec={**self.spec,'source_path':'snapshot/repo/'+runner.WRAPPER}
        for key,name in [('binary_sha256','binary/probe'),('sass_sha256','build/sass.txt'),
                         ('dependencies_sha256','build/dependencies.json'),('shared_libraries_sha256','build/shared_libraries.json')]:spec[key]=sha(self.root/name)
        command=[env['tools']['nvcc']['path'],*base['build']['flags'],'-MD','-MF',str(self.root/'build/probe.d'),str(self.root/spec['source_path']),'-o',str(self.root/'binary/probe')]
        atomic_json(self.root/'build/command.json',{'argv':command,'source_relative':spec['source_path'],'target_relative':'binary/probe'})
        for name,argv,timeout in [('compile',command,180),('disassemble',[env['tools']['cuobjdump']['path'],'--dump-sass',str(self.root/'binary/probe')],120),('ldd',['ldd',str(self.root/'binary/probe')],120)]:
            stdout=self.root/'build'/f'{name}.stdout';stderr=self.root/'build'/f'{name}.stderr'
            stdout.write_text('fixture command evidence');stderr.write_text('')
            atomic_json(self.root/'build'/f'{name}.receipt.json',self.write_receipt(argv,stdout,stderr,timeout))
        result=runner.verify_build(self.root,spec,base,baseline,low_precision_accumulation)
        self.assertEqual(result['status'],'identical_target_instructions')
        original=copy.deepcopy(deps)
        for field in ('local','external_toolchain'):
            changed=copy.deepcopy(original)
            if field=='local':changed[field].remove('repo/'+runner.WRAPPER)
            else:changed[field][next(iter(changed[field]))]='0'*64
            atomic_json(self.root/'build/dependencies.json',changed)
            altered={**spec,'dependencies_sha256':sha(self.root/'build/dependencies.json')}
            with self.assertRaises(ValueError):runner.verify_build(self.root,altered,base,baseline,low_precision_accumulation)
        atomic_json(self.root/'build/dependencies.json',original)
        atomic_json(self.root/'build/command.json',{'argv':command,'source_relative':'snapshot/repo/wrong.cu','target_relative':'binary/probe'})
        with self.assertRaisesRegex(ValueError,'source/flags'):runner.verify_build(self.root,spec,base,baseline,low_precision_accumulation)

    def test_compile_override_retains_long_kind_and_never_formal_dispatches(self):
        self.spec.pop('binary_sha256');self.spec['source_path']='snapshot/repo/'+runner.WRAPPER
        self.loaded=(self.spec,{'source':runner.BASE_CONTRACT}, {},self.repo/'baseline',self.device,self.resource,self.module)
        def compile_mock(root,spec,base,env):
            self.assertEqual(spec['kind'],runner.KIND)
            self.assertEqual(spec['source_path'],'snapshot/repo/'+runner.WRAPPER)
            spec['binary_sha256']='new compiler hash'
            atomic_json(root/'run_spec.json',spec)
            self.calls.append(('compile',))
        with contextlib.ExitStack() as stack:
            self.mocks(stack);stack.enter_context(patch.object(runner,'compile_probe',side_effect=compile_mock))
            self.assertEqual(runner.execute(self.root),0)
            self.assertEqual(json.loads((self.root/'run_spec.json').read_text())['kind'],runner.KIND)
            self.assertLess(self.calls.index(('compile',)),self.calls.index(('collect','collect',120)))

    def test_load_rejects_formal_kind_and_source_override_tampering(self):
        good={'schema_version':2,'kind':runner.KIND,'fixture':False,**{k:False for k in runner.FALSE_FLAGS},
              'source_path':'snapshot/repo/'+runner.WRAPPER,'contract_path':'snapshot/repo/'+runner.BASE_CONTRACT,
              'diagnostic_path':'snapshot/repo/'+runner.DIAGNOSTIC,'case_id':'wgmma_e4m3_g1_one_cta',
              'profile_id':'wgmma_e4m3_uniform_8192_full_output'}
        for field,value in [('kind','formal'),('fixture',True),('family_B3_eligible',True),
                            ('source_path','snapshot/repo/'+runner.CAMPAIGN+'/probes/low_precision.cu'),
                            ('contract_path','snapshot/repo/other.json'),('profile_id','short_uniform_nonuniform_1_2')]:
            atomic_json(self.root/'run_spec.json',{**good,field:value})
            with patch.object(runner,'check_snapshot') as snapshot:
                with self.assertRaises(ValueError):runner.load(self.root)
                snapshot.assert_not_called()

    def test_original_baseline_offline_integrity(self):
        baseline=REPO/'results/gh200_resource_campaign/20261001-resource-suite-v2'/runner.BASELINE
        _,_,device,resource=runner.baseline_evidence(baseline)
        self.assertEqual(resource['kernel_symbol'],'lp_wgmma_e4m3_g1')
        self.assertEqual(resource['registers_per_thread'],148)
        self.assertIn('GH200',device['name'])


if __name__=='__main__':unittest.main()
