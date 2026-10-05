"""Mocked boundary runner: at most four profile processes, including resume."""
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.suite_io import atomic_json,sha
from runners import accumulation_boundary as runner
from runners.environment import Checkpoint


class BoundaryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.repo=Path(self.temp.name)
        self.root=self.repo/'suite/low_precision/run'
        for n in ('environment','build','binary','attempts'):(self.root/n).mkdir(parents=True,exist_ok=True)
        self.device={'type':'device','uuid':'GPU-00000000-0000-0000-0000-000000000000'}
        self.resource={'kernel_symbol':'lp_wgmma_e4m3_g1'}
        self.env={'uuid':self.device['uuid'],'name':'GH200','driver':'same','compiler':'12.9','tools':{},'execution_uid':1000}
        self.profiles=['original_uniform_'+str(n) for n in (31,32,33,64)]
        self.spec={'schema_version':2,'kind':runner.KIND,'execution_root':str(self.root),
            'case_id':'wgmma_e4m3_g1_one_cta','profile_ids':self.profiles,'binary_sha256':'fixture',
            **{k:False for k in runner.FALSE_FLAGS}}
        module=types.SimpleNamespace(audit_values=lambda *a:{'status':'collected_finite_difference','profile_id':a[-1]})
        self.loaded=(self.spec,{}, {},self.repo/'baseline',self.device,self.resource,module)
        atomic_json(self.root/'environment/initial.json',self.env)
        atomic_json(self.root/'diagnostic_state.json',{'status':'pending'})
        self.calls=[]

    def tearDown(self):self.temp.cleanup()

    def receipt(self,argv,stdout,stderr,timeout):
        r={'argv':argv,'returncode':0,'timed_out':False,'cleanup_confirmed':True,'timeout_seconds':timeout,
           'host_start_ns':10,'host_stop_ns':20,'pid':100,'pgid':100,'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr)}
        r['gpu_process_registration']={'state':'cleanup_confirmed','gpu_uuid':self.device['uuid'].lower(),
            'argv':argv,'pid':100,'pgid':100,'cleanup_receipt':copy.deepcopy(r)}
        return r

    def query(self,root,name,argv,timeout,env,**kwargs):
        self.calls.append(('query',))
        self.assertEqual(argv,[str(self.root/'binary/probe'),'device']);self.assertEqual(timeout,30)
        stdout=root/(name+'.stdout');stderr=root/(name+'.stderr')
        stdout.write_text(json.dumps(self.device)+'\n'+json.dumps({'type':'resource','resource_identity':self.resource})+'\n');stderr.write_text('')
        r=self.receipt(argv,stdout,stderr,timeout);atomic_json(root/(name+'.receipt.json'),r);return r

    def collect(self,argv,cwd,stdout,stderr,timeout,**kwargs):
        self.assertEqual(argv[:2],[str(self.root/'binary/probe'),'collect'])
        self.assertEqual(len(argv),3);self.assertIn(argv[2],self.profiles);self.assertEqual(timeout,120)
        self.calls.append(('collect',argv[2]))
        stdout.write_text(json.dumps(self.device)+'\n'+json.dumps({'type':runner.KIND,'profile_id':argv[2],'iterations':int(argv[2].rsplit('_',1)[1]),'outputs':[{'index':i,'value':64.0,'f32_bits':1115684864} for i in range(8192)]})+'\n');stderr.write_text('')
        return self.receipt(argv,stdout,stderr,timeout)

    def mocks(self,stack):
        for name,value in [('load',self.loaded),('inspect_allocation',self.env),('verify_build',{'equal':True})]:
            stack.enter_context(patch.object(runner,name,return_value=value))
        stack.enter_context(patch.object(runner,'file_lock',side_effect=lambda p:contextlib.nullcontext()))
        stack.enter_context(patch.object(runner,'gpu_lock',side_effect=lambda u:contextlib.nullcontext([])))
        stack.enter_context(patch.object(runner,'verify_interrupted_gpu_groups',side_effect=lambda *a:self.calls.append(('cleanup',))))
        stack.enter_context(patch.object(runner,'budget',return_value=None))
        stack.enter_context(patch.object(runner,'run_command',side_effect=self.query))
        stack.enter_context(patch.object(runner,'bounded',side_effect=self.collect))

    def test_four_processes_relocation_and_no_rerun(self):
        with contextlib.ExitStack() as stack:
            self.mocks(stack);self.assertEqual(runner.execute(self.root),0)
            self.assertEqual([c[1] for c in self.calls if c[0]=='collect'],self.profiles)
            summary=runner.audit(self.root)
            self.assertEqual(set(summary['evidence']),set(self.profiles))
            moved=self.repo/'moved';shutil.copytree(self.root,moved)
            self.assertEqual(runner.audit(moved),summary)
            with self.assertRaisesRegex(ValueError,'audit-only'):runner.execute(moved)
            with self.assertRaisesRegex(ValueError,'cannot be rerun'):runner.execute(self.root)
            self.assertEqual(sum(c[0]=='collect' for c in self.calls),4)

    def test_checkpoint_resumes_only_unstarted_profiles(self):
        budgets=iter([None,None,Checkpoint('remaining budget')])
        def budget(*a):
            item=next(budgets)
            if isinstance(item,Exception):raise item
        with contextlib.ExitStack() as stack:
            self.mocks(stack)
            with patch.object(runner,'budget',side_effect=budget):self.assertEqual(runner.execute(self.root),75)
            first={p.name:sha(p/'raw.jsonl') for p in (self.root/'attempts').iterdir()}
            self.assertEqual(len(first),2)
            self.assertEqual(runner.execute(self.root),0)
            self.assertEqual([c[1] for c in self.calls if c[0]=='collect'],self.profiles)
            self.assertEqual(first,{n:sha(self.root/'attempts'/n/'raw.jsonl') for n in first})
            runner.audit(self.root)

    def test_uncertain_execution_and_unknown_profile_never_retry(self):
        for name in (self.profiles[0],'original_uniform_8192'):
            with self.subTest(name=name):
                folder=self.root/'attempts'/name;folder.mkdir();(folder/'raw.jsonl').write_text('partial')
                atomic_json(self.root/'diagnostic_state.json',{'status':'pending'})
                for f in ('failure_summary.json','failure_manifest.json'):(self.root/f).unlink(missing_ok=True)
                self.calls.clear()
                with contextlib.ExitStack() as stack:
                    self.mocks(stack)
                    with self.assertRaises(ValueError):runner.execute(self.root)
                    self.assertFalse(any(c[0] in ('query','collect') for c in self.calls))
                    self.assertEqual((folder/'raw.jsonl').read_text(),'partial')
                shutil.rmtree(folder)

    def test_failure_stops_remaining_profiles_and_preserves_raw(self):
        def fail(*args,**kwargs):
            r=self.collect(*args,**kwargs);r['returncode']=2;return r
        with contextlib.ExitStack() as stack:
            self.mocks(stack);stack.enter_context(patch.object(runner,'bounded',side_effect=fail))
            with self.assertRaisesRegex(ValueError,'collection failed'):runner.execute(self.root)
            self.assertEqual(sum(c[0]=='collect' for c in self.calls),1)
            self.assertEqual(runner.audit(self.root)['status'],'diagnostic-failed')
            with self.assertRaisesRegex(ValueError,'cannot be rerun'):runner.execute(self.root)

    def test_missing_review_blocks_before_allocation(self):
        with patch.object(runner,'verify_authorization',side_effect=ValueError('missing source B')),patch.object(runner,'inspect_allocation') as allocation:
            repo=Path(__file__).resolve().parents[3]
            suite=repo/'results/gh200_resource_campaign/20261001-resource-suite-v2'
            with self.assertRaisesRegex(ValueError,'missing source B'):
                runner.initialize(repo,suite,suite/'low_precision/forbidden-boundary-test')
            allocation.assert_not_called()

    def test_partial_profile_set_cannot_seal(self):
        with self.assertRaisesRegex(ValueError,'all four'):runner.seal(self.root,self.spec,{self.profiles[0]:{}})


if __name__=='__main__':unittest.main()
