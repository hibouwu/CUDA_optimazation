"""Controlled-runtime CPU tests with mocked allocation/profiler; no GPU work."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common.suite_io import atomic_json, read_json, sha, suite_lock_path
from runners.environment import counters


class RuntimeTests(unittest.TestCase):
    def test_suite_lock_is_on_shared_suite_filesystem(self):
        path=Path('/shared/results/suite')
        self.assertEqual(suite_lock_path(path),Path('/shared/results/.suite.suite-v2.lock'))
        self.assertNotEqual(suite_lock_path(path).parent,Path('/tmp'))

    def test_ncu_fallback_is_permission_only_and_cache_prevents_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'environment').mkdir()
            key='cpu-test-'+root.name
            cache=Path('/tmp')/('gh200-suite-v2-ncu-'+key)
            self.assertFalse(cache.exists())
            calls=[]
            def fake_bounded(argv,cwd,out,err,timeout,**kw):
                calls.append(argv); out.write_text('ERR_NVGPUCTRPERM\n'); err.write_text('')
                return {'returncode':1,'timed_out':False,'cleanup_confirmed':True}
            contract={'cases':[{'id':'first_case','iterations':1}]}
            try:
                with patch('runners.environment.shutil.which',return_value='/fake/ncu'),patch('runners.environment.permission_fingerprint',return_value={'test':key}),patch('runners.environment.digest',return_value=key),patch('runners.environment.budget'),patch('runners.environment.bounded',side_effect=fake_bounded):
                    one=counters(root,{},contract,{'uuid':'GPU-12345678-1234-1234-1234-123456789abc'}); two=counters(root,{},contract,{'uuid':'GPU-12345678-1234-1234-1234-123456789abc'})
                self.assertEqual(len(calls),1); self.assertEqual(one,two)
                self.assertIn('regex:.*',calls[0]); self.assertEqual(one['purpose'],'permission_capability_only')
                self.assertFalse(one['family_profile']); self.assertFalse(one['cache_residency_proven']); self.assertFalse(one['physical_hbm_bytes_proven'])
                self.assertEqual(one['state'],'permission_denied')
            finally:
                if cache.exists(): shutil.rmtree(cache)

    def test_unknown_ncu_error_is_not_permission_denial(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'environment').mkdir(); key='cpu-test-'+root.name
            cache=Path('/tmp')/('gh200-suite-v2-ncu-'+key)
            def fail(argv,cwd,out,err,timeout,**kw):
                out.write_text('');err.write_text('kernel launch error')
                return {'returncode':2,'timed_out':False,'cleanup_confirmed':True}
            try:
                with patch('runners.environment.shutil.which',return_value='/fake/ncu'),patch('runners.environment.permission_fingerprint',return_value={'test':key}),patch('runners.environment.digest',return_value=key),patch('runners.environment.budget'),patch('runners.environment.bounded',side_effect=fail):
                    with self.assertRaises(ValueError): counters(root,{}, {'cases':[{'id':'x','iterations':1}]},{'uuid':'GPU-12345678-1234-1234-1234-123456789abc'})
                self.assertEqual(read_json(root/'environment/ncu_status.json')['state'],'failed_unknown')
            finally:
                if cache.exists(): shutil.rmtree(cache)


class RecoveryTests(unittest.TestCase):
    def test_orphan_command_stdout_preserved_then_new_attempt_commits(self):
        from runners.suite_runner import run_command
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'environment').mkdir()
            prefix=root/'environment/device_query'
            Path(str(prefix)+'.stdout').write_text('old uncommitted output')
            Path(str(prefix)+'.stderr').write_text('old stderr')
            atomic_json(Path(str(prefix)+'.stdout.active.json'),{'host':os.uname().nodename,'pid':99999999,'pgid':99999999,'start_ticks':1})
            with patch('runners.suite_runner.budget'),patch('runners.suite_runner.group_alive',return_value=False):
                rec=run_command(root,'environment/device_query',[sys.executable,'-c','print("new")'],5,{'uuid':'GPU-12345678-1234-1234-1234-123456789abc'},gpu=True)
            self.assertEqual(rec['returncode'],0)
            old=list((root/'environment').glob('device_query.interrupted_*/device_query.stdout'))
            self.assertEqual(len(old),1); self.assertEqual(old[0].read_text(),'old uncommitted output')
            self.assertEqual(Path(str(prefix)+'.stdout').read_text().strip(),'new')

    def test_interrupted_device_commit_recovers_missing_hash_but_rejects_tamper(self):
        from runners.suite_runner import bind_verified_device
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); atomic_json(root/'environment/device.json',{'uuid':'GPU-12345678-1234-1234-1234-123456789abc'})
            spec={'schema_version':2}; atomic_json(root/'run_spec.json',spec)
            bind_verified_device(root,spec)
            self.assertEqual(read_json(root/'run_spec.json')['device_sha256'],sha(root/'environment/device.json'))
            atomic_json(root/'environment/device.json',{'uuid':'changed'})
            with self.assertRaises(ValueError): bind_verified_device(root,spec)

    def test_NCU_cleanup_failure_overrides_permission_denial_and_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'environment').mkdir(); key='cpu-test-'+root.name
            cache=Path('/tmp')/('gh200-suite-v2-ncu-'+key)
            def fail(argv,cwd,out,err,timeout,**kw):
                out.write_text('ERR_NVGPUCTRPERM');err.write_text('')
                return {'returncode':None,'timed_out':True,'cleanup_confirmed':False}
            try:
                with patch('runners.environment.shutil.which',return_value='/fake/ncu'),patch('runners.environment.permission_fingerprint',return_value={'test':key}),patch('runners.environment.digest',return_value=key),patch('runners.environment.budget'),patch('runners.environment.bounded',side_effect=fail):
                    with self.assertRaises(ValueError): counters(root,{}, {'cases':[{'id':'x','iterations':1}]},{'uuid':'GPU-12345678-1234-1234-1234-123456789abc'})
                self.assertEqual(read_json(root/'environment/ncu_status.json')['state'],'cleanup_unconfirmed')
            finally:
                if cache.exists(): shutil.rmtree(cache)


class ToolchainDependencyTests(unittest.TestCase):
    def test_install_alias_resolves_to_precise_compiler_root(self):
        from common.suite_snapshot import inventory_depfile
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); run=base/'run'; (run/'snapshot/repo').mkdir(parents=True); (run/'build').mkdir()
            source=run/'snapshot/repo/probe.cu'; source.write_text('// source')
            atomic_json(run/'snapshot/manifest.json',{'repo/probe.cu':sha(source)})
            install=base/'gpfs/app/cuda-12.9'; (install/'include').mkdir(parents=True)
            header=install/'include/cuda_runtime.h'; header.write_text('// system header')
            alias=base/'apps-cuda'; alias.symlink_to(install,target_is_directory=True)
            depfile=run/'build/probe.d'; depfile.write_text('probe: '+str(source)+' '+str(alias/'include/cuda_runtime.h')+'\n')
            result=inventory_depfile(run,depfile,toolchain_roots=[alias])
            self.assertEqual(result,{str(header.resolve()):sha(header)})
            self.assertIn(str(install.resolve()),read_json(run/'build/dependencies.json')['trusted_toolchain_roots'])

    def test_external_user_checkout_is_not_an_install_dependency(self):
        from common.suite_snapshot import inventory_depfile
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); run=base/'run'; (run/'snapshot/repo').mkdir(parents=True); (run/'build').mkdir()
            atomic_json(run/'snapshot/manifest.json',{'repo/frozen.h':'a'*64})
            install=base/'gpfs/app/cuda'; install.mkdir(parents=True)
            outsider=base/'gpfs/users/project/unfrozen.h'; outsider.parent.mkdir(parents=True); outsider.write_text('project header')
            depfile=run/'build/probe.d'; depfile.write_text('probe: '+str(outsider)+'\n')
            with self.assertRaisesRegex(ValueError,'non-toolchain dependency'):
                inventory_depfile(run,depfile,toolchain_roots=[install])


class IsolationTests(unittest.TestCase):
    def test_orphan_GPU_group_blocks_before_new_device_work(self):
        from runners.suite_runner import verify_interrupted_gpu_groups
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); active=root/'batches/case/attempt_00/raw.jsonl.active.json'
            atomic_json(active,{'gpu_uuid':'GPU-test','host':os.uname().nodename,'pid':111,'pgid':111,'start_ticks':1234})
            quarantine=root/'quarantine.json'
            with patch('runners.suite_runner.group_alive',return_value=True),patch('runners.suite_runner.proc_start_ticks',return_value=None),patch('runners.suite_runner.gpu_paths',return_value=(root/'gpu.lock',quarantine)):
                with self.assertRaisesRegex(ValueError,'no new device commands'):
                    verify_interrupted_gpu_groups(root,{'uuid':'GPU-test'})
            self.assertEqual(read_json(quarantine)['reason'],'orphan_gpu_group_unconfirmed')

    def test_same_UUID_lock_identity_is_independent_of_suite_and_job(self):
        from common.suite_io import gpu_paths
        uuid='GPU-12345678-1234-1234-1234-123456789abc'
        with patch.dict(os.environ,{'SLURM_JOB_ID':'100'}): first=gpu_paths(uuid)
        with patch.dict(os.environ,{'SLURM_JOB_ID':'200'}): second=gpu_paths(uuid)
        self.assertEqual(first,second); self.assertNotEqual(first[0],suite_lock_path('/shared/results/suite'))

    def test_gate_stale_source_and_self_review_are_rejected(self):
        from common.suite_io import DIMENSIONS,validate_gate
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'probe.cu'; source.write_text('source')
            review={'schema_version':2,'stage':'S04','phase':'A','status':'pass','reviewer':'independent','implementer':'author',
                    'checks':{k:{'status':'pass'} for k in DIMENSIONS},'findings':[],'gate_files':{'probe.cu':sha(source)}}
            atomic_json(root/'review.json',review); validate_gate(root/'review.json',root,'S04','A')
            source.write_text('changed')
            with self.assertRaises(ValueError): validate_gate(root/'review.json',root,'S04','A')
            source.write_text('source'); review['reviewer']='author'; atomic_json(root/'review.json',review)
            with self.assertRaises(ValueError): validate_gate(root/'review.json',root,'S04','A')
