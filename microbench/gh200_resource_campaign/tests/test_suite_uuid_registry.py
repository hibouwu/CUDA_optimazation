"""UUID lifecycle adversarial tests use CPU children only, never CUDA or NCU."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid as uuidlib
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from common.suite_io import atomic_json,bounded,gpu_lock,gpu_paths,read_json,sha
from common import gpu_registry
from auditors.suite import recompute,validate_counter_evidence
from runners.environment import counters


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.uuid='GPU-'+str(uuidlib.uuid4());self.child_pgid=None
    def tearDown(self):
        if self.child_pgid:
            try: os.killpg(self.child_pgid,signal.SIGKILL)
            except ProcessLookupError: pass
        registry=gpu_registry.registry_root(self.uuid)
        if registry.exists():shutil.rmtree(registry)
        for path in gpu_paths(self.uuid):
            if path.exists():path.unlink()
        self.temp.cleanup()

    def test_normal_bounded_process_has_intent_spawn_and_cleanup_history(self):
        with gpu_lock(self.uuid):
            rec=bounded([sys.executable,'-c','print("cpu only")'],self.root,self.root/'out',self.root/'err',5,uuid=self.uuid)
        registration=rec['gpu_process_registration']
        self.assertEqual([e['state'] for e in registration['events']],['launch_intent','spawned','cleanup_confirmed'])
        self.assertEqual(registration['pid'],registration['pgid']);self.assertGreater(registration['start_ticks'],0)
        self.assertFalse(list((gpu_registry.registry_root(self.uuid)/'active').glob('*.json')))
        self.assertEqual(len(list((gpu_registry.registry_root(self.uuid)/'closed').glob('*.json'))),1)

    def test_unknown_spawn_window_quarantines_before_next_Popen(self):
        gpu_registry.begin(self.uuid,['unused'],self.root,self.root/'old.out',self.root/'old.err')
        with patch('common.suite_io.subprocess.Popen',side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(ValueError,'quarantined'):
                bounded(['unused'],self.root,self.root/'new.out',self.root/'new.err',5,uuid=self.uuid)
        self.assertTrue(gpu_paths(self.uuid)[1].exists())

    def test_parent_SIGKILL_new_run_discovers_prior_CPU_child(self):
        old=self.root/'old_run';new=self.root/'new_run';old.mkdir();new.mkdir()
        code='''import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
from common.suite_io import bounded,gpu_lock
root=Path(sys.argv[2]);identity=sys.argv[3]
with gpu_lock(identity):
 bounded([sys.executable,'-c','import time; print("ready",flush=True); time.sleep(60)'],root,root/'out',root/'err',90,uuid=identity)
'''
        parent=subprocess.Popen([sys.executable,'-B','-c',code,str(HERE),str(old),self.uuid],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        registration=None
        try:
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                for path in (gpu_registry.registry_root(self.uuid)/'active').glob('*.json'):
                    candidate=read_json(path)
                    if candidate.get('state')=='spawned':registration=candidate
                if registration and (old/'out').exists() and 'ready' in (old/'out').read_text():break
                time.sleep(.02)
            self.assertIsNotNone(registration)
            self.child_pgid=registration['pgid'];parent.kill();parent.wait(timeout=5)
            marker=new/'would_execute'
            rejected=False
            try:
                with gpu_lock(self.uuid):
                    bounded([sys.executable,'-c',f'from pathlib import Path;Path({str(marker)!r}).write_text("new")'],new,new/'out',new/'err',5,uuid=self.uuid)
            except ValueError:
                rejected=True
            # Either proven cleanup permits the new command, or quarantine blocks it.
            if rejected:
                self.assertFalse(marker.exists());self.assertTrue(gpu_paths(self.uuid)[1].exists())
            else:
                self.assertTrue(marker.exists())
                closed=list((gpu_registry.registry_root(self.uuid)/'closed').glob('*.json'))
                old_record=next(read_json(p) for p in closed if read_json(p)['registry_id']==registration['registry_id'])
                self.assertIs(old_record['cleanup_receipt']['cleanup_confirmed'],True)
            if Path(f'/proc/{self.child_pgid}/stat').exists():
                self.assertEqual(Path(f'/proc/{self.child_pgid}/stat').read_text().split(') ',1)[1].split()[0],'Z')
        finally:
            if parent.poll() is None:parent.kill();parent.wait(timeout=5)
            if parent.stderr:parent.stderr.close()

    def test_old_NCU_fingerprint_unknown_identity_blocks_new_run(self):
        cache=Path('/tmp')/('gh200-suite-v2-ncu-cpu-registry-'+uuidlib.uuid4().hex);cache.mkdir()
        atomic_json(cache/'status.json',{'fingerprint':{'gpu_uuid':self.uuid,'driver_version':'old'},'state':'started_outcome_unknown'})
        atomic_json(cache/'stdout.active.json',{'gpu_uuid':self.uuid,'host':'unproven-host','pid':123,'pgid':123,'start_ticks':1})
        try:
            with self.assertRaisesRegex(ValueError,'legacy_NCU'):
                with gpu_lock(self.uuid):self.fail('must not acquire permission to measure')
            self.assertTrue((cache/'reconciliation.json').exists())
        finally:shutil.rmtree(cache)

    def test_current_NCU_unknown_cache_is_not_reused_after_confirmed_cleanup(self):
        cache=Path('/tmp')/('gh200-suite-v2-ncu-cpu-registry-'+uuidlib.uuid4().hex);cache.mkdir()
        fingerprint={'gpu_uuid':self.uuid,'driver_version':'old'}
        atomic_json(cache/'status.json',{'fingerprint':fingerprint,'state':'started_outcome_unknown'})
        atomic_json(cache/'stdout.active.json',{'gpu_uuid':self.uuid,'host':os.uname().nodename,'uid':os.geteuid(),'pid':99999999,'pgid':99999999,'start_ticks':1})
        (self.root/'environment').mkdir()
        key=cache.name.removeprefix('gh200-suite-v2-ncu-')
        try:
            with patch('runners.environment.shutil.which',return_value='/fake/ncu'),patch('runners.environment.permission_fingerprint',return_value=fingerprint),patch('runners.environment.digest',return_value=key),patch('runners.environment.bounded',side_effect=AssertionError('no blind retry')):
                with self.assertRaisesRegex(ValueError,'cached NCU outcome'):
                    counters(self.root,{}, {'cases':[]},{'uuid':self.uuid})
            self.assertIs(read_json(cache/'reconciliation.json')['cleanup_confirmed'],True)
        finally:shutil.rmtree(cache)


class CounterTerminalTests(unittest.TestCase):
    def test_all_unknown_or_execution_errors_are_not_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'batches').mkdir();(root/'environment').mkdir()
            spec={'kind':'formal','snapshot_manifest_sha256':'x','binary_sha256':'x'}
            for state in ('not_checked','started_outcome_unknown','cleanup_unconfirmed','failed_unknown','timeout'):
                atomic_json(root/'environment/ncu_status.json',{'state':state})
                with self.subTest(state=state),patch('auditors.suite.load_run',return_value=(spec,{'cases':[],'family':'CPU fixture','stage':'S03'},{},{},None)):
                    with self.assertRaisesRegex(ValueError,'counter state'):recompute(root,require_terminal=True)

    def test_tool_missing_requires_recorded_lookup_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);device={'uuid':'GPU-'+str(uuidlib.uuid4())}
            value={'state':'tool_missing','purpose':'permission_capability_only','family_profile':False,'cache_residency_proven':False,'physical_hbm_bytes_proven':False,'fingerprint':{'gpu_uuid':device['uuid']}}
            with self.assertRaises(ValueError):validate_counter_evidence(root,value,device)
            atomic_json(root/'environment/ncu_tool_lookup.json',{'executable':'ncu','resolved_path':None,'PATH':'/fixture/no-ncu'})
            value.update(unavailable_reason='executable_not_found',evidence_sha256={'tool_lookup.json':sha(root/'environment/ncu_tool_lookup.json')})
            validate_counter_evidence(root,value,device)
            (root/'environment/ncu_tool_lookup.json').write_text('{}')
            with self.assertRaises(ValueError):validate_counter_evidence(root,value,device)


class TelemetryFailureTests(unittest.TestCase):
    setUp = RegistryTests.setUp
    tearDown = RegistryTests.tearDown
    def check_failure(self, phase):
        from runners.suite_runner import telemetry
        (self.root/'environment').mkdir()
        real_popen=subprocess.Popen
        children=[]
        def cpu_child(*args,**kwargs):
            child=real_popen([sys.executable,'-c','import time;time.sleep(60)'],stdout=kwargs['stdout'],stderr=kwargs['stderr'],start_new_session=True)
            children.append(child);return child
        real_atomic=atomic_json
        def fail_active(path,value):
            if Path(path).name=='telemetry.active.json':raise OSError('injected active write failure')
            return real_atomic(path,value)
        env={'uuid':self.uuid,'tools':{'nvidia-smi':{'path':'/unused/mock'}}}
        patcher=patch('common.gpu_registry.spawned',side_effect=OSError('injected spawned record failure')) if phase=='spawned' else patch('runners.suite_runner.atomic_json',side_effect=fail_active)
        with patch('runners.suite_runner.subprocess.Popen',side_effect=cpu_child),patcher:
            with self.assertRaises(OSError):
                with telemetry(self.root,env):self.fail('setup must fail before yield')
        self.assertEqual(len(children),1);self.assertIsNotNone(children[0].poll())
        self.assertFalse(list((gpu_registry.registry_root(self.uuid)/'active').glob('*.json')))
        records=list((gpu_registry.registry_root(self.uuid)/'closed').glob('*.json'))
        self.assertEqual(len(records),1)
        self.assertIs(read_json(records[0])['cleanup_receipt']['cleanup_confirmed'],True)

    def test_spawned_registry_write_failure_still_stops_real_child(self):self.check_failure('spawned')
    def test_archive_active_write_failure_still_stops_real_child(self):self.check_failure('active')
