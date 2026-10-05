"""Adversarial CPU-only archive fixtures; tests do not attest actual GPU execution."""
import copy
import json
import random
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path(__file__).parent))
from test_suite_core import CONTRACT, PROTOCOL, DEVICE, raw
from common.suite_io import atomic_json, digest, read_json, sha, state
from common.suite_snapshot import freeze, check_snapshot
from auditors import memory_baseline
from auditors.suite import audit, case_evidence, load_run, recompute, report, trial_evidence
from runners.suite_runner import measurement_manifest, finalize


def create_sample(root, case, spec, batch, index, elapsed=1000, *, warm=True):
    folder = root/'batches'/case['id']/f'batch_{batch:02d}'/f'trial_{index:02d}'/'attempt_00'
    folder.mkdir(parents=True)
    row = raw(case); seed = 3 + index*19 + batch*1009; row['seed'] = seed
    row['stop_ns'] = row['start_ns'] + elapsed
    row['event_ms'] = elapsed / 1e6 * 2
    for block in row['blocks_detail']:
        block['stop_ns'] = block['start_ns'] + elapsed
        block['stop_cycle'] = block['start_cycle'] + elapsed * 2
    if not warm:
        row['warmup_samples_ns'] = [1,100]*15; row['warmup_converged'] = False
    (folder/'raw.jsonl').write_text(json.dumps(DEVICE)+'\n'+json.dumps(row)+'\n')
    (folder/'stderr').write_text('')
    number = batch*100 + index
    receipt = {'status':'valid' if warm else 'warmup_unconverged', 'case_id':case['id'], 'batch':batch, 'trial':index, 'seed':seed,
               'relative_command':['binary/probe',case['id'],str(case['iterations']),str(seed)], 'binary_sha256':spec['binary_sha256'],
               'raw_sha256':sha(folder/'raw.jsonl'), 'stderr_sha256':sha(folder/'stderr'), 'pid':1000+number, 'pgid':1000+number,
               'host_start_ns':10000+number*10000, 'host_stop_ns':15000+number*10000, 'returncode':0, 'timed_out':False, 'cleanup_confirmed':True}
    atomic_json(folder/'receipt.json',receipt)
    return folder


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root/'batches').mkdir(); atomic_json(self.root/'contract.json',CONTRACT)
        self.spec = {'binary_sha256':'a'*64,'contract_path':'contract.json','snapshot_manifest_sha256':'b'*64,'kind':'formal'}
        self.case = CONTRACT['cases'][0]
    def tearDown(self): self.temp.cleanup()

    def test_ten_independent_processes_one_complete_batch(self):
        for index in range(10): create_sample(self.root,self.case,self.spec,0,index)
        result = case_evidence(self.root,self.spec,self.case,DEVICE,PROTOCOL)
        self.assertEqual(result['status'],'stable'); self.assertEqual(result['merged']['samples'],10)
        self.assertEqual(len({s['pid'] for s in result['batches'][0]['samples']}),10)

    def test_missing_trial_is_pending_not_complete(self):
        for index in range(9): create_sample(self.root,self.case,self.spec,0,index)
        result = case_evidence(self.root,self.spec,self.case,DEVICE,PROTOCOL)
        self.assertEqual(result['status'],'pending'); self.assertEqual(result['merged']['samples'],9)

    def test_raw_tamper_and_duplicate_process_attempt_rejected(self):
        folder = create_sample(self.root,self.case,self.spec,0,0)
        source = folder/'raw.jsonl'; original = source.read_text(); source.write_text(original+'\n')
        with self.assertRaises(ValueError): case_evidence(self.root,self.spec,self.case,DEVICE,PROTOCOL)
        source.write_text(original)
        shutil.copytree(folder,folder.parent/'attempt_01')
        with self.assertRaises(ValueError): case_evidence(self.root,self.spec,self.case,DEVICE,PROTOCOL)

    def test_high_CV_requires_full_batches_and_never_selects_best(self):
        for index in range(10): create_sample(self.root,self.case,self.spec,0,index,500 if index%2 else 2000)
        for index in range(10): create_sample(self.root,self.case,self.spec,1,index,1000)
        result = case_evidence(self.root,self.spec,self.case,DEVICE,PROTOCOL)
        self.assertEqual(result['status'],'pending'); self.assertEqual(result['merged']['samples'],20)
        self.assertGreater(result['merged']['cv'],.05)
        for index in range(10): create_sample(self.root,self.case,self.spec,2,index,1000)
        result = case_evidence(self.root,self.spec,self.case,DEVICE,PROTOCOL)
        self.assertEqual(result['status'],'unstable_after_bounded_remeasurement')
        self.assertEqual(result['merged']['samples'],30); self.assertFalse(result['exportable'])

    def test_extra_batch_after_stable_rejected(self):
        for index in range(10): create_sample(self.root,self.case,self.spec,0,index)
        create_sample(self.root,self.case,self.spec,1,0)
        with self.assertRaises(ValueError): case_evidence(self.root,self.spec,self.case,DEVICE,PROTOCOL)

    def test_warmup_failure_consumes_case_attempt_without_valid_sample(self):
        for batch in range(3): create_sample(self.root,self.case,self.spec,batch,0,warm=False)
        result = case_evidence(self.root,self.spec,self.case,DEVICE,PROTOCOL)
        self.assertEqual(result['status'],'unstable_after_bounded_remeasurement'); self.assertIsNone(result['merged'])
        other = CONTRACT['cases'][1]
        for index in range(10): create_sample(self.root,other,self.spec,0,index)
        self.assertEqual(case_evidence(self.root,self.spec,other,DEVICE,PROTOCOL)['status'],'stable')

    def test_numeric_process_failure_cannot_be_reported_as_gap(self):
        folder = create_sample(self.root,self.case,self.spec,0,0)
        receipt = read_json(folder/'receipt.json'); receipt['status']='failed'; atomic_json(folder/'receipt.json',receipt)
        with self.assertRaises(ValueError): case_evidence(self.root,self.spec,self.case,DEVICE,PROTOCOL)

    def test_relocated_readonly_raw_replay_keeps_hashes(self):
        for index in range(10): create_sample(self.root,self.case,self.spec,0,index)
        destination = self.root/'copy'; destination.mkdir()
        shutil.copytree(self.root/'batches',destination/'batches'); shutil.copyfile(self.root/'contract.json',destination/'contract.json')
        before = {str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}
        for p in destination.rglob('*'): p.chmod(0o555 if p.is_dir() else 0o444)
        self.assertEqual(case_evidence(destination,self.spec,self.case,DEVICE,PROTOCOL)['status'],'stable')
        self.assertEqual(before,{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()})
        for p in destination.rglob('*'): p.chmod(0o755 if p.is_dir() else 0o644)

    def test_CPU_fixture_never_loads_as_hardware_evidence(self):
        atomic_json(self.root/'run_spec.json',{'schema_version':2,'kind':'formal','fixture':True})
        with self.assertRaisesRegex(ValueError,'CPU fixture'): load_run(self.root)

    def test_snapshot_relocation_and_extra_dependency_rejection(self):
        repo=HERE.parents[1]; suite=repo/'results/gh200_resource_campaign/20261001-resource-suite-v2'
        source=self.root/'source'; source.mkdir()
        # Test storage relocation, not admission of an unreviewed formal core.
        spec=freeze(repo,suite,source,HERE/'contracts/memory_baseline.json',
                    preflight=True,case_diagnostic=True)
        destination=self.root/'moved'; shutil.move(source,destination)
        self.assertEqual(check_snapshot(destination,spec)['family'],'memory_baseline')
        (destination/'snapshot/repo/extra.py').write_text('unreviewed')
        with self.assertRaises(ValueError): check_snapshot(destination,spec)


if __name__=='__main__': unittest.main()


class CompletionReplayTests(unittest.TestCase):
    """Mock only the previously tested hardware-provenance loader, never arithmetic/hash audit."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)/'suite/family/run'; self.root.mkdir(parents=True)
        self.spec={'binary_sha256':'a'*64,'contract_path':'contract.json','snapshot_manifest_sha256':'b'*64,'kind':'formal'}
        atomic_json(self.root/'contract.json',CONTRACT)
        envdir=self.root/'environment';envdir.mkdir()
        (envdir/'ncu_stdout').write_text('ERR_NVGPUCTRPERM');(envdir/'ncu_stderr').write_text('')
        atomic_json(envdir/'ncu_receipt.json',{'returncode':1,'timed_out':False,'cleanup_confirmed':True,
            'stdout_sha256':sha(envdir/'ncu_stdout'),'stderr_sha256':sha(envdir/'ncu_stderr'),
            'gpu_process_registration':{'state':'cleanup_confirmed','gpu_uuid':DEVICE['uuid'].lower(),'cleanup_receipt':{'cleanup_confirmed':True}}})
        atomic_json(envdir/'ncu_status.json',{'state':'permission_denied','purpose':'permission_capability_only',
            'family_profile':False,'cache_residency_proven':False,'physical_hbm_bytes_proven':False,'fingerprint':{'gpu_uuid':DEVICE['uuid']},
            'evidence_sha256':{name:sha(envdir/('ncu_'+name)) for name in ('stdout','stderr','receipt.json')}})
        cases={c['id']:c for c in CONTRACT['cases']}; rng=random.Random(PROTOCOL['shuffle_seed']); order=[]
        for index in range(10):
            shuffled=sorted(cases); rng.shuffle(shuffled); order.extend([[key,index] for key in shuffled])
        for sequence,(key,index) in enumerate(order):
            folder=create_sample(self.root,cases[key],self.spec,0,index)
            rec=read_json(folder/'receipt.json')
            rec.update(pid=2000+sequence,pgid=2000+sequence,host_start_ns=10000+sequence*10000,host_stop_ns=15000+sequence*10000)
            atomic_json(folder/'receipt.json',rec)
        atomic_json(self.root/'batches/order_00.json',order)
        telemetry=self.root/'environment/telemetry_cpu_fixture'; telemetry.mkdir()
        (telemetry/'samples.csv').write_text('timestamp, uuid, clocks.sm\nnow, '+DEVICE['uuid']+', 1000 MHz\n'); (telemetry/'stderr').write_text('')
        atomic_json(telemetry/'receipt.json',{'cleanup_confirmed':True,'samples_sha256':sha(telemetry/'samples.csv'),'stderr_sha256':sha(telemetry/'stderr')})
        self.loader=patch('auditors.suite.load_run',return_value=(self.spec,CONTRACT,PROTOCOL,DEVICE,memory_baseline)); self.loader.start()
        summary=recompute(self.root,require_terminal=True)
        atomic_json(self.root/'summary.json',summary); (self.root/'REPORT.md').write_text(report(summary))
        state(self.root,'collected_pending_review',hardware_qualification=False)
        measurement_manifest(self.root)
    def tearDown(self): self.loader.stop(); self.temp.cleanup()

    def test_pending_C_audit_and_readonly_report_no_writes(self):
        before={p.relative_to(self.root).as_posix():sha(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertTrue(audit(self.root)['terminal']); self.assertIn('14',str(len(audit(self.root)['cases'])))
        self.assertEqual(before,{p.relative_to(self.root).as_posix():sha(p) for p in self.root.rglob('*') if p.is_file()})
        with self.assertRaises(ValueError): audit(self.root,complete=True)

    def test_summary_missing_raw_and_manifest_corruption_are_rejected(self):
        summary=read_json(self.root/'summary.json'); summary['cases'][0]['merged']['median']*=2; atomic_json(self.root/'summary.json',summary)
        with self.assertRaises(ValueError): audit(self.root)

    def test_finalize_requires_independent_C_gate_and_binds_COMPLETE(self):
        from common.suite_io import DIMENSIONS
        external=Path(self.temp.name)/'review.json'
        gate={'schema_version':2,'stage':'S04','phase':'C','status':'pass','reviewer':'CPU-test-independent-reviewer','implementer':'CPU-test-fixture-author',
              'checks':{key:{'status':'pass'} for key in DIMENSIONS},'findings':[],
              'gate_files':{name:sha(self.root/name) for name in ('summary.json','measurement_manifest.json')}}
        atomic_json(external,gate)
        result=finalize(self.root,self.root.parent.parent,external)
        self.assertEqual(result['summary_sha256'],sha(self.root/'summary.json')); audit(self.root,complete=True)
        (self.root/'COMPLETE').write_text('{}')
        with self.assertRaises(ValueError): audit(self.root,complete=True)

    def test_open_review_finding_cannot_finalize(self):
        from common.suite_io import DIMENSIONS
        external=Path(self.temp.name)/'review.json'
        gate={'schema_version':2,'stage':'S04','phase':'C','status':'pass','reviewer':'reviewer','implementer':'author',
              'checks':{key:{'status':'pass'} for key in DIMENSIONS},'findings':[{'status':'open'}],
              'gate_files':{name:sha(self.root/name) for name in ('summary.json','measurement_manifest.json')}}
        atomic_json(external,gate)
        with self.assertRaises(ValueError): finalize(self.root,self.root.parent.parent,external)
        self.assertFalse((self.root/'COMPLETE').exists())
