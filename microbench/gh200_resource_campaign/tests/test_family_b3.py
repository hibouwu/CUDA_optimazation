"""CPU-only admission tests against copied, genuine S12 short archives.

Temporary integration gates below are test fixtures, never saved to the suite
or used to launch GPU commands. They exercise freeze/load without pretending
that the new implementation has already received an independent B review.
"""
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
import json
import shutil
import sys
import tempfile
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
sys.path.insert(0, str(BASE))
from common import family_b3 as b3
from common.suite_io import atomic_json, read_json, sha, DIMENSIONS
from common.suite_snapshot import freeze, required_reviews, check_snapshot
from auditors.suite import adapter, load_run
from runners import suite_runner
from runners import validation_diagnostic

SUITE_NAME = 'results/gh200_resource_campaign/20261001-resource-suite-v2'
CONTRACT = 'microbench/gh200_resource_campaign/contracts/async_copy_formal_v1.json'


class FamilyB3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = read_json(REPO/CONTRACT)
        cls.bundle = b3.validate_bundle(REPO, cls.original)
        cls.temporary = tempfile.TemporaryDirectory(prefix='family-b3-CPU-tests-')
        cls.repo = Path(cls.temporary.name)/'repo'
        names = cls.bundle['names'] | {CONTRACT}
        for stage, phase, name in required_reviews(cls.original, True):
            path = REPO/SUITE_NAME/name
            if path.exists():
                names.add(path.relative_to(REPO).as_posix())
                names.update(read_json(path)['gate_files'])
        # Copy the current implementation and tests too; no links to live files.
        for path in BASE.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                names.add(path.relative_to(REPO).as_posix())
        for name in names:
            target = cls.repo/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO/name, target)
        cls.suite = cls.repo/SUITE_NAME

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def changed(self, path, value):
        original = path.read_bytes()
        try:
            if isinstance(value, bytes):
                path.write_bytes(value)
            else:
                atomic_json(path, value)
            yield
        finally:
            path.write_bytes(original)

    def contract(self):
        return deepcopy(self.original)

    def test_complete_bundle_and_original_contract_unchanged(self):
        result = b3.validate_bundle(self.repo, self.contract())
        self.assertEqual(len(result['cases']), 24)
        self.assertEqual(len(result['names']), len(self.bundle['names']))
        old = read_json(self.repo/self.original['family_b3']['parent_contract']['path'])
        self.assertFalse(b3.enabled(old))
        self.assertEqual(old['cases'], self.original['cases'])

    def test_unknown_revision_and_missing_revision_are_rejected(self):
        for revision in (None, 'future', ''):
            c = self.contract()
            if revision is None:
                c.pop('execution_revision')
            else:
                c['execution_revision'] = revision
            with self.assertRaises(ValueError):
                b3.enabled(c)

    def test_no_formal_change_to_case_workload_source_or_build(self):
        for key in ('cases','source','build'):
            c = self.contract()
            if key == 'cases': c[key][0]['iterations'] = 1
            elif key == 'source': c[key] = 'other.cu'
            else: c[key]['flags'].append('-G')
            with self.assertRaisesRegex(ValueError, 'semantics'):
                b3.validate_bundle(self.repo, c)

    def test_missing_and_changed_review_rejected_before_environment_query(self):
        c = self.contract()
        c['family_b3']['review']['sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'contract.json'; atomic_json(path,c)
            with mock.patch.object(suite_runner,'inspect_allocation') as allocation:
                with self.assertRaises(ValueError):
                    suite_runner.initialize(self.repo,self.suite,self.suite/'async_copy/new',path,preflight=True)
                allocation.assert_not_called()

    def test_old_family_new_preflight_rejects_missing_current_core_before_queries(self):
        old = BASE/'contracts/memory_baseline.json'
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)/'suite'
            (suite/'reviews').mkdir(parents=True)
            for name in ('S02-A-review.json','S02-preflight-A-review-r2.json'):
                shutil.copyfile(REPO/SUITE_NAME/'reviews'/name,suite/'reviews'/name)
            with mock.patch.object(suite_runner,'inspect_allocation') as allocation:
                with self.assertRaisesRegex(ValueError, 'S02-formal-b3-v2-B-review'):
                    suite_runner.initialize(REPO,suite,suite/'memory_baseline/new',old,preflight=True)
                allocation.assert_not_called()
        # Standalone long/boundary diagnostics use the default freeze/preflight
        # context. Their separate source review must bind this current core too.
        from runners.accumulation_diagnostic import review_requirements
        base = read_json(BASE/'contracts/low_precision_lowering_v3.json')
        self.assertIn(('S02','B','reviews/S02-formal-b3-v2-B-review.json'),review_requirements(base))

    def test_nonindependent_or_nonpass_review_rejected(self):
        p = self.repo/self.original['family_b3']['review']['path']
        for update in ({'status':'revise'}, {'implementer':'/root (host); /root/gate_reviewer (collection)'}):
            review = read_json(p);review.update(update)
            with self.changed(p,review):
                c = self.contract();c['family_b3']['review']['sha256']=sha(p)
                with self.assertRaises(ValueError):b3.validate_bundle(self.repo,c)

    def test_partial_duplicate_or_extra_coverage_rejected_even_if_rehashed(self):
        cp=self.repo/self.original['family_b3']['coverage']['path']
        rp=self.repo/self.original['family_b3']['review']['path']
        for change in ('missing','duplicate','extra'):
            coverage=read_json(cp)
            if change=='missing':coverage['cases'].pop()
            elif change=='duplicate':coverage['cases'][-1]=deepcopy(coverage['cases'][0])
            else:coverage['cases'].append({**coverage['cases'][0],'case_id':'unexpected'})
            with self.changed(cp,coverage):
                review=read_json(rp);review['coverage_record']['sha256']=sha(cp)
                review['gate_files'][self.original['family_b3']['coverage']['path']]=sha(cp)
                with self.changed(rp,review):
                    c=self.contract();c['family_b3']['coverage']['sha256']=sha(cp);c['family_b3']['review']['sha256']=sha(rp)
                    with self.assertRaises(ValueError):b3.validate_bundle(self.repo,c)

    def test_source_reference_profile_and_transitive_file_drift_rejected(self):
        first=next(iter(self.bundle['cases'].values()))['root'].relative_to(REPO)
        targets=[self.repo/self.original['source'],
                 self.repo/'microbench/gh200_resource_campaign/common/async_copy_reference.hpp',
                 self.repo/self.original['family_b3']['profiles']['path'],
                 self.repo/first/'snapshot/repo/microbench/gh200_resource_campaign/common/probe_runtime.cuh']
        for path in targets:
            with self.changed(path,path.read_bytes()+b'\nchanged\n'):
                with self.assertRaises(ValueError):b3.validate_bundle(self.repo,self.contract())

    def test_failed_raw_output_rejected(self):
        first=next(iter(self.bundle['cases'].values()))['root'].relative_to(REPO)
        path=self.repo/first/'attempts/attempt_00/raw.jsonl'
        device,row=b3.raw_rows(path);row['checks'][0]['errors']=1;row['errors']=1
        with self.changed(path, (json.dumps(device)+'\n'+json.dumps(row)+'\n').encode()):
            with self.assertRaises(ValueError):b3.validate_bundle(self.repo,self.contract())

    def test_new_policy_requires_B3_and_integration_gates_even_for_preflight(self):
        deps=required_reviews(self.contract(),True)
        self.assertIn(('S12','early-validation-B3','reviews/S12-early-validation-B3-review.json'),deps)
        self.assertIn(('S12','formal-b3-source-B','reviews/S12-formal-b3-source-B-review-r2.json'),deps)
        self.assertIn(('S02','B','reviews/S02-formal-b3-v2-B-review.json'),deps)
        self.assertNotIn(('S12','B','reviews/S12-B-review-r2.json'),deps)
        self.assertIn(('S12','B','reviews/S12-B-review-r2.json'),required_reviews(self.contract(),False))

    def test_six_old_adapter_contracts_keep_arithmetic_and_no_B3_requirement(self):
        for name in ('memory_baseline','shared_memory','low_precision_lowering_v3','legacy_fma','legacy_mma','legacy_wgmma'):
            c=read_json(BASE/'contracts'/f'{name}.json')
            adapter(c).validate_contract(c)
            self.assertFalse(b3.enabled(c))
            self.assertFalse(any(phase=='early-validation-B3' for _,phase,_ in required_reviews(c,True)))
            self.assertIn(('S02','B','reviews/S02-formal-b3-v2-B-review.json'),required_reviews(c,True))
            self.assertIn(('S03','B','reviews/S03-formal-b3-v2-B-review.json'),required_reviews(c,True))
        short=read_json(BASE/'contracts/async_copy.json')
        manifest=read_json(BASE/'contracts/async_copy_validation_adapter_v1.json')
        policy=read_json(BASE/'contracts/early_validation_v1.json')
        self.assertFalse(any(phase=='early-validation-B3' for _,phase,_ in validation_diagnostic.review_requirements(short,manifest,policy)))
        self.assertEqual(validation_diagnostic.CORE_REVIEW,'reviews/S02-early-validation-B-review-r5.json')
        self.assertFalse(any(stage in ('S02','S03') and phase=='B' for stage,phase,_ in validation_diagnostic.review_requirements(short,manifest,policy)))

    def test_integrated_formal_adapter_checks_all_24_bound_resources(self):
        from test_async_copy_formal import AsyncCopyFormalTests
        fixture = AsyncCopyFormalTests()
        formal = adapter(self.contract())
        protocol = read_json(BASE/'contracts/protocol.json')
        device = self.bundle['identity']['device']
        for original in self.original['cases']:
            case = deepcopy(original)
            record = self.bundle['cases'][case['id']]
            case['b3_resource_identity'] = record['resource']
            case['b3_blocks'] = record['blocks']
            row = fixture.row(case)
            row.update({key:record['resource'][key] for key in b3.RESOURCE_FIELDS})
            formal.validate_trial(row,case,device,3,protocol)
            row['occupancy_limit_ctas_per_sm'] -= 1
            with self.assertRaises(ValueError):formal.validate_trial(row,case,device,3,protocol)
            with self.assertRaises(ValueError):formal.validate_trial(row,original,device,3,protocol)

    def test_current_entry_dispatches_old_completed_snapshot_before_new_policy(self):
        import run_suite
        root = REPO/SUITE_NAME/'memory_baseline/formal-v3-a'
        self.assertTrue((root/'COMPLETE').exists())
        with mock.patch.object(run_suite.os, 'execv', side_effect=SystemExit(97)) as execute:
            with self.assertRaises(SystemExit) as caught:
                run_suite.frozen_dispatch(root, ['audit', str(root), '--require-complete'])
            self.assertEqual(caught.exception.code, 97)
            self.assertEqual(execute.call_args.args[1][2], str(root/'snapshot/repo/microbench/gh200_resource_campaign/run_suite.py'))

    def make_run(self, directory):
        c=self.contract()
        # Test-only integration reviews: no run of any device program follows.
        created=[]
        for stage,phase,name in required_reviews(c,True):
            p=self.suite/name
            if not p.exists():
                atomic_json(p,{'schema_version':2,'stage':stage,'phase':phase,'status':'pass',
                    'implementer':'CPU fixture author','reviewer':'CPU fixture reviewer',
                    'checks':{key:{'status':'pass'} for key in DIMENSIONS},'findings':[],
                    'gate_files':{CONTRACT:sha(self.repo/CONTRACT)}})
                created.append(p)
        root=Path(directory)/'formal-fixture';root.mkdir()
        try:
            frozen=freeze(self.repo,self.suite,root,self.repo/CONTRACT,preflight=True)
        finally:
            for p in created:p.unlink()
        first=next(iter(self.bundle['cases'].values()))['root']
        for name in ('binary','build','environment'):
            shutil.copytree(first/name,root/name)
        shutil.copyfile(root/'build/sass.stdout',root/'build/sass.txt')
        shutil.copyfile(root/'build/sass.stdout',root/'build/disassemble.stdout')
        shutil.copyfile(root/'build/sass.stderr',root/'build/disassemble.stderr')
        shutil.copyfile(root/'build/sass.receipt.json',root/'build/disassemble.receipt.json')
        env=read_json(root/'environment/initial.json')
        source=frozen['source_path'];flags=c['build']['flags']
        atomic_json(root/'build/command.json',{'argv':[env['tools']['nvcc']['path'],*flags,'-MD','-MF',str(root/'build/probe.d'),str(root/source),'-o',str(root/'binary/probe')], 'source_relative':source,'target_relative':'binary/probe'})
        atomic_json(root/'build/sass_audit.json',adapter(c).audit_sass((root/'build/sass.txt').read_text(),c))
        spec={'schema_version':2,'kind':'preflight','fixture':False,**frozen,
              'binary_sha256':sha(root/'binary/probe'),'sass_sha256':sha(root/'build/sass.txt'),
              'dependencies_sha256':sha(root/'build/dependencies.json'),'shared_libraries_sha256':sha(root/'build/shared_libraries.json'),
              'environment_sha256':sha(root/'environment/initial.json'),'device_sha256':sha(root/'environment/device.json')}
        atomic_json(root/'run_spec.json',spec)
        return root

    def test_frozen_load_relocation_and_review_set_cannot_be_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_run(tmp)
            _,c,_,_,_=load_run(root)
            self.assertTrue(all('b3_resource_identity' in case for case in c['cases']))
            moved=Path(tmp)/'moved';root.rename(moved)
            # This corrupts the mutable test workspace, not the run snapshot.
            p=self.repo/self.original['source']
            with self.changed(p,b'not CUDA'):
                load_run(moved)
            spec=read_json(moved/'run_spec.json');spec['reviews']=[x for x in spec['reviews'] if x['phase']!='early-validation-B3']
            atomic_json(moved/'run_spec.json',spec)
            with self.assertRaises(ValueError):load_run(moved)
            with mock.patch.object(suite_runner,'inspect_allocation') as allocation:
                with self.assertRaises(ValueError):suite_runner.execute(moved,self.suite,preflight=True)
                allocation.assert_not_called()

    def test_build_device_toolchain_resources_and_resume_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.make_run(tmp);c=self.contract();device=read_json(root/'environment/device.json')
            b3.verify_build(root,c,device)
            changed={**device,'runtime_version':12080}
            with self.assertRaises(ValueError):b3.verify_build(root,c,changed)
            env=read_json(root/'environment/initial.json');env['tools']['nvcc']['sha256']='0'*64
            with self.assertRaises(ValueError):b3.verify_build(root,c,device,live_environment=env)
            for name in ('build/sass.txt','build/compile.stderr','build/dependencies.json','build/shared_libraries.json'):
                p=root/name
                if name.endswith('sass.txt'):value=p.read_bytes().replace(b'LDGSTS',b'BADSTS',1)
                elif name.endswith('stderr'):value=p.read_bytes().replace(b'Used 26 registers',b'Used 25 registers')
                else:
                    value=read_json(p)
                    if 'dependencies' in name:value['external_toolchain'][next(iter(value['external_toolchain']))]='0'*64
                    else:value[next(iter(value))]='0'*64
                with self.changed(p,value):
                    with self.assertRaises(ValueError):b3.verify_build(root,c,device)


if __name__=='__main__':
    unittest.main()
