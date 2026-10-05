"""S13 finite admission against actual copied B3 evidence; no GPU calls."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

BASE=Path(__file__).resolve().parents[1]
REPO=BASE.parents[1]
sys.path.insert(0,str(BASE))
from common import family_b3 as b3
from common.suite_io import atomic_json,read_json,sha,DIMENSIONS
from common.suite_snapshot import freeze,required_reviews
from auditors.suite import adapter,load_run
from runners import suite_runner
SUITE='results/gh200_resource_campaign/20261001-resource-suite-v2'
CONTRACT='microbench/gh200_resource_campaign/contracts/global_duplex_formal_v1.json'


class S13FamilyB3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=read_json(REPO/CONTRACT)
        cls.bundle=b3.validate_bundle(REPO,cls.original)
        cls.temp=tempfile.TemporaryDirectory(prefix='S13-B3-CPU-')
        cls.repo=Path(cls.temp.name)/'repo';names=cls.bundle['names']|{CONTRACT}
        for stage,phase,name in required_reviews(cls.original,True):
            path=REPO/SUITE/name
            if path.exists():names.add(path.relative_to(REPO).as_posix());names.update(read_json(path)['gate_files'])
        for path in BASE.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:names.add(path.relative_to(REPO).as_posix())
        for name in names:
            target=cls.repo/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,target)
        cls.suite=cls.repo/SUITE

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    @contextmanager
    def changed(self,path,value):
        old=path.read_bytes()
        try:
            if isinstance(value,bytes):path.write_bytes(value)
            else:atomic_json(path,value)
            yield
        finally:path.write_bytes(old)

    def test_exact18_bundle_and_version_dispatch(self):
        self.assertEqual(len(self.bundle['cases']),18)
        self.assertEqual(len(self.bundle['target_identities']),9)
        for key,value in [('execution_revision','family-b3-v1'),('adapter_id','async_copy_formal_v1'),('stage','S12'),('family','async_copy')]:
            bad={**self.original,key:value}
            with self.assertRaises(ValueError):b3.enabled(bad)
        self.assertFalse(b3.enabled(read_json(BASE/'contracts/global_duplex.json')))
        with self.assertRaises(ValueError):adapter(read_json(BASE/'contracts/global_duplex.json'))

    def test_S12_enabled_acceptance_domain_matches_saved_implementation(self):
        path=REPO/SUITE/'implementation/s13-formal-admission-b/initial/common/family_b3.py'
        spec=importlib.util.spec_from_file_location('original_S12_B3',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        original=read_json(BASE/'contracts/async_copy_formal_v1.json')
        for revision in (None,'family-b3-v1','unknown'):
            for stage in ('S12','S13','unknown'):
                for family in ('async_copy','global_duplex','unknown'):
                    c={**original,'execution_revision':revision,'stage':stage,'family':family}
                    def outcome(module):
                        try:return ('returned',module.enabled(c))
                        except ValueError:return ('rejected',)
                    self.assertEqual(outcome(b3),outcome(old))

    def test_new_and_legacy_review_choices(self):
        pre=required_reviews(self.original,True)
        self.assertIn(('S13','formal-b3-A','reviews/S13-formal-b3-A-review-r2.json'),pre)
        self.assertIn(('S13','early-validation-B3','reviews/S13-early-validation-B3-review.json'),pre)
        self.assertIn(('S13','formal-b3-source-B','reviews/S13-formal-b3-source-B-review.json'),pre)
        self.assertNotIn(('S13','B','reviews/S13-B-review.json'),pre)
        self.assertIn(('S13','B','reviews/S13-B-review.json'),required_reviews(self.original))
        old=read_json(BASE/'contracts/async_copy_formal_v1.json')
        self.assertIn(('S12','formal-b3-source-B','reviews/S12-formal-b3-source-B-review-r2.json'),required_reviews(old,True))
        self.assertIn(('S12','B','reviews/S12-B-review-r2.json'),required_reviews(old))
        self.assertNotIn(('S12','B','reviews/S12-B-review-r2.json'),required_reviews(old,True))

    def test_B3_failure_before_allocation(self):
        bad=deepcopy(self.original);bad['family_b3']['review']['sha256']='0'*64
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'contract.json';atomic_json(p,bad)
            with mock.patch.object(suite_runner,'inspect_allocation') as allocation:
                with self.assertRaises(ValueError):suite_runner.initialize(self.repo,self.suite,self.suite/'global_duplex/fixture',p,preflight=True)
                allocation.assert_not_called()

    def test_rehashed_partial_coverage_and_pure_write_false_data_reject(self):
        cp=self.repo/self.original['family_b3']['coverage']['path'];rp=self.repo/self.original['family_b3']['review']['path']
        for kind in ('missing','duplicate','profile','seed','pure_write','read_total'):
            cov=read_json(cp)
            if kind=='missing':cov['cases'].pop()
            elif kind=='duplicate':cov['cases'][-1]=deepcopy(cov['cases'][0])
            elif kind=='profile':cov['profile_id']='other'
            elif kind=='seed':cov['seed']=4
            elif kind=='pure_write':next(c for c in cov['cases'] if c['case_id'].startswith('write_'))['pure_write_zero_checksums_counted']=True
            else:cov['read_checksum_elements']+=1
            with self.changed(cp,cov):
                review=read_json(rp);review['coverage_record']['sha256']=sha(cp);review['gate_files'][self.original['family_b3']['coverage']['path']]=sha(cp)
                with self.changed(rp,review):
                    bad=deepcopy(self.original);bad['family_b3']['coverage']['sha256']=sha(cp);bad['family_b3']['review']['sha256']=sha(rp)
                    with self.subTest(kind=kind),self.assertRaises(ValueError):b3.validate_bundle(self.repo,bad)

    def test_exact_machine_encoding_and_target_multiplicity(self):
        first=next(iter(self.bundle['cases'].values()))['root'];text=(first/'build/sass.stdout').read_text();symbols=set(self.bundle['target_identities']);old=b3.target_identity(text,symbols)
        # Even an encoding-only change keeps all textual opcode/operand checks.
        import re
        match=re.search(r'/\* (0x[0-9a-f]{16}) \*/',text)
        changed=text[:match.start(1)]+('0x'+format(int(match[1],16)^1,'016x'))+text[match.end(1):]
        self.assertNotEqual(b3.target_identity(changed,symbols),old)
        # Text-only changes are equally material: preserve opcode counts while
        # altering the instruction's predicate or one register operand.
        operand=re.search(r'\bR(\d+)\b',text)
        changed=text[:operand.start()]+('R'+str(int(operand[1])+1))+text[operand.end():]
        self.assertNotEqual(b3.target_identity(changed,symbols),old)
        predicate=re.search(r'@(!?P\d+)\s',text)
        changed=text[:predicate.start(1)]+('!' if not predicate[1].startswith('!') else '')+predicate[1].lstrip('!')+text[predicate.end(1):]
        self.assertNotEqual(b3.target_identity(changed,symbols),old)
        name=sorted(symbols)[0]
        for wrong in (text.replace('Function : '+name,'Function : absent',1),text+'\nFunction : '+name+'\n/*0000*/ NOP; /* 0x0000000000000000 */\n/* 0x0000000000000000 */'):
            with self.assertRaises(ValueError):b3.target_identity(wrong,symbols)

    def test_rehashed_review_identity_purpose_and_binding_rejected(self):
        path=self.repo/self.original['family_b3']['review']['path']
        for kind in ('stage','phase','self_review','purpose','coverage_gate'):
            review=read_json(path)
            if kind=='stage':review['stage']='S12'
            elif kind=='phase':review['phase']='B'
            elif kind=='self_review':review['implementer']='/root plus '+review['reviewer']
            elif kind=='purpose':review['authorization']['family_B3_qualified']=False
            else:review['gate_files'].pop(self.original['family_b3']['coverage']['path'])
            with self.changed(path,review):
                contract=deepcopy(self.original);contract['family_b3']['review']['sha256']=sha(path)
                with self.subTest(kind=kind),self.assertRaises(ValueError):b3.validate_bundle(self.repo,contract)

    def make_run(self,directory):
        created=[]
        for stage,phase,name in required_reviews(self.original,True):
            p=self.suite/name
            if not p.exists():
                atomic_json(p,{'schema_version':2,'stage':stage,'phase':phase,'status':'pass','implementer':'CPU fixture only','reviewer':'independent fixture only','checks':{k:{'status':'pass'} for k in DIMENSIONS},'findings':[],'gate_files':{CONTRACT:sha(self.repo/CONTRACT)}});created.append(p)
        root=Path(directory)/'run';root.mkdir()
        try:frozen=freeze(self.repo,self.suite,root,self.repo/CONTRACT,preflight=True)
        finally:
            for p in created:p.unlink()
        first=next(iter(self.bundle['cases'].values()))['root']
        for name in ('binary','build','environment'):shutil.copytree(first/name,root/name)
        for a,b in [('sass.stdout','sass.txt'),('sass.stdout','disassemble.stdout'),('sass.stderr','disassemble.stderr'),('sass.receipt.json','disassemble.receipt.json')]:shutil.copyfile(root/'build'/a,root/'build'/b)
        env=read_json(root/'environment/initial.json');flags=self.original['build']['flags'];source=frozen['source_path']
        atomic_json(root/'build/command.json',{'argv':[env['tools']['nvcc']['path'],*flags,'-MD','-MF',str(root/'build/probe.d'),str(root/source),'-o',str(root/'binary/probe')],'source_relative':source,'target_relative':'binary/probe'})
        atomic_json(root/'build/sass_audit.json',adapter(self.original).audit_sass((root/'build/sass.txt').read_text(),self.original))
        spec={'schema_version':2,'kind':'preflight','fixture':False,**frozen,'binary_sha256':sha(root/'binary/probe'),'sass_sha256':sha(root/'build/sass.txt'),'dependencies_sha256':sha(root/'build/dependencies.json'),'shared_libraries_sha256':sha(root/'build/shared_libraries.json'),'environment_sha256':sha(root/'environment/initial.json'),'device_sha256':sha(root/'environment/device.json')}
        atomic_json(root/'run_spec.json',spec);return root

    def test_relocation_and_resume_cannot_skip_exact_review_set(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.make_run(d);_,c,_,_,_=load_run(run)
            self.assertTrue(all('b3_resource_identity' in case for case in c['cases']))
            moved=Path(d)/'relocated';run.rename(moved)
            with self.changed(self.repo/self.original['source'],b'changed live workspace'):
                load_run(moved)
            spec=read_json(moved/'run_spec.json');spec['reviews']=[x for x in spec['reviews'] if x['phase']!='early-validation-B3'];atomic_json(moved/'run_spec.json',spec)
            with self.assertRaises(ValueError):load_run(moved)
            with mock.patch.object(suite_runner,'inspect_allocation') as allocation:
                with self.assertRaises(ValueError):suite_runner.execute(moved,self.suite,preflight=True)
                allocation.assert_not_called()

    def test_build_device_resources_and_libraries_drift(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.make_run(d);device=self.bundle['identity']['device']
            b3.verify_build(run,self.original,device)
            with self.assertRaises(ValueError):b3.verify_build(run,self.original,{**device,'runtime_version':1})
            environment=read_json(run/'environment/initial.json')
            different=deepcopy(environment)
            different['tools']['nvcc']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'live toolchain/environment'):
                b3.verify_build(run,self.original,device,live_environment=different)
            for name in ('build/dependencies.json','build/shared_libraries.json','environment/initial.json'):
                p=run/name;value=read_json(p);value['unexpected']='drift'
                if name=='environment/initial.json':value['driver']='different'
                with self.changed(p,value),self.assertRaises(ValueError):b3.verify_build(run,self.original,device)
            p=run/'build/compile.stderr'
            with self.changed(p,p.read_bytes().replace(b'1024 bytes smem',b'2048 bytes smem')),self.assertRaises(ValueError):b3.verify_build(run,self.original,device)
            p=run/'build/sass.txt';text=p.read_text();import re
            hit=re.search(r'/\* (0x[0-9a-f]{16}) \*/',text);bad=text[:hit.start(1)]+'0x'+format(int(hit[1],16)^1,'016x')+text[hit.end(1):]
            with self.changed(p,bad.encode()),self.assertRaisesRegex(ValueError,'machine code'):b3.verify_build(run,self.original,device)


if __name__=='__main__':unittest.main()
