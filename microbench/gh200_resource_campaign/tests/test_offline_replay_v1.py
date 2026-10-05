"""CPU-only replay truth tables and mutations of copied real evidence."""
from __future__ import annotations
import copy
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from auditors.offline_replay_v1 import compare,cv_squared,exact_cv_pass,exact_warmup,inventory,replay,sha,ulp_distance
from auditors.offline_replay_worker_v1 import guard
POLICY=HERE/'contracts/offline_replay_v1.json'
REPO=HERE.parents[1]
SOURCE=REPO/'results/gh200_resource_campaign/20261001-resource-suite-v2/memory_baseline/preflight-v3'
PATTERNS=['/cases/*/warmup_cv','/cases/*/value']


def bump(value,count):
    for _ in range(count):value=math.nextafter(value,math.inf)
    return value


class FloatPolicyTests(unittest.TestCase):
    def test_white_list_zero_one_two_ULP_only(self):
        for count in (0,1,2):
            left={'cases':{'point':{'warmup_cv':.125}}};right=copy.deepcopy(left);right['cases']['point']['warmup_cv']=bump(.125,count)
            result=compare(left,right,PATTERNS);self.assertEqual(result[0]['ulp_distance'],count)
        with self.assertRaisesRegex(ValueError,'2 ULP'):compare({'cases':{'x':{'value':.125}}},{'cases':{'x':{'value':bump(.125,3)}}},PATTERNS)

    def test_zero_negative_zero_nonfinite_and_integer_bool_rejected(self):
        self.assertEqual(ulp_distance(0.0,0.0),0)
        for a,b in [(-0.0,0.0),(0.0,math.nextafter(0.0,1.0)),(math.nan,math.nan),(math.inf,math.inf),(-.1,-.1),(1,1.0),(True,1.0)]:
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):ulp_distance(a,b)

    def test_nonwhite_fields_types_shapes_hashes_units_and_states_strict(self):
        base={'uuid':'GPU-a','hash':'a'*64,'unit':'B/cycle','status':'stable','count':10,'valid':True,'other':.125}
        for key,value in [('uuid','GPU-b'),('hash','b'*64),('unit','TFLOP/s'),('status','pending'),('count',10.0),('valid',1),('other',bump(.125,1))]:
            bad={**base,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):compare(base,bad,PATTERNS)
        with self.assertRaises(ValueError):compare(base,{**base,'extra':0},PATTERNS)
        with self.assertRaises(ValueError):compare([base],[base,base],PATTERNS)

    def test_formal_paths_are_separate_from_preflight(self):
        patterns=json.loads(POLICY.read_text())['formal_float_paths']
        left={'cases':[{'merged':{'cv':.125},'batches':[{'stats':{'mean':4.0},'samples':[{'value':1.0,'warmup_cv':.1}]}]}]}
        right=copy.deepcopy(left);right['cases'][0]['merged']['cv']=bump(.125,2)
        self.assertTrue(compare(left,right,patterns))
        with self.assertRaises(ValueError):compare(left,right,PATTERNS)

    def test_exact_threshold_not_based_on_rounded_CV(self):
        # Construct fractions on both sides of CV=1/50 and 1/20 for a two-sample set.
        # CV^2 for [1-d,1+d] is 2*d^2; comparisons use integers/Fraction, not sqrt.
        for limit in (Fraction(1,50),Fraction(1,20)):
            lower=Fraction.from_float(float(limit)/math.sqrt(2))
            while 2*lower*lower>=limit*limit:lower-=Fraction(1,10**20)
            upper=lower+Fraction(1,10**12)
            self.assertTrue(exact_cv_pass([1-lower,1+lower],limit))
            self.assertFalse(exact_cv_pass([1-upper,1+upper],limit))
            with self.assertRaises(ValueError):compare({'stable':True},{'stable':False},PATTERNS)

    def test_exact_warmup_rejects_wrong_flag_even_if_display_CV_is_near_threshold(self):
        protocol={'warmup_min_windows':8,'warmup_max_windows':30,'warmup_tail_windows':5}
        # Five positive integers with CV just below 0.02, then just above.
        mean=10**18;prefix=[mean//2,mean*3//2,mean//2]
        decisions=[]
        for delta in (mean//50-1,mean//50+1):
            tail=[mean-delta,mean-delta,mean,mean+delta,mean+delta]
            expected=exact_cv_pass([Fraction(x) for x in tail],Fraction(1,50));decisions.append(expected)
            self.assertEqual(float(Fraction(delta,mean)),.02)
            row={'warmup_samples_ns':prefix+tail,'warmup_converged':not expected}
            with self.assertRaisesRegex(ValueError,'flag disagrees'):exact_warmup(row,protocol)
        self.assertEqual(decisions,[True,False])

    def test_worker_denies_mutation_and_process_execution(self):
        guard('open',('/tmp/read','r',os.O_RDONLY))
        for event,args in [('open',('/tmp/write','w',os.O_WRONLY)),('os.mkdir',('/tmp/x',0o700,-1)),('subprocess.Popen',('nvcc',[],None,None)),('os.rename',('a','b',-1,-1))]:
            with self.assertRaises(ValueError):guard(event,args)


@unittest.skipUnless(SOURCE.exists(),'real source archive unavailable')
class RealArchiveMutationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'relocated';shutil.copytree(SOURCE,self.root)
    def tearDown(self):
        for p in self.root.rglob('*'):
            if p.is_dir():p.chmod(0o755)
            elif p.is_file():p.chmod(0o644)
        self.root.chmod(0o755);self.temp.cleanup()
    def mutate_summary(self,field,value):
        path=self.root/'preflight_summary.json';data=json.loads(path.read_text());data['cases']['global_write_256m'][field]=value;path.write_text(json.dumps(data))

    def test_real_readonly_relocated_run_preserves_strict_failure_and_all_hashes(self):
        before=inventory(self.root)
        for p in self.root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        self.root.chmod(0o555)
        result=replay(self.root,POLICY)
        self.assertEqual(result['status'],'pass_under_bounded_float_replay_v1',result['failed_checks'])
        self.assertEqual(result['original_strict_audit']['returncode'],2)
        self.assertIn('preflight recomputation differs',result['original_strict_audit']['stderr'])
        self.assertEqual(len(result['float_differences']),1);self.assertEqual(len(result['threshold_checks']),14)
        self.assertEqual(result['input_artifacts_sha256'],before);self.assertTrue(result['input_unchanged'])
        self.assertEqual(inventory(self.root),before)

    def test_three_ULP_summary_change_rejected(self):
        result=replay(self.root,POLICY);target=next(r for r in result['float_comparisons'] if r['json_pointer']=='/cases/global_write_256m/warmup_cv')
        self.mutate_summary('warmup_cv',bump(target['recomputed'],3))
        bad=replay(self.root,POLICY);self.assertEqual(bad['status'],'replay_rejected');self.assertIn('2 ULP',' '.join(bad['failed_checks']))

    def test_unit_state_and_hash_mutations_rejected(self):
        original=(self.root/'preflight_summary.json').read_text()
        for field,value in [('unit','TFLOP/s'),('warmup_converged',False)]:
            self.mutate_summary(field,value);bad=replay(self.root,POLICY)
            self.assertEqual(bad['status'],'replay_rejected');(self.root/'preflight_summary.json').write_text(original)
        with (self.root/'binary/probe').open('ab') as f:f.write(b'corrupted')
        bad=replay(self.root,POLICY);self.assertEqual(bad['status'],'replay_rejected');self.assertIn('binary changed',' '.join(bad['failed_checks']))

    def test_wrong_workload_still_rejected_if_raw_hash_is_rebound(self):
        path=next((self.root/'preflight/smem_read_stride1').glob('batch_*/trial_*/attempt_*/raw.jsonl'))
        rows=[json.loads(line) for line in path.read_text().splitlines()];rows[1]['work_count']+=1
        path.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
        receipt=path.with_name('receipt.json');data=json.loads(receipt.read_text());data['raw_sha256']=sha(path);receipt.write_text(json.dumps(data))
        bad=replay(self.root,POLICY);self.assertEqual(bad['status'],'replay_rejected');self.assertIn('workload',' '.join(bad['failed_checks']))

    def test_missing_evidence_and_unsupported_policy_rejected(self):
        path=next((self.root/'preflight/smem_read_stride1').glob('batch_*/trial_*/attempt_*/raw.jsonl'));path.unlink()
        self.assertEqual(replay(self.root,POLICY)['status'],'replay_rejected')
        policy=Path(self.temp.name)/'loose.json';data=json.loads(POLICY.read_text());data['max_binary64_ULP_distance']=100;policy.write_text(json.dumps(data))
        bad=replay(SOURCE,policy);self.assertEqual(bad['status'],'replay_rejected');self.assertIn('unreviewed replay policy',' '.join(bad['failed_checks']))


if __name__=='__main__':unittest.main()
