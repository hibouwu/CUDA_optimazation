"""Versioned bounded-float replay; no dependency on mutable live framework modules."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
from fractions import Fraction

POLICY_SHA256='0e64c689ceee45a78625e49945af731c446b505a1f416c3383e26a8f21fc12db'
HERE=Path(__file__).resolve().parent
WORKER=HERE/'offline_replay_worker_v1.py'


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    def pairs(items):
        out={}
        for key,value in items:
            require(key not in out,'duplicate JSON key: '+key);out[key]=value
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def safe(root,name):
    rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,'unsafe artifact path')
    out=root
    for part in rel.parts:out=out/part;require(not out.is_symlink(),'symlink artifact')
    require(out.is_file(),'missing artifact: '+name);return out


def inventory(root):
    result={}
    for path in sorted(Path(root).rglob('*')):
        require(not path.is_symlink(),'symlink in archive: '+str(path))
        if path.is_file():result[path.relative_to(root).as_posix()]=sha(path)
    require(result,'empty archive')
    return result


def float_bits(value):
    require(type(value) is float and math.isfinite(value) and value>=0,'derived field must be finite nonnegative binary64 float')
    bits=struct.unpack('>Q',struct.pack('>d',value))[0]
    require(not bits>>63,'negative zero/sign forbidden')
    return bits


def ulp_distance(recorded,recomputed):
    a,b=float_bits(recorded),float_bits(recomputed)
    if recorded==0.0 or recomputed==0.0:
        require(a==b==0,'zero must match exactly')
    return abs(a-b)


def matching(pointer,patterns):
    tokens=pointer.split('/')
    return any(len(pattern.split('/'))==len(tokens) and all(a=='*' or a==b for a,b in zip(pattern.split('/'),tokens)) for pattern in patterns)


def compare(recorded,recomputed,patterns,pointer=''):
    """Complete shape/type checking; only named floating leaves receive a ULP allowance."""
    differences=[]
    def walk(left,right,path):
        if matching(path,patterns):
            distance=ulp_distance(left,right)
            differences.append({'json_pointer':path,'recorded':left,'recomputed':right,'ulp_distance':distance,'accepted':distance<=2})
            require(distance<=2,'derived float exceeds 2 ULP: '+path)
            return
        require(type(left) is type(right),'strict field type changed: '+path)
        if isinstance(left,dict):
            require(set(left)==set(right),'JSON object fields changed: '+path)
            for key in sorted(left):walk(left[key],right[key],path+'/'+key.replace('~','~0').replace('/','~1'))
        elif isinstance(left,list):
            require(len(left)==len(right),'JSON list length changed: '+path)
            for index,(a,b) in enumerate(zip(left,right)):walk(a,b,path+'/'+str(index))
        elif type(left) is float:
            require(math.isfinite(left) and math.isfinite(right),'nonfinite strict field')
            require(struct.pack('>d',left)==struct.pack('>d',right),'strict float changed: '+path)
        else:require(left==right,'strict field value changed: '+path)
    try:walk(recorded,recomputed,pointer)
    except ValueError as exc:
        exc.float_comparisons=list(differences)
        raise
    return differences


def cv_squared(values):
    require(len(values)>=2 and all(isinstance(x,Fraction) and x>0 for x in values),'positive exact CV samples required')
    n=len(values);s=sum(values);q=sum(x*x for x in values)
    return n*(n*q-s*s)/((n-1)*s*s)


def exact_cv_pass(values,limit):return cv_squared(values)<=limit*limit


def exact_warmup(row,protocol):
    basis=row.get('warmup_basis','elapsed_ns')
    require(basis in ('elapsed_ns','cta_clock64_cycles'),'unknown warmup basis')
    samples=row['warmup_samples_cycles'] if basis=='cta_clock64_cycles' else row['warmup_samples_ns']
    require(protocol['warmup_min_windows']==8 and protocol['warmup_max_windows']==30 and protocol['warmup_tail_windows']==5,'unknown warmup protocol')
    require(8<=len(samples)<=30 and all(type(x) is int and x>0 for x in samples),'invalid exact warmup samples')
    valid=exact_cv_pass([Fraction(x) for x in samples[-5:]],Fraction(1,50))
    require(type(row['warmup_converged']) is bool and row['warmup_converged']==valid,'warmup flag disagrees with exact threshold')
    require(valid or len(samples)==30,'unconverged warmup stopped early')
    require(all(not exact_cv_pass([Fraction(x) for x in samples[end-5:end]],Fraction(1,50)) for end in range(8,len(samples))),'warmup missed exact first stopping point')
    square=cv_squared([Fraction(x) for x in samples[-5:]])
    return {'basis':basis,'windows':len(samples),'limit_fraction':[1,50],'cv_squared_fraction':[square.numerator,square.denominator],'converged':valid,'first_stop_verified':True}


def exact_rate(row,metric):
    counts={key:row[key] for key in ('work_count','read_payload_bytes','write_payload_bytes')}
    require(all(type(v) is int and v>=0 for v in counts.values()),'work quantities must be integer')
    counts['payload_bytes']=counts['read_payload_bytes']+counts['write_payload_bytes']
    counts['elapsed_ns']=row['stop_ns']-row['start_ns'];counts['one']=1
    if row['scope']=='one_cta':
        require(len(row['blocks_detail'])==1,'local cycle scope')
        block=row['blocks_detail'][0];counts['cta_clock64_cycles']=block['stop_cycle']-block['start_cycle']
    require(metric['numerator'] in counts and metric['denominator'] in counts,'exact metric quantity requires a reviewed replay extension')
    require(type(metric['scale']) in (int,float) and math.isfinite(metric['scale']) and metric['scale']>0,'invalid metric scale')
    require(counts[metric['denominator']]>0,'exact metric denominator')
    value=Fraction(counts[metric['numerator']],counts[metric['denominator']])*Fraction(str(metric['scale']))
    require(value>0,'nonpositive exact rate');return value


def exact_thresholds(root,payload):
    checks=[];formal={}
    for binding in payload['raw_records']:
        lines=safe(root,binding['raw']).read_text().splitlines();require(len(lines)==2,'raw row count')
        # Frozen worker already did duplicate-key/schema/identity/numerical checks.
        row=json.loads(lines[1]);check=exact_warmup(row,payload['protocol'])
        checks.append({'raw':binding['raw'],'scope':binding['scope'],'case_id':binding['case_id'],**check})
        if binding['scope']=='formal':formal[(binding['case_id'],binding['batch'],binding['trial'])]=(exact_rate(row,binding['metric']),check['converged'])
    if payload['kind']=='formal':
        for case in payload['recomputed']['cases']:
            merged=[];status='pending';terminal=False
            for batch in case['batches']:
                require(not terminal,'batch after exact stability acceptance')
                values=[];warm_failed=False
                for sample in batch['samples']:
                    value,converged=formal[(case['case_id'],batch['batch'],sample['trial'])]
                    require(sample['warmup_converged']==converged,'sample threshold decision differs')
                    if converged:values.append(value);merged.append(value)
                    else:warm_failed=True
                complete=len(values)==10
                require(batch['complete']==complete and batch['warmup_failed']==warm_failed,'exact batch state changed')
                stable=complete and exact_cv_pass(values,Fraction(1,20)) and exact_cv_pass(merged,Fraction(1,20))
                checks.append({'case_id':case['case_id'],'batch':batch['batch'],'scope':'formal','limit_fraction':[1,20],
                               'complete':complete,'exact_stable':stable,'valid_samples':len(values),'merged_samples':len(merged)})
                if stable:status='stable';terminal=True
                elif batch['batch']==2 and (complete or warm_failed):status='unstable_after_bounded_remeasurement';terminal=True
            require(case['status']==status,'formal state disagrees with exact threshold: '+case['case_id'])
    return checks


def invoke(mode,root):
    command=[sys.executable,'-I','-B',str(WORKER),mode,str(root)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':''})
    return {'mode':mode,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
            'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr.encode()).hexdigest()}


def replay(root,policy_path):
    root=Path(root).resolve();policy_path=Path(policy_path).resolve()
    receipt={'schema_version':1,'extension_id':'bounded-float-replay-v1','status':'replay_rejected',
             'qualification':'offline_evidence_only_requires_independent_extension_B_and_family_review',
             'policy_sha256':sha(policy_path),'auditor_files_sha256':{p.name:sha(p) for p in (Path(__file__),WORKER,HERE.parent/'audit_replay.py')},
             'replay_environment':{'python':sys.version,'executable_sha256':sha(Path(sys.executable).resolve()),'machine':platform.machine(),'platform':platform.platform()},
             'input_artifacts_sha256':{},'original_strict_audit':None,'float_comparisons':[],'threshold_checks':[],'failed_checks':[],'GPU_execution':False}
    before=None
    try:
        require(receipt['policy_sha256']==POLICY_SHA256,'unreviewed replay policy hash')
        policy=read(policy_path);before=inventory(root);receipt['input_artifacts_sha256']=before
        spec=read(safe(root,'run_spec.json'));require(spec.get('schema_version')==2,'unsupported run schema')
        receipt['input_run_identity']={key:spec[key] for key in ('suite_id','family','kind')}
        receipt['input_manifest_sha256']=digest(before)
        receipt['original_strict_audit']=invoke('strict',root)
        outcome=invoke('recompute',root);receipt['recomputation_worker']={k:v for k,v in outcome.items() if k!='stdout'}
        require(outcome['returncode']==0,'frozen evidence recomputation rejected: '+outcome['stderr'])
        payload=json.loads(outcome['stdout']);require(payload['kind']==spec['kind'],'worker kind mismatch')
        receipt['threshold_checks']=exact_thresholds(root,payload)
        patterns=policy['preflight_float_paths'] if payload['kind']=='preflight' else policy['formal_float_paths']
        receipt['float_comparisons']=compare(payload['recorded'],payload['recomputed'],patterns)
        receipt['float_differences']=[row for row in receipt['float_comparisons'] if row['ulp_distance']]
        strict=receipt['original_strict_audit']
        if strict['returncode']!=0:
            expected='preflight recomputation differs' if payload['kind']=='preflight' else 'summary differs from raw recomputation'
            require(expected in strict['stderr'],'original strict failure is not explained by the approved summary comparison')
            require(receipt['float_differences'],'strict audit failed but no allowed floating difference explains it')
        receipt['status']='pass_under_bounded_float_replay_v1'
    except (ValueError,KeyError,OSError,subprocess.TimeoutExpired,json.JSONDecodeError) as exc:
        receipt['failed_checks'].append(str(exc));receipt['status']='replay_rejected'
        receipt['float_comparisons']=getattr(exc,'float_comparisons',receipt['float_comparisons'])
    finally:
        if before is not None:
            try:
                after=inventory(root);receipt['input_unchanged']=after==before
                require(after==before,'input archive changed during read-only replay')
            except (ValueError,OSError) as exc:
                receipt['failed_checks'].append(str(exc));receipt['status']='replay_rejected'
    return receipt
