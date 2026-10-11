#!/usr/bin/env python3
"""R16 residency/quota: freeze, compile/SASS, exact checks and paired sampling."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cv(values):return statistics.pstdev(values)/statistics.mean(values) if len(values)>1 else 0.


def matrix(subset='residency'):
    return [c for c in json.loads((ROOT/'configs/r16.json').read_text())['cases'] if c['subset']==subset]


def prepare(output,cutlass,subset='residency'):
    if not cutlass or not (cutlass/'include/cute/tensor.hpp').exists():
        raise ValueError('CUTLASS3.9.2 snapshot required')
    version=(cutlass/'include/cutlass/version.h').read_text()
    for name,value in [('MAJOR',3),('MINOR',9),('PATCH',2)]:
        if f'#define CUTLASS_{name} {value}' not in version:
            raise ValueError('expected CUTLASS 3.9.2')
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source';(source/'probes').mkdir(parents=True);(source/'configs').mkdir()
    for name in ['run_r16.py','analyze_r16.py','analyze_r16_quota.py']:shutil.copy2(ROOT/name,source/name)
    for name in ['r16_residency.cu','r16_quota.cu','r01_support.hpp','r00_common.hpp']:
        shutil.copy2(ROOT/'probes'/name,source/'probes'/name)
    shutil.copy2(ROOT/'configs/r16.json',source/'configs/r16.json')
    shutil.copytree(cutlass/'include',source/'cutlass/include')
    write(output/'source_hashes.json',{str(p.relative_to(output)):sha(p) for p in source.rglob('*') if p.is_file()})
    write(output/'cases.json',matrix(subset))
    write(output/'run_config.json',dict(subset=subset,cases_sha256=sha(output/'cases.json')))


def build(output,subset='residency'):
    b=output/'build';b.mkdir()
    version=subprocess.check_output(['nvcc','--version'],text=True)
    if 'release 12.9,' not in version:raise ValueError('CUDA12.9 formal build required')
    cmd=['nvcc','-std=c++17','-O3','-DNDEBUG','-gencode=arch=compute_90a,code=sm_90a',
         '-lineinfo','--ptxas-options=-v','-Isource/cutlass/include',f'source/probes/r16_{subset}.cu','-lcuda','-o','build/r16']
    write(b/'command.json',cmd);(b/'nvcc-version.txt').write_text(version)
    with (b/'compile.log').open('w') as log:
        subprocess.run(cmd,cwd=output,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    with (b/'sass.txt').open('w') as log:
        subprocess.run(['cuobjdump','--dump-sass','build/r16'],cwd=output,stdout=log,check=True)
    write(b/'binary.json',dict(sha256=sha(b/'r16'),sass_sha256=sha(b/'sass.txt')))


def run_one(output,case,label,tiles=32):
    directory=output/'samples'/case['id']/label
    args=dict(stage=case['stage'],resident=case['target_occupancy'],tiles=tiles,
              large=int(case['source']=='independent_large'))
    if (directory/'result.json').exists():
        result=json.loads((directory/'result.json').read_text())
        if any(result[k]!=v for k,v in args.items()):raise ValueError('resume request changed')
        return result
    directory.mkdir(parents=True,exist_ok=False)
    cmd=[str(output/'build/r16')]+[str(x) for item in args.items() for x in ('--'+item[0].replace('_','-'),item[1])]
    write(directory/'command.json',cmd)
    with (directory/'stdout.json').open('w') as out,(directory/'stderr.log').open('w') as err:
        start_ns=time.time_ns()
        process=subprocess.run(cmd,cwd=directory,stdout=out,stderr=err,timeout=120)
        stop_ns=time.time_ns()
    result=json.loads((directory/'stdout.json').read_text())
    result.update(case_id=case['id'],label=label,returncode=process.returncode,
                  host_start_ns=start_ns,host_stop_ns=stop_ns)
    if process.returncode not in [0,3] or result['event_errors'] or result['local_bytes']:
        write(directory/'result.json',result);raise ValueError('failed execution/events or spill: '+str(directory))
    files={}
    sizes={'accum.f32':result['blocks']*4096*4,'last.u16':result['blocks']*8192*2}
    for name,size in sizes.items():
        p=directory/name
        if p.stat().st_size!=size:raise ValueError('wrong witness size')
        files[name]=dict(bytes=size,sha256=sha(p))
        with p.open('rb') as src,gzip.open(str(p)+'.gz','wb') as dest:shutil.copyfileobj(src,dest)
        p.unlink()
    result['files']=files;write(directory/'result.json',result)
    return result


def run_quota_one(output,case,label):
    directory=output/'samples'/case['id']/label
    args=dict(delay=case['delay'],before=int(case['placement']=='before_dec'))
    if (directory/'result.json').exists():
        result=json.loads((directory/'result.json').read_text())
        if any(result[k]!=v for k,v in args.items()) or result.get('returncode')!=0:
            raise ValueError('invalid quota resume record')
        return result
    directory.mkdir(parents=True,exist_ok=False)
    cmd=[str(output/'build/r16')]+[str(x) for k,v in args.items() for x in ('--'+k,v)]
    write(directory/'command.json',cmd)
    with (directory/'stdout.json').open('w') as out,(directory/'stderr.log').open('w') as err:
        start=time.time_ns()
        process=subprocess.run(cmd,cwd=directory,stdout=out,stderr=err,timeout=120)
    result=json.loads((directory/'stdout.json').read_text())
    result.update(case_id=case['id'],label=label,returncode=process.returncode,
                  host_start_ns=start,host_stop_ns=time.time_ns())
    if process.returncode or result['local_bytes'] or result['registers']*384>65536:
        write(directory/'result.json',result);raise ValueError('quota launch/spill/pool failure')
    result['files']={}
    for name,size in [('values.f32',256*192*4),('integers.u32',384*4)]:
        path=directory/name
        if path.stat().st_size!=size:raise ValueError('wrong quota witness size')
        result['files'][name]=dict(bytes=size,sha256=sha(path))
        with path.open('rb') as src,gzip.open(str(path)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
        path.unlink()
    write(directory/'result.json',result)
    # Validate each process before spending time on the remaining coordinates.
    from analyze_r16_quota import replay
    replay(directory,result)
    return result


def sample_quota(output,cases,smoke_only,analyze,procs=3):
    for c in cases:run_quota_one(output,c,'smoke-00')
    analyze('--check-only')
    if smoke_only:return
    for delay in [0,64,256]:
        group=[c for c in cases if c['delay']==delay]
        records={c['id']:[] for c in group};target=procs;trial=0
        while trial<target:
            shuffled=group[:];random.Random(20261008+delay+trial).shuffle(shuffled)
            for c in shuffled:records[c['id']].append(run_quota_one(output,c,f'formal-{trial:02}'))
            trial+=1
            values=[[statistics.median(s[2]-s[1] for s in r['stamps'][4:]) for r in records[c['id']]] for c in group]
            if trial==3 and target==3:
                delta=abs(statistics.median(values[0])-statistics.median(values[1]))
                noise=max(statistics.pstdev(v) for v in values)
                if any(cv(v)>.01 for v in values) or delta<=max(2,3*noise):target=10
            if trial==target and target<30:
                bad=any(cv(v)>.05 for v in values) or any(not r['warmup_converged'] for rr in records.values() for r in rr)
                if bad:target=10 if target==3 else target+10
        print('quota delay',delay,'paired processes',target,flush=True)
    analyze()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--cutlass-root',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--subset',choices=['residency','quota'],default='residency')
    p.add_argument('--quota-procs',type=int,choices=[3,10],default=3,
                   help='use ten when comparing near-zero effects across delays')
    a=p.parse_args();cases=matrix(a.subset)
    if a.plan_only:print(json.dumps(cases,indent=2));return
    if not a.output:p.error('--output required')
    output=a.output.resolve()
    if not a.resume:prepare(output,a.cutlass_root,a.subset)
    config=json.loads((output/'run_config.json').read_text()) if (output/'run_config.json').exists() else {'subset':'residency'}
    if config['subset']!=a.subset:raise ValueError('resume subset changed')
    if json.loads((output/'cases.json').read_text())!=cases:raise ValueError('frozen cases changed')
    for name,h in json.loads((output/'source_hashes.json').read_text()).items():
        if sha(output/name)!=h:raise ValueError('frozen source changed: '+name)
    if a.prepare_only:print(output);return
    if not os.environ.get('SLURM_JOB_ID'):p.error('single-GPU Slurm allocation required')
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if '\n' in uuid:raise ValueError('one visible GPU required')
    model=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    if 'GH200' not in model:raise ValueError('ROMEO GH200 required')
    locks=[open('/tmp/gh200-measurement-'+uuid+'.lock','a'),(output/'.run.lock').open('a')]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,gpu_uuid=uuid)
    if (output/'environment.json').exists():
        if json.loads((output/'environment.json').read_text())!=env:raise ValueError('new allocation requires new run')
    else:write(output/'environment.json',env)
    (output/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    if not (output/'build/binary.json').exists():build(output,a.subset)
    if sha(output/'build/r16')!=json.loads((output/'build/binary.json').read_text())['sha256']:
        raise ValueError('binary changed')
    def analyze(*flags):
        subprocess.run([sys.executable,str(output/'source/analyze_r16.py'),'--input',str(output),*flags],check=True)
    analyze('--sass-only')
    if a.subset=='quota':
        sample_quota(output,cases,a.smoke_only,analyze,a.quota_procs)
        (output/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
        print('R16 quota archive:',output);return
    # Odd unit counts make the WGMMA's alternating-sign work numerically observable.
    for c in cases:
        for tiles in [1,2]:run_one(output,c,f'smoke-{tiles}',tiles)
    analyze('--check-only')
    if a.smoke_only:print('R16 residency all short cases passed');return
    # Whole-GPU cases use ten independent processes, paired residency controls.
    for stage in [1,2,4]:
        for source in ['shared_small','independent_large']:
            group=[c for c in cases if c['stage']==stage and c['source']==source]
            target=10;trial=0;records={c['id']:[] for c in group}
            while trial<target:
                shuffled=group[:];random.Random(20261008+stage*100+trial).shuffle(shuffled)
                for c in shuffled:
                    r=run_one(output,c,f'formal-{trial:02}')
                    records[c['id']].append(r)
                trial+=1
                if trial==target and target<30:
                    bad=any(cv([r['elapsed_ns'] for r in rr])>.05 or any(not r['warmup_converged'] for r in rr) for rr in records.values())
                    if bad:target+=10
            print('stage',stage,'source',source,'paired processes',target,flush=True)
    analyze()
    (output/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    print('R16 residency archive:',output)


if __name__=='__main__':main()
