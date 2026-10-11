#!/usr/bin/env python3
"""R12: freeze, compile/SASS, nonuniform checks, paired sampling and analysis."""
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


def matrix():
    d=json.loads((ROOT/'configs/r12.json').read_text())
    return [dict(id=f'{pair}_{layout}_{org}',pair=i,swizzle=j,order=k,units=32)
            for i,pair in enumerate(d['pairs']) for j,layout in enumerate(d['layouts'])
            for k,org in enumerate(d['organizations'])]


def prepare(output,cutlass):
    if not cutlass or not (cutlass/'include/cute/tensor.hpp').exists():
        raise ValueError('CUTLASS3.9.2 snapshot required')
    version=(cutlass/'include/cutlass/version.h').read_text()
    for name,value in [('MAJOR',3),('MINOR',9),('PATCH',2)]:
        if f'#define CUTLASS_{name} {value}' not in version:
            raise ValueError('expected CUTLASS 3.9.2')
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source';(source/'probes').mkdir(parents=True);(source/'configs').mkdir()
    for name in ['run_r12.py','analyze_r12.py']:shutil.copy2(ROOT/name,source/name)
    for name in ['r12.cu','r01_support.hpp','r00_common.hpp']:
        shutil.copy2(ROOT/'probes'/name,source/'probes'/name)
    shutil.copy2(ROOT/'configs/r12.json',source/'configs/r12.json')
    shutil.copytree(cutlass/'include',source/'cutlass/include')
    write(output/'source_hashes.json',{str(p.relative_to(output)):sha(p) for p in source.rglob('*') if p.is_file()})
    write(output/'cases.json',matrix())


def build(output):
    b=output/'build';b.mkdir()
    version=subprocess.check_output(['nvcc','--version'],text=True)
    if 'release 12.9,' not in version:raise ValueError('CUDA12.9 formal build required')
    cmd=['nvcc','-std=c++17','-O3','-DNDEBUG','-gencode=arch=compute_90a,code=sm_90a',
         '-lineinfo','--ptxas-options=-v','-Isource/cutlass/include','source/probes/r12.cu','-lcuda','-o','build/r12']
    write(b/'command.json',cmd);(b/'nvcc-version.txt').write_text(version)
    with (b/'compile.log').open('w') as log:
        subprocess.run(cmd,cwd=output,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    with (b/'sass.txt').open('w') as log:
        subprocess.run(['cuobjdump','--dump-sass','build/r12'],cwd=output,stdout=log,check=True)
    write(b/'binary.json',dict(sha256=sha(b/'r12'),sass_sha256=sha(b/'sass.txt')))


def run_one(output,case,label,units=32,order=None):
    directory=output/'samples'/case['id']/label
    args=dict(pair=case['pair'],order=case['order'] if order is None else order,
              units=units,swizzle=case['swizzle'])
    if (directory/'result.json').exists():
        result=json.loads((directory/'result.json').read_text())
        if any(result[k]!=v for k,v in args.items()):raise ValueError('resume request changed')
        return result
    directory.mkdir(parents=True,exist_ok=False)
    cmd=[str(output/'build/r12')]+[str(x) for item in args.items() for x in ('--'+item[0].replace('_','-'),item[1])]
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
    sizes={'source.u16':65536,'shared.u16':155648,'initial.f32':32768,'accum.f32':32768}
    for name,size in sizes.items():
        p=directory/name
        if p.stat().st_size!=size:raise ValueError('wrong witness size')
        files[name]=dict(bytes=size,sha256=sha(p))
        with p.open('rb') as src,gzip.open(str(p)+'.gz','wb') as dest:shutil.copyfileobj(src,dest)
        p.unlink()
    result['files']=files;write(directory/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--cutlass-root',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();cases=matrix()
    if a.plan_only:print(json.dumps(cases,indent=2));return
    if not a.output:p.error('--output required')
    output=a.output.resolve()
    if not a.resume:prepare(output,a.cutlass_root)
    for name,h in json.loads((output/'source_hashes.json').read_text()).items():
        if sha(output/name)!=h:raise ValueError('frozen source changed: '+name)
    if a.prepare_only:print(output);return
    if not os.environ.get('SLURM_JOB_ID'):p.error('single-GPU Slurm allocation required')
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if '\n' in uuid:raise ValueError('one visible GPU required')
    locks=[open('/tmp/gh200-measurement-'+uuid+'.lock','a'),(output/'.run.lock').open('a')]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,gpu_uuid=uuid)
    if (output/'environment.json').exists():
        if json.loads((output/'environment.json').read_text())!=env:raise ValueError('new allocation requires new run')
    else:write(output/'environment.json',env)
    (output/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    if not (output/'build/binary.json').exists():build(output)
    if sha(output/'build/r12')!=json.loads((output/'build/binary.json').read_text())['sha256']:
        raise ValueError('binary changed')
    def analyze(*flags):
        subprocess.run([sys.executable,str(output/'source/analyze_r12.py'),'--input',str(output),*flags],check=True)
    analyze('--sass-only')
    # Odd unit counts make the WGMMA's alternating-sign work numerically observable.
    for c in cases:
        for units in [1,3]:run_one(output,c,f'smoke-{units}',units)
    analyze('--check-only')
    if a.smoke_only:print('R12 all short cases passed');return
    for pair in range(3):
        for swizzle in range(2):
            group=[c for c in cases if c['pair']==pair and c['swizzle']==swizzle]
            by_order={c['order']:c for c in group};records={c['id']:[] for c in group}
            target=3;trial=0
            while trial<target:
                blocks=[[0,1],[2,3]];rng=random.Random(20261008+pair*100+swizzle*10+trial)
                rng.shuffle(blocks)
                for block in blocks:
                    rng.shuffle(block)
                    for org in block:
                        r=run_one(output,by_order[org],f'formal-{trial:02}')
                        records[r['case_id']].append(r)
                trial+=1
                if trial==target:
                    unstable=any(cv([r['elapsed_cycles'] for r in rr])>(.01 if target==3 else .05)
                                 or any(not r['warmup_converged'] for r in rr) for rr in records.values())
                    if unstable and target<30:target=10 if target==3 else target+10
            print('pair',pair,'layout',swizzle,'paired processes',target,flush=True)
    for pair in range(3):
        for swizzle in range(2):
            case=next(c for c in cases if c['pair']==pair and c['swizzle']==swizzle)
            for trial in range(3):
                controls=[('b-first',4),('a-second',5),('empty',6)]
                random.Random(20261008+pair*100+swizzle*10+trial).shuffle(controls)
                for label,org in controls:run_one(output,case,f'{label}-{trial:02}',order=org)
    analyze()
    (output/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    print('R12 archive:',output)


if __name__=='__main__':main()
