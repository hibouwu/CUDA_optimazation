#!/usr/bin/env python3
"""R13: build, paired representatives, finite 18-case supply/retirement matrix."""
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
ROOT=Path(__file__).resolve().parent

def matrix():
    return [dict(id=f'n{n}_s{s}_c{c}',n=n,stages=s,consumer=c)
            for n in (128,256) for s in (1,2,4) for c in (0,1,2)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def build(output,cutlass,extra):
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source';(source/'probes').mkdir(parents=True)
    for name in ('run_r13.py','analyze_r13.py'):shutil.copy2(ROOT/name,source/name)
    for name in ('r13.cu','gaps_common.hpp','r00_common.hpp'):
        shutil.copy2(ROOT/'probes'/name,source/'probes'/name)
    shutil.copytree(cutlass/'include',source/'cutlass/include')
    write(output/'source_hashes.json',{str(p.relative_to(source)):sha(p)
          for p in source.rglob('*') if p.is_file()})
    directory=output/'build';directory.mkdir()
    for variant in ('plain','trace_ready','trace_retire','trace_reuse'):
        cmd=['nvcc','-std=c++17','-O3','-DNDEBUG','-gencode=arch=compute_90a,code=sm_90a',
             '-lineinfo','--ptxas-options=-v','-I'+str(source/'cutlass/include'),
             str(source/'probes/r13.cu'),'-lcuda','-o',str(directory/variant)]
        if variant!='plain':cmd+=['-DR13_TRACE','-DR13_TRACE_PAIR='+str({'trace_ready':0,'trace_retire':1,'trace_reuse':2}[variant])]
        cmd+=extra;write(directory/(variant+'-command.json'),cmd)
        with (directory/(variant+'.log')).open('w') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        with (directory/(variant+'.sass')).open('w') as f:
            subprocess.run(['cuobjdump','--dump-sass',str(directory/variant)],stdout=f,check=True)
    write(directory/'binary_hashes.json',{v:sha(directory/v) for v in ('plain','trace_ready','trace_retire','trace_reuse')})

def sample(output,case,variant,trial,scope,trace_tile):
    folder=output/'samples'/case['id']/f'{variant}-{trial:02d}';folder.mkdir(parents=True)
    cmd=[str(output/'build'/variant)]
    for name in ('n','stages','consumer'):cmd+=['--'+name,str(case[name])]
    cmd+=['--scope',scope,'--trace-tile',str(trace_tile)];write(folder/'command.json',cmd)
    with (folder/'stdout.json').open('w') as f,(folder/'stderr.log').open('w') as e:
        proc=subprocess.run(cmd,cwd=folder,stdout=f,stderr=e,timeout=180)
    write(folder/'process.json',dict(returncode=proc.returncode))
    if proc.returncode:raise RuntimeError(str(folder))
    row=json.loads((folder/'stdout.json').read_text());assert row['status']=='measured'
    row.update(configuration=case,variant=variant,trial=trial,trace_cta0_only=True,
               gpu_uuid=json.loads((output/'environment.json').read_text())['gpu_uuid'],files={})
    for name in ('accum.f32','last-input.u32','trace.u64'):
        path=folder/name
        if path.exists():
            raw=path.read_bytes();row['files'][name]={'sha256':sha(path),'bytes':len(raw)}
            with gzip.open(str(path)+'.gz','wb',compresslevel=1) as f:f.write(raw)
            path.unlink()
    write(folder/'result.json',row)
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--cutlass-root',type=Path);p.add_argument('--build-only',action='store_true')
    p.add_argument('--sample-only',action='store_true');p.add_argument('--representatives',action='store_true')
    p.add_argument('--scope',choices=('one_cta','all_gpu'),default='one_cta')
    p.add_argument('--trace-tile',type=int,default=16)
    p.add_argument('--trace-pair',choices=('ready','retire','reuse'),default='ready')
    p.add_argument('--case',action='append',help='restrict to named representative cases')
    p.add_argument('--nvcc-extra',action='append',default=[])
    a=p.parse_args()
    if a.list:print(json.dumps(matrix(),indent=2));return
    if not a.output:p.error('--output required')
    if not 0<=a.trace_tile<32:p.error('--trace-tile must be 0..31')
    output=a.output.resolve()
    if not a.sample_only:
        if not a.cutlass_root:p.error('--cutlass-root required for build')
        build(output,a.cutlass_root.resolve(),a.nvcc_extra)
    if a.build_only:return
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('GPU sampling requires Slurm')
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if '\n' in uuid:raise ValueError('one GPU required')
    write(output/'environment.json',dict(gpu_uuid=uuid,job=os.environ['SLURM_JOB_ID'],
          hostname=os.uname().nodename,cuda=subprocess.check_output(['nvcc','--version'],text=True)))
    cases=matrix()
    if a.representatives:cases=[c for c in cases if c['n']==256 and c['stages'] in (1,4) and c['consumer'] in (0,2)]
    if a.case:cases=[c for c in cases if c['id'] in a.case]
    if not cases:raise ValueError('no selected cases')
    if a.trace_pair=='reuse' and any(a.trace_tile+c['stages']>=32 for c in cases):
        p.error('reuse selected tile must have a subsequent refill (tile + stages < 32)')
    write(output/'cases.json',cases);rng=random.Random(20261007);records={c['id']:[] for c in cases}
    counts=3 if a.scope=='one_cta' else 10
    with open('/tmp/gh200-access-'+uuid+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for trial in range(counts):
            pairs={}
            for c in cases:pairs.setdefault((c['n'],c['consumer']),[]).append(c)
            order=list(pairs.values());rng.shuffle(order)
            ordered=[]
            for pair in order:
                rng.shuffle(pair);ordered.extend(pair)
            for case in ordered:
                variants=['plain','trace_'+a.trace_pair];rng.shuffle(variants)
                for variant in variants:
                    records[case['id']].append(sample(output,case,variant,trial,a.scope,a.trace_tile))
        extend=set()
        for case in cases:
            rows=records[case['id']];plain=[r['elapsed'] for r in rows if r['variant']=='plain']
            if counts==3 and statistics.stdev(plain)/statistics.mean(plain)>.01:
                extend.add(case['id'])
        if counts==3:
            for n in (128,256):
                for consumer in (0,1,2):
                    paired=[c for c in cases if c['n']==n and c['consumer']==consumer]
                    for i,a_case in enumerate(paired):
                        for b_case in paired[i+1:]:
                            x=[r['elapsed'] for r in records[a_case['id']] if r['variant']=='plain']
                            y=[r['elapsed'] for r in records[b_case['id']] if r['variant']=='plain']
                            noise=3*(statistics.stdev(x)**2+statistics.stdev(y)**2)**.5
                            if abs(statistics.mean(x)-statistics.mean(y))<=max(64,noise):
                                extend.update((a_case['id'],b_case['id']))
        for trial in range(3,10):
            for case in cases:
                if case['id'] in extend:
                    for variant in ('plain','trace_'+a.trace_pair):
                        records[case['id']].append(sample(output,case,variant,trial,a.scope,a.trace_tile))
    subprocess.run([sys.executable,str(output/'source/analyze_r13.py'),'--input',str(output)],check=True)
if __name__=='__main__':main()
