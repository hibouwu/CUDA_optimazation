#!/usr/bin/env python3
"""B01: four missing microprobe coordinates; existing bandwidth evidence is reused."""
import argparse,fcntl,gzip,hashlib,json,os,random,shutil,subprocess,time
from pathlib import Path
from analyze_b01 import replay,sass,analyze
ROOT=Path(__file__).resolve().parent
CASES=[dict(id=f'smem_independent_read_write_w{w}',kind='smem',warps=w,iterations=256) for w in [1,8]]+[
 dict(id=f'tma_{direction}_one_cta_above_l2',kind='tma',input=i,slots=8192) for direction,i in [('input',1),('output',0)]]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def run_one(root,case,label,short=False):
    path=root/'samples'/case['id']/label
    if (path/'result.json').exists():
        r=json.loads((path/'result.json').read_text());replay(path,r);return r
    path.mkdir(parents=True,exist_ok=False);args={k:v for k,v in case.items() if k not in ['id','kind']}
    if short:args['iterations' if case['kind']=='smem' else 'slots']=1 if case['kind']=='smem' else 3
    cmd=[str(root/'build'/case['kind'])]+[str(x) for k,v in args.items() for x in ('--'+k,v)]
    start=time.time_ns();process=subprocess.run(cmd,cwd=path,capture_output=True,text=True,timeout=180)
    (path/'stdout.json').write_text(process.stdout);(path/'stderr.log').write_text(process.stderr);write(path/'command.json',cmd)
    if process.returncode:raise ValueError('B01 failed: '+process.stderr)
    r=json.loads(process.stdout);r.update(case_id=case['id'],label=label,returncode=process.returncode,host_start_ns=start,host_stop_ns=time.time_ns(),files={})
    names=['stored.u32','sums.u32'] if case['kind']=='smem' else ['transport.u16']
    for name in names:
        p=path/name;r['files'][name]=dict(bytes=p.stat().st_size,sha256=sha(p))
        with p.open('rb') as src,gzip.open(str(p)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
        p.unlink()
    write(path/'result.json',r);replay(path,r);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('step',choices=['prepare','build','sample']);p.add_argument('--output',required=True,type=Path);a=p.parse_args();root=a.output.resolve()
    if a.step=='prepare':
        (root/'source/probes').mkdir(parents=True,exist_ok=False);(root/'build').mkdir()
        for name in ['run_b01.py','analyze_b01.py']:shutil.copy2(ROOT/name,root/'source'/name)
        for name in ['b01_smem.cu','b01_tma_large.cu','r01_support.hpp','r00_common.hpp']:shutil.copy2(ROOT/'probes'/name,root/'source/probes'/name)
        write(root/'cases.json',CASES);write(root/'source_hashes.json',{str(p.relative_to(root)):sha(p) for p in (root/'source').rglob('*') if p.is_file()});return
    for n,h in json.loads((root/'source_hashes.json').read_text()).items():
        if sha(root/n)!=h:raise ValueError('source changed')
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Slurm required')
    if a.step=='build':
        version=subprocess.check_output(['nvcc','--version'],text=True)
        if 'release 12.9,' not in version:raise ValueError('CUDA12.9 required')
        (root/'build/nvcc-version.txt').write_text(version)
        for name,source in [('smem','b01_smem.cu'),('tma','b01_tma_large.cu')]:
            cmd=['nvcc','-std=c++17','-O3','-DNDEBUG','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','--ptxas-options=-v','source/probes/'+source,'-lcuda','-o','build/'+name]
            write(root/'build'/f'{name}-command.json',cmd)
            with (root/'build'/f'{name}.log').open('w') as f:subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
            with (root/'build'/f'{name}.sass').open('w') as f:subprocess.run(['cuobjdump','--dump-sass',str(root/'build'/name)],stdout=f,check=True)
        write(root/'build/hashes.json',{str(p.relative_to(root)):sha(p) for p in (root/'build').iterdir() if p.is_file()});sass(root);return
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,power.limit','--format=csv,noheader'],text=True).strip()
    if '\n' in gpu or 'GH200' not in gpu:raise ValueError('single GH200 required')
    env=dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,gpu=gpu)
    locks=[open('/tmp/gh200-measurement-'+gpu.split(',')[0].strip()+'.lock','a'),(root/'.run.lock').open('a')]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'environment.json').exists():
        if json.loads((root/'environment.json').read_text())!=env:raise ValueError('environment changed')
    else:write(root/'environment.json',env)
    for n,h in json.loads((root/'build/hashes.json').read_text()).items():
        if sha(root/n)!=h:raise ValueError('build changed')
    (root/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    for c in CASES:run_one(root,c,'smoke',True)
    for trial in range(10):
        group=CASES[:];random.Random(20261008+trial).shuffle(group)
        for c in group:run_one(root,c,f'formal-{trial:02}')
        print('B01 trial',trial,flush=True)
    analyze(root)
    (root/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
if __name__=='__main__':main()
