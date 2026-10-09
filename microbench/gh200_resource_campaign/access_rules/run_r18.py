#!/usr/bin/env python3
"""R18 matched physical grids; R19 reuses this actual-work-coordinate probe."""
import argparse,fcntl,gzip,json,os,random,shutil,statistics,subprocess,time
from pathlib import Path
import v06_run as common
import v06_model as model

ROOT=Path(__file__).resolve().parent
CONFIGS={'cfg_a':0,'cfg_b':1,'cfg_c':2,'cfg_a1':3,'cfg_c1':4}
common.EXPECTED_HGMMA.update(cfg_a1=8,cfg_c1=16)
model.CONFIGS.update(cfg_a1=dict(index=3,tile=(128,128,64),cluster=(1,1),schedule='cooperative'),
                     cfg_c1=dict(index=4,tile=(256,128,64),cluster=(1,1),schedule='cooperative'))


def cases(family='r18'):
    rows=[]
    if family=='b01_cache':
        return [dict(id=f'gemm_sw{sw}_'+('evict_prepared' if ev else 'repeat'),config='cfg_a',group='cache',kind='ordinary',
                     m=4096,n=4096,k=2048,lda=2048,ldb=4096,ldd=4096,storage_m=4096,storage_n=4096,zero_m=-1,zero_n=-1,swizzle=sw,evict=ev)
                for sw in [1,8] for ev in [0,1]]
    if family=='r19':
        for label,m,n,k in [('near_short',1408,1408,1024),('near_long',1408,1408,8192),
                            ('multi_short',4096,4096,1024),('multi_long',4096,4096,8192)]:
            for swizzle in [1,8]:
                rows.append(dict(id=f'cfg_b_{label}_sw{swizzle}',config='cfg_b',group=label,kind='ordinary',
                    m=m,n=n,k=k,lda=k,ldb=n,ldd=n,storage_m=m,storage_n=n,zero_m=-1,zero_n=-1,swizzle=swizzle))
        return rows
    for config in ['cfg_a','cfg_c']:
        for k in [1024,8192]:
            for kind,extent in [('ordinary',1536),('oob',1408),('explicit_zero',1536),('partial_even',1504),('partial_odd',1376)]:
                m,n=(extent,2048) if config=='cfg_a' else (3072,extent)
                pm,pn=(1536,2048) if config=='cfg_a' else (3072,1536)
                rows.append(dict(id=f'{config}_{kind}_k{k}',config=config,group=f'{config}_k{k}',kind=kind,
                  m=m,n=n,k=k,lda=k,ldb=pn,ldd=pn,storage_m=pm,storage_n=pn,
                  zero_m=1408 if kind=='explicit_zero' and config=='cfg_a' else -1,
                  zero_n=1408 if kind=='explicit_zero' and config=='cfg_c' else -1,swizzle=1))
        for kind,extent in [('oob',1408),('partial_odd',1376)]:
            m,n=(extent,2048) if config=='cfg_a' else (3072,extent)
            pm,pn=(1536,2048) if config=='cfg_a' else (3072,1536)
            rows.append(dict(id=f'{config}1_{kind}_k8192',config=config+'1',group=f'{config}_k8192',kind=kind,
                 m=m,n=n,k=8192,lda=8192,ldb=pn,ldd=pn,storage_m=pm,storage_n=pn,zero_m=-1,zero_n=-1,swizzle=1))
    if family=='r18_lite':
        rows=[r for r in rows if r['config'] in ['cfg_a','cfg_c'] and (r['kind']=='ordinary' or (r['config']=='cfg_a' and r['kind']=='partial_even') or (r['config']=='cfg_c' and r['kind']=='partial_odd'))]
    return rows


def prepare(root,cutlass,family,rows=None,dual_clock=False):
    rows=cases(family) if rows is None else rows
    root.mkdir(parents=True,exist_ok=False);source=root/'source';(source/'probes').mkdir(parents=True);(root/'build').mkdir()
    for name in ['run_r18.py','run_r18_lite.py','run_r19.py','run_b01_cache.py','analyze_r18.py','v06_run.py','v06_model.py']:
        shutil.copy2(ROOT/name,source/name)
    for name in ['r18.cu','r18_trace.hpp','gaps_common.hpp','r00_common.hpp']:
        shutil.copy2(ROOT/'probes'/name,source/'probes'/name)
    shutil.copy2(ROOT/'probes/r18_trace.hpp',source/'probes/v06_trace.hpp')
    for part in ['include','tools/util/include']:shutil.copytree(cutlass/part,source/'cutlass'/part)
    common.make_overlay(source/'cutlass',source/'overlay')
    shutil.copy2(ROOT/'probes/r18_trace.hpp',source/'overlay/v06_trace.hpp')
    for header in [common.COOP,common.PING]:
        path=source/'overlay'/header;text=path.read_text()
        site='        ++v06_tile_seq;\n'
        if text.count(site)!=1:raise ValueError('consumer coordinate site changed')
        text=text.replace(site,site+'        v06_begin(work_tile_info, v06_tile_seq);\n')
        path.write_text(text)
    commands={}
    variants=['plain','wide','stamped','dual'] if dual_clock else ['plain'] if family=='b01_cache' else ['plain','stamped']
    for config in sorted({r['config'] for r in rows}):
        for variant in variants:
            cmd=['nvcc','-std=c++17','-O3','-DNDEBUG','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','--ptxas-options=-v',f'-DV06_CFG={CONFIGS[config]}']
            if dual_clock and variant!='plain':cmd+=['-DR18_DUAL_CLOCK' if variant=='dual' else '-DR18_MATCH_DUAL_LAYOUT']
            if variant in ['stamped','dual']:
                cmd+=['-DV06_TRACE','-Isource/overlay']
                if family=='r18_lite':cmd+=['-DR18_LIGHT']
            cmd+=['-Isource/probes','-Isource/cutlass/include','-Isource/cutlass/tools/util/include','source/probes/r18.cu','-o',f'build/{config}_{variant}']
            commands[f'{config}_{variant}']=cmd
    common.write_json(root/'build/commands.json',commands);common.write_json(root/'cases.json',rows)
    common.write_json(root/'run_config.json',dict(family=family,cases_sha256=common.sha(root/'cases.json'),variants=variants))
    common.write_json(root/'source_hashes.json',{str(p.relative_to(root)):common.sha(p) for p in source.rglob('*') if p.is_file()})


def case_args(row):
    keys=['m','n','k','lda','ldb','ldd','storage_m','storage_n','zero_m','zero_n','swizzle']
    if 'evict' in row:keys.append('evict')
    for key in ['sm_count','input_mode','seed','alloc_lda','alloc_ldb']:
        if key in row:keys.append(key)
    return [str(x) for key in keys for x in ('--'+key.replace('_','-'),row[key])]


def setup(root):
    common.verify(root);records=[]
    for row in json.loads((root/'cases.json').read_text()):
        raw=subprocess.check_output([str(root/'build'/f"{row['config']}_plain"),'--mode','setup',*case_args(row)],text=True)
        event=next(json.loads(l) for l in raw.splitlines() if '"setup"' in l)
        expected_stages=4 if row['config'].startswith('cfg_c') else 6
        if event['stages']!=expected_stages:raise ValueError('stage count changed')
        records.append(dict(case=row['id'],setup=event))
    for group in {r['group'] for r in json.loads((root/'cases.json').read_text())}:
        ids={r['id'] for r in json.loads((root/'cases.json').read_text()) if r['group']==group and r['kind'] in ['ordinary','oob','explicit_zero'] and r['config'] in ['cfg_a','cfg_c']}
        grids={tuple(r['setup']['grid']) for r in records if r['case'] in ids}
        if len(grids)>1:raise ValueError('matched boundary grid changed')
    common.write_json(root/'static_setup.json',records);common.write_json(root/'environment.json',common.identity())
    (root/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    print('setup verified',len(records),'cases')


def run_one(root,row,variant,trial,attempt=0):
    folder=root/'samples'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    suffix=f'{variant}-{trial:02}'+(f'-retry{attempt}' if attempt else '')
    path=folder/(suffix+'.json')
    if path.exists():
        record=json.loads(path.read_text())
        if record['returncode'] and 'last five warmups CV exceeds 2%' in record['stderr'] and attempt<2:
            return run_one(root,row,variant,trial,attempt+1)
        if record['returncode'] or common.sha(root/record['raw'])!=record['raw_sha256']:raise ValueError('invalid prior process')
        return record
    cmd=[str(root/'build'/f"{row['config']}_{variant}"),*case_args(row)]
    start=time.time_ns();proc=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
    raw=folder/(suffix+'.txt.gz')
    with gzip.open(raw,'wt') as stream:stream.write(proc.stdout)
    record=dict(case=row['id'],variant=variant,trial=trial,attempt=attempt,command=cmd,returncode=proc.returncode,
       host_start_ns=start,host_stop_ns=time.time_ns(),stderr=proc.stderr,raw=str(raw.relative_to(root)),raw_sha256=common.sha(raw))
    for line in proc.stdout.splitlines():
        ev=json.loads(line)
        if ev.get('event')=='call':record.update(elapsed_us=ev['elapsed_us'],warmup=ev['warmup_us'])
    common.write_json(path,record)
    if proc.returncode and 'last five warmups CV exceeds 2%' in proc.stderr and attempt<2:
        return run_one(root,row,variant,trial,attempt+1)
    if proc.returncode:raise ValueError('failed GEMM process: '+str(path))
    from analyze_r18 import replay
    replay(root,record,row)
    return record


def main(family='r18'):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('step',choices=['prepare','build','setup','sample','list'])
    p.add_argument('--output',type=Path);p.add_argument('--cutlass-root',type=Path);p.add_argument('--procs',type=int,choices=[1,10],default=10)
    p.add_argument('--cases-file',type=Path,help='prepare: use a listed extension of an existing experiment')
    p.add_argument('--dual-clock',action='store_true',help='prepare: add same-call globaltimer events; all variants use matched scratch sizes')
    a=p.parse_args()
    if a.step=='list':print(json.dumps(cases(family),indent=2));return
    root=a.output.resolve()
    if a.step=='prepare':
        rows=json.loads(a.cases_file.read_text()) if a.cases_file else None
        prepare(root,a.cutlass_root.resolve(),family,rows,a.dual_clock);return
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Slurm allocation required')
    if a.step=='build':
        version=subprocess.check_output(['nvcc','--version'],text=True)
        if 'release 12.9,' not in version:raise ValueError('CUDA12.9 required')
        (root/'build/nvcc-version.txt').write_text(version);common.build(root)
        common.write_json(root/'build/sass_hashes.json',{p.name:common.sha(p) for p in (root/'build').glob('*.sass')});return
    env=common.identity();gpu=env['gpu'].split(',')[0].strip()
    locks=[open('/tmp/gh200-measurement-'+gpu+'.lock','a'),(root/'.run.lock').open('a')]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.step=='setup':setup(root);return
    common.verify(root)
    if env!=json.loads((root/'environment.json').read_text()):raise ValueError('allocation/device changed')
    config=json.loads((root/'run_config.json').read_text())
    if config['family']!=family or common.sha(root/'cases.json')!=config['cases_sha256']:raise ValueError('matrix changed')
    rows=json.loads((root/'cases.json').read_text())
    # Matching physical-grid controls stay adjacent; group and variant order randomized.
    for trial in range(a.procs):
        groups=sorted({r['group'] for r in rows});random.Random(20261008+trial).shuffle(groups)
        for group in groups:
            subset=[r for r in rows if r['group']==group];random.Random(20261008+trial).shuffle(subset)
            for row in subset:
                variants=config.get('variants',['plain'] if family=='b01_cache' else ['plain','stamped'])[:]
                random.Random(20261008+trial).shuffle(variants)
                for variant in variants:run_one(root,row,variant,trial)
        print(family,'trial',trial,'complete',flush=True)
    (root/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))

if __name__=='__main__':main()
