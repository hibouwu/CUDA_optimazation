#!/usr/bin/env python3
"""V08: wider same-card calibration, controls, then frozen predictions on new held-out shapes.

Variants per process: plain (scored), stamped (per-tile events, as V07), ends (CTA entry,
producer first work and final release only; low-perturbation window and clock).
  prepare --output RUN --v07 V07RUN      CPU: sources, CUTLASS copy, overlay, matrix
  build | setup | sample --set calib|ctrl|heldout [--predictions FROZEN]   on the node
The measured GPU is selected by UUID in V08_GPU (also exported as CUDA_VISIBLE_DEVICES).
"""
import argparse,fcntl,gzip,json,os,random,shutil,subprocess,time
from pathlib import Path
import v06_run as common
import v06_model as model

ROOT=Path(__file__).resolve().parent
CONFIGS={'cfg_a':0,'cfg_b':1,'cfg_c':2}
VARIANTS=['plain','stamped','ends']
V07_CALIB=[('c1_r_k512',1536,2816,512),('c2_k4096',1536,2816,4096),('c3_shortk',3072,5632,256),
           ('c4_k2048',3072,5632,2048),('c5_k1024',6144,5632,1024),('c6_longk',1536,2816,16384),
           ('c7_large',6144,11264,2048),('c8_tail',2816,2816,1024),('c9_thrash',8192,8192,2048)]
# New calibration: single/partial waves, small footprints, cfg_c long-K multi-tile, swizzle 8.
NEW_CALIB=[('g1',1280,1792,1024,1),('g2',2048,2048,512,1),('g3',1024,1536,4096,1),('g4',2560,2304,3072,1),
           ('g5',1792,2816,12288,1),('g6',3072,1536,8192,1),('g7',2304,3328,2048,8),('g8',4096,2048,1536,8)]
# cfg_a boundary magnitude: odd tile-row count (cluster 2x1 pads a whole out-of-bounds row) and a
# partial odd row; N columns and K varied.
BOUNDARY=[(m,n,k) for m in (1152,1664,2176) for n in (1536,2560) for k in (2048,8192)]+[(1624,2560,2048),(1624,2560,8192)]
# L2 control: evict before every timed call vs. warm (diagnostic, not fitted).
CONTROL=[('g1',1280,1792,1024),('g4',2560,2304,3072),('c4_k2048',3072,5632,2048)]
# Held-out shapes, identical for all configs; new (checked against history at prepare).
HELDOUT=[('h01',1408,2688,2048,1),('h02',2944,1920,5120,1),('h03',2400,3000,1000,1),('h04',1024,6144,3072,1),
         ('h05',6400,1280,1536,1),('h06',3584,3584,20480,1),('h07',5120,4608,2560,8),('h08',2048,2048,24576,1),
         ('h09',1536,1280,768,1),('h10',4480,5376,1024,8),('h11',2688,1664,4096,1),('h12',7168,6144,2048,1)]


def row(case_id,cfg,cset,m,n,k,swizzle=1,evict=0,kind='ordinary'):
    return dict(id=case_id,config=cfg,set=cset,kind=kind,m=m,n=n,k=k,lda=k,ldb=n,ldd=n,storage_m=m,storage_n=n,
                zero_m=-1,zero_n=-1,swizzle=swizzle,evict=evict)


def boundary_kind(cfg,m,n):
    """cfg_a pads along M (cluster 2x1), cfg_c along N (cluster 1x2)."""
    extent,tile=(m,128) if cfg=='cfg_a' else (n,128) if cfg=='cfg_c' else (None,None)
    if extent is None:return 'ordinary'
    rows=model.cdiv(extent,tile)
    if rows%2:return 'oob' if extent%tile==0 else 'partial_odd'
    return 'partial_even' if extent%tile else 'ordinary'


def matrix():
    rows=[]
    for cfg in CONFIGS:
        for label,m,n,k in V07_CALIB:rows.append(row(f'{cfg}_{label}',cfg,'calib',m,n,k,kind=boundary_kind(cfg,m,n)))
        for label,m,n,k,sw in NEW_CALIB:rows.append(row(f'{cfg}_{label}',cfg,'calib',m,n,k,sw,kind=boundary_kind(cfg,m,n)))
        for label,m,n,k in CONTROL:rows.append(row(f'{cfg}_{label}_evict',cfg,'ctrl',m,n,k,evict=1,kind=boundary_kind(cfg,m,n)))
        for label,m,n,k,sw in HELDOUT:rows.append(row(f'{cfg}_{label}',cfg,'heldout',m,n,k,sw,kind=boundary_kind(cfg,m,n)))
    for m,n,k in BOUNDARY:rows.append(row(f'cfg_a_b{m}x{n}_k{k}','cfg_a','calib',m,n,k,kind=boundary_kind('cfg_a',m,n)))
    return rows


def prepare(root,v07):
    if root.exists():raise ValueError('new V08 directory required')
    rows=matrix()
    prior=model.prior_shapes()
    clashes=sorted({(r['m'],r['n'],r['k']) for r in rows if r['set']=='heldout'}&prior)
    if clashes:raise ValueError(f'held-out shapes already measured: {clashes}')
    source=root/'source';(source/'probes').mkdir(parents=True);(root/'build').mkdir()
    for name in ['run_v08.py','v08_model.py','v08_fit.py','analyze_v08.py','v06_run.py','v06_model.py','v06_fit.py',
                 'analyze_r18.py','v07_model.py','run_r18.py']:
        shutil.copy2(ROOT/name,source/name)
    for name in ['r18.cu','r18_trace.hpp','gaps_common.hpp','r00_common.hpp']:shutil.copy2(ROOT/'probes'/name,source/'probes'/name)
    shutil.copytree(v07/'source/cutlass',source/'cutlass')
    common.make_overlay(source/'cutlass',source/'overlay')
    shutil.copy2(ROOT/'probes/r18_trace.hpp',source/'overlay/v06_trace.hpp')
    for header in [common.COOP,common.PING]:
        path=source/'overlay'/header;text=path.read_text();site='        ++v06_tile_seq;\n'
        if text.count(site)!=1:raise ValueError('consumer coordinate site changed')
        path.write_text(text.replace(site,site+'        v06_begin(work_tile_info, v06_tile_seq);\n'))
    commands={}
    for cfg,index in CONFIGS.items():
        for variant in VARIANTS:
            cmd=['nvcc','-std=c++17','-O3','-DNDEBUG','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','--ptxas-options=-v',f'-DV06_CFG={index}']
            if variant!='plain':cmd+=['-DV06_TRACE','-Isource/overlay']+(['-DV08_ENDS'] if variant=='ends' else [])
            cmd+=['-Isource/probes','-Isource/cutlass/include','-Isource/cutlass/tools/util/include','source/probes/r18.cu','-o',f'build/{cfg}_{variant}']
            commands[f'{cfg}_{variant}']=cmd
    common.write_json(root/'build/commands.json',commands);common.write_json(root/'cases.json',rows)
    common.write_json(root/'novelty.json',dict(prior_shapes_scanned=len(prior),heldout_clashes=clashes))
    common.write_json(root/'run_config.json',dict(family='v08',cases_sha256=common.sha(root/'cases.json'),v07=str(v07)))
    common.write_json(root/'source_hashes.json',{str(p.relative_to(root)):common.sha(p) for p in source.rglob('*') if p.is_file()})
    print('prepared',len(rows),'cases',{s:sum(r['set']==s for r in rows) for s in ['calib','ctrl','heldout']},'prior shapes',len(prior))


def identity():
    uuid=os.environ.get('V08_GPU','')
    if not uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:raise ValueError('set V08_GPU and CUDA_VISIBLE_DEVICES to one UUID')
    gpu=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    if '\n' in gpu or 'GH200' not in gpu:raise ValueError('one GH200 expected: '+gpu)
    return dict(gpu=gpu,job=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename)


def case_args(r):
    keys=['m','n','k','lda','ldb','ldd','storage_m','storage_n','zero_m','zero_n','swizzle','evict']
    for key in ['sm_count','input_mode','seed','alloc_lda','alloc_ldb','input_map_m','input_map_n']:
        if key in r:keys.append(key)
    return [str(x) for key in keys for x in ('--'+key.replace('_','-'),r[key])]


def setup(root):
    common.verify(root);records=[]
    for r in json.loads((root/'cases.json').read_text()):
        raw=subprocess.check_output([str(root/'build'/f"{r['config']}_plain"),'--mode','setup',*case_args(r)],text=True)
        event=next(json.loads(l) for l in raw.splitlines() if '"setup"' in l)
        if event['stages']!=(4 if r['config']=='cfg_c' else 6):raise ValueError('stage count changed')
        records.append(dict(case=r['id'],setup=event))
    common.write_json(root/'static_setup.json',records);common.write_json(root/'environment.json',identity())
    (root/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi','-q','-i',os.environ['V08_GPU']],text=True))
    print('setup',len(records),'cases')


def run_one(root,r,variant,trial,attempt=0):
    import v08_model
    folder=root/'samples'/r['id'];folder.mkdir(parents=True,exist_ok=True)
    suffix=f'{variant}-{trial:02}'+(f'-retry{attempt}' if attempt else '');path=folder/(suffix+'.json')
    if path.exists():
        rec=json.loads(path.read_text())
        if rec['returncode'] and 'last five warmups CV exceeds 2%' in rec['stderr'] and attempt<2:return run_one(root,r,variant,trial,attempt+1)
        if rec['returncode'] or common.sha(root/rec['raw'])!=rec['raw_sha256']:raise ValueError('invalid prior process '+str(path))
        return rec
    cmd=[str(root/'build'/f"{r['config']}_{variant}"),*case_args(r)]
    start=time.time_ns();proc=subprocess.run(cmd,text=True,capture_output=True,timeout=300)
    raw=folder/(suffix+'.txt.gz')
    with gzip.open(raw,'wt') as stream:stream.write(proc.stdout)
    rec=dict(case=r['id'],variant=variant,trial=trial,attempt=attempt,command=cmd,returncode=proc.returncode,host_start_ns=start,
             host_stop_ns=time.time_ns(),stderr=proc.stderr,raw=str(raw.relative_to(root)),raw_sha256=common.sha(raw))
    for line in proc.stdout.splitlines():
        ev=json.loads(line)
        if ev.get('event')=='call':rec.update(elapsed_us=ev['elapsed_us'],warmup=ev['warmup_us'])
    common.write_json(path,rec)
    if proc.returncode and 'last five warmups CV exceeds 2%' in proc.stderr and attempt<2:return run_one(root,r,variant,trial,attempt+1)
    if proc.returncode:raise ValueError('failed GEMM process: '+str(path)+' '+proc.stderr[-300:])
    if variant in ['wide','dual']:
        from analyze_r18 import replay
        replay(root,rec,r)
    else:
        v08_model.observe(root,rec,r)
    return rec


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step',choices=['prepare','build','setup','sample']);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--v07',type=Path);p.add_argument('--set',choices=['calib','ctrl','heldout']);p.add_argument('--predictions',type=Path)
    a=p.parse_args();root=a.output.resolve()
    if a.step=='prepare':prepare(root,a.v07.resolve());return
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Slurm allocation required')
    if a.step=='build':
        version=subprocess.check_output(['nvcc','--version'],text=True)
        if 'release 12.9,' not in version:raise ValueError('CUDA 12.9 required')
        (root/'build/nvcc-version.txt').write_text(version)
        common.build(root);common.write_json(root/'build/sass_hashes.json',{q.name:common.sha(q) for q in (root/'build').glob('*.sass')});return
    env=identity()
    locks=[open('/tmp/gh200-measurement-'+env['gpu'].split(',')[0]+'.lock','a'),(root/'.run.lock').open('a')]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.step=='setup':setup(root);return
    common.verify(root)
    if env!=json.loads((root/'environment.json').read_text()):raise ValueError('device or allocation changed')
    config=json.loads((root/'run_config.json').read_text())
    if common.sha(root/'cases.json')!=config['cases_sha256']:raise ValueError('matrix changed')
    rows=[r for r in json.loads((root/'cases.json').read_text()) if r['set']==a.set]
    if a.set=='heldout':
        if not a.predictions:raise ValueError('frozen predictions required')
        frozen=json.loads(a.predictions.read_text());now=time.time_ns()
        if frozen['status']!='frozen' or frozen['gpu']!=env['gpu'] or not frozen['frozen_unix_ns']<now:raise ValueError('invalid freeze')
        if set(frozen['predictions'])!={r['id'] for r in rows}:raise ValueError('frozen targets mismatch')
        if a.predictions.stat().st_mode&0o222:raise ValueError('predictions must be read-only')
        binding=root/'prediction_binding.json'
        if binding.exists():
            if json.loads(binding.read_text())['sha256']!=common.sha(a.predictions):raise ValueError('prediction changed on resume')
        else:
            if any((root/'samples'/r['id']).exists() for r in rows):raise ValueError('target samples predate binding')
            common.write_json(binding,dict(sha256=common.sha(a.predictions),frozen_unix_ns=frozen['frozen_unix_ns'],first_sample_unix_ns=now,gpu=env['gpu']))
    for trial in range(10):
        group=rows[:];random.Random(20261009+trial).shuffle(group)
        for r in group:
            variants=config.get('variants',VARIANTS)[:];random.Random(f"{trial}-{r['id']}").shuffle(variants)
            for variant in variants:run_one(root,r,variant,trial)
        print('V08',a.set,'trial',trial,'complete',time.strftime('%H:%M:%S'),flush=True)
    (root/f'nvidia-smi-after-{a.set}.txt').write_text(subprocess.check_output(['nvidia-smi','-q','-i',os.environ['V08_GPU']],text=True))

if __name__=='__main__':main()
