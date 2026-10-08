#!/usr/bin/env python3
"""V07 uses verified R18/R19 binaries on the same card; held-out timing requires a freeze."""
import argparse,fcntl,json,os,random,shutil,subprocess,time
from pathlib import Path
import v06_model as model
import v06_run as common
import run_r18
ROOT=Path(__file__).resolve().parent


def matrix():
    rows=[]
    # Existing V06 calibration shapes may be remeasured; the 24 targets remain new.
    for r in model.case_rows('calib'):
        if r['config'] not in ['cfg_a','cfg_b','cfg_c']:continue
        rows.append(dict(r,set='calib',group=r['id'],kind='ordinary',storage_m=r['m'],storage_n=r['n'],zero_m=-1,zero_n=-1,swizzle=1))
    for r in json.loads((ROOT/'configs/v07-cases.json').read_text())['cases']:
        rows.append(dict(r,set='heldout',label=r['kind'],group=r['id']))
    return rows


def prepare(root,r18,r19):
    if root.exists():raise ValueError('new V07 directory required')
    for origin in [r18,r19]:common.verify(origin)
    for name,h in json.loads((r18/'source_hashes.json').read_text()).items():
        if name.startswith(('source/probes/','source/cutlass/','source/overlay/')) and common.sha(r19/name)!=h:
            raise ValueError('R18/R19 compiled source dependencies differ: '+name)
    shutil.copytree(r18/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));(root/'source/configs').mkdir(exist_ok=True);(root/'build').mkdir()
    for name in ['run_v07.py','v07_fit.py','v07_model.py','analyze_v07.py','v06_fit.py','run_r18.py','analyze_r18.py']:
        shutil.copy2(ROOT/name,root/'source'/name)
    shutil.copy2(ROOT/'configs/v07-cases.json',root/'source/configs/v07-cases.json')
    provenance={};commands={};hashes={}
    for cfg in ['cfg_a','cfg_b','cfg_c']:
        origin=r19 if cfg=='cfg_b' else r18
        origin_commands=json.loads((origin/'build/commands.json').read_text())
        for variant in ['plain','stamped']:
            name=f'{cfg}_{variant}'
            for suffix in ['', '.sass','.log']:shutil.copy2(origin/'build'/(name+suffix),root/'build'/(name+suffix))
            hashes['build/'+name]=common.sha(root/'build'/name);commands[name]=origin_commands[name]
            provenance[name]=dict(origin=str(origin),binary_sha256=hashes['build/'+name],source_manifest_sha256=common.sha(origin/'source_hashes.json'))
    common.write_json(root/'build/commands.json',commands);common.write_json(root/'build/binary_hashes.json',hashes)
    common.write_json(root/'build/origins.json',provenance)
    common.write_json(root/'build/sass_hashes.json',{p.name:common.sha(p) for p in (root/'build').glob('*.sass')})
    common.write_json(root/'cases.json',matrix())
    common.write_json(root/'run_config.json',dict(family='v07',cases_sha256=common.sha(root/'cases.json')))
    common.write_json(root/'source_hashes.json',{str(p.relative_to(root)):common.sha(p) for p in (root/'source').rglob('*') if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('step',choices=['prepare','setup','sample'])
    p.add_argument('--output',required=True,type=Path);p.add_argument('--r18',type=Path);p.add_argument('--r19',type=Path)
    p.add_argument('--set',choices=['calib','heldout'],default='calib');p.add_argument('--predictions',type=Path)
    a=p.parse_args();root=a.output.resolve()
    if a.step=='prepare':prepare(root,a.r18.resolve(),a.r19.resolve());return
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Slurm required')
    env=common.identity();gpu=env['gpu'].split(',')[0].strip()
    locks=[open('/tmp/gh200-measurement-'+gpu+'.lock','a'),(root/'.run.lock').open('a')]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    common.verify(root)
    if a.step=='setup':run_r18.setup(root);return
    if env!=json.loads((root/'environment.json').read_text()):raise ValueError('calibration GPU/allocation changed')
    if common.sha(root/'cases.json')!=json.loads((root/'run_config.json').read_text())['cases_sha256']:raise ValueError('matrix changed')
    rows=[r for r in json.loads((root/'cases.json').read_text()) if r['set']==a.set]
    if a.set=='heldout':
        if not a.predictions:raise ValueError('frozen predictions required')
        frozen=json.loads(a.predictions.read_text());now=time.time_ns()
        if frozen['status']!='frozen' or frozen['gpu']!=env['gpu'] or not frozen['frozen_unix_ns']<now:
            raise ValueError('invalid time/card prediction freeze')
        if set(frozen['predictions'])!={r['id'] for r in rows}:raise ValueError('frozen targets mismatch')
        if a.predictions.stat().st_mode&0o222:raise ValueError('predictions must be read-only')
        binding=root/'prediction_binding.json'
        if binding.exists():
            if json.loads(binding.read_text())['sha256']!=common.sha(a.predictions):raise ValueError('prediction changed on resume')
        else:
            if any((root/'samples'/r['id']).exists() for r in rows):raise ValueError('target samples predate binding')
            common.write_json(binding,dict(sha256=common.sha(a.predictions),frozen_unix_ns=frozen['frozen_unix_ns'],first_sample_unix_ns=now,gpu=env['gpu']))
    for trial in range(10):
        group=rows[:];random.Random(20261008+trial).shuffle(group)
        for row in group:
            variants=['plain','stamped'];random.Random(20261008+trial).shuffle(variants)
            for variant in variants:run_r18.run_one(root,row,variant,trial)
        print('V07',a.set,'trial',trial,'complete',flush=True)
    (root/f'nvidia-smi-after-{a.set}.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
if __name__=='__main__':main()
