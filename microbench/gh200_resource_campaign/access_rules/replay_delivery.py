#!/usr/bin/env python3
"""Relocate and replay this GH200 delivery's changed/new evidence; no GPU is used."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
SOURCE=Path(__file__).resolve().parent
RUNS={
 'r16_quota':'20261008-R16-quota-job737122-v2',
 'r02_frame':'20261008-R02-frame-job737122-v1',
 'r18':'20261008-R18-job737122-v1',
 'r18_lite':'20261008-R18-lite-job737122-v1',
 'r19':'20261008-R19-job737122-v1',
 'b01':'20261008-B01-job737122-v4',
 'b01_cache':'20261008-B01-cache-job737122-v1',
 'v07':'20261008-V07-job737122-v1',
}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dest=a.output.resolve();dest.mkdir(parents=True,exist_ok=False);copies={};reports={};records=[]
    for key,name in RUNS.items():
        copied=dest/'archives'/name;shutil.copytree(a.archive_root/name,copied,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        copies[key]=copied;reports[key]=dest/'reports'/key
    env=dict(os.environ,PYTHONPATH=str(SOURCE));start=time.time_ns()
    def run(label,args):
        result=subprocess.run([sys.executable,*map(str,args)],cwd=dest,env=env,text=True,capture_output=True)
        (dest/(label+'.log')).write_text(result.stdout+result.stderr)
        records.append(dict(label=label,returncode=result.returncode))
        if result.returncode:raise ValueError('offline replay failed: '+label+'\n'+result.stderr[-1500:])
        print(label,'passed',flush=True)
    run('r16_quota',[SOURCE/'analyze_r16.py','--input',copies['r16_quota'],'--output',reports['r16_quota']])
    run('r02_frame',[SOURCE/'run_r02_source_frame.py','analyze','--output',copies['r02_frame'],'--analysis-output',reports['r02_frame']])
    for key in ['r18','r18_lite','r19']:run(key,[SOURCE/'analyze_r18.py','--input',copies[key],'--output',reports[key]])
    run('b01',[SOURCE/'analyze_b01.py','--input',copies['b01'],'--output',reports['b01']])
    run('b01_cache',[SOURCE/'analyze_b01_cache.py','--input',copies['b01_cache'],'--micro',copies['b01'],'--output',reports['b01_cache']])
    run('v07',[SOURCE/'analyze_v07.py','--run',copies['v07'],'--predictions',copies['v07']/'v07-predictions.json','--output',reports['v07']])
    run('frozen_predictions',[SOURCE/'reproduce_v07.py','--run',copies['v07'],'--r18-analysis',reports['r18'],'--r18-lite-analysis',reports['r18_lite'],'--output',dest/'prediction-reproduction'])
    summary=dict(status='passed',relocated_runs=RUNS,checks=records,start_unix_ns=start,stop_unix_ns=time.time_ns(),
      scope='CPU-only replay of new/changed evidence and frozen calibration; unchanged earlier R02/R11/R12/R16 residency evidence reused by identity')
    (dest/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
