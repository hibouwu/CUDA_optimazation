#!/usr/bin/env python3
"""Plot the four reviewed-contract S08 observations, with no performance export."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run_dir.resolve();out=args.output.resolve()
    if out.exists() or out==run or run in out.parents:raise ValueError('new output directory outside run required')
    entry=run/'snapshot/repo/microbench/gh200_resource_campaign/run_accumulation_boundary.py'
    audit=subprocess.run([sys.executable,'-B',str(entry),'audit',str(run)],text=True,capture_output=True,check=True)
    checked=json.loads(audit.stdout)
    if checked['status']!='diagnostic-collected' or checked['performance_eligible'] is not False:raise ValueError('unqualified diagnostic identity')
    rows=[]
    for n in (31,32,33,64):
        key='original_uniform_'+str(n);evidence=checked['evidence'][key];r=evidence['result']
        rows.append({'iterations':n,'mathematical_expected':r['mathematical_expected'],
            'minimum':r['minimum'],'maximum':r['maximum'],'output_count':r['output_count'],
            'difference_count':r['difference_count'],'raw_sha256':evidence['raw_sha256']})
    out.mkdir(parents=True)
    (out/'frozen-audit.json').write_text(audit.stdout)
    with (out/'observations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='gh200-boundary-v1'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for axis,selected,title in [(axes[0],rows,'All four lengths'),(axes[1],rows[:3],'Boundary detail')]:
        x=[r['iterations'] for r in selected]
        axis.plot(x,[r['mathematical_expected'] for r in selected],'o--',label='Mathematical reference')
        lo=[r['minimum'] for r in selected];hi=[r['maximum'] for r in selected]
        axis.errorbar(x,lo,yerr=[[0]*len(lo),[b-a for a,b in zip(lo,hi)]],fmt='s',capsize=4,label='Observed output range')
        axis.set(xlabel='Iterations (16 MMA per chain per iteration)',ylabel='Accumulator output value',title=title)
        axis.set_xticks([31,40,50,64] if len(selected)==4 else x);axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.suptitle('FP8 WGMMA diagnostic: one process per length, 8192 outputs per process',fontsize=11)
    fig.savefig(out/'boundary.png',dpi=170);fig.savefig(out/'boundary.svg',metadata={'Date':None});plt.close(fig)
    shutil.copy2(Path(__file__),out/'report_boundary.py')
    (out/'index.md').write_text('''# S08 四点诊断图表

每点只有一次 target 启动。图中 observed range 表示同次启动的全部8192个输出范围；不代表独立重复采样误差或置信区间。数学参考按每个输出所属的一条链计算，不能再乘chain数。

本图不导出性能参数，也不从平台期推断内部位宽。原始值、逐位置差分和链比较在完整run中，frozen-audit.json记录本次只读重放。
''')
    manifest={'kind':'boundary_diagnostic_analysis','performance_eligible':False,
      'matplotlib_version':matplotlib.__version__,
      'input_run_files':{n:sha(run/n) for n in ['run_spec.json','diagnostic_manifest.json','diagnostic_summary.json','boundary_aggregate.json']},
      'output_files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},'raw_sources':[r['raw_sha256'] for r in rows]}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'diagnostic_analysis_generated','output':str(out),'performance_eligible':False}))


if __name__=='__main__':main()
