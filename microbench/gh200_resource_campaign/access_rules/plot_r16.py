#!/usr/bin/env python3
"""Measured residency and quota services; error bars are process standard deviations."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--residency',type=Path,required=True)
p.add_argument('--quota',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
res=json.loads((a.residency/'analysis/rules.json').read_text())['conditions']
for r in res:
 values=[json.loads(p.read_text())['elapsed_ns'] for p in (a.residency/'samples'/r['case']).glob('formal-*/result.json')]
 r['stddev_ns']=statistics.pstdev(values)
quota=[]
for path in (a.quota/'samples').glob('*/formal-*/result.json'):
 r=json.loads(path.read_text());quota.append((r['delay'],r['before'],statistics.median(s[2]-s[1] for s in r['stamps'][4:]),statistics.median(s[7]-s[6] for s in r['stamps'][4:])))
fig,axes=plt.subplots(1,3,figsize=(13.4,4.1))
for large,style in [(0,'-'),(1,'--')]:
 for resident,color in [(1,'#ae5934'),(2,'#287491')]:
  rr=sorted([r for r in res if r['large']==large and r['resident']==resident],key=lambda r:r['stage'])
  axes[0].errorbar([r['stage'] for r in rr],[r['ns']/1000 for r in rr],
   yerr=[r['stddev_ns']/1000 for r in rr],fmt='o'+style,color=color,capsize=3,
   label=f'{"large" if large else "small"} source, capacity {resident}')
axes[0].set(xlabel='Software stages',ylabel='Whole-GPU envelope (us)',title='Residency: 264 CTAs',xticks=[1,2,4])
for before,color,label in [(0,'#287491','delay after release'),(1,'#ae5934','delay before release')]:
 for column,index in [(1,2),(2,3)]:
  groups=[[q[index] for q in quota if q[0]==d and q[1]==before] for d in [0,64,256]]
  axes[column].errorbar([0,64,256],[statistics.median(v) for v in groups],yerr=[statistics.pstdev(v) for v in groups],fmt='o-',capsize=3,color=color,label=label)
axes[1].set(xlabel='Producer dependent IMAD count / thread',ylabel='Consumer inc window (SM cycles)',title='Quota: request to return')
axes[2].set(xlabel='Producer dependent IMAD count / thread',ylabel='192 dependent FADD window (SM cycles)',title='After allocation: retained-value reduction')
for ax in axes:
 ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160)
