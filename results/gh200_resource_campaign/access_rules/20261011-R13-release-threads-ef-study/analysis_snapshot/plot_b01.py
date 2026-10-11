#!/usr/bin/env python3
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--micro',required=True,type=Path);p.add_argument('--cache',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
rows=json.loads((a.micro/'reanalysis/local-independent/rules.json').read_text())['conditions'];cache=json.loads((a.cache/'reanalysis/local-independent/rules.json').read_text())['conditions']
fig,axes=plt.subplots(1,3,figsize=(13,4.1))
for ax,rr,labels,title in [(axes[0],rows[:2],['1 warp','8 warps'],'Independent LDS.128 + STS.128'),(axes[1],rows[2:],['Input','Output'],'Tensor transport, 128 MiB addresses')]:
 values=[]
 for row in rr:
  records=[json.loads(p.read_text()) for p in (a.micro/'samples'/row['case']).glob('formal-*/result.json')]
  values.append([row['logical_bytes']/r['cycles'] for r in records])
 ax.bar(range(2),[statistics.median(v) for v in values],yerr=[statistics.pstdev(v) for v in values],capsize=4,color=['#287491','#b8613d'])
 ax.set(xticks=range(2),xticklabels=labels,title=title,ylabel='Requested B / SM cycle / CTA')
axes[2].bar(range(4),[r['microseconds'] for r in cache],yerr=[r['stddev'] for r in cache],capsize=4,color=['#287491','#b8613d']*2)
axes[2].set(xticks=range(4),xticklabels=['sw1 repeat','sw1 evict','sw8 repeat','sw8 evict'],title='CUTLASS 4096 x 4096 x 2048',ylabel='Plain complete GEMM (us)')
axes[2].tick_params(axis='x',rotation=25)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160)
