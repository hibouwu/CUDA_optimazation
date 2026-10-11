#!/usr/bin/env python3
"""Measured R18 boundary conditions or R19 tail distributions; process SD error bars."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--family',choices=['r18','r19'],required=True)
a=p.parse_args();rows=json.loads((a.analysis/'rules.json').read_text())['conditions']
for r in rows:
 vals=[json.loads(p.read_text())['elapsed_us'] for p in (a.run/'samples'/r['id']).glob('plain-*.json') if json.loads(p.read_text())['returncode']==0]
 r['sd']=statistics.pstdev(vals)
if a.family=='r18':
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for ix,cfg in enumerate(['cfg_a','cfg_c']):
  for iy,k in enumerate([1024,8192]):
   ax=axes[ix,iy];rr=[r for r in rows if r['config'].startswith(cfg) and r['k']==k]
   labels=[('cluster1 '+r['kind']) if r['config'].endswith('1') else r['kind'] for r in rr]
   ax.bar(range(len(rr)),[r['plain_us'] for r in rr],yerr=[r['sd'] for r in rr],capsize=3,color=['#287491' if r['trace_qualified'] else '#b8613d' for r in rr])
   ax.set_xticks(range(len(rr)));ax.set_xticklabels(labels,rotation=27,ha='right',fontsize=8)
   ax.set(ylabel='Plain complete GEMM (us)',title=f'{cfg}, K={k}')
   ax.grid(axis='y',alpha=.2)
 fig.suptitle('R18: equal physical-grid controls; orange = full trace perturbation >5%',fontsize=11)
else:
 tails=json.loads((a.analysis/'tails.json').read_text());fig,axes=plt.subplots(1,3,figsize=(14,4.5))
 for sw,color in [(1,'#287491'),(8,'#b8613d')]:
  rr=[r for r in rows if r['swizzle']==sw]
  x=[i+(-.18 if sw==1 else .18) for i in range(len(rr))]
  axes[0].bar(x,[r['plain_us'] for r in rr],.34,yerr=[r['sd'] for r in rr],capsize=3,color=color,label=f'swizzle {sw}')
  for idx,key in [(1,'max_cta_cycles'),(2,'entry_spread_ns')]:
   groups=[[t[key] for t in tails if t['case']==r['id']] for r in rr]
   axes[idx].bar(x,[statistics.median(v) for v in groups],.34,yerr=[statistics.pstdev(v) for v in groups],capsize=3,color=color)
 for ax,title,ylabel in zip(axes,['Complete plain GEMM','Maximum per-CTA duration','CTA entry spread'],['us','SM cycles','globaltimer ns']):
  ax.set(title=title,ylabel=ylabel,xticks=range(4));ax.set_xticklabels([r['group'] for r in rows if r['swizzle']==1],rotation=25,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
 axes[0].legend(fontsize=8)
fig.tight_layout();fig.savefig(a.analysis/(a.family+'-service.png'),dpi=160)
