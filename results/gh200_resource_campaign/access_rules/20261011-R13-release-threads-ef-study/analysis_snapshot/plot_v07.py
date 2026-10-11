#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();d=json.loads(a.input.read_text());rows=d['cases']
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={'width_ratios':[1.35,1]})
colors={'cfg_a':'#287491','cfg_b':'#b8613d','cfg_c':'#786298'}
ax.barh(range(len(rows)),[r['us_error']*100 for r in rows],color=[colors[r['config']] for r in rows])
ax.set_yticks(range(len(rows)));ax.set_yticklabels([r['case'] for r in rows],fontsize=8);ax.invert_yaxis()
for x in [-10,10]:ax.axvline(x,color='#a74330',ls='--',lw=1)
for x in [-5,5]:ax.axvline(x,color='#8c979e',ls=':',lw=1)
ax.axvline(0,color='#344b5a',lw=.6);ax.set(xlim=(-11,11),xlabel='Frozen complete-time error (%)',title='All 24 new conditions; no post-hoc refit')
phase=[('Critical CTA',d['critical_cycles']),('Supply',d['component_assessment']['supply']['statistics']),('Mainloop',d['component_assessment']['mainloop']['statistics']),('Last epilogue',d['component_assessment']['last_epilogue']['statistics']),('Handoff h (17)',d['component_assessment']['handoff_interval']['statistics'])]
y=range(len(phase));bx.barh([v-.18 for v in y],[r['median']*100 for _,r in phase],.34,label='Median absolute error',color='#287491');bx.barh([v+.18 for v in y],[r['maximum']*100 for _,r in phase],.34,label='Maximum absolute error',color='#b8613d')
bx.set_yticks(list(y));bx.set_yticklabels([k for k,_ in phase]);bx.invert_yaxis();bx.axvline(10,color='#8c979e',ls=':');bx.axvline(20,color='#a74330',ls='--');bx.set(xlabel='Stage error (%)',title='Component criteria are independent');bx.legend(fontsize=8,loc='lower right')
for axis in [ax,bx]:axis.grid(axis='x',alpha=.15)
fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160)
