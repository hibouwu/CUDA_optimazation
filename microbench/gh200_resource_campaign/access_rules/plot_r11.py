#!/usr/bin/env python3
"""R11 exact-work comparison; raw max and common-cost-corrected sum are references."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
a=p.parse_args();root=a.input.resolve();d=json.loads((root/'analysis/rules.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,4.2));names=['FFMA + IMAD.WIDE','LDS.128 + CVT action','WGMMA + IMAD']
labels=['A','B','AB','BA','interleave','cross']
for pair,ax in enumerate(axes):
 rows=[r for r in d['conditions'] if r['pair']==pair]
 values=[r['cycles']/1000 for r in rows]
 ax.bar(range(6),values,color=['#819fc0','#a5b5c7','#d1a977','#d1a977','#609e77','#397b56'])
 ref=next(r for r in d['joint_service'] if r['pair']==pair and r['order']==4)
 ax.axhline(ref['max']/1000,color='black',ls=':',label='max(A, B)')
 ax.axhline(ref['common_cost_corrected_sum']/1000,color='#943c3c',ls='--',label='A+B-empty')
 ax.set_xticks(range(6),labels);ax.set_title(names[pair]);ax.set_ylabel('complete window, thousand SM cycles')
 ax.legend(fontsize=8)
fig.tight_layout();out=root/'analysis/plots';out.mkdir(exist_ok=True)
fig.savefig(out/'r11-joint-service.png',dpi=160)
(out/'source.json').write_text(json.dumps(dict(source='../rules.json',count=18,statistic='process median',
 error_bars='none; CV and all raw samples in cases.csv and samples'),indent=2)+'\n')
print(out/'r11-joint-service.png')
