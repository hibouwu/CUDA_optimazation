#!/usr/bin/env python3
"""R12 complete windows with phase-matched controls; raw process medians."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
a=p.parse_args();root=a.input.resolve();d=json.loads((root/'analysis/rules.json').read_text())
rs=[json.loads(p.read_text()) for p in (root/'samples').glob('*/*/result.json')]
controls=[]
for g in d['joint_service']:
 mine=[r for r in rs if r['pair']==g['pair'] and r['swizzle']==g['swizzle']]
 c={label:statistics.median(r['elapsed_cycles'] for r in mine if r['label'].startswith(label+'-')) for label in ['b-first','a-second','empty']}
 c.update(pair=g['pair'],swizzle=g['swizzle'],a_first=g['a'],b_second=g['b'],concurrent=g['concurrent'],
  phase_matched_max=max(g['a'],c['b-first']),concurrent_excess=g['concurrent']/max(g['a'],c['b-first'])-1)
 controls.append(c)
(root/'analysis/phase-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(15,4.2));names=['TMA + WGMMA','WGMMA + STS.128','TMA + STS.128']
for pair,ax in enumerate(axes):
 for offset,sw in enumerate([0,1]):
  g=next(g for g in d['joint_service'] if g['pair']==pair and g['swizzle']==sw)
  val=[g[k]/1000 for k in ['a','b','serial','concurrent']]
  ax.bar([i+(offset-.5)*.35 for i in range(4)],val,.35,label='SW128' if sw else 'INTER')
 ax.set_xticks(range(4),['A','B','serial','concurrent']);ax.set_ylabel('complete window, thousand SM cycles')
 ax.set_title(names[pair]);ax.legend()
fig.tight_layout();out=root/'analysis/plots';out.mkdir(exist_ok=True);fig.savefig(out/'r12-paths.png',dpi=160)
(out/'source.json').write_text(json.dumps(dict(source='../rules.json',statistic='process median',
 errors='CV in cases.csv; no error bars',controls='../phase-controls.json'),indent=2)+'\n')
print(out/'r12-paths.png');print(json.dumps(controls,indent=1))
