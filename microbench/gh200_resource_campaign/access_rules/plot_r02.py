#!/usr/bin/env python3
"""Plots from R02 process records; error bars are process standard deviations."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',required=True,type=Path)
a=p.parse_args();root=a.input.resolve()
records=[json.loads(p.read_text()) for p in (root/'samples').glob('*/*/result.json')]
output=root/'analysis/plots';output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(1,3,figsize=(15,4.2))
for offset,warps in enumerate([1,4]):
 values=[];errors=[]
 for source in ['constant_multiplier','reuse_pair','rotate_eight_pairs']:
  cid=f'source_{source}_c8_w{warps}'
  sample=[r for r in records if r['case_id']==cid and r['label'].startswith('final-')]
  rate=[r['threads']*r['steps']*2/r['consumed_cycles'] for r in sample]
  values.append(statistics.median(rate));errors.append(statistics.pstdev(rate))
 axes[0].bar([i+(offset-.5)*.35 for i in range(3)],values,.35,yerr=errors,label=f'{warps} warp')
axes[0].set_xticks([0,1,2],['constant','reuse pair','8 pairs'])
axes[0].set_ylabel('producer FLOP / SM cycle / CTA')
axes[0].set_title('FFMA, 8 accumulator chains');axes[0].legend()
for idx,family in enumerate(['fadd_f32','integer_u64']):
 tails=[]
 for pool in [8,32]:
  cid=f'pool_{family}_p{pool}_w1'
  sample=[r for r in records if r['case_id']==cid and r['label'].startswith('formal-')]
  tails.append(statistics.median(r['stamps'][0][2]-r['stamps'][0][1] for r in sample))
 axes[1].plot([8,32],tails,'o-',label=family)
axes[1].set_xticks([8,32]);axes[1].set_xlabel('retained destinations')
axes[1].set_ylabel('observed consumer tail, cycles');axes[1].legend()
axes[1].set_title('Consumer cost, not RF write bandwidth')
for offset,banked in enumerate([False,True]):
 times=[]
 for org in ['same_warp','split_warps']:
  cid=f'mixed_fadd_lds128_{org}_consume'
  prefix='banked-final-' if banked else 'final-'
  sample=[r for r in records if r['case_id']==cid and r['label'].startswith(prefix)]
  times.append(statistics.median(r['consumed_cycles'] for r in sample)/1000)
 axes[2].bar([i+(offset-.5)*.35 for i in range(2)],times,.35,label='lane stride 128 B' if banked else 'lane stride 16 B')
axes[2].set_xticks([0,1],['same warp','split warps'])
axes[2].set_ylabel('complete window, thousand SM cycles')
axes[2].set_title('Same code and returned values; address control');axes[2].legend(fontsize=8)
fig.tight_layout();fig.savefig(output/'r02-service.png',dpi=160)
(output/'source.json').write_text(json.dumps(dict(source='samples/*/*/result.json',
    error_bars='population standard deviation across independent processes',
    figures=['r02-service.png']),indent=2)+'\n')
print(output/'r02-service.png')
