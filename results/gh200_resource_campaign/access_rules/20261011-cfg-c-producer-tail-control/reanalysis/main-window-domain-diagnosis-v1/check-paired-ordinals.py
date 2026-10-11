from pathlib import Path
from fractions import Fraction as F
import gzip,json,statistics
out=Path(__file__).resolve().parent
with gzip.open(out/'windows.json.gz','rt') as f:rows=json.load(f)
indexed={(r['pair'],r['cta'],r['wg'],r['ordinal']):r for r in rows}
summary=[]
for pair in range(10):
 contrasts=[];opposite=0
 for r in rows:
  if r['pair']!=pair or r['ordinal']!=1:continue
  first=indexed[pair,r['cta'],r['wg'],0]
  a=F(first['ratio_change']);b=F(r['ratio_change'])
  contrasts.append((1+b)/(1+a)-1)
  opposite+=(r['on_ns']-r['off_ns'])*(first['on_ns']-first['off_ns'])<0
 assert len(contrasts)==168
 summary.append(dict(pair=pair,matched_CTA_roles=168,opposite_ns_shift_roles=opposite,second_vs_first_ratio_change_median=float(statistics.median(contrasts)),minimum=float(min(contrasts)),maximum=float(max(contrasts))))
result=dict(matched_two_output_role_pairs=1680,groups=summary,observation_qualified=False,limits='Exact observed ratio contrasts, no tolerance-derived physical clock rejection. No scalar correction fitted or applied.')
(out/'ordinal-contrast.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
