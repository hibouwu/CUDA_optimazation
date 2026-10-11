from pathlib import Path
from fractions import Fraction as F
import json,gzip,hashlib
out=Path(__file__).resolve().parent;run=out.parents[1]
a=run/'reanalysis/A-switch-multiplier-transfer-v1'
report=json.loads((a/'transfer.json').read_text());inputs={};rows=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for trial in range(10):
 path=run/'samples/r09_history_common_sw8'/f'ends-{trial:02d}.json'
 record=json.loads(path.read_text());raw=run/record['raw']
 for p in (path,raw):
  inputs[str(p)]=sha(p);assert inputs[str(p)]==report['input_sha256'][str(p)]
 assert sha(raw)==record['raw_sha256']
 with gzip.open(raw,'rt') as f:packets=[json.loads(line) for line in f if line.strip()]
 assert len(packets)==len({p['event'] for p in packets})
 events={p['event']:p for p in packets};call=events['call']
 assert call['warmup_swizzle']==1 and len(call['warmup_us'])==16
 assert call['warmup_us']==record['warmup'] and call['elapsed_us']==record['elapsed_us']
 previous=F(str(call['warmup_us'][-1]));target=F(str(call['elapsed_us']))
 rows.append(dict(trial=trial,previous_us=str(previous),target_us=str(target),q=previous/target))
lo=min(rows,key=lambda r:r['q']);hi=max(rows,key=lambda r:r['q'])
lower=F(9,10)/lo['q'];upper=F(11,10)/hi['q'];assert lower>upper
optimum=2/(lo['q']+hi['q']);floor=(hi['q']-lo['q'])/(hi['q']+lo['q'])
assert max(abs(optimum*r['q']-1) for r in rows)==floor
assert floor==F('1279801964400078195/12163953950764594318')
assert (lo['trial'],hi['trial'])==(9,5)
result=dict(input_sha256=inputs,raw_processes=10,ten_percent_multiplier_lower=str(lower),ten_percent_multiplier_upper=str(upper),ten_percent_feasible=False,exact_minimax_error=str(floor),minimax_error=float(floor),minimum_q_trial=9,maximum_q_trial=5,rows=[dict(r,q=str(r['q'])) for r in rows],scope='Same historical batch, one common positive multiplier times last prefix event. Not a general history-model or noise bound.',observation_qualified=False)
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','rows')}))
