from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,statistics
out=Path(__file__).resolve().parent;run=out.parents[1]
inputs={};packets={};series={}
for record in json.loads((run/'returned/records.json').read_text()):
 raw=run/'returned'/record['raw'];digest=hashlib.sha256(raw.read_bytes()).hexdigest();assert digest==record['sha256'];inputs[str(raw)]=digest
 with gzip.open(raw,'rt') as f:events={v['event']:v for v in map(json.loads,f)}
 t=events['call']['trace'];assert events['call']['producer_tail_observer']==0
 for cta in range(132):
  head=cta*1296
  for wg in range(2):
   previous=t[head+3]
   for j in range(t[head+6+wg*3]):
    k=head+16+wg*640+10*j;v=t[k:k+4]
    for kind,value in (('L',v[1]-v[0]),('W',v[2]-v[1]),('E',v[3]-v[2]),('GAP',v[0]-previous)):
     assert value>=0
     series.setdefault((cta,wg,j,kind),[]).append(dict(pair=record['pair'],arm=record['arm'],cycles=value,sm=t[head+2]-1))
    previous=v[3]
   series.setdefault((cta,wg,-1,'TAIL'),[]).append(dict(pair=record['pair'],arm=record['arm'],cycles=t[head+4+wg*3]-previous,sm=t[head+2]-1))
results=[];summary={}
for (cta,wg,j,kind),rows in sorted(series.items()):
 assert len(rows)==20
 by={(r['pair'],r['arm']):r for r in rows};deltas=[];relative=[]
 for pair in range(10):
  a,b=by[pair,0],by[pair,1];assert a['sm']==b['sm']
  deltas.append(b['cycles']-a['cycles'])
  relative.append(str(F(b['cycles']-a['cycles'],a['cycles'])) if a['cycles'] else None)
 low,high=min(r['cycles'] for r in rows),max(r['cycles'] for r in rows)
 # One positive fixed prediction versus all positive observations: harmonic center.
 floor=F(high-low,high+low) if low>0 else None
 optimum=F(2*high*low,high+low) if low>0 else None
 if floor is not None:assert max(abs(optimum/F(r['cycles'])-1) for r in rows)==floor
 result=dict(cta=cta,wg=wg,ordinal=j,kind=kind,samples=rows,signed_pair_cycles=deltas,pair_relative_changes=relative,min_cycles=low,max_cycles=high,constant_prediction_minimax_relative=str(floor) if floor is not None else None,oracle_constant_cycles=str(optimum) if optimum is not None else None)
 results.append(result)
for kind in ('L','W','E','GAP','TAIL'):
 rows=[r for r in results if r['kind']==kind]
 delta=[abs(x) for r in rows for x in r['signed_pair_cycles']]
 rel=[abs(F(x)) for r in rows for x in r['pair_relative_changes'] if x is not None]
 floors=[F(r['constant_prediction_minimax_relative']) for r in rows if r['constant_prediction_minimax_relative'] is not None]
 summary[kind]=dict(series=len(rows),paired_windows=len(delta),absolute_cycle_difference_median=statistics.median(delta),absolute_cycle_difference_max=max(delta),relative_pair_difference_median=float(statistics.median(rel)) if rel else None,relative_pair_difference_max=float(max(rel)) if rel else None,zero_reference_pairs=sum(x is None for r in rows for x in r['pair_relative_changes']),constant_oracle_floor_median=float(statistics.median(floors)) if floors else None,constant_oracle_floor_max=float(max(floors)) if floors else None,positive_series_with_floor_above_ten_percent=sum(x>F(1,10) for x in floors),zero_minimum_series=sum(r['min_cycles']==0 for r in rows))
with gzip.open(out/'series.json.gz','wt') as f:json.dump(results,f)
report=dict(input_sha256=inputs,processes=20,summary=summary,observation_qualified=False,limits=['Cycle-domain retrospective diagnosis, not replacement of failed ns gates or new acceptance.', 'Pair deltas use within-SM intervals and preserve all roles, outputs and tails. Zero intervals retained without invented relative errors.', 'Fixed-value oracle reads all observations; its empirical floor only excludes one constant per coordinate with unchanged exposed static inputs.', 'Complete measured histories differ and may be inputs to richer models; this does not prove a general unpredictability bound.', 'TAIL is consumer FINAL minus last DONE, not producer-tail precision. No conversion to ns or physical clock claim.'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))
