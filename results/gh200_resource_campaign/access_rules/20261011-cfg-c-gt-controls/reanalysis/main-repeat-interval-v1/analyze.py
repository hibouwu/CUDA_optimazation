"""Conditional interval intersection across the ten existing enabled targets."""
from pathlib import Path
from fractions import Fraction as Q
import gzip,hashlib,json,statistics,sys
ROOT=Path(__file__).resolve().parent
RUN=ROOT.parents[1]
REPO=RUN.parents[3]
sys.path.insert(0,str(RUN/'analysis_snapshot'))
from decode_cfg_c_gt_bracket import decode
inputs={}
def read(path):
 data=path.read_bytes();inputs[str(path.resolve())]=hashlib.sha256(data).hexdigest();return data
def load(path):return json.loads(read(path))
manifest=load(RUN/'manifest.json');records=load(RUN/'returned/records.json')
assert [{k:r[k] for k in ('id','comparison','pair','arm','flag','command')} for r in records]==manifest['samples']
freeze=load(RUN/'freeze.json')
for name,digest in freeze['files'].items():
 assert hashlib.sha256(read(RUN/name)).hexdigest()==digest
 assert hashlib.sha256(read(RUN/'returned'/name)).hexdigest()==digest
receipt=load(RUN/'receipt.json');assert receipt['returncode']==0
assert hashlib.sha256(read(RUN/'output.tar.gz')).hexdigest()==receipt['sha256']
series={};enabled_ids=[]
for record in records:
 raw=read(RUN/'returned'/record['raw']);assert hashlib.sha256(raw).hexdigest()==record['sha256']
 assert record['returncode']==0
 if record['comparison']!='off_on' or record['flag']!=1:continue
 enabled_ids.append(record['id'])
 packets={p['event']:p for p in map(json.loads,gzip.decompress(raw).splitlines())}
 decoded=decode(packets['setup'],packets['call'],manifest['gpu_uuid']);assert decoded['record_valid']
 trace=packets['call']['trace'];shadow=packets['call']['gt_shadow']
 for cta in decoded['records']:
  for role in cta['roles']:
   for ordinal,tile in enumerate(role['tiles']):
    key=f"{cta['cta']}/{role['wg']}/{ordinal}/"+','.join(map(str,tile['work']))
    pos=cta['cta']*1296+16+role['wg']*640+ordinal*10
    pre=cta['cta']*9+role['wg']*4+ordinal*2
    low=shadow[pre+1]-trace[pos+6];high=trace[pos+7]-shadow[pre]
    assert [low,high]==tile['conditional_elapsed_gt'] and 0<low<=high
    series.setdefault(key,[]).append(dict(id=record['id'],sm=cta['sm'],low=low,high=high,
     cycles=trace[pos+1]-trace[pos],post_gt=trace[pos+7]-trace[pos+6]))
assert len(enabled_ids)==10 and len(series)==432
rows=[]
for key,windows in sorted(series.items()):
 assert [w['id'] for w in windows]==enabled_ids
 high_lower=max(w['low'] for w in windows);low_upper=min(w['high'] for w in windows)
 spread=max(0,high_lower-low_upper)
 # Every selection must span at least spread. Projecting low_upper into each
 # closed interval attains that bound, so this is the exact marginal minimum.
 selected=[max(w['low'],min(low_upper,w['high'])) for w in windows]
 assert max(selected)-min(selected)==spread
 relative_floor=Q(spread,high_lower+low_upper)
 selected_center=Q(2*high_lower*low_upper,high_lower+low_upper) if spread else Q(low_upper)
 assert max(abs(selected_center/Q(t)-1) for t in selected)==relative_floor
 box_lo=min(w['low'] for w in windows);box_hi=max(w['high'] for w in windows)
 box_center=Q(2*box_lo*box_hi,box_lo+box_hi)
 box_radius=Q(box_hi-box_lo,box_hi+box_lo)
 assert max(abs(box_center/Q(t)-1) for w in windows for t in (w['low'],w['high']))==box_radius
 budget=Q(spread,4)
 # One GT read error e expands each duration endpoint by 2e; 4e closes gap.
 common=max(Q(w['low'])-2*budget for w in windows)
 assert common<=min(Q(w['high'])+2*budget for w in windows)
 if budget:
  less=budget-Q(1,4)
  assert max(Q(w['low'])-2*less for w in windows)>min(Q(w['high'])+2*less for w in windows)
 rows.append(dict(series=key,windows=windows,same_sm=len({w['sm'] for w in windows})==1,
  common_elapsed_at_zero_extra_error=spread==0,
  minimum_elapsed_spread_at_zero_extra_error=spread,
  native_cycle_spread=max(w['cycles'] for w in windows)-min(w['cycles'] for w in windows),
  constant_relative_error_lower_bound=str(Q(spread,high_lower+low_upper)),
  retrospective_constant_box_radius=str(Q(max(w['high'] for w in windows)-min(w['low'] for w in windows),max(w['high'] for w in windows)+min(w['low'] for w in windows))),
  required_per_GT_error_for_common_elapsed=str(budget),
  witness_high_lower=next(w['id'] for w in windows if w['low']==high_lower),
  witness_low_upper=next(w['id'] for w in windows if w['high']==low_upper),
  attaining_duration_assignment=selected))
spreads=[r['minimum_elapsed_spread_at_zero_extra_error'] for r in rows]
budgets=[Q(r['required_per_GT_error_for_common_elapsed']) for r in rows]
summary=dict(enabled_records=10,series=432,windows=4320,same_sm_series=sum(r['same_sm'] for r in rows),
 constant_elapsed_infeasible_at_zero_extra_error=sum(not r['common_elapsed_at_zero_extra_error'] for r in rows),
 minimum_elapsed_spread=dict(minimum=min(spreads),median=statistics.median(spreads),maximum=max(spreads)),
 required_per_GT_error=dict(minimum=str(min(budgets)),median=str(statistics.median(budgets)),maximum=str(max(budgets))),
 constant_relative_error_lower_bound_max=str(max(Q(r['constant_relative_error_lower_bound']) for r in rows)),
 retrospective_constant_box_radius_max=str(max(Q(r['retrospective_constant_box_radius']) for r in rows)),
 native_cycle_spread=dict(minimum=min(r['native_cycle_spread'] for r in rows),median=statistics.median(r['native_cycle_spread'] for r in rows),maximum=max(r['native_cycle_spread'] for r in rows)))
report=dict(summary=summary,rows=rows,input_sha256=inputs,
 assumptions=['Clock sampling actually enclosed by its two dependent GT reads.','GT read coordinate error bounded independently by explicit e; e is not measured here.'],
 interpretation='Marginal per-coordinate feasibility only. The minimum e makes a constant duration mathematically possible; it is not an estimate or chosen accuracy budget. No claim that all coordinates admit one shared physical clock trajectory. Relative bounds/radii only contextualize absolute spread; retrospective per-coordinate constants are not proposed predictions or fitted service prices.',
 limits='Existing ten enabled records only, no new pairing/gates, no attribution to physical frequency, no observer or null subtraction, no model score or qualification.',
 observation_qualified=False,prediction_qualified=False)
(ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
