from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,subprocess
out=Path(__file__).resolve().parent;other=out.parent/'B-20261011-shared-path-prefix-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(other/'B-shared-path-prefix.json');b=read(out/'B-shared-path-prefix.json')
resolved=[]
for r in (a,b):
 for path,digest in r['input_sha256'].items():
  if sha(Path(path))==digest:continue
  assert path.endswith('/0785/CUDA_optimazation/microbench/gh200_resource_campaign/access_rules/supply_model.py')
  blob=subprocess.check_output(['git','show','043d06b8:microbench/gh200_resource_campaign/access_rules/supply_model.py'])
  assert hashlib.sha256(blob).hexdigest()==digest
  (out/'B-frozen-supply_model.py').write_bytes(blob)
  resolved.append(dict(path=path,source_commit='043d06b8',sha256=digest,reason='B worktree source advanced; verify original commit blob instead'))
assert {k:v for k,v in a.items() if k!='input_sha256'}=={k:v for k,v in b.items() if k!='input_sha256'}
assert (other/'B-initial-demands.json.gz').read_bytes()==(out/'B-initial-demands.json.gz').read_bytes()
common=set(a['input_sha256'])&set(b['input_sha256']);assert all(a['input_sha256'][k]==b['input_sha256'][k] for k in common)
assert {Path(k).name:v for k,v in a['input_sha256'].items() if k not in common}=={Path(k).name:v for k,v in b['input_sha256'].items() if k not in common}
with gzip.open(out/'B-initial-demands.json.gz','rt') as f:raw=json.load(f)
f=lambda x:F.from_float(x) if isinstance(x,float) else F(x)
assert len(raw)==2376 and sum(f(r['work']) for r in raw)==1584
for cta in range(132):
 rows=sorted([r for r in raw if r['cta']==cta],key=lambda r:r['arrival_ns'])
 assert len(rows)==18
 assert [r['index'] for r in rows]==[g for g in range(6) for _ in range(3)]
 assert [r['source_ready_ns'] for r in rows]==sorted(r['source_ready_ns'] for r in rows)
 if cta==55:assert all(r['merged'] for r in rows)
tau=f(b['parameters']['read_latency_ns']);service=f(b['parameters']['destination_service_ns'])
def direct_uncompressed(placement,delta,price):
 rows=[]
 for r in raw:
  target=r['cta']==55
  if delta and target and r['index']>0:continue
  issue=f(r['arrival_ns'])+(delta if target else 0)
  ready=f(r['source_ready_ns'])
  arrival=max(ready,issue+tau) if placement=='return' else issue
  tie=(int(target),r['demand_id']) if delta else (0,r['demand_id'])
  rows.append((arrival,tie,r,ready))
 rows.sort(key=lambda x:(x[0],x[1]))
 # Prefix-sum max formula for the shared server, not a recurrent frontier implementation.
 work=F(0);maximum=F(0);local=F(0);full={}
 for arrival,tie,r,ready in rows:
  maximum=max(maximum,arrival-work*price)
  work+=f(r['work']);finish=work*price+maximum
  if r['cta']==55:
   incoming=finish if placement=='return' else max(ready,finish+tau)
   local=max(local,incoming)+service*f(r['work'])
   if r['kind']=='issue_b1':full[r['index']]=local
 return full
checks=0;witnesses=[]
for r in b['results']:
 envelopes={k:[(F(v['slope']),F(v['intercept'])) for v in values] for k,values in r['baseline_envelopes'].items()}
 points={F(0),F(292,165)}
 for lines in envelopes.values():
  points.update((x[1]-y[1])/(y[0]-x[0]) for x,y in zip(lines,lines[1:]))
 for d in r['delta_results']:
  points.add(F(d['maximum_at_price']))
  for region in d['joint_price_intervals']:points.update((F(region['lower']),F(region['upper'])))
 ordered=sorted(x for x in points if x>=0)
 points=set(ordered)|{(x+y)/2 for x,y in zip(ordered,ordered[1:])}|{ordered[-1]+1}
 for price in points:
  full=direct_uncompressed(r['placement'],F(0),price)
  for g,v in full.items():assert v==max(s*price+c for s,c in envelopes[f'full{g}'])
  checks+=1
 slope=envelopes['full5'][-1][0]-envelopes['full0'][-1][0]
 intercept=envelopes['full5'][-1][1]-envelopes['full0'][-1][1]
 assert slope==1320 and intercept==0
 upper=(F(r['baseline_L_upper_ns'])-11*f(b['parameters']['compute_ns']))/slope
 assert upper==F(292,165)
 for d in r['delta_results']:
  price=F(d['maximum_at_price']);delta=f(d['delta_ns'])
  response=direct_uncompressed(r['placement'],delta,price)[0]-direct_uncompressed(r['placement'],F(0),price)[0]-delta
  assert response==F(d['maximum_S_change_exact']);checks+=1
  if delta==512:witnesses.append(dict(placement=r['placement'],response=str(response),joint_regions=d['joint_price_intervals']))
report=dict(advanced_worktree_sources_resolved=resolved,common_input_hashes=len(common),path_only_modules=len(a['input_sha256'])-len(common),prefix_byte_equal=True,uncompressed_fifo_checks=checks,ordered_ctas=132,price_upper='292/165',prescribed_shift_witnesses=witnesses,scope='Independent uncompressed FIFO boundary/interior and high-price checks; complete partition projection additionally replayed by original analyzer.',observation_qualified=False)
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
