from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,statistics
out=Path(__file__).resolve().parent;run=out.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((run/'manifest.json').read_text());records=json.loads((run/'returned/records.json').read_text())
inputs={str(run/'manifest.json'):sha(run/'manifest.json'),str(run/'analysis.json'):sha(run/'analysis.json')}
packets={}
for row in records:
 raw=run/'returned'/row['raw'];assert sha(raw)==row['sha256'];inputs[str(raw)]=sha(raw)
 with gzip.open(raw,'rt') as f:events={p['event']:p for p in map(json.loads,f)}
 assert events['setup']['gpu_uuid']==manifest['gpu_uuid']
 packets[row['pair'],row['flag']]=events['call']['trace']
values=[];groups={}
for pair in range(10):
 off,on=packets[pair,0],packets[pair,1]
 for cta in range(132):
  h=cta*1296;assert off[h+2]==on[h+2]
  for wg in range(2):
   count=off[h+6+3*wg];assert count==on[h+6+3*wg]
   for j in range(count):
    i=h+16+wg*640+j*10;a=off[i:i+10];b=on[i:i+10];assert a[4:6]==b[4:6]
    c0,c1=a[1]-a[0],b[1]-b[0];n0,n1=a[7]-a[6],b[7]-b[6]
    assert min(c0,c1,n0,n1)>0
    cycle_ratio=F(c1,c0);timer_ratio=F(n1,n0);rate_ratio=cycle_ratio/timer_ratio
    assert cycle_ratio==timer_ratio*rate_ratio
    row=dict(pair=pair,cta=cta,wg=wg,ordinal=j,work=[v-1 for v in a[4:6]],off_cycles=c0,on_cycles=c1,off_ns=n0,on_ns=n1,cycle_change=str(cycle_ratio-1),ns_change=str(timer_ratio-1),ratio_change=str(rate_ratio-1))
    values.append(row);groups.setdefault((pair,wg,j),[]).append(row)
summary=[]
for (pair,wg,j),rows in sorted(groups.items()):
 r=dict(pair=pair,wg=wg,ordinal=j,windows=len(rows))
 for key in ('cycle_change','ns_change','ratio_change'):
  v=[F(x[key]) for x in rows]
  r[key]=dict(median=float(statistics.median(v)),minimum=float(min(v)),maximum=float(max(v)))
 r['median_delta_cycles']=statistics.median(x['on_cycles']-x['off_cycles'] for x in rows)
 r['median_delta_ns']=statistics.median(x['on_ns']-x['off_ns'] for x in rows)
 summary.append(r)
assert len(values)==4320
with gzip.open(out/'windows.json.gz','wt') as f:json.dump(values,f)
result=dict(input_sha256=inputs,pairs=10,window_pairs=4320,exact_identities=4320,groups=summary,observation_qualified=False,correction_applied=False,limits=['Posthoc paired differences; no clock precision, physical frequency, service or observer overhead inference.','Native differences are each within one SM epoch. Pair SM mapping unchanged.','Cycle/ns is an observed window ratio, not an independent measured conversion or prediction input.','All roles and output ordinals retained; original failed gates unchanged.'])
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
