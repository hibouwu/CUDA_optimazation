from pathlib import Path
import gzip,hashlib,json,statistics,sys
r=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'manifest.json').read_text());data=r/'returned'
for p,h in json.loads((r/'freeze.json').read_text())['files'].items():
 assert sha(r/p)==h and sha(data/p)==h,p
for p,h in m['analysis_sha256'].items():assert sha(r/'analysis_snapshot'/p)==h,p
sys.path.insert(0,str(r/'analysis_snapshot'))
from decode_r18_first_output import decode
from analyze_r18_first_output import SELECTED_CTAS
from analyze_r18 import reference
records=json.loads((data/'records.json').read_text());assert [{k:x[k] for k in ('id','trial','flag')} for x in records]==m['samples']
assert json.loads((data/'complete.json').read_text())==dict(manifest_sha256=sha(r/'manifest.json'),records=20)
processes=[];errors=[]
for row in records:
 try:
  assert not row['returncode'];path=data/row['raw'];assert sha(path)==row['sha256']
  with gzip.open(path,'rt') as f:es={x['event']:x for line in f for x in [json.loads(line)]}
  s,c=es['setup'],es['call'];assert s['gpu_uuid']==m['gpu_uuid'] and s['input_map_m']==1024 and s['first_output_observer']==row['flag']
  decode(s,c,row['id'])
  check=es['check'];assert check['samples']==4096 and not check['nonfinite'] and not check['padding_errors']
  assert len(check['checked_values'])==4096
  for index,value in zip(check['checked_indices'],check['checked_values']):assert value==reference(index//2816,index%2816,4096,17)
  assert es['input_zero_storage']['nonzero']==0
  local=[]
  for cta in SELECTED_CTAS:
   for role in (1,2):
    head=cta*1318+16+(role-1)*640;t=c['trace'];ns=t[head+7]-t[head+6];cy=t[head+1]-t[head]
    assert ns>=0 and cy>=0 and t[head+6]>0
    local.append(dict(cta=cta,role=role,ns=ns,cycles=cy,sm=t[cta*1318+2]-1))
  processes.append(dict(row,elapsed_us=c['elapsed_us'],local=local))
 except Exception as error:errors.append(dict(id=row['id'],error=repr(error)))
def metric(values,median_limit,maximum_limit):
 a=[abs(x) for x in values]
 return dict(signed=values,median_absolute=statistics.median(a),maximum_absolute=max(a),passed=statistics.median(a)<=median_limit and max(a)<=maximum_limit)
result=dict(errors=errors,processes=processes,common_L_control_passed=False,internal_endpoints_qualified=False,observation_qualified=False)
if not errors:
 pairs=[(next(p for p in processes if p['trial']==j and p['flag']==0),next(p for p in processes if p['trial']==j and p['flag']==1)) for j in range(10)]
 g=m['gates'];local=[]
 for i in range(24):
  first=pairs[0][0]['local'][i];d=[b['local'][i]['ns']-a['local'][i]['ns'] for a,b in pairs]
  local.append(dict(cta=first['cta'],role=first['role'],**metric(d,g['per_role_L_median_absolute_ns'],g['per_role_L_maximum_absolute_ns']),sm_changes=[a['local'][i]['sm']!=b['local'][i]['sm'] for a,b in pairs]))
 event=metric([b['elapsed_us']/a['elapsed_us']-1 for a,b in pairs],g['event_median_absolute_ratio'],g['event_maximum_absolute_ratio'])
 off=[a['elapsed_us'] for a,b in pairs];cv=statistics.pstdev(off)/statistics.fmean(off)
 result.update(local=local,event=event,off_cv=cv,common_L_control_passed=all(x['passed'] for x in local) and event['passed'] and cv<=g['off_event_cv'])
(r/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('processes','local')}))
