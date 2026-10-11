from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,statistics,sys
out=Path(__file__).resolve().parent;run=out.parents[1];base=run.parent;repo=run.parents[3]
sys.path.insert(0,str(repo/'microbench/gh200_resource_campaign/access_rules'))
from cfg_c_native_compute import compute_reference
inputs={}
def read(p):inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
manifest=read(run/'manifest.json');verification=read(run/'verification.json')
reference=dict(protocol='cfg-c-mma-reference-v1',iterations=1024,mma_shape=[64,128,16],warpgroups=2,chains_per_wg=2,per_chain_batch=4,b_major='MN',wait_group=1,gpu_uuid=manifest['gpu_uuid'],cycle_samples=[])
for sample in read(run/'returned/records.json'):
 if sample['iterations']!=1024:continue
 p=run/'returned/samples'/sample['id']/'stdout.json';v=read(p)
 assert inputs[str(p)]==sample['files'][str(p.relative_to(run/'returned'))]
 assert v['gpu_uuid']==reference['gpu_uuid'] and v['work_flop']==4294967296
 reference['cycle_samples'].append(v['elapsed'])
reference['source_manifest_sha256']=inputs[str(run/'manifest.json')]
(out/'reference.json').write_text(json.dumps(reference,indent=2)+'\n')
target=base/'20261011-cfg-c-producer-tail-off-off';results=[];predictions=[]
for sample in read(target/'returned/records.json'):
 p=target/'returned'/sample['raw'];inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();assert inputs[str(p)]==sample['sha256']
 with gzip.open(p,'rt') as f:events={v['event']:v for v in map(json.loads,f)}
 prediction=compute_reference(events['setup'],reference);predictions.append(prediction)
 predicted=F(*prediction['reference_cycles_fraction']);t=events['call']['trace']
 for cta in range(132):
  h=cta*1296;assert t[h+11]==t[h+12]==t[h+2]
  for j in range(t[h+6]):
   a=h+16+10*j;b=a+640;assert t[a+4:a+6]==t[b+4:b+6]
   observed=max(t[a+1],t[b+1])-min(t[a],t[b]);error=predicted/observed-1
   results.append(dict(process=sample['id'],cta=cta,ordinal=j,observed_cycles=observed,predicted_cycles=str(predicted),relative_error=str(error)))
rejected=[]
for name in ('20261009-R13-R15-R18-composition-job738496','20261009-R18-first-fill-job738865'):
 for row in read(base/name/'static_setup.json'):
  if row['setup']['config']!='cfg_c':continue
  try:compute_reference(row['setup'],reference)
  except ValueError as e:rejected.append(dict(run=name,case=row['case'],gpu_uuid=row['setup']['gpu_uuid'],reason=str(e)))
  else:raise AssertionError('unexpected supported historical case')
assert len(results)==4320 and len(rejected)==15
with gzip.open(out/'windows.json.gz','wt') as f:json.dump(results,f)
errors=[abs(F(r['relative_error'])) for r in results]
code=repo/'microbench/gh200_resource_campaign/access_rules/cfg_c_native_compute.py';inputs[str(code)]=hashlib.sha256(code.read_bytes()).hexdigest()
report=dict(input_sha256=inputs,reference=reference,target_processes=20,target_windows=4320,predicted_cycles=predictions[0]['reference_cycles_fraction'],median_absolute_relative_error=float(statistics.median(errors)),maximum_absolute_relative_error=float(max(errors)),historical_rejections=rejected,observation_qualified=False,prediction_qualified=False,limits=['Development same-card compute-envelope hypothesis; targets were already revealed before reference selection.', 'No target residual, overhead subtraction, clock conversion, source price or output price fitted.', 'Independent instruction reference is SM124 resident repeated operands; application to all target SMs remains a transfer hypothesis.', 'The 15 historical cfg_c conditions on GPU432 are preserved as unsupported direct price transfers, not silently rescored.', 'No whole-time, critical-CTA or full cfg_a/b/c support implied.'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','historical_rejections','reference')}))
